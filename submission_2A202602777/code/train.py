"""train.py — Huấn luyện, đánh giá validation và xuất dự đoán.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import csv
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # Được chọn bằng validation trong lab_workflow.py.
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (7, 7):
        raise ValueError(f"confusion matrix phải có shape (7, 7), nhận {cm.shape}")
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    if len(X) == 0:
        return torch.empty(0, dtype=torch.int64, device=X.device)
    chunks = [
        model(X[start:start + batch_size]).argmax(dim=1)
        for start in range(0, len(X), batch_size)
    ]
    return torch.cat(chunks, dim=0).to(torch.int64)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    if len(X) != len(y) or len(y) == 0:
        raise ValueError("X và y phải có cùng số mẫu và không rỗng")
    if loss_name not in {"ce", "mse"}:
        raise ValueError(f"loss không hỗ trợ: {loss_name}")
    model.eval()
    loss_sum = 0.0
    correct = 0
    cm = torch.zeros((7, 7), dtype=torch.int64, device=y.device)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size]
            yb = y[start:start + batch_size]
            logits = model(xb)
            loss_sum += float(compute_loss(logits, yb, loss_name, reduction="sum").item())
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            encoded = yb * 7 + pred
            cm += torch.bincount(encoded, minlength=49).reshape(7, 7)
    divisor = len(y) if loss_name == "ce" else len(y) * 7
    return {
        "loss": loss_sum / divisor,
        "acc": correct / len(y),
        "macro_f1": macro_f1_from_confusion(cm.cpu().numpy()),
    }


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        targets = F.one_hot(y, num_classes=7).to(dtype=logits.dtype)
        return F.mse_loss(logits, targets, reduction=reduction)
    raise ValueError(f"loss không hỗ trợ: {loss_name}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(cfg["hidden"])
    if cfg["lr"] is None or cfg["lr"] <= 0:
        raise ValueError("cfg['lr'] phải là giá trị dương đã chọn bằng validation")
    if cfg["epochs"] <= 0 or cfg["batch"] <= 0:
        raise ValueError("epochs và batch phải lớn hơn 0")
    if cfg["loss"] not in {"ce", "mse"}:
        raise ValueError("loss phải là 'ce' hoặc 'mse'")
    if cfg["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision phải là 'fp32', 'fp16' hoặc 'bf16'")

    set_seed(int(cfg["seed"]))
    device = data["X_tr"].device
    if cfg["precision"] != "fp32" and device.type != "cuda":
        raise ValueError("Thí nghiệm mixed precision của lab yêu cầu GPU CUDA")
    if cfg["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU hiện tại không hỗ trợ BF16")

    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"]).to(device)
    expected = EXPECTED_PARAMS.get(cfg["hidden"])
    if expected is not None and count_params(model) != expected:
        raise AssertionError(f"{cfg['hidden']}: cần {expected} tham số, nhận {count_params(model)}")
    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(), lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], momentum=cfg["momentum"],
        betas=tuple(cfg.get("betas", (0.9, 0.999))), eps=cfg.get("eps", 1e-8),
    )
    scaler = torch.amp.GradScaler("cuda") if cfg["precision"] == "fp16" else None
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(cfg["precision"])
    generator = torch.Generator(device=device).manual_seed(int(cfg["seed"]))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step0 = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])
    train_size = min(50_000, len(data["X_tr"]))
    train_X, train_y = data["X_tr"][:train_size], data["y_tr"][:train_size]
    initial_train = evaluate(model, train_X, train_y, cfg["loss"])
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best = {**step0, "epoch": 0}
    history = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_acc": [],
        "val_macro_f1": [], "grad_norm": [], "grad_norm_p95": [],
        "grad_norm_max": [], "epoch_time_s": [],
    }
    diverged = not math.isfinite(step0["loss"])

    for epoch in range(1, cfg["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model.train()
        step_norms: list[float] = []
        epoch_diverged = False

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], cfg["batch"], generator):
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_dtype is not None else nullcontext()
            )
            with autocast:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
            if not torch.isfinite(loss):
                epoch_diverged = True
                break

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = clip_gradients(model.parameters(), cfg["clip_norm"])
            if not math.isfinite(grad_norm):
                epoch_diverged = True
                optimizer.zero_grad(set_to_none=True)
                break
            step_norms.append(grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - started
        train_metrics = evaluate(model, train_X, train_y, cfg["loss"])
        val_metrics = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])
        norms = np.asarray(step_norms, dtype=np.float64)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(float(norms.mean()) if norms.size else 0.0)
        history["grad_norm_p95"].append(float(np.quantile(norms, 0.95)) if norms.size else 0.0)
        history["grad_norm_max"].append(float(norms.max()) if norms.size else 0.0)
        history["epoch_time_s"].append(epoch_time)

        if math.isfinite(val_metrics["loss"]) and val_metrics["loss"] < best["loss"]:
            best = {**val_metrics, "epoch": epoch}
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        diverged = diverged or epoch_diverged or not math.isfinite(val_metrics["loss"])
        print(
            f"{cfg['exp_id']} epoch {epoch:02d}/{cfg['epochs']}: "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if diverged:
            print(f"{cfg['exp_id']}: dừng sớm do loss/gradient không hữu hạn")
            break

    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mem_mb = None
    last = {
        "train_loss": history["train_loss"][-1] if history["epoch"] else initial_train["loss"],
        "val_loss": history["val_loss"][-1] if history["epoch"] else step0["loss"],
    }
    times = history["epoch_time_s"]
    summary = {
        "step0_loss": step0["loss"], "best_val_loss": best["loss"],
        "best_epoch": best["epoch"], "final_train_loss": last["train_loss"],
        "final_val_loss": last["val_loss"], "val_acc": best["acc"],
        "val_macro_f1": best["macro_f1"],
        "time_per_epoch_s": float(np.mean(times)) if times else None,
        "peak_mem_MB": peak_mem_mb, "diverged": bool(diverged),
    }
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_id = np.asarray(row_id)
    preds = np.asarray(preds)
    if row_id.ndim != 1 or preds.ndim != 1 or row_id.size != preds.size:
        raise ValueError("row_id và preds phải là vector cùng số phần tử")
    if not np.issubdtype(row_id.dtype, np.integer) or not np.issubdtype(preds.dtype, np.integer):
        raise ValueError("row_id và pred phải là số nguyên")
    if np.unique(row_id).size != row_id.size:
        raise ValueError("row_id bị trùng")
    if np.any((preds < 0) | (preds > 6)):
        raise ValueError("pred phải nằm trong 0..6")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "pred"))
        writer.writerows(zip(row_id.astype(np.int64), preds.astype(np.int64)))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"])
    model.load_state_dict(result["best_state"])
    model.to(data["X_eval"].device)
    predictions = predict(model, data["X_eval"]).cpu().numpy().astype(np.int64)
    if len(predictions) != 116_203 or len(data["eval_row_id"]) != 116_203:
        raise ValueError("Kết quả cuối phải chứa toàn bộ 116203 mẫu eval")
    write_predictions(data["eval_row_id"], predictions, pred_path)
    print(f"Đã ghi {len(predictions)} dự đoán vào {pred_path}; tiếp theo bạn tự chạy scripts/evaluate.py.")
