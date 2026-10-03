"""plots.py — Vẽ lịch sử train và so sánh các thí nghiệm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg, history = result["cfg"], result["history"]
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    if epochs:
        axes[0].plot(epochs, history["train_loss"], label="train loss")
        axes[0].plot(epochs, history["val_loss"], label="val loss")
        axes[1].plot(epochs, history["val_acc"], label="val accuracy")
        axes[1].plot(epochs, history["val_macro_f1"], label="val macro-F1")
        axes[2].plot(epochs, history["grad_norm"], label="mean grad norm")
        best_epoch = result["summary"]["best_epoch"]
        if best_epoch in epochs:
            for axis in axes:
                axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.45)
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No completed epochs", ha="center", va="center")

    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score")
    axes[2].set(title="Gradient norm before clipping", xlabel="Epoch", ylabel="L2 norm")
    for axis in axes:
        axis.grid(alpha=0.25)
        if epochs:
            axis.legend()
    fig.suptitle(
        f"{cfg['exp_id']} | {cfg['optimizer']} | lr={cfg['lr']} | "
        f"batch={cfg['batch']} | hidden={'-'.join(map(str, cfg['hidden']))} | "
        f"dropout={cfg['dropout']} | init={cfg['init']}"
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        raise ValueError("Cần ít nhất một kết quả để vẽ")
    if any(metric not in result["history"] for result in results):
        raise ValueError(f"Không phải mọi kết quả đều có metric {metric!r}")
    fig, axis = plt.subplots(figsize=(8, 5))
    for result in results:
        history = result["history"]
        axis.plot(history["epoch"], history[metric], marker="o", markersize=3,
                  label=result["cfg"]["exp_id"])
    axis.set(title=title or metric, xlabel="Epoch", ylabel=metric)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
