"""data.py — Nạp split, tách validation và chuẩn hóa dữ liệu.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pathlib import Path

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7
N_TRAIN = 464_809
N_EVAL = 116_203


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    processed_dir = Path(processed_dir)
    train_path, eval_path = processed_dir / "train.npz", processed_dir / "eval.npz"
    for path in (train_path, eval_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy {path}. Hãy tự chạy scripts/split_data.py từ thư mục gốc repo."
            )

    with np.load(train_path) as train, np.load(eval_path) as evaluation:
        X_train, y_train = train["X"], train["y"]
        X_eval, y_eval, eval_row_id = evaluation["X"], evaluation["y"], evaluation["row_id"]

    for name, X, y, expected_n in (
        ("train", X_train, y_train, N_TRAIN),
        ("eval", X_eval, y_eval, N_EVAL),
    ):
        if X.shape != (expected_n, N_FEATURES) or X.dtype != np.float32:
            raise ValueError(f"{name}: cần X float32 shape ({expected_n}, {N_FEATURES}); nhận {X.shape} {X.dtype}")
        if y.shape != (expected_n,) or y.dtype != np.int64:
            raise ValueError(f"{name}: cần y int64 shape ({expected_n},); nhận {y.shape} {y.dtype}")
        if not np.isfinite(X).all() or (y.min() < 0 or y.max() >= N_CLASSES):
            raise ValueError(f"{name}: dữ liệu có giá trị không hợp lệ hoặc nhãn ngoài 0..6")

    if eval_row_id.shape != (N_EVAL,) or eval_row_id.dtype != np.int64 or np.unique(eval_row_id).size != N_EVAL:
        raise ValueError("eval.npz: row_id phải là int64, đủ mẫu và không trùng")
    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction phải nằm giữa 0 và 1")
    if len(X) != len(y) or len(y) < 2:
        raise ValueError("X và y phải có cùng số mẫu và ít nhất hai mẫu")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    X_tr = np.asarray(X_tr, dtype=np.float32)
    if X_tr.ndim != 2 or X_tr.shape[1] != N_FEATURES or len(X_tr) == 0:
        raise ValueError(f"X_tr phải có shape (N, {N_FEATURES}) và không rỗng")
    mean = X_tr[:, :N_NUMERIC].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_tr[:, :N_NUMERIC].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X = np.asarray(X, dtype=np.float32)
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"X phải có shape (N, {N_FEATURES})")
    if mean.shape != (N_NUMERIC,) or std.shape != (N_NUMERIC,) or np.any(std <= 0):
        raise ValueError(f"mean/std phải có shape ({N_NUMERIC},) và std dương")
    result = X.copy()
    result[:, :N_NUMERIC] = (result[:, :N_NUMERIC] - mean) / std
    return result


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )
    mean, std = fit_standardizer(X_tr)
    X_tr, X_val, X_eval = (
        apply_standardizer(X, mean, std) for X in (X_tr, X_val, X_eval)
    )

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã chọn CUDA nhưng PyTorch không thấy GPU CUDA")
    if target_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Đã chọn MPS nhưng thiết bị MPS không khả dụng")

    data = {
        "X_tr": torch.as_tensor(X_tr, dtype=torch.float32, device=target_device),
        "y_tr": torch.as_tensor(y_tr, dtype=torch.int64, device=target_device),
        "X_val": torch.as_tensor(X_val, dtype=torch.float32, device=target_device),
        "y_val": torch.as_tensor(y_val, dtype=torch.int64, device=target_device),
        "X_eval": torch.as_tensor(X_eval, dtype=torch.float32, device=target_device),
        "y_eval": torch.as_tensor(y_eval, dtype=torch.int64, device=target_device),
        "eval_row_id": eval_row_id.copy(),
        "numeric_mean": mean,
        "numeric_std": std,
        "device": target_device,
    }
    majority_class = int(np.bincount(y_tr, minlength=N_CLASSES).argmax())
    majority_acc = float(np.mean(y_val == majority_class))
    print(
        f"train={tuple(data['X_tr'].shape)}, val={tuple(data['X_val'].shape)}, "
        f"eval={tuple(data['X_eval'].shape)}, majority_val_acc={majority_acc:.4f} "
        f"(class {majority_class})"
    )
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0")
    if len(X) != len(y):
        raise ValueError("X và y phải có cùng số mẫu")
    indices = (
        torch.randperm(len(X), generator=generator, device=X.device)
        if shuffle else torch.arange(len(X), device=X.device)
    )
    for start in range(0, len(X), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield X[batch_indices], y[batch_indices]
