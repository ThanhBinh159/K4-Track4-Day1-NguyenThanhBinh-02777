"""Notebook orchestration for the Track 4 Day 1 lab."""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from model import MLP, activation_stats, count_params, EXPECTED_PARAMS
from plots import plot_compare, plot_run
from results_table import save_result, to_row, write_xlsx
from train import DEFAULT_CFG, compute_loss, evaluate, final_eval, run_experiment, set_seed


def initialize_output(repo_root: str | Path, out_dir: str | Path) -> tuple[Path, Path, Path]:
    repo_root, out_dir = Path(repo_root).resolve(), Path(out_dir).resolve()
    figures_dir, results_dir = out_dir / "figures", out_dir / "results"
    for directory in (out_dir, figures_dir, results_dir, out_dir / "code"):
        directory.mkdir(parents=True, exist_ok=True)

    source_code, submission_code = repo_root / "code", out_dir / "code"
    if source_code.resolve() != submission_code.resolve():
        for source in source_code.glob("*.py"):
            destination = submission_code / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
        notebook = source_code / "lab.ipynb"
        destination = submission_code / notebook.name
        if notebook.is_file() and not destination.exists():
            shutil.copy2(notebook, destination)
    report = out_dir / "REPORT.md"
    template = repo_root / "templates" / "REPORT_TEMPLATE.md"
    if not report.exists() and template.is_file():
        shutil.copy2(template, report)
    return out_dir, figures_dir, results_dir


def run_health_checks(data: dict, figures_dir: str | Path) -> dict:
    figures_dir = Path(figures_dir)
    device = data["X_tr"].device
    print(f"PyTorch {torch.__version__} | device={device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    expected_counts = (371_847, 92_962, 116_203)
    actual_counts = (len(data["X_tr"]), len(data["X_val"]), len(data["X_eval"]))
    print(f"train / val / eval counts: {actual_counts}")
    if actual_counts != expected_counts:
        raise ValueError(f"Sau khi tách val cần {expected_counts}, nhận {actual_counts}")

    x_mean = data["X_tr"][:, :10].mean(dim=0).detach().cpu().numpy()
    x_std = data["X_tr"][:, :10].std(dim=0, unbiased=False).detach().cpu().numpy()
    print(f"standardized train mean max abs={np.abs(x_mean).max():.6g}")
    print(f"standardized train std range={x_std.min():.6g}..{x_std.max():.6g}")

    set_seed(42)
    model = MLP(hidden=(256, 128), dropout=0.0, init="he").to(device)
    assert count_params(model) == EXPECTED_PARAMS[(256, 128)] == 47_879
    logits = model(torch.randn(8, 54, dtype=torch.float32, device=device))
    assert logits.shape == (8, 7)
    step0 = evaluate(model, data["X_val"], data["y_val"])["loss"]
    print(f"M-base parameters={count_params(model)}, logits={tuple(logits.shape)}, step0_loss={step0:.6f} (ln 7={math.log(7):.6f})")

    model.zero_grad(set_to_none=True)
    loss = compute_loss(model(data["X_tr"][:64]), data["y_tr"][:64], "ce")
    loss.backward()
    gradient_norms = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.count_nonzero(parameter.grad).item():
            raise AssertionError(f"Gradient không chảy tới {name}")
        gradient_norms[name] = float(parameter.grad.norm().item())
    print("gradient norms:", {name: round(value, 5) for name, value in gradient_norms.items()})

    set_seed(42)
    small_x, small_y = data["X_tr"][:20], data["y_tr"][:20]
    overfit_model = MLP(hidden=(256, 128), dropout=0.0, init="he").to(device)
    optimizer = torch.optim.Adam(overfit_model.parameters(), lr=0.01)
    losses = []
    for step in range(500):
        overfit_model.train()
        optimizer.zero_grad(set_to_none=True)
        small_loss = compute_loss(overfit_model(small_x), small_y, "ce")
        small_loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == 499:
            overfit_model.eval()
            with torch.no_grad():
                losses.append(float(compute_loss(overfit_model(small_x), small_y, "ce").item()))
            if losses[-1] < 1e-4:
                break
    overfit_model.eval()
    with torch.no_grad():
        overfit_acc = float((overfit_model(small_x).argmax(1) == small_y).float().mean().item())
    print(f"20-sample overfit: final_loss={losses[-1]:.6f}, accuracy={overfit_acc:.3f}")

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(np.arange(len(losses)) * 10, losses)
    axis.set(title="M-base: overfit 20 training samples", xlabel="Update", ylabel="Cross-entropy")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures_dir / "health_overfit_20.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return {
        "step0_loss": step0, "gradient_norms": gradient_norms,
        "overfit_loss": losses[-1], "overfit_accuracy": overfit_acc,
    }


def _run_and_save(cfg: dict, data: dict, figures_dir: Path, results_dir: Path,
                  prediction: str = "") -> dict:
    cfg = {**cfg, "notes": prediction}
    print(f"\nPrediction before {cfg['exp_id']}: {prediction}")
    result = run_experiment(cfg, data)
    save_result(result, results_dir)
    plot_run(result, figures_dir / f"{cfg['exp_id']}.png")
    print(
        f"After {cfg['exp_id']}: best epoch={result['summary']['best_epoch']}, "
        f"val_acc={result['summary']['val_acc']:.4f}, "
        f"val_macro_f1={result['summary']['val_macro_f1']:.4f}, "
        f"diverged={result['summary']['diverged']}"
    )
    return result


def run_baseline_suite(data: dict, figures_dir: str | Path, results_dir: str | Path,
                       probe_epochs: int = 5, seeds=(1, 2, 3)) -> dict:
    figures_dir, results_dir = Path(figures_dir), Path(results_dir)
    grids = {
        "sgd": (0.01, 0.03, 0.1),
        "sgd_momentum": (0.01, 0.03, 0.1),
        "adam": (1e-4, 1e-3, 3e-3),
        "adamw": (1e-4, 1e-3, 3e-3),
    }
    predictions = {
        "sgd": "Without momentum, SGD may need a more careful learning-rate choice and may progress less smoothly.",
        "sgd_momentum": "Momentum should smooth updates; one of the three rates should reduce validation loss faster than the others.",
        "adam": "Adaptive per-parameter steps should make early progress across a wider range of learning rates.",
        "adamw": "With weight decay zero, AdamW should be close to Adam; the rate sweep tests this on the same split and epoch count.",
    }
    optimizer_probes = []
    best_by_optimizer = {}
    for optimizer_name, learning_rates in grids.items():
        group_results = []
        for lr in learning_rates:
            label = f"{lr:g}".replace(".", "p").replace("-", "m")
            cfg = {
                **DEFAULT_CFG, "exp_id": f"opt-{optimizer_name}-lr{label}",
                "group": "optimizer", "description": f"{optimizer_name} learning-rate probe",
                "optimizer": optimizer_name, "lr": lr, "epochs": probe_epochs,
                "seed": 1, "weight_decay": 0.0,
            }
            result = _run_and_save(cfg, data, figures_dir, results_dir, predictions[optimizer_name])
            group_results.append(result)
            optimizer_probes.append(result)
        viable = [result for result in group_results if not result["summary"]["diverged"]]
        if not viable:
            raise RuntimeError(f"Mọi learning rate của {optimizer_name} đều diverged")
        best_by_optimizer[optimizer_name] = max(viable, key=lambda result: result["summary"]["val_macro_f1"])
        print(
            f"Best {optimizer_name}: lr={best_by_optimizer[optimizer_name]['cfg']['lr']}, "
            f"val_macro_f1={best_by_optimizer[optimizer_name]['summary']['val_macro_f1']:.4f}"
        )

    baseline_lr = best_by_optimizer["sgd_momentum"]["cfg"]["lr"]
    baseline_cfg = {
        **DEFAULT_CFG, "lr": baseline_lr, "group": "baseline",
        "description": "M-base baseline; lr selected on validation",
    }
    baseline_results = []
    for seed in seeds:
        cfg = {**baseline_cfg, "seed": int(seed), "exp_id": f"base-s{seed}"}
        result = _run_and_save(
            cfg, data, figures_dir, results_dir,
            "He + Cross-Entropy + SGD momentum 0.9; same validation split and 20 epochs.",
        )
        baseline_results.append(result)

    macro_scores = np.asarray([r["summary"]["val_macro_f1"] for r in baseline_results], dtype=float)
    acc_scores = np.asarray([r["summary"]["val_acc"] for r in baseline_results], dtype=float)
    noise = {
        "val_macro_f1_mean": float(macro_scores.mean()),
        "val_macro_f1_std": float(macro_scores.std(ddof=1)) if len(macro_scores) > 1 else None,
        "val_macro_f1_2sigma": float(2 * macro_scores.std(ddof=1)) if len(macro_scores) > 1 else None,
        "val_acc_mean": float(acc_scores.mean()),
        "val_acc_std": float(acc_scores.std(ddof=1)) if len(acc_scores) > 1 else None,
    }
    print(f"Baseline val macro-F1 mean ± std: {noise['val_macro_f1_mean']:.4f} ± {noise['val_macro_f1_std'] or 0:.4f}")
    print(f"Seed-noise threshold 2σ: {noise['val_macro_f1_2sigma']}")
    if any(result["summary"]["val_acc"] <= 0.488 for result in baseline_results):
        print("Check baseline: a seed did not beat the majority-class accuracy mark (0.488).")
    return {
        "optimizer_probes": optimizer_probes, "best_by_optimizer": best_by_optimizer,
        "baseline_cfg": baseline_cfg, "baseline_results": baseline_results, "noise": noise,
    }


def run_topic_experiments(data: dict, baseline_suite: dict,
                          figures_dir: str | Path, results_dir: str | Path) -> dict:
    figures_dir, results_dir = Path(figures_dir), Path(results_dir)
    baseline_cfg = baseline_suite["baseline_cfg"]
    baseline_reference = baseline_suite["baseline_results"][0]
    results = []

    trials = [
        ("loss-mse", "loss", "MSE vs cross-entropy", {"loss": "mse"},
         "MSE may make less useful updates on confidently wrong classes; compare validation metrics and convergence, not raw loss values."),
        ("batch-2048", "hparam", "Batch size 2048", {"batch": 2048},
         "A larger batch should reduce gradient noise, but with 20 epochs it also means fewer parameter updates."),
        ("dropout-0p3", "dropout", "Dropout 0.3", {"dropout": 0.3},
         "Dropout may help if the baseline overfits; if train and validation curves stay close, it may slow learning instead."),
    ]
    for exp_id, group, description, changes, prediction in trials:
        cfg = {**baseline_cfg, **changes, "exp_id": exp_id, "group": group, "description": description, "seed": 1}
        results.append(_run_and_save(cfg, data, figures_dir, results_dir, prediction))

    p95_values = [
        value for run in baseline_suite["baseline_results"]
        for value in run["history"]["grad_norm_p95"]
        if math.isfinite(value) and value > 0
    ]
    clip_norm = float(np.median(p95_values)) if p95_values else 1.0
    high_lr = baseline_cfg["lr"] * 10
    print(f"Clipping threshold from baseline batch-grad p95 summaries: c={clip_norm:.6g}; high lr={high_lr:g}")
    clipping_prediction = "At 10x lr, the unclipped run may oscillate or diverge; clipping should limit large updates if c activates."
    for label, threshold in (("none", None), ("p95", clip_norm)):
        cfg = {
            **baseline_cfg, "exp_id": f"clip-high-lr-{label}", "group": "clipping",
            "description": f"High learning rate, clip={threshold}", "lr": high_lr,
            "clip_norm": threshold, "seed": 1,
        }
        results.append(_run_and_save(cfg, data, figures_dir, results_dir, clipping_prediction))

    device = data["X_tr"].device
    if device.type == "cuda":
        amp_trials = [("amp-fp16", "fp16")]
        if torch.cuda.is_bf16_supported():
            amp_trials.append(("amp-bf16", "bf16"))
        else:
            print("BF16 experiment skipped: this CUDA device does not support BF16.")
        for exp_id, precision in amp_trials:
            cfg = {
                **baseline_cfg, "exp_id": exp_id, "group": "amp",
                "description": f"Mixed precision {precision}", "precision": precision, "seed": 1,
            }
            results.append(_run_and_save(
                cfg, data, figures_dir, results_dir,
                "Mixed precision should reduce memory; speed and score changes must be judged from measurements.",
            ))
    else:
        print("Mixed-precision experiment skipped: the selected device is not CUDA; record this hardware limit in the report.")

    for init_name, prediction in (
        ("zeros", "Zero weights preserve symmetry; hidden-layer weight gradients should be zero and learning should fail."),
        ("normal", "Small normal weights may keep initial logits small but can make gradients/activations shrink."),
        ("xavier", "Xavier should keep activation scales reasonable, though He is tailored to ReLU layers."),
    ):
        probe = MLP(hidden=baseline_cfg["hidden"], dropout=0.0, init=init_name).to(device)
        stats = activation_stats(probe, data["X_val"][:1024])
        print(f"Init {init_name} activation stds (hidden ReLUs, output logits): {[round(v, 6) for v in stats]}")
        cfg = {
            **baseline_cfg, "exp_id": f"init-{init_name}", "group": "init",
            "description": f"Initialization {init_name}", "init": init_name, "seed": 1,
        }
        results.append(_run_and_save(cfg, data, figures_dir, results_dir, prediction))

    comparison_groups = {"optimizer": baseline_suite["optimizer_probes"]}
    for group in ("loss", "hparam", "dropout", "clipping", "amp", "init"):
        group_runs = [run for run in results if run["cfg"]["group"] == group]
        if group_runs:
            comparison_groups[group] = [baseline_reference, *group_runs]
    for group, runs in comparison_groups.items():
        plot_compare(
            runs, "val_macro_f1", figures_dir / f"compare_{group}.png",
            title=f"Validation macro-F1: {group}",
        )
    return {"results": results, "comparison_groups": comparison_groups, "clip_norm": clip_norm}


def finalize_predictions(data: dict, baseline_suite: dict, topic_suite: dict,
                         figures_dir: str | Path, results_dir: str | Path,
                         out_dir: str | Path, repo_root: str | Path) -> dict:
    figures_dir, results_dir = Path(figures_dir), Path(results_dir)
    out_dir, repo_root = Path(out_dir), Path(repo_root)
    candidates = [
        *baseline_suite["baseline_results"], *topic_suite["results"],
    ]
    candidates = [
        result for result in candidates
        if result["cfg"]["epochs"] == DEFAULT_CFG["epochs"]
        and not result["summary"]["diverged"]
        and math.isfinite(result["summary"]["val_macro_f1"])
    ]
    if not candidates:
        raise RuntimeError("Không có cấu hình 20 epoch hợp lệ để chọn bằng validation")
    selected = max(candidates, key=lambda result: result["summary"]["val_macro_f1"])
    final_cfg = {
        **selected["cfg"], "exp_id": "final-s1", "group": "final",
        "description": f"Final configuration selected by validation from {selected['cfg']['exp_id']}",
        "seed": 1,
        "notes": f"Selected on validation from {selected['cfg']['exp_id']}; retrained with seed 1.",
    }
    print(f"Selected by validation: {selected['cfg']['exp_id']} (val macro-F1={selected['summary']['val_macro_f1']:.4f})")
    final_result = _run_and_save(
        final_cfg, data, figures_dir, results_dir,
        f"Selected on validation from {selected['cfg']['exp_id']}; no eval scores used for selection.",
    )
    baseline_result = next(
        result for result in baseline_suite["baseline_results"] if result["cfg"]["seed"] == 1
    )
    final_eval(baseline_result["cfg"], baseline_result, data, out_dir / "predictions_baseline.csv")
    final_eval(final_result["cfg"], final_result, data, out_dir / "predictions_eval.csv")

    all_results = [
        *baseline_suite["optimizer_probes"], *baseline_suite["baseline_results"],
        *topic_suite["results"], final_result,
    ]
    refresh_eval_artifacts(all_results, baseline_suite, final_result, out_dir, repo_root)
    print_next_evaluation_command(out_dir)
    return {"final_result": final_result, "selected_from": selected["cfg"]["exp_id"], "all_results": all_results}


def _current_json(path: Path, prediction_path: Path) -> dict | None:
    if not path.is_file() or not prediction_path.is_file() or path.stat().st_mtime < prediction_path.stat().st_mtime:
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def refresh_eval_artifacts(all_results: list[dict], baseline_suite: dict,
                           final_result: dict, out_dir: str | Path,
                           repo_root: str | Path) -> None:
    out_dir, repo_root = Path(out_dir), Path(repo_root)
    baseline_scores = _current_json(out_dir / "eval_result_baseline.json", out_dir / "predictions_baseline.csv")
    final_scores = _current_json(out_dir / "eval_result.json", out_dir / "predictions_eval.csv")
    rows = []
    for result in all_results:
        exp_id = result["cfg"]["exp_id"]
        eval_scores = baseline_scores if exp_id == "base-s1" else final_scores if exp_id == "final-s1" else None
        rows.append(to_row(result, eval_scores=eval_scores, notes=result["cfg"].get("notes", "")))
    write_xlsx(
        rows, repo_root / "templates" / "experiment_table_template.xlsx",
        out_dir / "experiments.xlsx",
    )
    if baseline_scores is not None and final_scores is not None:
        _append_report_evidence(out_dir / "REPORT.md", baseline_suite, final_result, baseline_scores, final_scores)
    print(f"Workbook refreshed: {out_dir / 'experiments.xlsx'}")


def _append_report_evidence(report_path: Path, baseline_suite: dict,
                            final_result: dict, baseline_scores: dict,
                            final_scores: dict) -> None:
    marker = "## Kết quả đã đo (tự sinh)"
    content = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Báo cáo Lab Day 1\n"
    if marker in content:
        content = content.split(marker, 1)[0].rstrip()
    noise = baseline_suite["noise"]
    baseline_mean = noise["val_macro_f1_mean"]
    baseline_std = noise["val_macro_f1_std"]
    lines = [
        marker, "",
        f"- Baseline validation macro-F1: {baseline_mean:.4f} ± {baseline_std:.4f} (2σ={noise['val_macro_f1_2sigma']:.4f}).",
        f"- Baseline eval: accuracy={baseline_scores['accuracy']:.4f}, macro-F1={baseline_scores['macro_f1']:.4f}.",
        f"- Final configuration: `{final_result['cfg']['exp_id']}`; val macro-F1={final_result['summary']['val_macro_f1']:.4f}.",
        f"- Final eval: accuracy={final_scores['accuracy']:.4f}, macro-F1={final_scores['macro_f1']:.4f}.",
        "", "### Final per-class metrics", "", "| Class | Support | Precision | Recall | F1 |", "|---:|---:|---:|---:|---:|",
    ]
    for item in final_scores["per_class"]:
        lines.append(
            f"| {item['cls']} | {item['support']} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |"
        )
    lines.extend(["", "### Final confusion matrix", "", "```text"])
    lines.extend(" ".join(str(value) for value in row) for row in final_scores["confusion_matrix"])
    lines.extend(["```", "", "Phần diễn giải cơ chế, dự đoán trước/sau và hạn chế vẫn cần được hoàn thiện dựa trên các biểu đồ và kết quả của bạn."])
    report_path.write_text(content + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def print_next_evaluation_command(out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    baseline_done = _current_json(out_dir / "eval_result_baseline.json", out_dir / "predictions_baseline.csv") is not None
    final_done = _current_json(out_dir / "eval_result.json", out_dir / "predictions_eval.csv") is not None
    relative = out_dir.name
    command = "!python" if "google.colab" in sys.modules else "python"
    location = "Colab code cell" if command.startswith("!") else "terminal"
    if not baseline_done:
        print(f"Next {location} command (run from repo root): {command} scripts/evaluate.py --pred {relative}/predictions_baseline.csv --out {relative}/eval_result_baseline.json")
    elif not final_done:
        print(f"Next {location} command (run from repo root): {command} scripts/evaluate.py --pred {relative}/predictions_eval.csv --out {relative}/eval_result.json")
    else:
        print("Both evaluator outputs are current; run refresh_eval_artifacts(...) to update the workbook and report.")
