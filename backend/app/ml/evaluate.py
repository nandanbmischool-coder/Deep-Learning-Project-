"""Evaluation utilities. Every number is computed from real held-out predictions (PNEUMONIA = positive class)."""

from __future__ import annotations

import html
import time
from datetime import UTC, datetime

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from app.ml.model import CLASS_NAMES, count_parameters, model_size_mb


def _div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(((y == 1) & (pred == 1)).sum()),
        "tn": int(((y == 0) & (pred == 0)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
    }


def basic_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    c = confusion(y, pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    both = len(np.unique(y)) == 2
    return {
        "accuracy": _div(tp + tn, len(y)),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": _div(tn, tn + fp),
        "f1": _div(2 * precision * recall, precision + recall),
        "roc_auc": float(roc_auc_score(y, scores)) if both else None,
        "pr_auc": float(average_precision_score(y, scores)) if both else None,
        "confusion": c,
    }


def per_class_report(c: dict[str, int]) -> dict:
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]

    def row(p, r, support):
        return {"precision": p, "recall": r, "f1": _div(2 * p * r, p + r), "support": support}

    pn = row(_div(tp, tp + fp), _div(tp, tp + fn), tp + fn)  # PNEUMONIA (positive)
    nm = row(_div(tn, tn + fn), _div(tn, tn + fp), tn + fp)  # NORMAL
    total = pn["support"] + nm["support"]
    macro = {k: (pn[k] + nm[k]) / 2 for k in ("precision", "recall", "f1")}
    weighted = {k: _div(pn[k] * pn["support"] + nm[k] * nm["support"], total) for k in ("precision", "recall", "f1")}
    return {
        "NORMAL": nm,
        "PNEUMONIA": pn,
        "macro avg": {**macro, "support": total},
        "weighted avg": {**weighted, "support": total},
    }


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, threshold: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """Percentile bootstrap (resampling test images with replacement, 95% interval)."""
    rng = np.random.default_rng(seed)
    keys = ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc")
    samples: dict[str, list[float]] = {k: [] for k in keys}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        m = basic_metrics(y[idx], scores[idx], threshold)
        for k in keys:
            if m[k] is not None:
                samples[k].append(m[k])
    return {
        k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 20 else None) for k, v in samples.items()
    } | {"method": f"percentile bootstrap, {n_boot} resamples, 95% interval"}


def _downsample(*arrays: np.ndarray, max_points: int = 200) -> list[list[float]]:
    n = len(arrays[0])
    idx = np.linspace(0, n - 1, min(n, max_points)).astype(int) if n > max_points else np.arange(n)
    return [np.asarray(a)[idx].round(5).tolist() for a in arrays]


def curves(y: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return {"roc": None, "pr": None}
    fpr, tpr, _ = roc_curve(y, scores)
    prec, rec, _ = precision_recall_curve(y, scores)
    f, t = _downsample(fpr, tpr)
    p, r = _downsample(prec, rec)
    return {"roc": {"fpr": f, "tpr": t}, "pr": {"precision": p, "recall": r}}


def calibration(y: np.ndarray, scores: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (scores >= lo) & ((scores < hi) if hi < 1 else (scores <= hi))
        if m.any():
            ece += m.mean() * abs(scores[m].mean() - y[m].mean())
    return {
        "evaluated": True,
        "brier_score": float(np.mean((scores - y) ** 2)),
        "ece_10_bins": float(ece),
        "note": "No post-hoc calibration was applied; probabilities are raw softmax scores.",
    }


def select_threshold(y_val: np.ndarray, scores_val: np.ndarray) -> tuple[float, str]:
    """Pick the threshold on VALIDATION data only: maximise Youden's J (sensitivity + specificity - 1)."""
    if len(np.unique(y_val)) < 2:
        return 0.5, "default 0.5 (validation set contained a single class)"
    fpr, tpr, thr = roc_curve(y_val, scores_val)
    finite = np.isfinite(thr)
    j = (tpr - fpr)[finite]
    best = float(thr[finite][int(np.argmax(j))])
    best = float(np.clip(best, 0.05, 0.95))
    return best, "maximises Youden's J (sensitivity + specificity - 1) on the validation set; test set not used"


@torch.no_grad()
def predict_scores(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ss = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
        ys.append(y.numpy())
        ss.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ss)


@torch.no_grad()
def measure_latency(model: torch.nn.Module, device: torch.device, size: int = 224, runs: int = 20) -> dict:
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(3):
        model(x)
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "device": device.type,
        "batch_size": 1,
        "runs": runs,
    }


def build_report(
    model, y, scores, threshold, threshold_method, device, split_name="test", seed=0, extra: dict | None = None
) -> dict:
    m = basic_metrics(y, scores, threshold)
    total, _ = count_parameters(model)
    return {
        "split": split_name,
        "created_at": datetime.now(UTC).isoformat(),
        "positive_class": "PNEUMONIA",
        "class_names": CLASS_NAMES,
        "n_samples": int(len(y)),
        "threshold": threshold,
        "threshold_method": threshold_method,
        "metrics": m,
        "per_class": per_class_report(m["confusion"]),
        "confidence_intervals": bootstrap_ci(y, scores, threshold, seed=seed),
        "curves": curves(y, scores),
        "calibration": calibration(y, scores),
        "latency": measure_latency(model, device),
        "parameter_count": total,
        "model_size_mb": round(model_size_mb(model), 2),
        **(extra or {}),
    }


def render_html(report: dict, title: str = "PneumoVision AI - Evaluation report") -> str:
    m, c = report["metrics"], report["metrics"]["confusion"]
    ci = report.get("confidence_intervals", {})

    def f(v):
        return "n/a" if v is None else f"{v:.4f}"

    def cif(k):
        v = ci.get(k)
        return "n/a" if not v else f"[{v[0]:.3f}, {v[1]:.3f}]"

    rows = "".join(
        f"<tr><td>{k}</td><td>{f(m[k])}</td><td>{cif(k)}</td></tr>"
        for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc")
    )
    pc = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{f(v['precision'])}</td><td>{f(v['recall'])}</td><td>{f(v['f1'])}</td>"
        f"<td>{v['support']}</td></tr>"
        for k, v in report["per_class"].items()
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;color:#12222a}}table{{border-collapse:collapse;margin:1rem 0}}
td,th{{border:1px solid #c9d3d6;padding:.35rem .7rem;text-align:left}}th{{background:#eef3f3}}</style></head><body>
<h1>{html.escape(title)}</h1><p>Split: <b>{report["split"]}</b> &middot; n = {report["n_samples"]} &middot; positive class: PNEUMONIA &middot;
threshold {report["threshold"]:.3f} ({html.escape(report["threshold_method"])})</p>
<h2>Metrics (95% bootstrap CI)</h2><table><tr><th>Metric</th><th>Value</th><th>95% CI</th></tr>{rows}</table>
<h2>Confusion matrix</h2><table><tr><th></th><th>Pred NORMAL</th><th>Pred PNEUMONIA</th></tr>
<tr><th>True NORMAL</th><td>TN {c["tn"]}</td><td>FP {c["fp"]}</td></tr><tr><th>True PNEUMONIA</th><td>FN {c["fn"]}</td><td>TP {c["tp"]}</td></tr></table>
<h2>Per-class report</h2><table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>{pc}</table>
<p>Parameters: {report["parameter_count"]:,} &middot; size {report["model_size_mb"]} MB &middot; median latency
{report["latency"]["median_ms"]:.1f} ms ({report["latency"]["device"]}, batch 1).</p>
<p><i>Research use only; not a clinically validated device. Created {report["created_at"]}.</i></p></body></html>"""
