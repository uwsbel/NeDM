"""NumPy-only frozen reporting metrics; no trainer or Torch import.

Exact binary_metrics extraction from scripts/traverse_fdm_train.py.
Original function SHA256: 4ce43ea449d2f4d5c0e92d8532837e68c830cd093ec60e76bd451e7856035248
Training implementation is unchanged.
"""
from __future__ import annotations
from typing import Any
import numpy as np


def binary_metrics(label: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    label = np.asarray(label, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    n = len(label)
    positive = int(label.sum())
    negative = n - positive
    result: dict[str, Any] = {"count": n, "positive": positive, "negative": negative,
                             "auroc": None, "average_precision": None, "brier": None,
                             "ece": None, "unsupported": None}
    if n == 0:
        result["unsupported"] = "no observed labels"
        return result
    result["brier"] = float(np.mean((score - label) ** 2))
    bins = []
    ece = 0.0
    for lower in np.linspace(0, 0.9, 10):
        upper = lower + 0.1
        included = (score >= lower) & ((score < upper) if upper < 0.999 else (score <= 1))
        if included.any():
            prediction, frequency = float(score[included].mean()), float(label[included].mean())
            ece += included.mean() * abs(prediction - frequency)
            bins.append({"lower": float(lower), "count": int(included.sum()), "prediction": prediction, "frequency": frequency})
    result["ece"], result["calibration_bins"] = float(ece), bins
    if positive and negative:
        order = np.argsort(score, kind="stable")
        sorted_score = score[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1]
        ends = np.r_[starts[1:], n]
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.repeat((starts + 1 + ends) / 2.0, ends - starts)
        result["auroc"] = float((ranks[label == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))
        descending = np.argsort(-score, kind="stable")
        desc_score, desc_label = score[descending], label[descending]
        end = np.r_[np.flatnonzero(np.diff(desc_score)) + 1, n]
        tp = np.cumsum(desc_label)[end - 1]
        recall, precision = tp / positive, tp / end
        result["average_precision"] = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    else:
        result["unsupported"] = "zero positive events" if not positive else "zero negative events"
    for threshold in (0.1, 0.5):
        accepted = score < threshold
        result[f"threshold_{threshold}"] = {
            "accepted": int(accepted.sum()), "false_accepts": int(label[accepted].sum()),
            "false_accept_rate": float(label[accepted].mean()) if accepted.any() else None,
            "failure_recall": float((~accepted & (label == 1)).sum() / positive) if positive else None,
        }
    return result
