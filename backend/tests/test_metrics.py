import numpy as np

from app.ml import evaluate as ev


def test_confusion_and_basic_metrics():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.4, 0.2, 0.6, 0.1, 0.3, 0.7])
    m = ev.basic_metrics(y, s, 0.5)
    assert m["confusion"] == {"tp": 3, "tn": 3, "fp": 1, "fn": 1}
    assert m["accuracy"] == 6 / 8 and m["recall"] == 0.75 and m["specificity"] == 0.75 and m["precision"] == 0.75
    assert 0 < m["roc_auc"] <= 1 and 0 < m["pr_auc"] <= 1


def test_safe_when_class_absent():
    y, s = np.zeros(5, dtype=int), np.linspace(0, 1, 5)
    m = ev.basic_metrics(y, s, 0.5)
    assert m["roc_auc"] is None and m["precision"] == 0.0  # no ZeroDivisionError / NaN
    assert ev.curves(y, s) == {"roc": None, "pr": None}
    assert ev.select_threshold(y, s)[0] == 0.5


def test_threshold_from_validation_and_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 300), 0, 1)
    thr, why = ev.select_threshold(y, s)
    assert 0.05 <= thr <= 0.95 and "validation" in why
    ci = ev.bootstrap_ci(y, s, thr, n_boot=200)
    lo, hi = ci["accuracy"]
    assert lo < ev.basic_metrics(y, s, thr)["accuracy"] < hi
    c = ev.per_class_report(ev.basic_metrics(y, s, thr)["confusion"])
    assert c["NORMAL"]["support"] + c["PNEUMONIA"]["support"] == 300
