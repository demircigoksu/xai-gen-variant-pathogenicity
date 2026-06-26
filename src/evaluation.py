"""
evaluation.py
=============
Deney Protokolü ve Değerlendirme (PDF Bölüm 4.1 / 4.2 / 4.3).
ADIM 5: reverse_dist_monte_carlo_cv eklendi (yarisma final dagilimi).

İşlevler
--------
- 100-Repeated Monte Carlo Cross Validation (stratified split).
- Panel bazlı metrik hesaplaması: F1-Macro, Accuracy, ROC-AUC, PR-AUC,
  Cohen's Kappa, Sensitivity, Specificity, Precision-Macro.
- Hatalı sınıflandırılan örneklerin (FP/FN) panel bazlı toplanması.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FoldResult:
    fold_id: int
    metrics: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    fp_indices: List[int] = field(default_factory=list)
    fn_indices: List[int] = field(default_factory=list)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Panel bazlı metrikleri hesaplar."""
    from sklearn.metrics import (
        f1_score,
        accuracy_score,
        roc_auc_score,
        average_precision_score,
        cohen_kappa_score,
        recall_score,
        precision_score,
        confusion_matrix,
    )

    metrics: Dict[str, float] = {}
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_pos"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision_macro"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["sensitivity"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    metrics["specificity"] = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = (
            int(tn),
            int(fp),
            int(fn),
            int(tp),
        )
    return metrics


def aggregate_metrics(folds: List[FoldResult]) -> Dict[str, Dict[str, float]]:
    """100 fold'un mean / std / min / max raporu."""
    if not folds:
        return {}
    keys = [k for k in folds[0].metrics.keys() if isinstance(folds[0].metrics[k], float)]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.array([f.metrics.get(k, np.nan) for f in folds], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[k] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
            continue
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
        }
    return out


def monte_carlo_cv(
    fit_predict_fn,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_repeats: int = 100,
    test_size: float = 0.2,
    random_state_base: int = 1000,
    progress: bool = True,
) -> List[FoldResult]:
    """
    100-Repeated Monte Carlo Cross Validation (PDF Bölüm 4.1).

    Parameters
    ----------
    fit_predict_fn : Callable
        İmza: (X_tr, y_tr, X_va, y_va) -> (y_pred, y_proba, threshold)
        - X_tr, y_tr : eğitim seti (SMOTE, fit, threshold tuning içeride yapılır)
        - X_va, y_va : doğrulama seti (sadece tahmin)
        - Geri dönüş: (np.ndarray pred, np.ndarray proba, float threshold)
    X, y : tüm panel verisi (preprocess edilmiş)
    n_repeats : iterasyon sayısı (PDF: 100)
    test_size : doğrulama oranı (default 0.2)
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(
        n_splits=n_repeats, test_size=test_size, random_state=random_state_base
    )

    folds: List[FoldResult] = []
    iterator = sss.split(X, y)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=n_repeats, desc="Monte Carlo CV")
        except ImportError:
            pass

    for i, (tr_idx, va_idx) in enumerate(iterator):
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        X_va = X.iloc[va_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        y_pred, y_proba, threshold = fit_predict_fn(X_tr, y_tr, X_va, y_va)
        metrics = compute_metrics(y_va, y_pred, y_proba)

        fp_idx = list(np.where((y_va == 0) & (y_pred == 1))[0])
        fn_idx = list(np.where((y_va == 1) & (y_pred == 0))[0])

        folds.append(
            FoldResult(
                fold_id=i,
                metrics=metrics,
                threshold=threshold,
                fp_indices=fp_idx,
                fn_indices=fn_idx,
            )
        )
    return folds



def reverse_dist_monte_carlo_cv(
    fit_predict_fn,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_repeats: int = 10,
    test_size: float = 0.2,
    target_test_benign_frac: float = 0.8,
    random_state_base: int = 2000,
    progress: bool = True,
) -> List[FoldResult]:
    """
    Reverse-distribution Monte Carlo CV (ADIM 5 / ADIM 7 duzeltmesi).

    Yarisma sartnamesi: egitim ~%80 patojenik, test ~%80 benign (TERSI).

    ADIM 7 (2026-06-21) DUZELTME — benign-starvation gideriliyor:
      Eski surum test'e neredeyse tum benign'leri cekiyordu (PAH'ta egitime
      2 benign kaliyordu -> ROC ~0.5 ARTEFAKT, gercek yarisma kosulu degil).
      Gercekte egitim seti SABIT (tum benign'ler elde), test AYRI benign-agirlikli.
      Yeni surum bunu dogru simule eder:
        1) Her sinifin yalnizca test_size kadarini test HAVUZUNA ayirir
           -> egitim orijinal orani + benign'lerin %80'ini KORUR (starvation yok).
        2) Test fold'unu: ayrilan patojenikleri TAM kullanip, ayrilan essiz
           benign'leri bootstrap (replace=True) ile cogaltarak %80 benign yapar.
           Egitime dokunulmaz, train/test ayrik (sizinti yok).
    """
    from collections import Counter

    benign_idx = np.where(y == 0)[0]
    pat_idx = np.where(y == 1)[0]

    # Her siniftan test havuzuna ayrilacak miktar (orani egitimde korur)
    n_hold_b = max(1, int(round(len(benign_idx) * test_size)))
    n_hold_p = max(1, int(round(len(pat_idx) * test_size)))
    # Test fold benign sayisi: ayrilan patojenikleri %20 yapacak sekilde bootstrap
    ratio = target_test_benign_frac / max(1e-9, (1.0 - target_test_benign_frac))

    folds: List[FoldResult] = []
    iterator = range(n_repeats)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=n_repeats, desc="ReverseDist MC")
        except ImportError:
            pass

    printed = False
    for i in iterator:
        rng = np.random.default_rng(random_state_base + i + 1)
        b_perm = rng.permutation(benign_idx)
        p_perm = rng.permutation(pat_idx)
        hold_b = b_perm[:n_hold_b]   # test havuzu benign (essiz, egitimde yok)
        hold_p = p_perm[:n_hold_p]   # test havuzu patojenik (essiz)
        train_idx = np.concatenate([b_perm[n_hold_b:], p_perm[n_hold_p:]])

        n_test_b = max(1, int(round(len(hold_p) * ratio)))
        boot_b = rng.choice(hold_b, size=n_test_b, replace=True)  # benign bootstrap
        test_idx = np.concatenate([boot_b, hold_p])
        rng.shuffle(test_idx)

        X_tr = X.iloc[train_idx].reset_index(drop=True)
        X_va = X.iloc[test_idx].reset_index(drop=True)
        y_tr = y[train_idx]
        y_va = y[test_idx]

        if not printed:
            trc = Counter(y_tr.tolist())
            te_b = int(np.sum(y_va == 0)); te_p = int(np.sum(y_va == 1))
            print(
                f"[ReverseDistCV-v2] egitim: {trc.get(1, 0)} pat / {trc.get(0, 0)} benign "
                f"(benign KORUNDU) | test: {te_b} benign + {te_p} pat "
                f"({te_b / max(1, te_b + te_p):.0%} benign; {len(hold_b)} essiz benign'den bootstrap)"
            )
            printed = True

        y_pred, y_proba, threshold = fit_predict_fn(X_tr, y_tr, X_va, y_va)
        metrics = compute_metrics(y_va, y_pred, y_proba)
        fp_idx = list(np.where((y_va == 0) & (y_pred == 1))[0])
        fn_idx = list(np.where((y_va == 1) & (y_pred == 0))[0])
        folds.append(
            FoldResult(
                fold_id=i, metrics=metrics, threshold=threshold,
                fp_indices=fp_idx, fn_indices=fn_idx,
            )
        )
    return folds
