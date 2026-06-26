"""
model.py
========
CatBoost modelinin sarmalayıcısı.

PDF Bölüm 3.6 / 5.1 / 5.3:
- Tabular veri için CatBoost (Ordered Boosting + L2 Reg + Early Stopping).
- CatBoost'un yerleşik kategorik handler'ı kullanılır (One-Hot YOK).
- Hiperparametre optimizasyonu CatBoost'un built-in `grid_search` mekanizmasıyla
  yapılır.
- Cost-Sensitive Learning için class_weights uygulanır.
- F1-Macro maksimize edecek karar eşiği (threshold) tuning'i yapılır.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TrainedModel:
    estimator: object
    selected_features: List[str]
    categorical_columns: List[str]
    medians: pd.Series
    best_threshold: float
    best_params: Dict
    class_weights: Dict
    train_metrics: Dict


# -----------------------------
# Hiperparametre Grid Search
# -----------------------------

def grid_search_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    categorical_columns: List[str],
    *,
    grid: Dict,
    fixed_params: Dict,
    class_weights: Optional[Dict] = None,
    cv_folds: int = 3,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    PDF 5.3: CatBoost'un yerleşik built-in grid_search mekanizması.

    Returns
    -------
    best_params : dict
    """
    from catboost import CatBoostClassifier

    cat_idx = [X.columns.get_loc(c) for c in categorical_columns if c in X.columns]

    base_params = dict(fixed_params)
    base_params.setdefault("loss_function", "Logloss")
    base_params.setdefault("eval_metric", "F1")
    base_params.setdefault("verbose", 0)
    base_params.setdefault("random_seed", random_state)
    base_params["allow_writing_files"] = False
    # Kategorik öznitelikleri yapıcıya geçiriyoruz — grid_search içindeki Pool
    # bunu okur (aksi halde AA_/CAT_ sütunlarını float'a çevirmeye çalışır).
    if cat_idx:
        base_params["cat_features"] = cat_idx
    if class_weights:
        # CatBoost class_weights expects list ordered by class label
        labels = sorted(class_weights.keys())
        base_params["class_weights"] = [class_weights[l] for l in labels]

    model = CatBoostClassifier(**base_params)
    if verbose:
        print(f"[GridSearch] grid: {grid}")
    result = model.grid_search(
        param_grid=grid,
        X=X,
        y=y,
        cv=cv_folds,
        partition_random_seed=random_state,
        verbose=False,
        plot=False,
        refit=False,
    )
    best_params = result["params"] if isinstance(result, dict) and "params" in result else result
    if verbose:
        print(f"[GridSearch] best: {best_params}")
    return best_params


# -----------------------------
# Probabilistic Kalibrasyon (Adim 2)
# -----------------------------

class PlattCalibrator:
    """
    Platt scaling (sigmoid kalibrasyon).
    Logit(raw_proba) ustunde tek-degiskenli logistic regression fit eder.
    Kucuk veri icin robust (sklearn 'sigmoid' yontemiyle ayni mantik).
    """

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        # Effectively unregularized (C cok buyuk) - Platt orjinal MLE.
        self.lr = LogisticRegression(C=1e6, max_iter=10000, solver="lbfgs")

    @staticmethod
    def _to_logit(probas):
        eps = 1e-7
        p = np.clip(probas, eps, 1 - eps)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw_probas, y):
        logits = self._to_logit(np.asarray(raw_probas).ravel())
        self.lr.fit(logits, np.asarray(y).ravel())
        return self

    def transform(self, raw_probas):
        logits = self._to_logit(np.asarray(raw_probas).ravel())
        return self.lr.predict_proba(logits)[:, 1]


class IsotonicCalibrator:
    """
    Isotonic regression kalibrasyonu. Monotonic herhangi bir distorsiyonu
    duzeltir; >1000 ornek icin Platt'tan ustun olabilir ama kucuk veride
    overfit riski var.
    """

    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, raw_probas, y):
        self.iso.fit(np.asarray(raw_probas).ravel(), np.asarray(y).ravel())
        return self

    def transform(self, raw_probas):
        return self.iso.transform(np.asarray(raw_probas).ravel())


def fit_calibrator(method: str, raw_probas, y):
    """Factory. method: 'none' | 'platt' | 'sigmoid' | 'isotonic'"""
    if method is None:
        return None
    m = method.lower()
    if m in ("none", ""):
        return None
    if m in ("platt", "sigmoid"):
        return PlattCalibrator().fit(raw_probas, y)
    if m == "isotonic":
        return IsotonicCalibrator().fit(raw_probas, y)
    raise ValueError(f"Bilinmeyen kalibrasyon yontemi: {method!r}")


# -----------------------------
# Threshold Tuning
# -----------------------------

def tune_threshold_for_macro_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    step: float = 0.005,
) -> Tuple[float, float]:
    """
    PDF 4.2 / 4.5 / 5.3: F1-Macro skorunu maksimize edecek karar eşiğini bulur.
    Returns: (best_threshold, best_f1)
    """
    from sklearn.metrics import f1_score

    best_t, best_f1 = 0.5, -1.0
    thresholds = np.arange(0.05, 0.95 + 1e-9, step)
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def tune_threshold_for_balanced_accuracy(
    y_true,
    y_proba,
    *,
    step: float = 0.005,
):
    """
    Balanced Accuracy = (Sensitivity + Specificity) / 2'yi maksimize edecek esigi bulur.
    Sinif dengesizliginin extra oldugu panellerde (PAH, CFTR) tercih edilir.
    """
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score

    best_t, best_ba = 0.5, -1.0
    thresholds = np.arange(0.05, 0.95 + 1e-9, step)
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        ba = balanced_accuracy_score(y_true, pred)
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_t, best_ba


def prior_shift_sample_weights(y_true, test_pos_frac: float) -> np.ndarray:
    """
    ADIM 8 (2026-06-22) — Label-shift / prior-shift duzeltmesi.
    Egitim prior'u ~%80 patojenik, ama TEST ~%20 patojenik (yarisma TERSI).
    Esigi test dagilimina gore secmek icin ornek agirliklari uretir: her sinif
    test prior'una gore agirliklanir, boylece agirlikli F1-pos test kosulunu
    yansitir. (Lipton 2014: F1 esigi test base-rate'inde secilmeli; egitim
    base-rate'inde secmek suboptimaldir.)
    """
    y = np.asarray(y_true).ravel()
    n1 = max(1, int((y == 1).sum()))
    n0 = max(1, int((y == 0).sum()))
    p = float(test_pos_frac)
    return np.where(y == 1, p / n1, (1.0 - p) / n0).astype(float)


def tune_threshold_for_pos_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    step: float = 0.005,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    YARISMA GUNCEL (18 Haz 2026 toplantisi): F1 patojenik-odakli hesaplanacak.
    Macro yerine pos_label=1 (patojenik) F1'i maksimize eden karar esigini bulur.
    ADIM 8: sample_weight verilirse agirlikli F1-pos maksimize edilir (prior-shift).
    Returns: (best_threshold, best_f1_pos)
    """
    from sklearn.metrics import f1_score

    best_t, best_f1 = 0.5, -1.0
    thresholds = np.arange(0.05, 0.95 + 1e-9, step)
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1, zero_division=0, sample_weight=sample_weight)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def tune_threshold(
    y_true,
    y_proba,
    *,
    strategy: str = "f1_pos_max",
    step: float = 0.005,
    test_pos_frac: Optional[float] = None,
):
    """
    Threshold tuning dispatcher.
    strategy:
        "f1_pos_max" (yeni varsayilan, 18 Haz 2026 toplantisi: patojenik-odakli F1)
        "macro_f1_max" (eski varsayilan, dengeli F1)
        "balanced_accuracy_max"
    ADIM 8: test_pos_frac verilirse (orn 0.2), f1_pos esigi prior-shift agirligiyla
    secilir — yani esik TEST dagilimina (%20 patojenik) gore optimize edilir.
    """
    sw = None
    if test_pos_frac is not None and 0.0 < float(test_pos_frac) < 1.0:
        sw = prior_shift_sample_weights(y_true, test_pos_frac)
    if strategy in ("f1_pos_max", "pos_f1_max", "f1_positive_max"):
        return tune_threshold_for_pos_f1(y_true, y_proba, step=step, sample_weight=sw)
    if strategy == "balanced_accuracy_max":
        return tune_threshold_for_balanced_accuracy(y_true, y_proba, step=step)
    if strategy in ("macro_f1_max", "f1_macro_max"):
        return tune_threshold_for_macro_f1(y_true, y_proba, step=step)
    # Bilinmeyen strategy -> yeni varsayilan (f1_pos)
    return tune_threshold_for_pos_f1(y_true, y_proba, step=step, sample_weight=sw)


# -----------------------------
# Final Model Eğitimi
# -----------------------------

def fit_final_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    categorical_columns: List[str],
      *,
    best_params: Dict,
    fixed_params: Dict,
    class_weights: Optional[Dict] = None,
    eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
    early_stopping_rounds: int = 50,
):
    """
    En iyi hiperparametreler ile final CatBoost modelini egitir.
    PDF 4.5: Early Stopping + Ordered Boosting + L2 Regularization aktif.
    """
    from catboost import CatBoostClassifier

    cat_idx = [X.columns.get_loc(c) for c in categorical_columns if c in X.columns]

    params = dict(fixed_params)
    params.update(best_params)
    params["allow_writing_files"] = False
    params.setdefault("loss_function", "Logloss")
    params.setdefault("eval_metric", "F1")

    if class_weights:
        labels = sorted(class_weights.keys())
        params["class_weights"] = [class_weights[l] for l in labels]

    model = CatBoostClassifier(**params)
    fit_kwargs = dict(cat_features=cat_idx)
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        fit_kwargs["use_best_model"] = True
    model.fit(X, y, **fit_kwargs)
    return model
