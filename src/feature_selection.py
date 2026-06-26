"""
feature_selection.py
====================
Sirali Ileri Ozellik Secimi (Sequential Forward Selection - SFS)
PDF Bolum 3.6: "Bos bir kumeyle yola cikip, modelin Patojenik/Benign ayirimindaki
F1 skoruna en yuksek matematiksel katkiyi saglayan 'isimsiz' degiskenleri tek tek
bularak gurultuyu eleyecektir."
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class SFSResult:
    selected_features: List[str] = field(default_factory=list)
    selection_history: List[dict] = field(default_factory=list)
    final_score: float = 0.0


def sequential_forward_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    categorical_columns: List[str],
    *,
    n_features_to_select: int = 25,
    cv_folds: int = 3,
    scoring: str = "f1_macro",
    iterations: int = 200,
    depth: int = 4,
    learning_rate: float = 0.1,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
    early_stop_patience: int = 3,
    min_score_improvement: float = 0.001,
) -> SFSResult:
    """
    Sequential Forward Selection: empty -> greedy add -> stop at n_features_to_select.
    Cap edilmis: sabit feature'lar elenir, fold'da CatBoost hatasi gelirse aday atlanir.
    PDF 3.6 ile uyumlu plateau-early-stop: art arda 'early_stop_patience' adimda
    'min_score_improvement' degerinden daha az iyilesme olursa durulur.
    """
    try:
        from catboost import CatBoostClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
    except ImportError as e:
        raise ImportError("SFS icin catboost & scikit-learn gerekli.") from e

    cat_set = set(categorical_columns)
    all_features = list(X.columns)

    # Sabit (constant) feature'lari ele
    constant_features: List[str] = []
    informative_features: List[str] = []
    for col in all_features:
        try:
            n_unique = int(X[col].nunique(dropna=False))
        except Exception:
            n_unique = 2
        if n_unique <= 1:
            constant_features.append(col)
        else:
            informative_features.append(col)
    if verbose and constant_features:
        print(f"[SFS] {len(constant_features)} sabit ozellik elendi (unique<=1).")

    selected: List[str] = []
    history: List[dict] = []
    best_overall = -np.inf
    plateau_count = 0

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    target_n = min(n_features_to_select, len(informative_features))

    while len(selected) < target_n:
        remaining = [f for f in informative_features if f not in selected]
        if not remaining:
            break
        best_feat = None
        best_score = -np.inf
        n_failed = 0
        for feat in remaining:
            trial_feats = selected + [feat]
            trial_cats = [f for f in trial_feats if f in cat_set]
            cat_idx = [trial_feats.index(c) for c in trial_cats]

            scores = []
            failed = False
            for tr_idx, va_idx in skf.split(X, y):
                X_tr = X.iloc[tr_idx][trial_feats]
                X_va = X.iloc[va_idx][trial_feats]
                y_tr = y[tr_idx]
                y_va = y[va_idx]
                try:
                    model = CatBoostClassifier(
                        iterations=iterations,
                        depth=depth,
                        learning_rate=learning_rate,
                        cat_features=cat_idx,
                        loss_function="Logloss",
                        verbose=0,
                        random_seed=random_state,
                        thread_count=n_jobs if n_jobs and n_jobs > 0 else -1,
                        allow_writing_files=False,
                    )
                    model.fit(X_tr, y_tr)
                    # ADIM 7 (2026-06-21): esik-bagimsiz scoring (roc_auc/pr_auc).
                    #   Kucuk + dengesiz panellerde (PAH/CFTR) f1_pos doygundur
                    #   (cogunluk patojenik -> skor ~0.91'de duz kalir, gradyan yok).
                    #   roc_auc/pr_auc proba uzerinden ayirt ediciligi olcer -> SFS
                    #   gercekten ayirici ozellikleri secebilir.
                    # ADIM 6 (2026-06-19): scoring "f1_pos" -> pos_label=1 (patojenik)
                    # "f1_macro" -> eski makro davranis (geri uyum icin).
                    if scoring in ("roc_auc", "auc", "roc"):
                        proba = model.predict_proba(X_va)[:, 1]
                        scores.append(roc_auc_score(y_va, proba))
                    elif scoring in ("pr_auc", "average_precision", "ap", "auprc"):
                        proba = model.predict_proba(X_va)[:, 1]
                        scores.append(average_precision_score(y_va, proba))
                    elif scoring in ("f1_pos", "pos_f1", "f1_positive"):
                        pred = model.predict(X_va).astype(int).ravel()
                        scores.append(f1_score(y_va, pred, pos_label=1, zero_division=0))
                    else:
                        pred = model.predict(X_va).astype(int).ravel()
                        scores.append(f1_score(y_va, pred, average="macro", zero_division=0))
                except Exception:
                    failed = True
                    break

            if failed or not scores:
                n_failed += 1
                continue

            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_feat = feat

        if best_feat is None:
            if verbose:
                print(f"[SFS] gecerli ozellik kalmadi; {n_failed} aday basarisiz.")
            break

        selected.append(best_feat)
        history.append({"step": len(selected), "added": best_feat, "score": best_score})
        if verbose:
            print(f"[SFS] step {len(selected):>3} | +{best_feat:<12} | score={best_score:.4f}")

        # Plateau early stop (PDF 3.6 ile uyumlu: katki azaldiysa dur)
        if best_score > best_overall + min_score_improvement:
            best_overall = best_score
            plateau_count = 0
        else:
            plateau_count += 1
            if plateau_count >= early_stop_patience:
                if verbose:
                    print(f"[SFS] plateau: {early_stop_patience} adimdir <={min_score_improvement:.3f} iyilesme yok. Durduruluyor.")
                break

    return SFSResult(
        selected_features=selected,
        selection_history=history,
        final_score=best_overall if best_overall > -np.inf else 0.0,
    )
