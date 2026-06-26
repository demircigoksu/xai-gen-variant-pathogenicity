"""
stacking.py
===========
Heterojen GBDT Stacking Ensemble — Adim 4 yapisal degisiklik (2026-05-24).

Mimari (PDF gerekcesi):
    [CatBoost ] ─┐
    [LightGBM ] ─┼─> 3-fold OOF probalari ─┐
    [XGBoost  ] ─┘                          ▼
                                    [Meta-CatBoost] (ana karar verici)
                                              ▼
                                    Final tahmin

- PDF "CatBoost kullanilacak" maddesi korunur: meta-learner CatBoost'tur,
  base'ler (LightGBM, XGBoost) yardimci sinyal saglar.
- Tabular medikal veride GBDT stack'inin tek-modele +0.02-0.04 F1 verdigi
  gosterilmistir (Research Square 2024, Arxiv 2410.03705).
- Multi-seed (n_seeds=2-3): Full-train refit'te K seed ortalanir, varyans
  duser (ozellikle CFTR icin).
- Train-Only SMOTE her inner fold icinde uygulanir (PDF Bolum 3.5 uyumu).
- Cost-Sensitive Learning her base ve meta'da class_weights/sample_weight.

API (sklearn benzeri):
    ensemble = StackingEnsemble(...)
    ensemble.fit(X_train, y_train)
    probas = ensemble.predict_proba(X_val)
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .balance import apply_train_only_smote, compute_class_weights


# --------------------------------------------------------------
# Yardimci: dict class_weights -> CatBoost listesi
# --------------------------------------------------------------
def _class_weights_to_list(cw: Optional[Dict]) -> Optional[list]:
    if not cw:
        return None
    labels = sorted(cw.keys())
    return [cw[l] for l in labels]


def _sample_weights_from_dict(y: np.ndarray, cw: Optional[Dict]) -> Optional[np.ndarray]:
    """LightGBM/XGBoost icin sample_weight uretir (sinif agirligindan)."""
    if not cw:
        return None
    return np.array([cw[int(yi)] for yi in y], dtype=float)


def _encode_for_lgb_xgb(
    X: pd.DataFrame,
    categorical_columns: List[str],
    cat_maps: Optional[Dict[str, Dict[str, int]]] = None,
) -> pd.DataFrame:
    """
    LightGBM/XGBoost icin pandas DataFrame'i tum-numeric hale getirir.

    SMOTE-NC sonrasi kategoriler string olabilir; bunlari label-encode ediyoruz.
    cat_maps verildi ise mevcut encoding'i kullanir (train/predict tutarliligi).
    Aksi halde X uzerinde yeni map kurar ve dondurur (caller kaydedebilir).
    """
    X_enc = X.copy().reset_index(drop=True)
    new_maps: Dict[str, Dict[str, int]] = {}

    for col in categorical_columns:
        if col not in X_enc.columns:
            continue
        ser = X_enc[col].astype(str)

        if cat_maps is not None and col in cat_maps:
            # Mevcut map'i kullan; gormedigi degeri -1'e yolla
            mp = cat_maps[col]
            X_enc[col] = ser.map(mp).fillna(-1).astype(int)
        else:
            uniques = sorted(ser.unique().tolist())
            mp = {v: i for i, v in enumerate(uniques)}
            new_maps[col] = mp
            X_enc[col] = ser.map(mp).astype(int)

    # XGBoost numpy float ister; LightGBM da iyi calisir
    X_enc = X_enc.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    if cat_maps is None:
        return X_enc, new_maps
    return X_enc


# --------------------------------------------------------------
# Ana sinif
# --------------------------------------------------------------
class StackingEnsemble:
    """
    Heterojen GBDT stacking ensemble.

    Parametreler
    ------------
    catboost_params : dict
        Grid search'ten gelen best params (depth, learning_rate, l2_leaf_reg, ...)
    catboost_fixed : dict
        loss_function, eval_metric, bootstrap_type vb.
    cat_features : list[str]
        Kategorik kolon isimleri (CatBoost natif; LGB/XGB encode edilir)
    cb_balance : dict
        SMOTE ve class_weights ayarlari (config.class_balance bloku).
    meta_params : dict, optional
        Meta-CatBoost hiperparametreleri. Varsayilan sig L2 yuksek model
        (overfit'e karsi).
    n_inner_folds : int
        OOF icin StratifiedKFold sayisi. Varsayilan 3 (hizli, 5'e gore %40 daha
        az fit). 5 daha stabil meta-feature uretir ama 1.7x maliyet.
    n_seeds : int
        Full-train refit'te kac seed (multi-seed averaging). Varsayilan 2.
    base_iterations : int
        LightGBM/XGBoost ve fold-CatBoost'unun iteration sayisi. 300 hizli +
        early-stopping ile yeterli. Best_params'in iterations'i full refit'te
        kullanilir (genelde 500).
    """

    def __init__(
        self,
        catboost_params: Dict,
        catboost_fixed: Dict,
        cat_features: List[str],
        cb_balance: Dict,
        meta_params: Optional[Dict] = None,
        n_inner_folds: int = 3,
        n_seeds: int = 2,
        base_iterations: int = 300,
        random_state: int = 42,
    ):
        self.catboost_params = dict(catboost_params)
        self.catboost_fixed = dict(catboost_fixed)
        self.cat_features = list(cat_features)
        self.cb_balance = dict(cb_balance)
        self.n_inner_folds = int(n_inner_folds)
        self.n_seeds = max(1, int(n_seeds))
        self.base_iterations = int(base_iterations)
        self.random_state = int(random_state)

        self.meta_params = meta_params or {
            "iterations": 200,
            "learning_rate": 0.05,
            "depth": 3,
            "l2_leaf_reg": 5.0,
            "verbose": 0,
            "random_seed": self.random_state,
            "allow_writing_files": False,
            "loss_function": "Logloss",
        }

        # Fit sonrasi doldurulur
        self.full_base_models_: Dict[str, list] = {"cat": [], "lgb": [], "xgb": []}
        self.meta_model_ = None
        self.calibrator_ = None  # ADIM 10: PSR 4.5 kalibrasyon (CV-tabanli izotonik)
        self.cat_maps_: Optional[Dict[str, Dict[str, int]]] = None
        self.feature_columns_: Optional[List[str]] = None

    # ---- Base learner factory ----

    def _make_catboost(self, seed: int, iterations: Optional[int] = None):
        from catboost import CatBoostClassifier

        params = dict(self.catboost_params)
        params.update(self.catboost_fixed)
        if iterations is not None:
            params["iterations"] = iterations
        params["random_seed"] = seed
        params.setdefault("verbose", 0)
        params["allow_writing_files"] = False
        # cat_features kolon-indeksi olarak ayri verilecek (fit_with_idx)
        return CatBoostClassifier(**params)

    def _make_lightgbm(self, seed: int):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "lightgbm kurulu degil. Colab'da: `!pip install lightgbm -q`"
            ) from e

        return lgb.LGBMClassifier(
            n_estimators=self.base_iterations,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=5,
            reg_lambda=3.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            force_row_wise=True,
        )

    def _make_xgboost(self, seed: int):
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "xgboost kurulu degil. Colab'da: `!pip install xgboost -q`"
            ) from e

        return xgb.XGBClassifier(
            n_estimators=self.base_iterations,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            reg_lambda=3.0,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
            tree_method="hist",
        )

    # ---- Inner fit helpers ----

    def _fit_catboost(self, X: pd.DataFrame, y: np.ndarray, seed: int, iterations: Optional[int] = None):
        model = self._make_catboost(seed, iterations=iterations)
        cat_idx = [X.columns.get_loc(c) for c in self.cat_features if c in X.columns]
        cw = compute_class_weights(
            y,
            manual_multiplier=self.cb_balance.get("manual_class_weight_multiplier"),
        ) if self.cb_balance.get("cost_sensitive", True) else None
        cw_list = _class_weights_to_list(cw)
        if cw_list is not None:
            model.set_params(class_weights=cw_list)
        if cat_idx:
            model.fit(X, y, cat_features=cat_idx, verbose=False)
        else:
            model.fit(X, y, verbose=False)
        return model

    def _fit_lightgbm(self, X_enc: pd.DataFrame, y: np.ndarray, seed: int):
        model = self._make_lightgbm(seed)
        cw = compute_class_weights(
            y,
            manual_multiplier=self.cb_balance.get("manual_class_weight_multiplier"),
        ) if self.cb_balance.get("cost_sensitive", True) else None
        sw = _sample_weights_from_dict(y, cw)
        if sw is not None:
            model.fit(X_enc, y, sample_weight=sw)
        else:
            model.fit(X_enc, y)
        return model

    def _fit_xgboost(self, X_enc: pd.DataFrame, y: np.ndarray, seed: int):
        model = self._make_xgboost(seed)
        cw = compute_class_weights(
            y,
            manual_multiplier=self.cb_balance.get("manual_class_weight_multiplier"),
        ) if self.cb_balance.get("cost_sensitive", True) else None
        sw = _sample_weights_from_dict(y, cw)
        if sw is not None:
            model.fit(X_enc, y, sample_weight=sw)
        else:
            model.fit(X_enc, y)
        return model

    # ---- Public API ----

    def fit(self, X: pd.DataFrame, y: np.ndarray, verbose: bool = False) -> "StackingEnsemble":
        X = X.reset_index(drop=True)
        y = np.asarray(y).astype(int)
        n = len(X)
        self.feature_columns_ = list(X.columns)

        # --- 1) Inner OOF: meta-feature uret ---
        oof_cat = np.zeros(n)
        oof_lgb = np.zeros(n)
        oof_xgb = np.zeros(n)

        skf = StratifiedKFold(
            n_splits=self.n_inner_folds, shuffle=True, random_state=self.random_state
        )

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr_raw = X.iloc[tr_idx].reset_index(drop=True)
            y_tr_raw = y[tr_idx]
            X_va = X.iloc[va_idx].reset_index(drop=True)

            # Train-Only SMOTE (fold icinde, leakage yok)
            if self.cb_balance.get("use_smote", True):
                X_tr, y_tr = apply_train_only_smote(
                    X_tr_raw, y_tr_raw,
                    categorical_columns=self.cat_features,
                    k_neighbors=self.cb_balance["smote_k_neighbors"],
                    random_state=self.cb_balance["smote_random_state"],
                    method=self.cb_balance.get("smote_method", "SMOTENC"),
                )
            else:
                X_tr, y_tr = X_tr_raw, y_tr_raw

            # Bu fold'a ozel encoding (LGB/XGB icin)
            X_tr_enc, fold_cat_maps = _encode_for_lgb_xgb(X_tr, self.cat_features)
            X_va_enc = _encode_for_lgb_xgb(X_va, self.cat_features, cat_maps=fold_cat_maps)

            # --- CatBoost OOF (tek seed, hizli iterations) ---
            cat_m = self._fit_catboost(X_tr, y_tr, seed=self.random_state, iterations=self.base_iterations)
            oof_cat[va_idx] = cat_m.predict_proba(X_va)[:, 1]

            # --- LightGBM OOF ---
            lgb_m = self._fit_lightgbm(X_tr_enc, y_tr, seed=self.random_state)
            oof_lgb[va_idx] = lgb_m.predict_proba(X_va_enc)[:, 1]

            # --- XGBoost OOF ---
            xgb_m = self._fit_xgboost(X_tr_enc, y_tr, seed=self.random_state)
            oof_xgb[va_idx] = xgb_m.predict_proba(X_va_enc)[:, 1]

            if verbose:
                print(f"[Stacking] OOF fold {fold_idx + 1}/{self.n_inner_folds} done "
                      f"(cat={oof_cat[va_idx].mean():.3f} lgb={oof_lgb[va_idx].mean():.3f} "
                      f"xgb={oof_xgb[va_idx].mean():.3f})")

        # --- 2) Meta-feature matrisini olustur ---
        meta_X = np.column_stack([oof_cat, oof_lgb, oof_xgb])

        # --- 3) Meta-CatBoost'u fit et ---
        from catboost import CatBoostClassifier

        meta_cw = compute_class_weights(
            y,
            manual_multiplier=self.cb_balance.get("manual_class_weight_multiplier"),
        ) if self.cb_balance.get("cost_sensitive", True) else None
        meta_cw_list = _class_weights_to_list(meta_cw)

        meta_params = dict(self.meta_params)
        if meta_cw_list is not None:
            meta_params["class_weights"] = meta_cw_list

        self.meta_model_ = CatBoostClassifier(**meta_params)
        self.meta_model_.fit(meta_X, y, verbose=False)

        if verbose:
            print(f"[Stacking] meta-learner fit OK, train acc ~{self.meta_model_.score(meta_X, y):.3f}")

        # --- ADIM 10 (2026-06-25): PSR 4.5 KALİBRASYON (CV-tabanli izotonik) ---
        # Holdout israfi YOK: meta-seviye OOF olasiliklari uzerinde IsotonicRegression.
        # Monoton oldugu icin siralama (ROC) ve esikle birlikte F1 degismez; amaci
        # olasilik guvenilirligi (PSR taahhudu). calibrate_stacking=false ise atlanir.
        if self.cb_balance.get("calibrate_stacking", False):
            from sklearn.isotonic import IsotonicRegression
            oof_meta = np.zeros(n)
            skf_c = StratifiedKFold(n_splits=self.n_inner_folds, shuffle=True,
                                    random_state=self.random_state + 7)
            for tr_i, va_i in skf_c.split(meta_X, y):
                mm = CatBoostClassifier(**meta_params)
                mm.fit(meta_X[tr_i], y[tr_i], verbose=False)
                oof_meta[va_i] = mm.predict_proba(meta_X[va_i])[:, 1]
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip").fit(oof_meta, y)
            if verbose:
                print("[Stacking] kalibrasyon: CV-tabanli izotonik fit OK")

        # --- 4) Full-train refit (multi-seed) — prediction icin ---
        # SMOTE on full train (predictions will be on UNSEEN data so this is leakage-free)
        if self.cb_balance.get("use_smote", True):
            X_full, y_full = apply_train_only_smote(
                X, y,
                categorical_columns=self.cat_features,
                k_neighbors=self.cb_balance["smote_k_neighbors"],
                random_state=self.cb_balance["smote_random_state"],
                method=self.cb_balance.get("smote_method", "SMOTENC"),
            )
        else:
            X_full, y_full = X, y

        X_full_enc, self.cat_maps_ = _encode_for_lgb_xgb(X_full, self.cat_features)

        self.full_base_models_ = {"cat": [], "lgb": [], "xgb": []}

        for s in range(self.n_seeds):
            seed = self.random_state + s * 1000

            # CatBoost — best_params iterations (genelde 500) ile
            cat_iters = int(self.catboost_params.get("iterations", 500))
            cat_m_full = self._fit_catboost(X_full, y_full, seed=seed, iterations=cat_iters)
            self.full_base_models_["cat"].append(cat_m_full)

            lgb_m_full = self._fit_lightgbm(X_full_enc, y_full, seed=seed)
            self.full_base_models_["lgb"].append(lgb_m_full)

            xgb_m_full = self._fit_xgboost(X_full_enc, y_full, seed=seed)
            self.full_base_models_["xgb"].append(xgb_m_full)

            if verbose:
                print(f"[Stacking] full-train refit seed {s + 1}/{self.n_seeds} done")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Iki sutunlu numpy array dondurur: [P(class=0), P(class=1)].
        Multi-seed proba'lari ortalar, sonra meta-learner ile final tahmini verir.
        """
        if self.meta_model_ is None:
            raise RuntimeError("fit() once cagrilmali.")

        X = X.reset_index(drop=True)
        X_enc = _encode_for_lgb_xgb(X, self.cat_features, cat_maps=self.cat_maps_)

        # Multi-seed averaged base probas
        cat_probas = np.mean(
            [m.predict_proba(X)[:, 1] for m in self.full_base_models_["cat"]],
            axis=0,
        )
        lgb_probas = np.mean(
            [m.predict_proba(X_enc)[:, 1] for m in self.full_base_models_["lgb"]],
            axis=0,
        )
        xgb_probas = np.mean(
            [m.predict_proba(X_enc)[:, 1] for m in self.full_base_models_["xgb"]],
            axis=0,
        )

        meta_X = np.column_stack([cat_probas, lgb_probas, xgb_probas])
        proba = self.meta_model_.predict_proba(meta_X)
        # ADIM 10: kalibrasyon (varsa) — sinif-1 olasiligini izotonik ile duzelt
        if getattr(self, "calibrator_", None) is not None:
            p1 = self.calibrator_.transform(proba[:, 1])
            p1 = np.clip(p1, 0.0, 1.0)
            proba = np.column_stack([1.0 - p1, p1])
        return proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    # ---- Persistence ----

    def save(self, path: str) -> None:
        """Pickle olarak diske yaz (test/inference icin)."""
        import pickle
        with open(path, "wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, path: str) -> "StackingEnsemble":
        import pickle
        with open(path, "rb") as fp:
            obj = pickle.load(fp)
        return obj
