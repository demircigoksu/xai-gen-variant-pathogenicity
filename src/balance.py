"""
balance.py
==========
Sınıf Dengesi ve Risk Perspektifi (PDF Bölüm 3.5).

PDF: "Yalnızca eğitim setine özgü Train-Only SMOTE sentetik veri üretme
yöntemi uygulanacaktır."

Veride hem sayısal hem de kategorik değişkenler bulunduğundan SMOTE-NC
(SMOTE for Nominal & Continuous) kullanılır.

ADIM 1 İYİLEŞTİRME (2026-05-23):
- `method` parametresi eklendi. Varsayılan "SMOTENC" (PDF baseline).
- "BorderlineSMOTE" seçeneği eklendi: karar sınırına yakın azınlık örneklerini
  sentez kaynağı olarak kullanır. Tıbbi/ekstrem dengesiz verilerde
  specificity'yi belirgin iyileştirdiği gösterilmiş (IEEE 2023, Han et al. 2005).
- Kategorik feature'lar label-encode edilip BorderlineSMOTE sonrası nearest
  valid category'ye snap edilir.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import pandas as pd


def apply_train_only_smote(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    categorical_columns: List[str],
    *,
    k_neighbors: int = 5,
    random_state: int = 42,
    method: str = "SMOTENC",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Eğitim setine SMOTE varyantı uygular.

    PDF Bölüm 3.5 ile uyumlu:
    - Yalnızca train fold'a uygulanır (test/validation dokunulmaz).
    - Kategorik değişkenler indeks olarak SMOTE-NC'ye geçirilir.

    Parameters
    ----------
    method : str
        "SMOTENC" (varsayilan, PDF baseline)
            Klasik SMOTE-NC. Kategorikleri natif olarak isler.
        "BorderlineSMOTE"
            Karar sinirina yakin azinlik orneklerini kaynak olarak kullanir.
            Kategorikler label-encode edilip islem sonrasi snap edilir.
        "ADASYN"
            Yogunluk-bazli adaptif sentez. Zor bolgelere daha cok sentez uretir.

    Notlar
    ------
    - Eğer her sınıftan örnek sayısı k_neighbors'tan azsa SMOTE çalışmaz;
      bu durumda k_neighbors otomatik olarak küçültülür.
    - Eğer sınıflar zaten dengeli ise pas geçilir.
    """
    from collections import Counter

    counts = Counter(y_train.tolist())
    if len(counts) < 2:
        return X_train.copy(), y_train.copy()

    minority_count = min(counts.values())
    majority_count = max(counts.values())
    if minority_count == majority_count:
        return X_train.copy(), y_train.copy()

    # ADIM 5 patch: SMOTE icin minimum 2 azinlik ornegi gerekli (kneighbors>=1).
    # Reverse-distribution CV'de inner fold'lar bazi panellerde 1 azinlik
    # ornegi birakabilir (PAH: 4 benign train, 3-fold inner CV -> bazi fold'da 1).
    # Bu durumda SMOTE yerine class_weight'a guvenip ham veriyi donduruyoruz.
    if minority_count < 2:
        print(
            f"[SMOTE-SKIP] Azinlik sinifinda yalnizca {minority_count} ornek var "
            f"(her sinif sayisi: {dict(counts)}). SMOTE atlandi; class_weight ile devam."
        )
        return X_train.copy(), y_train.copy()

    # k_neighbors azınlık sınıfından küçük olmalı
    k = min(k_neighbors, max(minority_count - 1, 1))
    if k < 1:
        return X_train.copy(), y_train.copy()

    # imbalanced-learn import (lazy)
    try:
        from imblearn.over_sampling import SMOTENC, SMOTE, BorderlineSMOTE, ADASYN  # noqa
    except ImportError as e:
        raise ImportError(
            "imbalanced-learn kurulu değil. `pip install imbalanced-learn`"
        ) from e

    method_upper = (method or "SMOTENC").upper()

    # ---------- Yol 1: Klasik SMOTENC (PDF baseline) ----------
    if method_upper in ("SMOTENC", "SMOTE"):
        cat_idx = [
            X_train.columns.get_loc(c)
            for c in categorical_columns
            if c in X_train.columns
        ]
        if cat_idx:
            sampler = SMOTENC(
                categorical_features=cat_idx,
                k_neighbors=k,
                random_state=random_state,
            )
        else:
            sampler = SMOTE(k_neighbors=k, random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        if isinstance(X_res, np.ndarray):
            X_res = pd.DataFrame(X_res, columns=X_train.columns)
        for c in categorical_columns:
            if c in X_res.columns:
                X_res[c] = X_res[c].astype(str)
        return X_res, np.asarray(y_res).astype(int)

    # ---------- Yol 2 & 3: BorderlineSMOTE / ADASYN ----------
    # imblearn'in BorderlineSMOTE ve ADASYN'i kategorik destek vermiyor —
    # label-encode + resample + snap-back yapiyoruz.
    if method_upper in ("BORDERLINESMOTE", "BORDERLINE", "BSMOTE", "ADASYN"):
        X_enc = X_train.copy().reset_index(drop=True)
        cat_maps = {}
        for c in categorical_columns:
            if c in X_enc.columns:
                X_enc[c] = X_enc[c].astype(str)
                uniques = sorted(X_enc[c].unique().tolist())
                cat_maps[c] = {v: i for i, v in enumerate(uniques)}
                X_enc[c] = X_enc[c].map(cat_maps[c]).astype(float)

        # m_neighbors: borderline'i tespit ederken kac komsuya bakacagiz.
        # Cok kucuk azinlik sayilarinda (PAH benign=62) varsayilan m=10 OK;
        # CFTR icin minority - 1 ile kap edilir.
        m_neighbors = min(10, max(minority_count - 1, 2))

        if method_upper == "ADASYN":
            sampler = ADASYN(
                n_neighbors=k,
                random_state=random_state,
            )
        else:
            sampler = BorderlineSMOTE(
                k_neighbors=k,
                m_neighbors=m_neighbors,
                kind="borderline-1",
                random_state=random_state,
            )

        try:
            X_res_arr, y_res = sampler.fit_resample(X_enc, y_train)
        except (ValueError, RuntimeError) as e:
            # Borderline bulamayabilir (cok temiz veya cok karisik fold) —
            # klasik SMOTENC'e fallback.
            print(f"[Balance] {method_upper} basarisiz ({e}); SMOTENC'e fallback.")
            return apply_train_only_smote(
                X_train, y_train, categorical_columns,
                k_neighbors=k_neighbors, random_state=random_state,
                method="SMOTENC",
            )

        X_res = pd.DataFrame(X_res_arr, columns=X_train.columns)
        # Kategorikleri en yakin gecerli kategoriye snap et
        for c in categorical_columns:
            if c in X_res.columns and c in cat_maps:
                reverse_map = {i: v for v, i in cat_maps[c].items()}
                n_cats = len(reverse_map)
                X_res[c] = (
                    X_res[c].round()
                    .clip(lower=0, upper=n_cats - 1)
                    .astype(int)
                    .map(reverse_map)
                    .astype(str)
                )
        return X_res, np.asarray(y_res).astype(int)

    raise ValueError(
        f"Bilinmeyen SMOTE method: {method!r}. "
        "Gecerli: 'SMOTENC', 'BorderlineSMOTE', 'ADASYN'."
    )


def compute_class_weights(y: np.ndarray, manual_multiplier: dict = None) -> dict:
    """
    PDF Bölüm 3.5: "Maliyete Duyarlı Öğrenme (Cost-Sensitive Learning)" için
    sınıf ağırlıkları. CatBoost `class_weights` parametresine geçirilir.

    Parameters
    ----------
    y : np.ndarray
    manual_multiplier : dict, opsiyonel
        Belirli siniflarin agirligini ekstra carparak yukseltir.
        Ornek: {0: 2.0} -> beni        Belirli siniflarin agirligini ekstra carparak yukseltir.
        Ornek: {0: 2.0} -> benign agirligini balanced'in 2 kati yapar.
        PAH gibi extra agresif cost-sensitive isteyen paneller icin.
    """
    from collections import Counter

    counts = Counter(y.tolist())
    n = len(y)
    n_classes = len(counts)
    if n_classes < 2:
        return {int(list(counts.keys())[0]): 1.0}
    weights = {int(c): n / (n_classes * cnt) for c, cnt in counts.items()}
    if manual_multiplier:
        for cls, mult in manual_multiplier.items():
            cls_int = int(cls)
            if cls_int in weights:
                weights[cls_int] *= float(mult)
    return weights
