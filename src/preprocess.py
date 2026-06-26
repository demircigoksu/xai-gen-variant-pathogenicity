"""
preprocess.py
=============
Veri Ön İşleme ve Temsilleme Stratejisi (PDF Bölüm 3.3) +
Etiket Güvenilirliği ve Veri Kalitesi Kontrolü (PDF Bölüm 3.4)

İşlevler
--------
- Sayısal değişkenler için medyan imputasyonu.
- Kategorik değişkenler için NaN -> "__MISSING__" string'i (CatBoost native handling).
- Tam kopya (duplicate) satırların temizlenmesi.
- Aynı özelliklere sahip ama farklı etiketli (çelişen) satırların ayıklanması.
- IQR temelli outlier tespiti (yalnızca raporlama; PDF: "ezberlemeyi engellemek için").
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


CAT_MISSING_TOKEN = "__MISSING__"


@dataclass
class PreprocessReport:
    n_rows_before: int = 0
    n_rows_after: int = 0
    n_full_duplicates_removed: int = 0
    n_conflicting_label_groups: int = 0
    n_conflicting_rows_removed: int = 0
    outlier_columns: List[str] = None
    outlier_total_cells: int = 0

    def to_dict(self) -> dict:
        return {
            "n_rows_before": self.n_rows_before,
            "n_rows_after": self.n_rows_after,
            "n_full_duplicates_removed": self.n_full_duplicates_removed,
            "n_conflicting_label_groups": self.n_conflicting_label_groups,
            "n_conflicting_rows_removed": self.n_conflicting_rows_removed,
            "n_outlier_columns_flagged": len(self.outlier_columns or []),
            "n_outlier_cells_flagged": self.outlier_total_cells,
        }


def median_imputation(
    X: pd.DataFrame,
    numeric_columns: List[str],
    medians: pd.Series | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    PDF 3.3: Sürekli değişkenlerde "medyan (ortanca) imputasyonu".

    Eğitimde medianlar hesaplanır ve döndürülür; inference'da bu medianlar
    parametre olarak verilerek tutarlı imputasyon yapılır.
    """
    X = X.copy()
    if medians is None:
        medians = X[numeric_columns].median(numeric_only=True)
    for col in numeric_columns:
        if col not in X.columns:
            continue
        # Bazı kolonlar tamamen NaN olabilir; o durumda 0 ile doldur
        med = medians.get(col, 0.0)
        if pd.isna(med):
            med = 0.0
            medians[col] = 0.0
        X[col] = X[col].fillna(med)
    return X, medians


def fill_categorical_missing(
    X: pd.DataFrame,
    categorical_columns: List[str],
) -> pd.DataFrame:
    """
    PDF 3.3: Kategorik değişkenler CatBoost'un yerleşik handler'ına verilir;
    NaN değerler özel bir token ile doldurulur (CatBoost string kategorik bekler).
    """
    X = X.copy()
    for col in categorical_columns:
        if col not in X.columns:
            continue
        X[col] = X[col].astype("object").fillna(CAT_MISSING_TOKEN).astype(str)
    return X


def remove_full_duplicates(
    X: pd.DataFrame, y: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """PDF 3.4: tam kopya kayıtların temizlenmesi."""
    df = X.copy()
    df["__y__"] = y
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    y_new = df["__y__"].to_numpy()
    df = df.drop(columns=["__y__"])
    return df, y_new, n_removed


def remove_conflicting_label_duplicates(
    X: pd.DataFrame, y: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray, int, int]:
    """
    PDF 3.4: "Birbirine tamamen zıt etiketlenmiş ancak özellikleri %100 örtüşen
    tutarsız veri profilleri tespit edilerek eğitim setinden izole edilecektir."

    Strateji: aynı X-row, farklı y'ye sahip tüm grupları sil.
    """
    df = X.copy()
    df["__y__"] = y
    grouped = df.groupby(list(X.columns), dropna=False)["__y__"].nunique()
    conflicting_keys = grouped[grouped > 1].index
    n_groups = len(conflicting_keys)

    if n_groups == 0:
        return X, y, 0, 0

    # mask: tutarsız gruplara ait satırları bul (numpy ndarray döner)
    df_idx = df.set_index(list(X.columns))
    mask = np.asarray(df_idx.index.isin(conflicting_keys))
    n_removed = int(mask.sum())
    keep = ~mask  # numpy bool array
    X_new = X.iloc[keep].reset_index(drop=True)
    y_new = np.asarray(y)[keep]
    return X_new, y_new, n_groups, n_removed


def detect_outliers_iqr(
    X: pd.DataFrame, numeric_columns: List[str], iqr_multiplier: float = 3.0
) -> Tuple[List[str], int]:
    """
    PDF 3.4: "Aşırı uç değerlerin (outliers) tespiti"
    Outlier'ları silmiyoruz – yalnızca raporluyoruz (CatBoost ağaç tabanlı,
    outlier'a karşı dirençlidir; SHAP analizi için de korunması faydalı).
    """
    flagged_cols: List[str] = []
    total_outlier_cells = 0
    for col in numeric_columns:
        if col not in X.columns:
            continue
        s = X[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low = q1 - iqr_multiplier * iqr
        high = q3 + iqr_multiplier * iqr
        n_out = int(((X[col] < low) | (X[col] > high)).sum())
        if n_out > 0:
            flagged_cols.append(col)
            total_outlier_cells += n_out
    return flagged_cols, total_outlier_cells


def add_missingness_features(
    X: pd.DataFrame,
    numeric_columns: List[str],
) -> pd.DataFrame:
    """
    ADIM 8 (2026-06-22) — Bilgilendirici eksiklik (informative missingness).
    Yarisma juri: "missing != 0, bilincli bir tercih" (MNAR sinyali). Eksiklik
    deseni patojeniteyle iliskili olabilir (orn. in-silico arac skoru
    uretememisse = nadir/atipik varyant). Literatur: Missing Indicator Method
    (Van Ness 2022). Imputasyondan ONCE cagrilmali (NaN deseni yakalanir).

    Eklenen ozellikler (hepsi numeric; sabit olanlar SFS'te nunique<=1 ile elenir):
      - MISS__count_all / MISS__frac_all : satir basi toplam eksik
      - MISS__count_AL / MISS__count_EK  : prefix bazli eksik sayisi
      - <EK_k>__isna                     : in-silico skor kolonlari icin gosterge
    """
    X = X.copy()
    num_present = [c for c in numeric_columns if c in X.columns]
    if not num_present:
        return X
    na = X[num_present].isna()
    al_cols = [c for c in num_present if c.startswith("AL_")]
    ek_cols = [c for c in num_present if c.startswith("EK_")]
    new = {}
    new["MISS__count_all"] = na.sum(axis=1).astype(float)
    new["MISS__frac_all"] = (na.sum(axis=1) / float(len(num_present))).astype(float)
    if al_cols:
        new["MISS__count_AL"] = na[al_cols].sum(axis=1).astype(float)
    if ek_cols:
        new["MISS__count_EK"] = na[ek_cols].sum(axis=1).astype(float)
        for c in ek_cols:
            new[f"{c}__isna"] = na[c].astype(float)
    miss_df = pd.DataFrame(new, index=X.index)
    return pd.concat([X, miss_df], axis=1)


def add_engineered_features(
    X: pd.DataFrame,
    numeric_columns: List[str],
) -> pd.DataFrame:
    """
    ADIM 9 (2026-06-24) — Sinyal-maksimizasyonu: alan-anlamli agrega ozellikler.
    EK_ (in-silico patojenite skorlari) ve AL_ (populasyon frekanslari) kolonlarindan
    NaN-duyarli ozetler uretir. Imputasyondan ONCE cagrilir; sabitler SFS'te elenir.
      - ENG__EK_max/mean/min : en guclu / ortalama / en dusuk in-silico sinyal
      - ENG__EK_n            : skor uretilen arac sayisi
      - ENG__AL_max/mean     : en yuksek / ortalama populasyon frekansi (yuksek=yaygin=benign egilimi)
      - ENG__AL_nonzero      : varyantin gozlendigi populasyon sayisi
    """
    X = X.copy()
    num = [c for c in numeric_columns if c in X.columns]
    ek = [c for c in num if c.startswith("EK_")]
    al = [c for c in num if c.startswith("AL_")]
    new = {}
    if ek:
        E = X[ek]
        new["ENG__EK_max"] = E.max(axis=1)
        new["ENG__EK_mean"] = E.mean(axis=1)
        new["ENG__EK_min"] = E.min(axis=1)
        new["ENG__EK_n"] = E.notna().sum(axis=1).astype(float)
    if al:
        A = X[al]
        new["ENG__AL_max"] = A.max(axis=1)
        new["ENG__AL_mean"] = A.mean(axis=1)
        new["ENG__AL_nonzero"] = (A.fillna(0) > 0).sum(axis=1).astype(float)
    if not new:
        return X
    df = pd.DataFrame(new, index=X.index)
    # ADIM 9 fix: tum-NaN satirlarda agrega NaN kalir -> SMOTE reddeder; medyanla doldur
    df = df.fillna(df.median(numeric_only=True)).fillna(0.0)
    return pd.concat([X, df], axis=1)


def preprocess_train(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric_columns: List[str],
    categorical_columns: List[str],
    *,
    drop_full_duplicates: bool = False,
    drop_conflict_label_duplicates: bool = True,
    outlier_iqr_multiplier: float = 3.0,
    add_missingness: bool = False,
    add_engineered: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, PreprocessReport]:
    # ADIM 6 (2026-06-19) — 18 Haz 2026 toplantisi:
    # "Tekrar eden varyantlar var. Evet, bu bilincli bir tercih."
    # drop_full_duplicates artik varsayilan FALSE — yarisma duplicate'leri
    # bilincli sunuyor (orneklenmis istatistiksel agirlik). Conflict label
    # duplicate temizligi devam (etiket celiskisi gercek gurultu).
    """
    Eğitim verisi için tam ön işleme akışı.

    Returns
    -------
    X_clean : pd.DataFrame
    y_clean : np.ndarray
    medians : pd.Series  (inference'da kullanılmak üzere)
    report : PreprocessReport
    """
    report = PreprocessReport()
    report.n_rows_before = len(X)

    # 1. Tam kopyaları temizle
    if drop_full_duplicates:
        X, y, n_dup = remove_full_duplicates(X, y)
        report.n_full_duplicates_removed = n_dup

    # 2. Çelişen etiketli aynı satırları temizle
    if drop_conflict_label_duplicates:
        X, y, n_groups, n_rows = remove_conflicting_label_duplicates(X, y)
        report.n_conflicting_label_groups = n_groups
        report.n_conflicting_rows_removed = n_rows

    # 3. Outlier tespiti (yalnızca rapor)
    outlier_cols, n_out_cells = detect_outliers_iqr(
        X, numeric_columns, iqr_multiplier=outlier_iqr_multiplier
    )
    report.outlier_columns = outlier_cols
    report.outlier_total_cells = n_out_cells

    # ADIM 9: Agrega (engineered) ozellikler — imputasyondan ONCE
    if add_engineered:
        X = add_engineered_features(X, numeric_columns)

    # ADIM 8: Bilgilendirici eksiklik ozellikleri — imputasyondan ONCE (NaN deseni)
    if add_missingness:
        X = add_missingness_features(X, numeric_columns)

    # 4. Imputasyon (yalnizca orijinal numeric_columns; MISS__ kolonlari NaN'siz)
    X, medians = median_imputation(X, numeric_columns)
    X = fill_categorical_missing(X, categorical_columns)

    report.n_rows_after = len(X)
    return X.reset_index(drop=True), y, medians, report


def preprocess_inference(
    X: pd.DataFrame,
    numeric_columns: List[str],
    categorical_columns: List[str],
    medians: pd.Series,
    add_missingness: bool = False,
    add_engineered: bool = False,
) -> pd.DataFrame:
    """
    Inference / test seti icin on isleme - egitimden gelen medianlari kullanir.
    Outlier silme veya duplicate temizleme YAPILMAZ (test seti dokunulmazdir).
    ADIM 8/9: add_missingness / add_engineered ile egitimle ayni ozellikler uretilir.
    """
    if add_engineered:
        X = add_engineered_features(X, numeric_columns)
    if add_missingness:
        X = add_missingness_features(X, numeric_columns)
    X, _ = median_imputation(X, numeric_columns, medians=medians)
    X = fill_categorical_missing(X, categorical_columns)
    return X.reset_index(drop=True)
