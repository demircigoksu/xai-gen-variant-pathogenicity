"""
data_loader.py
==============
CSV verilerini yükler ve feature gruplarını otomatik tespit eder.

PDF Bölüm 3.1: Yarışma komitesi tarafından sağlanan dengeli veri setlerinin yüklenmesi.
PDF Bölüm 3.2: Veri kısıtları – Variant_ID dışında bir tanımlayıcı kullanılmaz.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class PanelData:
    """Tek bir panele ait yüklenmiş veri ve meta bilgileri tutar."""
    name: str
    raw_df: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    feature_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    id_column: str = "Variant_ID"
    target_column: str = "Label"

    def n_samples(self) -> int:
        return len(self.X)

    def n_features(self) -> int:
        return self.X.shape[1]

    def class_distribution(self) -> dict:
        unique, counts = np.unique(self.y, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    def categorical_indices(self) -> List[int]:
        """CatBoost'un beklediği şekilde kategorik kolonların indexleri."""
        return [self.feature_columns.index(c) for c in self.categorical_columns]


def detect_feature_groups(
    df: pd.DataFrame,
    categorical_prefixes: List[str],
    numeric_prefixes: List[str],
    id_column: str,
    target_column: str,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Kolon isimleri prefix'lerine göre numeric / categorical olarak ayırır.

    Returns
    -------
    feature_columns : list[str]
    numeric_columns : list[str]
    categorical_columns : list[str]
    """
    skip = {id_column, target_column}
    feature_columns = [c for c in df.columns if c not in skip]

    categorical_columns: List[str] = []
    numeric_columns: List[str] = []
    for c in feature_columns:
        if any(c.startswith(p) for p in categorical_prefixes):
            categorical_columns.append(c)
        elif any(c.startswith(p) for p in numeric_prefixes):
            numeric_columns.append(c)
        else:
            # prefix dışı – dtype'a bak
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_columns.append(c)
            else:
                categorical_columns.append(c)

    return feature_columns, numeric_columns, categorical_columns


def load_panel(
    panel_name: str,
    csv_path: str | Path,
    *,
    categorical_prefixes: List[str],
    numeric_prefixes: List[str],
    id_column: str = "Variant_ID",
    target_column: str = "Label",
) -> PanelData:
    """
    Tek bir paneli yükler ve PanelData nesnesi döndürür.

    Notlar
    ------
    - Yarışma verisinde sayısal kolonlar `AL_*`, `EK_*`; kategorik kolonlar
      `CAT_*`, `AA_*` öneki taşımaktadır.
    - Eğer hedef kolon yoksa (örn. test seti), y boş array olarak döner.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    feature_columns, numeric_columns, categorical_columns = detect_feature_groups(
        df,
        categorical_prefixes=categorical_prefixes,
        numeric_prefixes=numeric_prefixes,
        id_column=id_column,
        target_column=target_column,
    )

    X = df[feature_columns].copy()

    if target_column in df.columns:
        y = df[target_column].astype(int).to_numpy()
    else:
        y = np.array([], dtype=int)

    return PanelData(
        name=panel_name,
        raw_df=df,
        X=X,
        y=y,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        id_column=id_column,
        target_column=target_column,
    )


def summarize_panel(p: PanelData) -> str:
    """Panel hakkında insan tarafından okunabilir özet üretir."""
    dist = p.class_distribution()
    lines = [
        f"=== {p.name} ===",
        f"Toplam satır: {p.n_samples()}",
        f"Toplam öznitelik: {p.n_features()}",
        f"Sayısal: {len(p.numeric_columns)} | Kategorik: {len(p.categorical_columns)}",
        f"Sınıf dağılımı: {dist}",
    ]
    if p.y.size > 0:
        n_pos = int((p.y == 1).sum())
        n_neg = int((p.y == 0).sum())
        ratio = n_pos / max(n_neg, 1)
        lines.append(f"Patojenik / Benign oranı: {ratio:.2f}  ({n_pos} / {n_neg})")
    nan_ratio = float(p.X.isna().mean().mean())
    lines.append(f"Ortalama NaN oranı (X): {nan_ratio:.3f}")
    return "\n".join(lines)
