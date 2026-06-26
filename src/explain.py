"""
explain.py
==========
Açıklanabilirlik / XAI – PDF Bölüm 4.4 ve 5.5.

PDF: "Modelin kararlarını yönlendiren isimsiz özelliklerin SHAP özet
graf iklerindeki yönelimlerine bakılacaktır. Elde edilen görsel ve matematiksel
veriler moleküler genetik uzmanlığı ile haritalandırılarak (Feature Mapping),
yüksek katkı sağlayan bu özelliklerin 'Evrimsel Korunmuşluk',
'Popülasyon Frekansı' veya 'Amino Asit Değişimi' gibi biyolojik karşılıkları
klinisyenler için anlaşılabilir bir formata açıklanacaktır."
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def shap_summary(
    model,
    X: pd.DataFrame,
    *,
    feature_names: Optional[List[str]] = None,
    output_dir: str | Path,
    panel_name: str,
    max_display: int = 25,
    background_sample_size: int = 200,
):
    """
    SHAP TreeExplainer ile özet grafiği üretir ve `output_dir/SHAP_summary.png`
    dosyasına kaydeder. Aynı zamanda öznitelik bazlı |SHAP| ortalamalarını
    `SHAP_feature_importance.csv` olarak yazar.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[SHAP] Bağımlılık eksik: {e}. SHAP atlanıyor.")
        return None

    # Arka plan örneği (SHAP hızı için)
    if len(X) > background_sample_size:
        bg = X.sample(background_sample_size, random_state=42)
    else:
        bg = X

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(bg)
    except Exception as e:
        print(f"[SHAP] TreeExplainer başarısız: {e}; KernelExplainer denenmiyor (yavaş).")
        return None

    # Bazı SHAP versiyonları (n_samples, n_features, n_classes) döner
    if isinstance(shap_values, list):
        # multi-class durumu — bizimkisi binary, ikinci sınıfı al
        shap_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_arr = shap_values

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_arr,
        bg,
        feature_names=feature_names or list(X.columns),
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Özet – {panel_name}")
    plt.tight_layout()
    out_png = output_dir / "SHAP_summary.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()

    # Feature importance (mean(|SHAP|))
    importance = np.abs(shap_arr).mean(axis=0)
    fi_df = pd.DataFrame(
        {"feature": feature_names or list(X.columns), "mean_abs_shap": importance}
    ).sort_values("mean_abs_shap", ascending=False)
    fi_df.to_csv(output_dir / "SHAP_feature_importance.csv", index=False)

    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_arr,
        bg,
        feature_names=feature_names or list(X.columns),
        max_display=max_display,
        plot_type="bar",
        show=False,
    )
    plt.title(f"SHAP Bar – {panel_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "SHAP_bar.png", dpi=140, bbox_inches="tight")
    plt.close()

    return fi_df
