"""
predict.py
==========
Eğitilmiş bir CatBoost modeli + meta dosyasını kullanarak yeni varyant
verisi üzerinde tahmin üretir.

Kullanım
--------
    python predict.py --panel MASTER --input test.csv --output preds.csv

Beklenen Girdi
--------------
- CSV dosyası, eğitim verisiyle aynı kolon yapısına sahip (Variant_ID + AL_*,
  CAT_*, EK_*, AA_*). 'Label' kolonu olabilir veya olmayabilir.

Çıktı
-----
- CSV: Variant_ID, predicted_label (0/1), predicted_probability
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.preprocess import (  # noqa: E402
    preprocess_inference, add_missingness_features, add_engineered_features,
    median_imputation, fill_categorical_missing, CAT_MISSING_TOKEN,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Genetik Varyant Sınıflandırma – Tahmin")
    parser.add_argument("--panel", required=True, choices=["MASTER", "KANSER", "PAH", "CFTR"])
    parser.add_argument("--input", required=True, help="Tahmin edilecek CSV")
    parser.add_argument("--output", required=True, help="Çıktı CSV")
    parser.add_argument(
        "--models-dir",
        default=str(Path(__file__).parent / "models"),
        help="Eğitilmiş modellerin bulunduğu klasör",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = Path(args.models_dir) / args.panel
    meta_path = models_dir / "model_meta.json"
    cbm_path = models_dir / "catboost_model.cbm"
    stack_path = models_dir / "stacking_ensemble.pkl"  # ADIM 6 stacking destegi

    if not meta_path.exists() or (not cbm_path.exists() and not stack_path.exists()):
        raise SystemExit(
            f"Model veya meta bulunamadı.\n"
            f"  Meta:     {meta_path}  exists={meta_path.exists()}\n"
            f"  CatBoost: {cbm_path}   exists={cbm_path.exists()}\n"
            f"  Stacking: {stack_path} exists={stack_path.exists()}\n"
            f"Önce `python train.py --panels {args.panel}` çalıştırın."
        )

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise SystemExit("catboost kurulu değil: `pip install catboost`")

    with open(meta_path, "r", encoding="utf-8") as fp:
        meta = json.load(fp)

    selected = meta["selected_features"]
    cat_cols = meta["categorical_columns"]
    medians = pd.Series(meta["medians"])
    threshold = float(meta["final_threshold"])
    model_type = meta.get("model_type", "catboost")

    # Sayısal vs. kategorik tespiti (selected içinden)
    numeric_in_sel = [c for c in selected if c not in cat_cols]

    # Yükle
    df = pd.read_csv(args.input, low_memory=False)
    id_col = "Variant_ID" if "Variant_ID" in df.columns else None

    # ADIM 8/9 DÜZELTME: Türetilmiş özellikler (MISS__/ENG__) ham test verisinden
    # YENİDEN ÜRETİLMELİDİR. Aksi halde eksik sayılıp 0 atanır ve KANSER/CFTR gibi
    # bu özellikleri kullanan paneller yanlış tahmin üretir. Türetme, eğitimdeki ile
    # birebir aynı şekilde TÜM orijinal sayısal kolonlar üzerinden yapılır.
    numeric_full = [str(c) for c in medians.index]  # eğitimde medyani kaydedilen orijinal sayısal kolonlar
    for c in numeric_full:
        if c not in df.columns:
            df[c] = np.nan

    need_miss = any(str(c).startswith("MISS__") for c in selected)
    need_eng = any(str(c).startswith("ENG__") for c in selected)
    work = df.copy()
    if need_eng:
        work = add_engineered_features(work, numeric_full)
    if need_miss:
        work = add_missingness_features(work, numeric_full)
    work, _ = median_imputation(work, numeric_full, medians=medians)
    work = fill_categorical_missing(work, cat_cols)

    for c in selected:               # güvenlik: hâlâ eksik bir seçili kolon varsa
        if c not in work.columns:
            work[c] = 0.0
    X = work[selected].copy()

    # ADIM 6 (2026-06-19): Stacking ensemble destegi.
    # meta.model_type == "stacking_ensemble" ise pickle yukle (StackingEnsemble.predict_proba).
    # Aksi halde klasik tek-CatBoost yolu (geriye uyum).
    use_stack = (model_type == "stacking_ensemble") or (stack_path.exists() and not cbm_path.exists())
    if use_stack:
        import pickle
        with open(stack_path, "rb") as fp:
            ensemble = pickle.load(fp)
        print(f"[Predict] Stacking ensemble yuklendi: {stack_path}")
        proba = ensemble.predict_proba(X)[:, 1]
    else:
        model = CatBoostClassifier()
        model.load_model(str(cbm_path))
        print(f"[Predict] CatBoost modeli yuklendi: {cbm_path}")
        proba = model.predict_proba(X)[:, 1]

    pred = (proba >= threshold).astype(int)

    out = pd.DataFrame(
        {
            "Variant_ID": df[id_col] if id_col else range(len(df)),
            "predicted_label": pred,
            "predicted_probability": proba,
        }
    )
    out.to_csv(args.output, index=False)
    print(f"[OK] {len(out)} tahmin yazildi -> {args.output}")
    print(f"  Threshold: {threshold:.4f} | pos: {int(pred.sum())} | neg: {int((1-pred).sum())}")


if __name__ == "__main__":
    main()
