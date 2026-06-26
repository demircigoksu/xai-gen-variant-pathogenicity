"""
report_figures.py
=================
Eğitim çıktılarından (outputs/ veya outputs_reverse_eval/) RAPOR görsellerini üretir:
  - Panel başına karmaşıklık matrisi      -> cm_<PANEL>.png
  - Panel karşılaştırma grafiği            -> panel_comparison.png
  - MCC dahil metrik özeti                 -> report_metrics.json
  - train.py'nin ürettiği SHAP grafiklerini toplar -> SHAP_summary_<PANEL>.png

NOT: SHAP grafikleri eğitim sırasında train.py tarafından zaten
`<results-dir>/<PANEL>/SHAP_summary.png` olarak üretilir; bu betik onları rapor
klasörüne kopyalar ve karmaşıklık matrisi / karşılaştırma / MCC tablosunu ekler.

Kullanım:
    python report_figures.py --results-dir outputs_reverse_eval --out outputs_report
    python report_figures.py --results-dir outputs --out outputs_report   # eqdist için
"""
from __future__ import annotations
import argparse, json, math, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = ["MASTER", "KANSER", "PAH", "CFTR"]


def _mcc(tn, fp, fn, tp):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Rapor görselleri üretici")
    ap.add_argument("--results-dir", default="outputs_reverse_eval",
                    help="Eğitim sonuç klasörü (outputs veya outputs_reverse_eval)")
    ap.add_argument("--out", default="outputs_report", help="Görsellerin yazılacağı klasör")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    REV = base / args.results_dir
    OUT = base / args.out
    OUT.mkdir(parents=True, exist_ok=True)

    summ = pd.read_csv(REV / "all_panels_summary.csv")
    rows = {}
    plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

    for p in PANELS:
        fmp = REV / p / "fold_metrics.csv"
        if not fmp.exists():
            print(f"[atla] {p}: fold_metrics.csv yok")
            continue
        fm = pd.read_csv(fmp)
        tn, fp, fn, tp = [int(fm[c].sum()) for c in ["tn", "fp", "fn", "tp"]]
        sr = summ[summ["panel"] == p].iloc[0]
        rows[p] = dict(
            f1_pos=float(sr["f1_pos_mean"]), mcc=float(_mcc(tn, fp, fn, tp)),
            pr_auc=float(sr["pr_auc_mean"]), roc_auc=float(sr["roc_auc_mean"]),
            sensitivity=float(sr["sensitivity_mean"]), specificity=float(sr["specificity_mean"]),
            threshold=float(sr["final_threshold"]), tn=tn, fp=fp, fn=fn, tp=tp,
        )
        # Karmaşıklık matrisi (satır-normalize)
        cm = np.array([[tn, fp], [fn, tp]], float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "Patojenik"]); ax.set_yticklabels(["Benign", "Patojenik"])
        ax.set_xlabel("Tahmin"); ax.set_ylabel("Gercek")
        ax.set_title(f"{p} - Karmasiklik Matrisi")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n(n={int(cm[i, j])})", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=10)
        plt.tight_layout(); plt.savefig(OUT / f"cm_{p}.png"); plt.close()
        # SHAP grafiklerini topla (train.py üretmiş olur)
        for nm in ("SHAP_summary.png", "SHAP_bar.png"):
            sp = REV / p / nm
            if sp.exists():
                shutil.copy(sp, OUT / f"{nm.split('.')[0]}_{p}.png")

    json.dump(rows, open(OUT / "report_metrics.json", "w"), indent=2)
    print("Metrikler:")
    for p, r in rows.items():
        print(f"  {p}: F1={r['f1_pos']:.3f} MCC={r['mcc']:.3f} PR-AUC={r['pr_auc']:.3f} "
              f"ROC={r['roc_auc']:.3f} esik={r['threshold']:.3f}")

    # Panel karşılaştırma grafiği
    if rows:
        metrics = ["f1_pos", "mcc", "pr_auc", "roc_auc"]
        labels = ["F1-pos", "MCC", "PR-AUC", "ROC-AUC"]
        ps = [p for p in PANELS if p in rows]
        x = np.arange(len(ps)); w = 0.2
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for i, (mm, l) in enumerate(zip(metrics, labels)):
            ax.bar(x + (i - 1.5) * w, [rows[p][mm] for p in ps], w, label=l)
        ax.set_xticks(x); ax.set_xticklabels(ps); ax.set_ylim(0, 1); ax.set_ylabel("Skor")
        ax.axhline(0.333, ls="--", c="gray", lw=1, label="'Hep patojenik' F1 (0.33)")
        ax.set_title("Panel bazli basarim")
        ax.legend(ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        plt.tight_layout(); plt.savefig(OUT / "panel_comparison.png"); plt.close()

    print("Rapor gorselleri ->", OUT)


if __name__ == "__main__":
    main()
