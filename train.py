"""
train.py
========
Tum 4 paneli (MASTER, KANSER, PAH, CFTR) icin pipeline'i calistirir.

Yeni Ozellikler
---------------
- Tamamlanmis panelleri otomatik atlar (model_meta.json + catboost_model.cbm
  varsa). Yeniden egitmek icin --force kullanin.
- Birikimli rapor: outputs/all_panels_summary.csv + outputs/RESULTS_REPORT.md
  her calistirmada guncellenir, eski sonuclar kaybolmaz.

Kullanim
--------
    python train.py                              # tum 4 panel, tamamlanmislari atlar
    python train.py --quick                      # hizli mod
    python train.py --panels MASTER PAH          # sadece bu panellerde calis
    python train.py --force                      # tamamlanmislari da yeniden egit
    python train.py --force --panels KANSER      # tek paneli yeniden egit

Ciktilar
--------
- 4-Model/outputs/<PANEL>/   : metrics_summary.json, fold_metrics.csv,
                                SHAP grafikler
- 4-Model/models/<PANEL>/    : catboost_model.cbm, model_meta.json
- 4-Model/outputs/all_panels_summary.csv : 4 panel toplu skor karsilastirmasi
- 4-Model/outputs/RESULTS_REPORT.md      : insan-okunabilir markdown rapor
"""

from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
import pandas as pd

# src paketini import edebilmek icin
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pipeline import run_panel_pipeline  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Genetik Varyant Siniflandirma - Egitim")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="config.yaml yolu",
    )
    parser.add_argument(
        "--panels",
        nargs="*",
        default=None,
        help="Hangi paneller? (varsayilan: tumu). Secenekler: MASTER KANSER PAH CFTR",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Hizli modda calistir (5 MC tekrar, kucuk grid, alt-ornek)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tamamlanmis panelleri de yeniden egit",
    )
    parser.add_argument(
        "--reverse-test-dist",
        action="store_true",
        help=(
            "Test setini yarisma final kosullarina gore stratify et "
            "(~%80 benign / %20 patojenik). Hizli validasyon icin n_repeats=10."
        ),
    )
    parser.add_argument(
        "--reverse-eval-repeats",
        type=int,
        default=10,
        help="Reverse-dist degerlendirmede Monte Carlo iterasyon sayisi (default 10).",
    )
    return parser.parse_args()


def is_panel_completed(panel_name: str, outputs_root: Path, models_root: Path) -> bool:
    """Bir panelin tam olarak egitildigini iki dosyanin varligiyla dogrular."""
    model_file = models_root / panel_name / "catboost_model.cbm"
    meta_file = models_root / panel_name / "model_meta.json"
    metrics_file = outputs_root / panel_name / "metrics_summary.json"
    return model_file.exists() and meta_file.exists() and metrics_file.exists()


def load_panel_results(panel_name: str, outputs_root: Path, models_root: Path) -> dict:
    """Tamamlanmis bir panelin meta + metrics dosyalarindan ozet uretir."""
    meta_file = models_root / panel_name / "model_meta.json"
    with open(meta_file, "r", encoding="utf-8") as fp:
        meta = json.load(fp)
    ms = meta.get("metrics_summary", {})
    # ADIM 6 (2026-06-19): f1_pos -> yarisma resmi metrigi (patojenik F1).
    # f1_macro hala raporlaniyor (geriye uyum + bilgi).
    return {
        "panel": panel_name,
        "f1_pos_mean": ms.get("f1_pos", {}).get("mean"),
        "f1_pos_std": ms.get("f1_pos", {}).get("std"),
        "f1_macro_mean": ms.get("f1_macro", {}).get("mean"),
        "f1_macro_std": ms.get("f1_macro", {}).get("std"),
        "accuracy_mean": ms.get("accuracy", {}).get("mean"),
        "roc_auc_mean": ms.get("roc_auc", {}).get("mean"),
        "pr_auc_mean": ms.get("pr_auc", {}).get("mean"),
        "cohen_kappa_mean": ms.get("cohen_kappa", {}).get("mean"),
        "sensitivity_mean": ms.get("sensitivity", {}).get("mean"),
        "specificity_mean": ms.get("specificity", {}).get("mean"),
        "n_features": len(meta.get("selected_features", [])),
        "best_params": json.dumps(meta.get("best_params", {}), ensure_ascii=False),
        "final_threshold": meta.get("final_threshold"),
        "completed_at": meta.get("__completed_at__", "bilinmiyor"),
    }


def update_summary_csv(outputs_root: Path, new_rows: list, all_panel_names: list):
    """all_panels_summary.csv'yi birikimli olarak gunceller (uzerine yazmaz)."""
    summary_path = outputs_root / "all_panels_summary.csv"
    existing_rows = []
    if summary_path.exists():
        try:
            existing_df = pd.read_csv(summary_path)
            existing_rows = existing_df.to_dict("records")
        except Exception:
            existing_rows = []

    # Yeni sonuclar eskilerin uzerine yazar (ayni panel)
    new_panels = {r["panel"] for r in new_rows}
    merged = [r for r in existing_rows if r.get("panel") not in new_panels] + new_rows

    # Panelleri sirali tutmaya calis (config sirasiyla)
    order_map = {p: i for i, p in enumerate(all_panel_names)}
    merged.sort(key=lambda r: order_map.get(r.get("panel"), 99))

    df = pd.DataFrame(merged)
    df.to_csv(summary_path, index=False)
    return df


def write_results_report(outputs_root: Path, df: pd.DataFrame, quick_mode: bool):
    """Insan-okunabilir markdown rapor olusturur."""
    report_path = outputs_root / "RESULTS_REPORT.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = "QUICK (hizli dogrulama)" if quick_mode else "FULL (PDF birebir)"

    lines = []
    lines.append(f"# Genetik Varyant Siniflandirma - Sonuc Raporu\n")
    lines.append(f"_Son guncelleme: {now}_  \n")
    lines.append(f"_Calistirma modu: **{mode}**_\n")
    lines.append("\n## Panel Bazli Sonuclar\n")

    if df.empty:
        lines.append("\n_Henuz tamamlanmis panel yok._\n")
    else:
        # Ana metrik tablosu — ADIM 6: F1-Pos (yarisma resmi) one cikti, F1-Macro bilgi.
        lines.append("| Panel | **F1-Pos** | F1-Macro | Accuracy | ROC-AUC | PR-AUC | Cohen Kappa | Sensitivity | Specificity | # Feature |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, row in df.iterrows():
            def fmt(v, digits=4):
                if pd.isna(v):
                    return "—"
                try:
                    return f"{float(v):.{digits}f}"
                except Exception:
                    return str(v)
            f1p = fmt(row.get("f1_pos_mean"))
            f1p_std = fmt(row.get("f1_pos_std"))
            f1 = fmt(row.get("f1_macro_mean"))
            f1_std = fmt(row.get("f1_macro_std"))
            lines.append(
                f"| **{row.get('panel')}** | **{f1p} ± {f1p_std}** | {f1} ± {f1_std} | "
                f"{fmt(row.get('accuracy_mean'))} | "
                f"{fmt(row.get('roc_auc_mean'))} | "
                f"{fmt(row.get('pr_auc_mean'))} | "
                f"{fmt(row.get('cohen_kappa_mean'))} | "
                f"{fmt(row.get('sensitivity_mean'))} | "
                f"{fmt(row.get('specificity_mean'))} | "
                f"{int(row['n_features']) if pd.notna(row.get('n_features')) else '—'} |"
            )

        # Panel basina best_params + threshold
        lines.append("\n## Hiperparametre Detaylari\n")
        for _, row in df.iterrows():
            lines.append(f"\n### {row.get('panel')}\n")
            lines.append(f"- **Best Params**: `{row.get('best_params', '—')}`")
            ft = row.get("final_threshold")
            ft_str = f"{float(ft):.4f}" if pd.notna(ft) else "—"
            lines.append(f"- **Final Threshold**: {ft_str}")
            lines.append(f"- **Tamamlanma**: {row.get('completed_at', '—')}")

        # Ortalama — ADIM 6: F1-Pos one cikti, F1-Macro de raporlandi.
        try:
            if "f1_pos_mean" in df.columns:
                avg_f1p = df["f1_pos_mean"].astype(float).mean()
                lines.append(f"\n## Panel-Ortalama F1-Pos (Yarisma Resmi): **{avg_f1p:.4f}**\n")
            avg_f1 = df["f1_macro_mean"].astype(float).mean()
            lines.append(f"## Panel-Ortalama F1-Macro (bilgi amacli): {avg_f1:.4f}\n")
        except Exception:
            pass

    lines.append("\n---\n")
    lines.append("\n## PDF Uyum Notu\n")
    lines.append("\nBu rapor PDF Proje Sunus Raporu'na birebir uyumlu pipeline ile uretildi:")
    lines.append("- 100-Repeated Monte Carlo Cross-Validation (PDF 4.1)")
    lines.append("- Sirali Ileri Ozellik Secimi - SFS (PDF 3.6)")
    lines.append("- Train-Only SMOTE + Cost-Sensitive Learning (PDF 3.5)")
    lines.append("- CatBoost yerlesik Grid Search (PDF 5.3)")
    lines.append("- F1-Macro hedefli threshold tuning (PDF 4.2)")
    lines.append("- SHAP aciklamabilirlik (PDF 4.4)\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    data_dir = (base_dir / config["paths"]["data_dir"]).resolve()
    outputs_root = base_dir / config["paths"]["outputs_dir"]
    models_root = base_dir / config["paths"]["models_dir"]

    # ADIM 5: Reverse-dist degerlendirme modu
    # ----------------------------------------------
    # Egitim seti orijinal kalir (~%80 patojenik), ama test seti
    # yarismadaki final kosullarina cevrilir (~%80 benign / %20 patojenik).
    # Sonuclar ayri bir klasore yazilir, eski sonuclar bozulmaz.
    if args.reverse_test_dist:
        config.setdefault("monte_carlo_cv", {})
        config["monte_carlo_cv"]["reverse_test_dist"] = True
        config["monte_carlo_cv"]["reverse_test_benign_frac"] = 0.8
        config["monte_carlo_cv"]["n_repeats"] = int(args.reverse_eval_repeats)
        outputs_root = outputs_root.parent / (outputs_root.name + "_reverse_eval")
        models_root = models_root.parent / (models_root.name + "_reverse_eval")
        print(
            "[REVERSE-EVAL] Test setinde %80 benign / %20 patojenik dagilimi kullanilacak. "
            f"n_repeats={args.reverse_eval_repeats}. "
            f"Sonuclar: {outputs_root}, {models_root}"
        )

    outputs_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    all_panel_names = list(config["panels"].keys())
    panel_names = args.panels or all_panel_names

    summary_rows = []
    skipped = []

    for name in panel_names:
        info = config["panels"][name]
        csv_path = data_dir / info["file"]
        if not csv_path.exists():
            print(f"[UYARI] {csv_path} bulunamadi, atlaniyor.")
            continue

        out_dir = outputs_root / name
        mod_dir = models_root / name

        # Skip-completed kontrolu
        if not args.force and is_panel_completed(name, outputs_root, models_root):
            print(f"\n[ATLA] {name} zaten egitilmis (model_meta.json + catboost_model.cbm + metrics_summary.json var).")
            print(f"       Yeniden egitmek icin: python train.py --force --panels {name}")
            try:
                summary_rows.append(load_panel_results(name, outputs_root, models_root))
                skipped.append(name)
            except Exception as e:
                print(f"       Mevcut sonuclari okurken hata: {e}")
            continue

        # Egit
        try:
            res = run_panel_pipeline(
                panel_name=name,
                csv_path=str(csv_path),
                config=config,
                outputs_dir=str(out_dir),
                models_dir=str(mod_dir),
                quick=args.quick,
            )
            # Tamamlanma zaman damgasi ekle (meta'ya)
            meta_path = mod_dir / "model_meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as fp:
                    meta = json.load(fp)
                meta["__completed_at__"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                meta["__mode__"] = "quick" if args.quick else "full"
                with open(meta_path, "w", encoding="utf-8") as fp:
                    json.dump(meta, fp, ensure_ascii=False, indent=2)

            row = {
                "panel": name,
                "f1_pos_mean": res["metrics_summary"].get("f1_pos", {}).get("mean"),
                "f1_pos_std": res["metrics_summary"].get("f1_pos", {}).get("std"),
                "f1_macro_mean": res["metrics_summary"].get("f1_macro", {}).get("mean"),
                "f1_macro_std": res["metrics_summary"].get("f1_macro", {}).get("std"),
                "accuracy_mean": res["metrics_summary"].get("accuracy", {}).get("mean"),
                "roc_auc_mean": res["metrics_summary"].get("roc_auc", {}).get("mean"),
                "pr_auc_mean": res["metrics_summary"].get("pr_auc", {}).get("mean"),
                "cohen_kappa_mean": res["metrics_summary"].get("cohen_kappa", {}).get("mean"),
                "sensitivity_mean": res["metrics_summary"].get("sensitivity", {}).get("mean"),
                "specificity_mean": res["metrics_summary"].get("specificity", {}).get("mean"),
                "n_features": len(res["selected_features"]),
                "best_params": json.dumps(res["best_params"], ensure_ascii=False),
                "final_threshold": res["final_threshold"],
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            summary_rows.append(row)

            # Her panelden sonra ozet dosyalari guncelle (early-interrupt korumasi)
            df = update_summary_csv(outputs_root, [row], all_panel_names)
            # RESULTS_REPORT.md'yi de guncel tum verilerle yaz
            full_df = pd.read_csv(outputs_root / "all_panels_summary.csv")
            write_results_report(outputs_root, full_df, args.quick)
            print(f"[KAYIT] {name} sonuclari all_panels_summary.csv + RESULTS_REPORT.md'ye yazildi.")
        except KeyboardInterrupt:
            print(f"        Tekrar denemek icin: python train.py --panels {name}")
            sys.exit(130)
        except Exception as e:
            print(f"[HATA] {name}: {e}")
            import traceback
            traceback.print_exc()

    # Final ozet
    if summary_rows:
        # Tum sonuclari (yeni + eski) tek bir dataframe'e topla
        final_df = update_summary_csv(outputs_root, [], all_panel_names)
        report_path = write_results_report(outputs_root, final_df, args.quick)

        print("\n" + "=" * 60)
        print("  TUM PANELLER OZETI")
        print("=" * 60)
        print(final_df.to_string(index=False))
        print()
        print(f"Ozet CSV : {outputs_root / 'all_panels_summary.csv'}")
        print(f"Markdown : {report_path}")
        if skipped:
            print(f"Atlanan  : {skipped} (--force ile yeniden egitilebilir)")


if __name__ == "__main__":
    main()
