"""
pipeline.py
===========
Tüm pipeline'ı orkestre eden ana modül.

Akış (PDF Bölüm 3):
  1. Veri yükleme (data_loader)
  2. Ön işleme (preprocess) – medyan imputasyon, kalite kontrolü
  3. Train-Only SMOTE + Cost-Sensitive (balance)        -> CV içinde
  4. SFS özellik seçimi (feature_selection)             -> bir kez (tüm panel üzerinde)
  5. CatBoost grid search + final fit (model)
  6. 100-Repeated Monte Carlo CV (evaluation)
  7. SHAP analizi (explain)
  8. Modeli ve raporları diske yazma
"""

from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import PanelData, load_panel, summarize_panel
from .preprocess import preprocess_train, preprocess_inference, PreprocessReport
from .balance import apply_train_only_smote, compute_class_weights
from .feature_selection import sequential_forward_selection, SFSResult
from .model import (
    grid_search_catboost,
    fit_final_catboost,
    tune_threshold_for_macro_f1,
    tune_threshold,
    fit_calibrator,
    TrainedModel,
)
from .evaluation import monte_carlo_cv, aggregate_metrics, compute_metrics, FoldResult
# ADIM 5: reverse_dist_monte_carlo_cv conditional olarak alttaki branch'ta import edilir.
from .explain import shap_summary


def _serialize(obj):
    """JSON dump'lanabilir yapıya çevirir."""
    if is_dataclass(obj):
        return _serialize(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


# ---------------------------------------------------------------
# Tek-panel akışı
# ---------------------------------------------------------------

def run_panel_pipeline(
    panel_name: str,
    csv_path: str,
    *,
    config: Dict,
    outputs_dir: str,
    models_dir: str,
    quick: bool = False,
) -> Dict:
    """
    Tek bir panel için uçtan uca pipeline'ı çalıştırır ve sonuç sözlüğü döner.
    """
    outputs_dir = Path(outputs_dir)
    models_dir = Path(models_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    pre = dict(config["preprocessing"])
    fs = dict(config["feature_selection"])
    mc = dict(config["monte_carlo_cv"])
    grid = dict(config["hyperparameter_grid"])
    fixed = config["catboost_fixed"]
    expl = config["explainability"]
    rt = config["runtime"]
    cb_balance = dict(config["class_balance"])

    # Panel-bazli overrides (PAH, CFTR gibi paneller icin ozel ayarlar)
    panel_cfg = config["panels"].get(panel_name, {})
    panel_overrides = panel_cfg.get("overrides", {}) or {}
    if panel_overrides:
        print(f"[Override] {panel_name} icin ozel ayarlar: {list(panel_overrides.keys())}")
        if "feature_selection" in panel_overrides:
            fs.update(panel_overrides["feature_selection"])
        if "class_balance" in panel_overrides:
            cb_balance.update(panel_overrides["class_balance"])
        if "preprocessing" in panel_overrides:
            pre.update(panel_overrides["preprocessing"])
        if "hyperparameter_grid" in panel_overrides:
            grid = dict(panel_overrides["hyperparameter_grid"])
        if "monte_carlo_cv" in panel_overrides:
            mc.update(panel_overrides["monte_carlo_cv"])

    if quick:
        ov = rt["quick_overrides"]
        mc = dict(mc)
        mc["n_repeats"] = ov["monte_carlo_cv_n_repeats"]
        fs = dict(fs)
        fs["n_features_to_select"] = ov["feature_selection_n_features_to_select"]
        fs["early_stop_patience"] = ov.get("feature_selection_early_stop_patience", 3)
        fs["min_score_improvement"] = ov.get("feature_selection_min_score_improvement", 0.002)
        # SFS hızlandırma override'ları
        fs["sfs_estimator_iterations"] = ov.get("feature_selection_sfs_iterations", fs.get("sfs_estimator_iterations", 200))
        fs["cv_folds"] = ov.get("feature_selection_sfs_cv_folds", fs.get("cv_folds", 3))
        fs["__subsample"] = ov.get("feature_selection_subsample", None)
        grid = ov["hyperparameter_grid"]

    # 1. Yükle
    print(f"\n========== [{panel_name}] Pipeline Başladı ==========")
    panel = load_panel(
        panel_name,
        csv_path,
        categorical_prefixes=pre["categorical_prefixes"],
        numeric_prefixes=pre["numeric_prefixes"],
        id_column=pre["id_columns"][0] if pre["id_columns"] else "Variant_ID",
        target_column=pre["target_column"],
    )
    print(summarize_panel(panel))

    # 2. Ön işleme (eğitim)
    X_clean, y_clean, medians, prep_report = preprocess_train(
        panel.X,
        panel.y,
        numeric_columns=panel.numeric_columns,
        categorical_columns=panel.categorical_columns,
        drop_full_duplicates=pre["drop_full_duplicates"],
        drop_conflict_label_duplicates=pre["drop_conflict_label_duplicates"],
        outlier_iqr_multiplier=pre["outlier_detection"]["iqr_multiplier"],
        add_missingness=pre.get("add_missingness_features", False),
        add_engineered=pre.get("add_engineered_features", False),
    )
    print(f"[Preprocess] {prep_report.to_dict()}")

    # 3. SFS – bir kez tüm panel üzerinde
    if fs.get("enabled", True):
        # Quick mod için isteğe bağlı alt-örnekleme
        sub_n = fs.get("__subsample")
        if sub_n and len(X_clean) > sub_n:
            print(f"[SFS] quick subsample: {sub_n} satır (toplam {len(X_clean)})")
            from sklearn.model_selection import StratifiedShuffleSplit
            sss_sub = StratifiedShuffleSplit(
                n_splits=1, train_size=sub_n, random_state=rt["random_seed"]
            )
            sub_idx, _ = next(sss_sub.split(X_clean, y_clean))
            X_sfs = X_clean.iloc[sub_idx].reset_index(drop=True)
            y_sfs = y_clean[sub_idx]
        else:
            X_sfs = X_clean
            y_sfs = y_clean
        print(f"[SFS] {fs['n_features_to_select']} özellik seçilecek...")
        sfs_res = sequential_forward_selection(
            X_sfs,
            y_sfs,
            categorical_columns=panel.categorical_columns,
            n_features_to_select=fs["n_features_to_select"],
            cv_folds=fs["cv_folds"],
            scoring=fs["scoring"],
            iterations=fs["sfs_estimator_iterations"],
            depth=fs["sfs_estimator_depth"],
            learning_rate=fs["sfs_estimator_learning_rate"],
            random_state=rt["random_seed"],
            n_jobs=rt["n_jobs"],
            verbose=True,
            early_stop_patience=fs.get("early_stop_patience", 3),
            min_score_improvement=fs.get("min_score_improvement", 0.001),
        )
        selected_features = sfs_res.selected_features
        sfs_history = sfs_res.selection_history
        print(f"[SFS] seçilen {len(selected_features)} özellik: {selected_features[:10]}...")
    else:
        selected_features = list(X_clean.columns)
        sfs_history = []

    cat_cols_in_sel = [c for c in selected_features if c in panel.categorical_columns]

    X_sel = X_clean[selected_features].copy()

    # 4. Sınıf ağırlıkları (cost-sensitive)
    class_weights = (
        compute_class_weights(
            y_clean,
            manual_multiplier=cb_balance.get("manual_class_weight_multiplier"),
        ) if cb_balance.get("cost_sensitive", True) else None
    )
    print(f"[Balance] class_weights = {class_weights}")

    # 5. Hiperparametre Grid Search (tek seferlik, full-data üzerinde)
    print("[GridSearch] başlıyor...")
    best_params = grid_search_catboost(
        X_sel,
        y_clean,
        categorical_columns=cat_cols_in_sel,
        grid=grid,
        fixed_params=fixed,
        class_weights=class_weights,
        cv_folds=fs["cv_folds"],
        random_state=rt["random_seed"],
        verbose=True,
    )

    # ADIM 4 (2026-05-24): Stacking ensemble bayragi.
    # use_stacking: true ise CatBoost+LGBM+XGB -> Meta-CatBoost.
    use_stacking = bool(cb_balance.get("use_stacking", False))
    if use_stacking:
        print(f"[Stacking] Adim 4 aktif: heterojen GBDT stack (3 base + meta-CatBoost)")
        from .stacking import StackingEnsemble
        stacking_inner_folds = int(cb_balance.get("stacking_inner_folds", 3))
        stacking_n_seeds = int(cb_balance.get("stacking_n_seeds", 2))
        stacking_base_iters = int(cb_balance.get("stacking_base_iterations", 300))

    # 6. 100-Repeated Monte Carlo CV
    # ADIM 8 (2026-06-22) PRIOR-SHIFT: esik, TEST dagilimina (deploy_pos_frac =
    # patojenik orani, orn 0.2) gore EGITIM verisi uzerinde secilir. Boylece
    # test fold'una sizma olmaz ve esik gercek yarisma kosulunu (%20 patojenik)
    # hedefler. None ise eski davranis (esik val fold'unda secilir).
    deploy_pos_frac = cb_balance.get("deployment_test_pos_frac")

    def _fit_predict(X_tr, y_tr, X_va, y_va):
        # --- ADIM 4: Stacking ensemble dali ---
        if use_stacking:
            ensemble = StackingEnsemble(
                catboost_params=best_params,
                catboost_fixed=fixed,
                cat_features=cat_cols_in_sel,
                cb_balance=cb_balance,
                n_inner_folds=stacking_inner_folds,
                n_seeds=stacking_n_seeds,
                base_iterations=stacking_base_iters,
                random_state=int(fixed.get("random_seed", 42)),
            )
            ensemble.fit(X_tr.reset_index(drop=True), np.asarray(y_tr), verbose=False)
            proba = ensemble.predict_proba(X_va.reset_index(drop=True))[:, 1]
            if deploy_pos_frac is not None:
                # ADIM 8: esik egitim folduna gore (prior-shift agirlikli) secilir
                proba_tr = ensemble.predict_proba(X_tr.reset_index(drop=True))[:, 1]
                best_t, _ = tune_threshold(
                    y_tr, proba_tr,
                    strategy=cb_balance.get("threshold_strategy", "f1_pos_max"),
                    step=cb_balance["threshold_search_step"],
                    test_pos_frac=deploy_pos_frac,
                )
            else:
                best_t, _ = tune_threshold(
                    y_va, proba,
                    strategy=cb_balance.get("threshold_strategy", "macro_f1_max"),
                    step=cb_balance["threshold_search_step"],
                )
            pred = (proba >= best_t).astype(int)
            return pred, proba, best_t

        # --- Klasik CatBoost-only yol (Adim 4 kapali) ---
        # ADIM 2: Probabilistic kalibrasyon (Platt/Isotonic) — opsiyonel.
        # Eger calibration != "none" ise train icinden holdout ayir.
        calib_method = (cb_balance.get("calibration", "none") or "none").lower()
        holdout_frac = float(cb_balance.get("calibration_holdout_fraction", 0.2))
        min_minority = int(cb_balance.get("calibration_min_minority", 5))

        X_tr_main, y_tr_main = X_tr, y_tr
        X_calib, y_calib = None, None
        do_calibration = False

        if calib_method not in ("none", ""):
            try:
                from sklearn.model_selection import train_test_split
                from collections import Counter
                X_main_try, X_calib_try, y_main_try, y_calib_try = train_test_split(
                    X_tr, y_tr,
                    test_size=holdout_frac,
                    stratify=y_tr,
                    random_state=42,
                )
                calib_counts = Counter(y_calib_try.tolist())
                if calib_counts and min(calib_counts.values()) >= min_minority:
                    X_tr_main = X_main_try.reset_index(drop=True)
                    y_tr_main = np.asarray(y_main_try)
                    X_calib = X_calib_try.reset_index(drop=True)
                    y_calib = np.asarray(y_calib_try)
                    do_calibration = True
            except (ValueError, KeyError):
                # Stratify basarisiz veya yetersiz ornek — kalibrasyonsuz devam
                pass

        # Train-Only SMOTE (sadece train_main uzerinde)
        if cb_balance.get("use_smote", True):
            X_tr_b, y_tr_b = apply_train_only_smote(
                X_tr_main,
                y_tr_main,
                categorical_columns=cat_cols_in_sel,
                k_neighbors=cb_balance["smote_k_neighbors"],
                random_state=cb_balance["smote_random_state"],
                method=cb_balance.get("smote_method", "SMOTENC"),
            )
        else:
            X_tr_b, y_tr_b = X_tr_main, y_tr_main

        local_class_weights = (
            compute_class_weights(
                y_tr_b,
                manual_multiplier=cb_balance.get("manual_class_weight_multiplier"),
            ) if cb_balance.get("cost_sensitive", True) else None
        )
        model = fit_final_catboost(
            X_tr_b,
            y_tr_b,
            categorical_columns=cat_cols_in_sel,
            best_params=best_params,
            fixed_params=fixed,
            class_weights=local_class_weights,
            eval_set=(X_va, y_va),
            early_stopping_rounds=fixed.get("od_wait", 50),
        )

        # Kalibrator fit (varsa) ve val probalarini transform et
        calibrator = None
        if do_calibration:
            try:
                raw_calib = model.predict_proba(X_calib)[:, 1]
                calibrator = fit_calibrator(calib_method, raw_calib, y_calib)
            except Exception as e:
                print(f"[Calibration] fit basarisiz ({e}); raw proba ile devam.")
                calibrator = None

        raw_val = model.predict_proba(X_va)[:, 1]
        if calibrator is not None:
            proba = calibrator.transform(raw_val)
        else:
            proba = raw_val

        # Threshold tuning — ADIM 8: prior-shift ise esik egitim folduna gore secilir
        if deploy_pos_frac is not None:
            raw_tr = model.predict_proba(X_tr_main)[:, 1]
            proba_tr = calibrator.transform(raw_tr) if calibrator is not None else raw_tr
            best_t, _ = tune_threshold(
                y_tr_main, proba_tr,
                strategy=cb_balance.get("threshold_strategy", "f1_pos_max"),
                step=cb_balance["threshold_search_step"],
                test_pos_frac=deploy_pos_frac,
            )
        else:
            best_t, _ = tune_threshold(
                y_va, proba,
                strategy=cb_balance.get("threshold_strategy", "macro_f1_max"),
                step=cb_balance["threshold_search_step"],
            )
        pred = (proba >= best_t).astype(int)
        return pred, proba, best_t

    use_reverse_dist = bool(mc.get("reverse_test_dist", False))
    if use_reverse_dist:
        from .evaluation import reverse_dist_monte_carlo_cv
        target_b = float(mc.get("reverse_test_benign_frac", 0.8))
        print(
            f"[ReverseDistCV] n_repeats={mc['n_repeats']} test_size={mc['test_size']} "
            f"test_benign_frac={target_b}"
        )
        folds = reverse_dist_monte_carlo_cv(
            _fit_predict,
            X_sel,
            y_clean,
            n_repeats=mc["n_repeats"],
            test_size=mc["test_size"],
            target_test_benign_frac=target_b,
            random_state_base=mc["random_state_base"],
            progress=True,
        )
    else:
        print(f"[MonteCarloCV] n_repeats={mc['n_repeats']} test_size={mc['test_size']}")
        folds = monte_carlo_cv(
            _fit_predict,
            X_sel,
            y_clean,
            n_repeats=mc["n_repeats"],
            test_size=mc["test_size"],
            random_state_base=mc["random_state_base"],
            progress=True,
        )
    metrics_summary = aggregate_metrics(folds)
    # ADIM 6 (2026-06-19): F1-pos (patojenik) yarisma resmi metrigi; F1-macro bilgi amacli.
    print(
        f"[Metrics-Mean] F1-Pos: {metrics_summary.get('f1_pos', {}).get('mean', float('nan')):.4f} "
        f"± {metrics_summary.get('f1_pos', {}).get('std', float('nan')):.4f}  "
        f"| F1-Macro: {metrics_summary.get('f1_macro', {}).get('mean', float('nan')):.4f} "
        f"± {metrics_summary.get('f1_macro', {}).get('std', float('nan')):.4f}"
    )

    # --- ADIM 4: Final model stacking dali ---
    if use_stacking:
        print("[Stacking] Final model: full-data stacking ensemble fit ediliyor...")
        final_ensemble = StackingEnsemble(
            catboost_params=best_params,
            catboost_fixed=fixed,
            cat_features=cat_cols_in_sel,
            cb_balance=cb_balance,
            n_inner_folds=stacking_inner_folds,
            n_seeds=stacking_n_seeds,
            base_iterations=stacking_base_iters,
            random_state=rt["random_seed"],
        )
        final_ensemble.fit(X_sel.reset_index(drop=True), np.asarray(y_clean), verbose=True)

        proba_full = final_ensemble.predict_proba(X_sel)[:, 1]
        final_threshold, _ = tune_threshold(
            y_clean, proba_full,
            strategy=cb_balance.get("threshold_strategy", "f1_pos_max"),
            step=cb_balance["threshold_search_step"],
            test_pos_frac=deploy_pos_frac,
        )
        if deploy_pos_frac is not None:
            print(f"[PriorShift] deployment esigi (test prior={deploy_pos_frac}): {final_threshold:.3f}")
        final_pred = (proba_full >= final_threshold).astype(int)
        final_train_metrics = compute_metrics(y_clean, final_pred, proba_full)

        # ADIM 10 (2026-06-25): SHAP (PSR 4.4/5.5 taahhudu). Stacking'te temsil
        # olarak full-data base CatBoost uzerinde TreeExplainer ile uretilir.
        if expl.get("enabled", True):
            try:
                base_cat = final_ensemble.full_base_models_.get("cat", [None])
                base_cat = base_cat[0] if base_cat else None
                if base_cat is not None:
                    shap_summary(
                        base_cat,
                        X_sel,
                        feature_names=selected_features,
                        output_dir=outputs_dir,
                        panel_name=panel_name,
                        max_display=expl["shap_max_display"],
                        background_sample_size=expl["background_sample_size"],
                    )
                    print("[SHAP] Stacking base-CatBoost uzerinde SHAP uretildi.")
                else:
                    print("[SHAP] base CatBoost bulunamadi; atlandi.")
            except Exception as e:
                print(f"[SHAP] hata: {e}")

        # Stacking model + meta
        import pickle as _pickle
        stack_path = models_dir / "stacking_ensemble.pkl"
        with open(stack_path, "wb") as fp:
            _pickle.dump(final_ensemble, fp)
        print(f"[Stacking] kayit: {stack_path}")

        meta = {
            "panel": panel_name,
            "selected_features": selected_features,
            "sfs_history": sfs_history,
            "categorical_columns": cat_cols_in_sel,
            "medians": medians.to_dict(),
            "best_params": best_params,
            "fixed_params": fixed,
            "class_weights": class_weights,
            "final_threshold": float(final_threshold),
            "metrics_summary": metrics_summary,
            "fold_metrics": [
                {"fold": f.fold_id, "threshold": f.threshold, **f.metrics}
                for f in folds
            ],
            "preprocessing_report": prep_report.to_dict(),
            "final_train_metrics": final_train_metrics,
            "model_type": "stacking_ensemble",
            "stacking": {
                "inner_folds": stacking_inner_folds,
                "n_seeds": stacking_n_seeds,
                "base_iterations": stacking_base_iters,
            },
        }
        with open(models_dir / "model_meta.json", "w", encoding="utf-8") as fp:
            json.dump(_serialize(meta), fp, ensure_ascii=False, indent=2)
        with open(outputs_dir / "metrics_summary.json", "w", encoding="utf-8") as fp:
            json.dump(_serialize(metrics_summary), fp, ensure_ascii=False, indent=2)

        fold_df = pd.DataFrame(
            [{"fold": f.fold_id, "threshold": f.threshold, **f.metrics} for f in folds]
        )
        fold_df.to_csv(outputs_dir / "fold_metrics.csv", index=False)

        print(f"========== [{panel_name}] Pipeline Tamamlandi (stacking) ==========")
        return {
            "panel": panel_name,
            "model_path": str(stack_path),
            "metrics_summary": metrics_summary,
            "selected_features": selected_features,
            "best_params": best_params,
            "final_threshold": float(final_threshold),
            "model_type": "stacking_ensemble",
        }

    # --- Klasik CatBoost-only final model yolu (Adim 4 kapali) ---
    # 7. Final model: kalibrasyon stratejisi
    # Eger calibration aktifse:
    #   - Train'i 80/20 holdout'a ayir
    #   - 80% uzerinde SMOTE + CatBoost fit -> deployment model
    #   - 20% (holdout, SMOTE'siz) uzerinde Platt/Isotonic fit -> calibrator
    #   - Boylece MC CV'deki davranis tutarli sekilde deployment'a tasinir.
    # Aksi halde eski yol (full data + raw proba) korunur.
    final_calib_method = (cb_balance.get("calibration", "none") or "none").lower()
    final_calibrator = None

    if final_calib_method not in ("none", ""):
        from sklearn.model_selection import train_test_split
        from collections import Counter
        holdout_frac = float(cb_balance.get("calibration_holdout_fraction", 0.2))
        min_minority = int(cb_balance.get("calibration_min_minority", 5))
        try:
            X_main_f, X_calib_f, y_main_f, y_calib_f = train_test_split(
                X_sel, y_clean,
                test_size=holdout_frac,
                stratify=y_clean,
                random_state=rt["random_seed"],
            )
            calib_counts = Counter(np.asarray(y_calib_f).tolist())
            if calib_counts and min(calib_counts.values()) >= min_minority:
                X_train_for_final = X_main_f.reset_index(drop=True)
                y_train_for_final = np.asarray(y_main_f)
                X_calib_final = X_calib_f.reset_index(drop=True)
                y_calib_final = np.asarray(y_calib_f)
            else:
                print(f"[FinalCalibration] holdout azinligi yetersiz ({min(calib_counts.values()) if calib_counts else 0}<{min_minority}); kalibrasyonsuz fit.")
                X_train_for_final = X_sel
                y_train_for_final = y_clean
                X_calib_final = None
                final_calib_method = "none"
        except (ValueError, KeyError) as e:
            print(f"[FinalCalibration] split basarisiz ({e}); kalibrasyonsuz fit.")
            X_train_for_final = X_sel
            y_train_for_final = y_clean
            X_calib_final = None
            final_calib_method = "none"
    else:
        X_train_for_final = X_sel
        y_train_for_final = y_clean
        X_calib_final = None

    # Train-Only SMOTE (sadece training porsiyonu uzerinde)
    if cb_balance.get("use_smote", True):
        X_final, y_final = apply_train_only_smote(
            X_train_for_final,
            y_train_for_final,
            categorical_columns=cat_cols_in_sel,
            k_neighbors=cb_balance["smote_k_neighbors"],
            random_state=cb_balance["smote_random_state"],
            method=cb_balance.get("smote_method", "SMOTENC"),
        )
    else:
        X_final, y_final = X_train_for_final, y_train_for_final

    final_class_weights = (
        compute_class_weights(
            y_final,
            manual_multiplier=cb_balance.get("manual_class_weight_multiplier"),
        ) if cb_balance.get("cost_sensitive", True) else None
    )
    final_model = fit_final_catboost(
        X_final,
        y_final,
        categorical_columns=cat_cols_in_sel,
        best_params=best_params,
        fixed_params=fixed,
        class_weights=final_class_weights,
        eval_set=None,
        early_stopping_rounds=fixed.get("od_wait", 50),
    )

    # Kalibratoru fit et (holdout uzerinde)
    if final_calib_method not in ("none", "") and X_calib_final is not None:
        try:
            raw_calib = final_model.predict_proba(X_calib_final)[:, 1]
            final_calibrator = fit_calibrator(final_calib_method, raw_calib, y_calib_final)
            print(f"[FinalCalibration] {final_calib_method} fitted on {len(y_calib_final)} samples")
        except Exception as e:
            print(f"[FinalCalibration] fit hata ({e}); kalibrasyonsuz devam.")
            final_calibrator = None

    # 7.1 Final threshold (kalibre edilmis proba'lar uzerinde)
    raw_full = final_model.predict_proba(X_sel)[:, 1]
    if final_calibrator is not None:
        proba_full = final_calibrator.transform(raw_full)
    else:
        proba_full = raw_full
    final_threshold, _ = tune_threshold(
        y_clean, proba_full,
        strategy=cb_balance.get("threshold_strategy", "f1_pos_max"),
        step=cb_balance["threshold_search_step"],
        test_pos_frac=deploy_pos_frac,
    )
    if deploy_pos_frac is not None:
        print(f"[PriorShift] deployment esigi (test prior={deploy_pos_frac}): {final_threshold:.3f}")
    final_pred = (proba_full >= final_threshold).astype(int)
    final_train_metrics = compute_metrics(y_clean, final_pred, proba_full)

    # 8. SHAP
    if expl.get("enabled", True):
        try:
            shap_summary(
                final_model,
                X_sel,
                feature_names=selected_features,
                output_dir=outputs_dir,
                panel_name=panel_name,
                max_display=expl["shap_max_display"],
                background_sample_size=expl["background_sample_size"],
            )
        except Exception as e:
            print(f"[SHAP] hata: {e}")

    # 9. Model + meta kaydı
    model_path = models_dir / "catboost_model.cbm"
    final_model.save_model(str(model_path))

    meta = {
        "panel": panel_name,
        "selected_features": selected_features,
        "sfs_history": sfs_history,
        "categorical_columns": cat_cols_in_sel,
        "medians": medians.to_dict(),
        "best_params": best_params,
        "fixed_params": fixed,
        "class_weights": class_weights,
        "final_threshold": float(final_threshold),
        "metrics_summary": metrics_summary,
        "fold_metrics": [
            {"fold": f.fold_id, "threshold": f.threshold, **f.metrics}
            for f in folds
        ],
        "preprocessing_report": prep_report.to_dict(),
        "final_train_metrics": final_train_metrics,
    }
    with open(models_dir / "model_meta.json", "w", encoding="utf-8") as fp:
        json.dump(_serialize(meta), fp, ensure_ascii=False, indent=2)
    with open(outputs_dir / "metrics_summary.json", "w", encoding="utf-8") as fp:
        json.dump(_serialize(metrics_summary), fp, ensure_ascii=False, indent=2)

    fold_df = pd.DataFrame(
        [{"fold": f.fold_id, "threshold": f.threshold, **f.metrics} for f in folds]
    )
    fold_df.to_csv(outputs_dir / "fold_metrics.csv", index=False)

    print(f"========== [{panel_name}] Pipeline Tamamlandi ==========")
    return {
        "panel": panel_name,
        "model_path": str(model_path),
        "metrics_summary": metrics_summary,
        "selected_features": selected_features,
        "best_params": best_params,
        "final_threshold": float(final_threshold),
    }
