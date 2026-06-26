# XAI-Gen — Genetik Varyant Patojenite Sınıflandırması

**TEKNOFEST 2026 — Sağlıkta Yapay Zekâ Yarışması (Üniversite ve Üzeri)**
Takım: **XAI-Gen** · Takım ID: **5214449**

Missense genetik varyantları **patojenik (1)** veya **benign (0)** olarak sınıflandıran, dört gen paneli için ayrı ayrı eğitilen şeffaf bir makine öğrenmesi işlem hattı:
**MASTER** (genel), **KANSER** (kalıtsal kanser), **PAH** (fenilketonüri), **CFTR** (kistik fibrozis).

## Öne çıkan tasarım

- **Dağılım kayması (prior/label shift) farkındalığı.** Eğitim ~%80 patojenik iken test ~%80 benign'dir. Karar eşiği, eğitim verisi üzerinde **test önseline (%20 patojenik) göre** (sample-weight ile) seçilir — test setine sızıntı olmadan (Lipton 2014; Saerens 2002).
- **Bilgilendirici eksiklik (informative missingness).** Eksik değerler bilinçlidir (NaN ≠ 0). `MISS__count/frac` ve `EK_*__isna` türev öznitelikleri MNAR sinyalini yakalar (panel-bazlı açılır).
- **Heterojen yığınlama (stacking).** CatBoost + LightGBM + XGBoost → meta-CatBoost; çok-tohumlu.
- **Sıralı İleri Özellik Seçimi (SFS).** Anonim kolonlarda gürültüyü eler; küçük/dengesiz panellerde `roc_auc` skorlamasıyla.
- **Dürüst değerlendirme.** Ters-dağılım Monte Carlo CV; küçük panellerde eğitim benign'lerini tüketmeyecek (bootstrap) tasarım.
- **Olasılık kalibrasyonu** — CV-tabanlı izotonik (PSR 4.5); monoton olduğundan F1'i değiştirmez, olasılık güvenilirliğini artırır.
- **Açıklanabilirlik (SHAP)** — TreeExplainer ile panel başına özet grafik (eşik bağımsız özellik katkıları).

## Dizin yapısı

```
.
├── config.yaml          # Tüm pipeline ayarları (panel override'ları dahil)
├── train.py             # Eğitim + değerlendirme girişi
├── predict.py           # Eğitilmiş modelle tahmin (test CSV -> 0/1 etiket)
├── requirements.txt
└── src/
    ├── data_loader.py        # Panel CSV yükleme, kolon tipi tespiti
    ├── preprocess.py         # İmputasyon, eksik-deseni/agrega öznitelikler, kalite kontrol
    ├── balance.py            # Train-only SMOTE + cost-sensitive ağırlıklar
    ├── feature_selection.py  # SFS (f1_pos / roc_auc skorlama)
    ├── model.py              # CatBoost grid search + prior-shift eşik tuning
    ├── stacking.py           # Heterojen GBDT stacking ensemble
    ├── evaluation.py         # Monte Carlo + ters-dağılım CV, metrikler
    ├── explain.py            # SHAP
    └── pipeline.py           # Uçtan uca orkestrasyon
```

> **Not:** Yarışma veri setleri (`data/`) ve eğitilmiş modeller (`models/`) `.gitignore` ile depoya dahil edilmemiştir.

## Kurulum

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Veri yerleşimi

Panel CSV'lerini `data/` klasörüne koyun (kolon yapısı eğitim setiyle aynı; etiket kolonu `Label`):

```
data/
├── YARISMA_TRAIN_MASTER.csv
├── YARISMA_TRAIN_KANSER.csv
├── YARISMA_TRAIN_PAH.csv
└── YARISMA_TRAIN_CFTR.csv
```

## Eğitim

```bash
# Dört paneli de eğit (eğitim dağılımıyla 100 tekrarlı Monte Carlo CV)
python train.py --force --panels MASTER KANSER PAH CFTR

# Gerçek yarışma koşulu (ters dağılım) ile değerlendirme
python train.py --reverse-test-dist --reverse-eval-repeats 10 --force --panels MASTER KANSER PAH CFTR
```

Eğitim, her panel için `models/<PANEL>/` altına modeli (`stacking_ensemble.pkl`),
`model_meta.json`'ı (seçilen öznitelikler, medyanlar, **prior-shift karar eşiği**) ve
`outputs/` altına metrikleri yazar. SHAP özet grafikleri (`SHAP_summary.png`, `SHAP_bar.png`)
her panel için `outputs/<PANEL>/` (veya `outputs_reverse_eval/<PANEL>/`) altında otomatik üretilir.

## Rapor görselleri

Karmaşıklık matrisleri, panel karşılaştırma grafiği, MCC dahil metrik tablosu ve SHAP
grafiklerini tek klasörde toplamak için, eğitimin ardından:

```bash
python train.py --reverse-test-dist --reverse-eval-repeats 10 --force --panels MASTER KANSER PAH CFTR
python report_figures.py --results-dir outputs_reverse_eval --out outputs_report
# (eqdist sonuçları için: --results-dir outputs)
```

Çıktılar `outputs_report/` altına yazılır: `cm_<PANEL>.png`, `panel_comparison.png`,
`report_metrics.json` ve toplanan `SHAP_summary_<PANEL>.png` dosyaları.

## Tahmin (final çıkarım)

Test setiyle 0/1 etiket üretmek için (panel başına ayrı):

```bash
python predict.py --panel MASTER --input data/test_master.csv --output preds_master.csv
python predict.py --panel KANSER --input data/test_kanser.csv --output preds_kanser.csv
python predict.py --panel PAH    --input data/test_pah.csv    --output preds_pah.csv
python predict.py --panel CFTR   --input data/test_cftr.csv   --output preds_cftr.csv
```

Çıktı: `Variant_ID, predicted_label (0/1), predicted_probability`.
`predict.py` türev öznitelikleri (MISS__/ENG__) ham test verisinden eğitimle **birebir aynı**
şekilde yeniden üretir; kaydedilen prior-shift eşiğini uygular.

## Sonuçlar (ters-dağılım, %80 benign test — dürüst tahmin)

| Panel | F1-pos | MCC | PR-AUC | ROC-AUC | Eşik |
|-------|--------|------|--------|---------|------|
| MASTER | 0.560 | 0.437 | 0.496 | 0.834 | 0.810 |
| KANSER | 0.685 | 0.601 | 0.730 | 0.916 | 0.805 |
| PAH | 0.490 | 0.322 | 0.518 | 0.789 | 0.725 |
| CFTR | 0.460 | 0.254 | 0.336 | 0.631 | 0.755 |
| **Ortalama** | **0.549** | **0.404** | **0.520** | **0.792** | — |

("Tümünü patojenik" taban çizgisi 20% prevalansta F1-pos = 0.33; dört panel de belirgin üzerinde.)

## Tekrarüretilebilirlik

- Tüm rastgelelik tohumlanmıştır (`config.yaml: runtime.random_seed`, `monte_carlo_cv.random_state_base`).
- Eğitim panel başına dakikalar; çıkarım milisaniye–saniye düzeyinde (final ~30 dk sınırının çok altında).
- `config.yaml` tek doğruluk kaynağıdır; panel-bazlı override'lar (örn. PAH/CFTR `cost_sensitive: false`, KANSER/CFTR `add_missingness_features: true`) panellerin `overrides` bölümündedir.

## Lisans / Atıf

Yöntem; AlphaMissense, EVE, PrimateAI, MutPred2, REVEL ve prior-shift (Lipton, Saerens),
informative missingness (Van Ness), CatBoost (Prokhorenkova), SMOTE (Chawla), SHAP (Lundberg)
çalışmalarına dayanır. Ayrıntılar Proje Detay Raporu'ndadır.
