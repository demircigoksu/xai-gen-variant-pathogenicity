"""
Genetik Varyant Sınıflandırma Pipeline
======================================
Proje Sunuş Raporuna birebir uyumlu uçtan uca makine öğrenmesi pipeline'ı.

Aşamalar (PDF Bölüm 3):
1. data_loader  -> CSV yükleme + feature group tespiti
2. preprocess   -> Medyan imputasyon, veri kalite kontrolü
3. balance      -> Train-Only SMOTE-NC
4. feature_selection -> Sıralı İleri Özellik Seçimi (SFS)
5. model        -> CatBoost + Grid Search + Threshold tuning
6. evaluation   -> 100-Repeated Monte Carlo CV + panel metrikleri
7. explain      -> SHAP analizi
8. pipeline     -> Orkestratör
"""

__version__ = "1.0.0"
__author__ = "Genetik Varyant Sınıflandırma Takımı"
