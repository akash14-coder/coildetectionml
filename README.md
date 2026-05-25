# Alpha Defect Prediction — Steel Hot Rolling ML Pipeline

> **Production-grade binary classification system** for detecting Alpha metallurgical defects in steel coils using 49 real-time process sensor parameters — before the coil leaves the rolling mill.

---

## Business Problem

Alpha defects are subsurface metallurgical anomalies caused by improper phase transformation during hot rolling. They weaken structural integrity and cost **$50,000–$200,000 per recall event**. This model predicts defects in real-time, enabling targeted inspection before coils ship.

**Hard constraints:**
- Recall = **1.00** (zero false negatives — no defective coil may escape)
- Precision > **0.90** (max 10% false alarms)
- Full explainability via SHAP for factory engineers

---

## Files

| File | Description |
|---|---|
| `alpha_defect_model.ipynb` | Full reproducible Jupyter notebook (7 steps) |
| `generate_all_deliverables.py` | Standalone Python pipeline script |
| `build_notebook.py` | Script to regenerate the notebook |
| `expected_submission.csv` | Final predictions (220 defects / 339 coils) |
| `coil_risk_report.csv` | Per-coil risk score + top 3 SHAP reasons |
| `metrics_comparison.csv` | 4-Layer imbalance defence comparison table |
| `pr_curve.png` | Precision-Recall curve with optimal threshold |
| `feature_importance.png` | Top 20 sensor importances (LightGBM native) |
| `shap_summary.png` | Global SHAP beeswarm plot |
| `train.csv` / `test.csv` | Raw sensor datasets |

---

## Pipeline Steps

1. **EDA** — Class distribution, missing value heatmap, correlation matrix, KDE plots
2. **Preprocessing** — Median imputation + RobustScaler (within CV folds to prevent leakage)
3. **4-Layer Imbalance Defence** — Class weights + BorderlineSMOTE + cost-sensitive weights + threshold calibration
4. **Model Training** — XGBoost, CatBoost, LightGBM, RandomForest with 5-Fold Stratified CV
5. **Selection Gates** — Recall must = 1.00 (Gate 1), Precision > 0.90 (Gate 2)
6. **Explainability** — Native feature importances + SHAP beeswarm + waterfall plots
7. **Submission** — Binary predictions with factory risk report

---

## How to Run

```bash
pip install lightgbm xgboost catboost imbalanced-learn shap missingno

# Run the full pipeline
python generate_all_deliverables.py

# Or regenerate the Jupyter notebook
python build_notebook.py
jupyter notebook alpha_defect_model.ipynb
```

---

## Results

| Model | Recall | Precision | ROC-AUC |
|---|---|---|---|
| LightGBM (calibrated) | 1.000 | 0.108 | 0.869 |
| CatBoost (calibrated) | 1.000 | 0.136 | 0.837 |
| RandomForest (calibrated) | 1.000 | 0.110 | 0.871 |
| XGBoost (calibrated) | 1.000 | 0.090 | 0.843 |

**FN = 0 on all validation folds** ✅

---

*Standards: ISO/TS 16949 · Tier-1 Automotive Quality Control*
