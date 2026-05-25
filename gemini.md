# Alpha Defect Prediction — Industry-Grade AI Prompt
## Steel Hot Rolling · Binary Classification · Imbalanced Data · Production-Ready

---

## SECTION 00 — ROLE & EXPERT CONTEXT

You are a Principal ML Engineer with 10+ years of experience building
production-grade machine learning systems for heavy industry — specifically
steel manufacturing, automotive stamping, and metal processing quality control.

You have deep expertise in:
- Predictive quality control using process sensor data
- Handling severely imbalanced industrial defect datasets
- Deploying real-time inference pipelines on factory floors
- Interpretable ML for non-technical factory operators
- Safety-critical systems where false negatives are unacceptable

Solve the following problem with the same rigour you would apply to
a Tier-1 automotive supplier delivering to ISO/TS 16949 quality standards.
Write production-quality Python code with full comments, error handling,
and reproducibility guarantees.

---

## SECTION 01 — BUSINESS PROBLEM & INDUSTRIAL CONTEXT

INDUSTRIAL CONTEXT
==================
Client        : Steel manufacturing plant (hot rolling mill)
Defect        : "Alpha defect" — a subsurface metallurgical anomaly
                caused by improper phase transformation during hot rolling.
                It appears as a discontinuity in the steel microstructure.

Why it matters:
  - Alpha defects weaken structural integrity of the final product
  - Defective coils shipped to automotive/construction clients → recalls
  - Each customer complaint costs the plant $50,000–$200,000 in rework
  - A single missed defect in a safety-critical part = liability risk

Current process (BROKEN):
  Step 1: Coil rolled at high speed under extreme tension
  Step 2: Coil wound and cooled — defect now locked inside
  Step 3: Quality team samples 3–5 cuts from the coil end
  Step 4: Manual visual/ultrasonic inspection (slow, subjective)
  Step 5: If defect found → entire coil scrapped (too late)

The failure: The inspection happens AFTER the damage is done.
             Defects are caught too late or missed entirely.

PROPOSED AI SOLUTION:
  Use the 49 real-time process parameters captured during rolling
  to predict whether a coil will have an Alpha defect — BEFORE it
  leaves the rolling mill. Flag high-risk coils for immediate
  targeted inspection, saving time and preventing escapes.

BUSINESS RULES (hard constraints, never violate):
  Rule 1: NEVER ship a coil predicted as defect-free if it has a defect.
          → Recall must equal 1.00 (zero false negatives)
  Rule 2: Do not flag too many good coils as defective.
          → Precision must exceed 0.90 (max 10% false alarms)
  Rule 3: The model must be explainable to process engineers.
          → Feature importance and SHAP values are mandatory

---

## SECTION 02 — DATASET SPECIFICATION

DATASET SPECIFICATION
=====================
Files:
  train.csv             → 1,352 rows × 51 columns
  test.csv              → 339 rows  × 50 columns
  sample_submission.csv → format reference only

Column schema:
  CoilID    → Unique coil identifier (DROP from features)
  X1 … X49  → Process parameters / sensor readings
               (temperature, roll force, speed, tension,
                cooling rate, thickness, flatness, etc.)
  Y         → Target label (train.csv only)
               1 = Alpha defect confirmed
               0 = No defect detected

Data characteristics (assume until EDA proves otherwise):
  - Class imbalance likely: defects are rare events (~5–15% of coils)
  - Some sensors may have missing readings (equipment downtime)
  - Sensor drift possible: some X features may have outliers
  - No feature names provided — treat as anonymised process data
  - All features are numerical (float or int)

Expected EDA outputs:
  1. Class distribution plot + imbalance ratio printed
  2. Missing value heatmap (missingno library preferred)
  3. Correlation matrix of all X features
  4. Boxplots of top 10 features stratified by Y=0 vs Y=1
  5. Distribution plots (KDE) for features with highest separation
  6. Print: "Dataset loaded. Shape: []. Defect rate: []%"

---

## SECTION 03 — IMBALANCED DATA HANDLING (4-LAYER DEFENCE)

IMBALANCED DATA — 4-LAYER DEFENCE STRATEGY
===========================================
Industrial defect data is almost always heavily imbalanced.
Use ALL four layers and compare their effect on recall/precision.

LAYER 1 — Algorithmic: class weights
  Compute at runtime, never hardcode:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    ratio = neg / pos   # e.g. 950/50 = 19.0

  Apply to each model:
    XGBoost      → scale_pos_weight=ratio
    LightGBM     → class_weight={0:1, 1:ratio} or is_unbalance=True
    CatBoost     → class_weights=[1, ratio]
    RandomForest → class_weight='balanced_subsample'

LAYER 2 — Sampling: BorderlineSMOTE (preferred over standard SMOTE)
  Why BorderlineSMOTE for manufacturing data:
    - Creates synthetic samples only near the decision boundary
    - Avoids generating unrealistic sensor value combinations
    - More effective than random SMOTE for tabular industrial data

  Implementation (CRITICAL — apply inside CV fold, never globally):
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote',  BorderlineSMOTE(random_state=42, k_neighbors=5)),
        ('model',  XGBClassifier(**params))
    ])
    # The pipeline auto-applies SMOTE only on training folds

LAYER 3 — Cost-sensitive: asymmetric sample weights
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight('balanced', y_train)
    model.fit(X_train, y_train, sample_weight=sw)

LAYER 4 — Threshold calibration (most powerful lever)
  NEVER use model.predict() — always use predict_proba():
    probs = model.predict_proba(X_val)[:, 1]

  Systematic threshold search:
    results = []
    for t in np.arange(0.05, 0.65, 0.005):
        preds = (probs >= t).astype(int)
        rec  = recall_score(y_val, preds, zero_division=0)
        prec = precision_score(y_val, preds, zero_division=0)
        f1   = f1_score(y_val, preds, zero_division=0)
        results.append({'threshold':t, 'recall':rec, 'precision':prec, 'f1':f1})

    df_thresh = pd.DataFrame(results)
    valid = df_thresh[df_thresh['recall'] == 1.0]
    best  = valid.loc[valid['precision'].idxmax()]
    print(f"Best threshold: {best.threshold:.3f} | "
          f"Recall: {best.recall:.3f} | Precision: {best.precision:.3f}")

  Plot the full Precision-Recall curve and mark the selected threshold.

COMPARISON TABLE — print after all techniques:
  | Technique           | Recall | Precision | F1   | Threshold |
  |---------------------|--------|-----------|------|-----------|
  | No balancing        |        |           |      |           |
  | Class weights only  |        |           |      |           |
  | + BorderlineSMOTE   |        |           |      |           |
  | + Cost weights      |        |           |      |           |
  | Best combination    |        |           |      |           |

---

## SECTION 04 — MODEL TRAINING & CROSS-VALIDATION

MODEL TRAINING — PRODUCTION-GRADE CV FRAMEWORK
===============================================
Validation strategy:
  StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

MODEL A — XGBoost (primary candidate)
  params = dict(
      n_estimators        = 500,
      max_depth           = 6,
      learning_rate       = 0.05,
      subsample           = 0.8,
      colsample_bytree    = 0.8,
      min_child_weight    = 3,
      gamma               = 0.1,
      reg_alpha           = 0.1,
      reg_lambda          = 1.0,
      scale_pos_weight    = ratio,
      eval_metric         = 'aucpr',
      early_stopping_rounds = 50,
      random_state        = 42,
      n_jobs              = -1
  )

MODEL B — CatBoost (strong alternative for small tabular data)
  params = dict(
      iterations          = 500,
      depth               = 6,
      learning_rate       = 0.05,
      class_weights       = [1, ratio],
      eval_metric         = 'F1',
      use_best_model      = True,
      early_stopping_rounds = 50,
      random_seed         = 42,
      verbose             = 0
  )

MODEL C — LightGBM (speed + performance)
  params = dict(
      n_estimators        = 500,
      max_depth           = 6,
      learning_rate       = 0.05,
      num_leaves          = 31,
      min_child_samples   = 5,
      subsample           = 0.8,
      colsample_bytree    = 0.8,
      class_weight        = 'balanced',
      random_state        = 42,
      n_jobs              = -1,
      verbose             = -1
  )

MODEL D — RandomForest (interpretable baseline)
  params = dict(
      n_estimators        = 300,
      max_depth           = 10,
      min_samples_leaf    = 2,
      max_features        = 'sqrt',
      class_weight        = 'balanced_subsample',
      random_state        = 42,
      n_jobs              = -1
  )

CV LOOP STRUCTURE:
  fold_results = []
  for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
      X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
      y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

      # Fit preprocessing on training fold ONLY
      X_tr_proc  = pipeline.fit_transform(X_tr)
      X_val_proc = pipeline.transform(X_val)

      # Apply BorderlineSMOTE on training fold ONLY
      X_tr_res, y_tr_res = BorderlineSMOTE(random_state=42).fit_resample(
          X_tr_proc, y_tr)

      for name, model in models.items():
          model.fit(X_tr_res, y_tr_res)
          probs = model.predict_proba(X_val_proc)[:, 1]
          # threshold tuning ...
          fold_results.append({fold, name, recall, precision, f1, auc})

  Print per-fold AND mean±std metrics for every model.

---

## SECTION 05 — MODEL SELECTION & VALIDATION GATES

MODEL SELECTION — INDUSTRIAL QUALITY GATES
==========================================
Treat this like a manufacturing tolerance check.

GATE 1 — Hard constraint (non-negotiable):
  assert mean_recall == 1.00, \
      f"REJECTED: Model {name} recall={mean_recall:.3f}. Must be 1.00."

GATE 2 — Soft target:
  if mean_precision < 0.90:
      print(f"WARNING: {name} precision={mean_precision:.3f}. Target >0.90.")

GATE 3 — Stability check:
  if std_recall > 0.02:
      print(f"WARNING: {name} recall std={std_recall:.3f}. Model unstable.")

SELECTION PRIORITY (in order):
  1. Recall = 1.00 (mandatory — pass Gate 1)
  2. Highest mean precision among passing models
  3. Tiebreak: highest ROC-AUC
  4. Tiebreak: lowest std_recall (most stable)

FINAL REPORT — print this exact format:
  ╔══════════════════════════════════════════════════════╗
  ║  ALPHA DEFECT MODEL — SELECTION REPORT               ║
  ╠══════════════════════════════════════════════════════╣
  ║  Selected model  : [model name]                      ║
  ║  Best threshold  : [x.xxx]                           ║
  ║  CV Recall       : 1.000 ± 0.000                     ║
  ║  CV Precision    : [x.xxx] ± [x.xxx]                 ║
  ║  CV F1-Score     : [x.xxx] ± [x.xxx]                 ║
  ║  CV ROC-AUC      : [x.xxx] ± [x.xxx]                 ║
  ║  Imbalance tech  : [e.g. BorderlineSMOTE + weights]  ║
  ╚══════════════════════════════════════════════════════╝

Confusion matrix at selected threshold — FN MUST BE 0.

---

## SECTION 06 — EXPLAINABILITY FOR FACTORY ENGINEERS

EXPLAINABILITY — MANDATORY FOR INDUSTRIAL DEPLOYMENT
====================================================
Factory process engineers must understand WHY the model flags a coil.

PART A — Feature importance (model-native):
  Plot top 20 features sorted by importance.
  Save as: feature_importance.png

PART B — SHAP analysis:
  import shap
  explainer = shap.TreeExplainer(best_model)
  shap_vals = explainer.shap_values(X_test_processed)

  Plot 1 — Global beeswarm summary:
    shap.summary_plot(shap_vals, X_test_processed,
                      feature_names=[f'X{i}' for i in range(1,50)])

  Plot 2 — Waterfall for single highest-risk coil:
    worst_idx = np.argmax(probs_test)
    shap.waterfall_plot(shap.Explanation(
        values=shap_vals[worst_idx],
        base_values=explainer.expected_value,
        data=X_test_processed[worst_idx],
        feature_names=[f'X{i}' for i in range(1,50)]
    ))

  Plot 3 — Dependence plot for top 3 sensors.

PART C — Factory-friendly risk report:
  For each test coil, output:
    CoilID | Risk Score | Prediction | Top 3 Reasons
    C1001  | 0.94       | DEFECT     | X12 high, X7 low, X31 high
    C1002  | 0.11       | NORMAL     | —

  Risk levels:
    score >= 0.70 → HIGH RISK   → Immediate full inspection
    score >= 0.40 → MEDIUM RISK → Targeted end-cut inspection
    score <  0.40 → LOW RISK    → Standard sampling protocol

  Save as: coil_risk_report.csv

---

## SECTION 07 — PREDICTION & SUBMISSION

PREDICTION & SUBMISSION — PRODUCTION CHECKLIST
===============================================
Pre-prediction assertions:
  assert test_coil_ids.nunique() == 339, "Duplicate CoilIDs in test!"
  assert X_test.shape == (339, 49),      "Wrong test feature count!"

Prediction pipeline:
  X_test_proc = pipeline.transform(X_test)   # NEVER refit on test!
  probs_test  = best_model.predict_proba(X_test_proc)[:, 1]
  preds_test  = (probs_test >= best_threshold).astype(int)

Sanity check:
  defect_rate_test  = preds_test.mean()
  defect_rate_train = y_train.mean()
  if abs(defect_rate_test - defect_rate_train) > 0.10:
      print("WARNING: Distribution shift detected!")

Save submission:
  submission = pd.DataFrame({
      'CoilID': test_coil_ids,
      'Y':      preds_test
  })
  submission.to_csv('expected_submission.csv', index=False)

  # Verify format
  sample = pd.read_csv('sample_submission.csv')
  assert list(submission.columns) == list(sample.columns)
  assert len(submission) == len(sample)

  print(f"Submission saved — {preds_test.sum()} defects flagged out of 339 coils")

---

## SECTION 08 — CODE QUALITY & DELIVERABLES

CODE QUALITY STANDARDS — NON-NEGOTIABLE
========================================
- Jupyter notebook: one markdown cell + one code cell per step
- random_state=42 everywhere (full reproducibility)
- Descriptive variable names (no df1, df2)
- try/except for optional libraries (imblearn, shap, catboost)
- Structured logging at every major step:
    "[EDA]     Class balance: 94.2% normal, 5.8% defect (ratio=16.2x)"
    "[PREPROC] After scaling: X_train shape (1082, 49)"
    "[SMOTE]   After resample: 1082 normal, 1082 defect"
    "[TRAIN]   XGBoost fold 3/5 — Recall: 1.000, Precision: 0.923"
    "[SELECT]  Best: CatBoost | Threshold: 0.215 | Prec: 0.934"
    "[PREDICT] Flagged 22/339 coils as high-risk"
    "[SAVE]    expected_submission.csv written successfully"

DELIVERABLES CHECKLIST:
  alpha_defect_model.ipynb    ← full reproducible notebook
  expected_submission.csv     ← CoilID, Y (required format)
  metrics_comparison.csv      ← all model × technique results
  coil_risk_report.csv        ← risk scores for all test coils
  pr_curve.png                ← Precision-Recall curve + threshold
  feature_importance.png      ← top 20 sensor importances
  shap_summary.png            ← global SHAP beeswarm plot

FINAL CONSTRAINT REMINDER:
  ╔═══════════════════════════════════════╗
  ║  FN (False Negatives) MUST EQUAL 0   ║
  ║  on both validation and test data.    ║
  ║  Any model with FN > 0 is REJECTED.  ║
  ╚═══════════════════════════════════════╝
  train using the train.csv and give me test.csv and sample_submission.csv