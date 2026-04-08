# UCI-HAR ML Workflow Summary (Baseline -> Improved)

## Project Objective
This project builds and evaluates a Human Activity Recognition (HAR) pipeline on the UCI-HAR dataset, then iteratively improves the workflow from a baseline notebook to a feature-engineered and model-extended notebook.

---

## Project Overview (Paper-Ready)

### 1) Problem Definition
- Goal: classify human activities from smartphone inertial sensor signals.
- Prediction target: activity class label.
- Dataset: UCI HAR Dataset.
- Task type: multiclass classification.
- Number of classes: 6 activities.

Target classes:
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

### 2) Dataset Details
- Original feature count: 561 engineered time/frequency-domain features.
- Train samples: 7352.
- Test samples: 2947.
- Total samples: 10299.
- Split used in workflow: original UCI HAR train/test split (not random re-split).

Feature-space evolution in this project:
- 561 original features.
- 525 features after near-zero variance filtering.
- 257 features after correlation pruning.

### 3) Preprocessing Steps
1. Label handling and encoding consistency
  - Baseline pipeline converts activity IDs to zero-based indexing where needed.
  - For XGBoost compatibility, labels must be contiguous and start at 0 for multiclass training. In this project context, this means handling potential 1..6 labels before XGBoost fitting.
2. Low-variance filtering
  - `VarianceThreshold(threshold=0.01)` removed 37 low-information features.
3. Univariate feature scoring
  - `SelectKBest(f_classif)` used to score and rank discriminative features.
4. K-sweep feature-count selection
  - Tried k values from 50 to 525 to identify strong feature counts.
5. Redundancy reduction
  - Correlation filter with `|r| > 0.95` reduced duplicated signal information.

Notes on other preprocessing methods:
- Standardization (`StandardScaler`): present in earlier experimentation, but the final reported improved run is based on the variance + SelectKBest + correlation workflow.
- PCA: explored during experimentation history, not used in the final comparison table in this report.
- SMOTE/imbalance handling: not used in the final workflow.

### 4) Models Used
Baseline models:
- Decision Tree
- Random Forest
- Logistic Regression
- Linear SVC
- RBF SVM
- K-Nearest Neighbor

Improved/additional models:
- Extra Trees
- Hist Gradient Boosting
- Tuned Random Forest
- Tuned Logistic Regression
- Tuned RBF SVM
- XGBoost

### 5) Evaluation Metrics
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Training time
- Classification report for selected best model analysis
- Confusion matrix visualization (classification diagnostics in notebook workflow)

Regression-only metrics such as RMSE are not applicable because this project is strictly classification.

### 6) Results Snapshot
Representative results from this project:
- Baseline best: Linear SVC, Accuracy = 0.97.
- Improved final best group: Logistic Regression / Linear SVC / Extra Trees / Tuned Logistic Regression, Accuracy = 0.95.
- Improved Random Forest: Accuracy = 0.94 (up from 0.92 in baseline).
- XGBoost in final improved table: Accuracy = 0.94.
- Best ensemble (latest): Stacking (Top models), Accuracy = 0.9511.
- Hyperparameter optimization workflow added in a dedicated notebook; execution results pending.

### 7) Key Observations
- The best absolute accuracy was achieved in the baseline configuration (Linear SVC: 0.97).
- Feature-selection and redundancy-pruning reduced dimensionality substantially (561 -> 257) and improved compactness.
- Tree ensembles benefited in some cases (Random Forest improved from 0.92 to 0.94).
- Some linear/SVM peaks slightly decreased after aggressive feature reduction (for example, Linear SVC 0.97 -> 0.95).
- Tuned Logistic Regression matched top improved accuracy but had high training cost, indicating weaker accuracy-time tradeoff.
- No major imbalance-correction strategy was required in the final run.
- The latest ensemble notebook confirms complementarity across model families: stacking slightly improves over most single improved models (0.9511 vs ~0.95), but still does not exceed the baseline 0.97 peak.

### 8) Folder Structure (Practical Layout)
Suggested paper/reproducibility layout:

```text
HAR/
  data/
   X_train.csv
   X_test.csv
   y_train.csv
   y_test.csv
   UCI-HAR-Dataset/
  models.py
  baseline.ipynb
  Improved_nb.ipynb
  ensemble.ipynb
  hyper_parameter.ipynb
  baseline_results.csv
  result_after_feature_selection.csv
  ensemble_results.csv
  hyperparameter_tuning_cv.csv
  hyperparameter_tuned_test.csv
  workflow_summary.md
```

---

## 1) Baseline Workflow (baseline.ipynb)

### 1.1 Data preparation and sanity checks
- Loaded UCI-HAR metadata (`features.txt`, `activity_labels.txt`) and train/test splits.
- Checked duplicate feature names.
- Remapped activity labels to zero-based indexing in baseline preprocessing.
- Exported processed train/test CSV files for reuse:
  - `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.

### 1.2 Baseline model set
The baseline notebook trains these models using the shared trainer:
- Decision Tree
- Random Forest
- Logistic Regression
- Linear SVC
- RBF SVM
- K-Nearest Neighbor

### 1.3 Baseline results (from baseline.ipynb)

| Model | Precision | Recall | F1-Score | Accuracy | Training Time (s) |
|---|---:|---:|---:|---:|---:|
| Linear SVC | 0.97 | 0.97 | 0.96 | **0.97** | 1.70 |
| Logistic Regression | 0.96 | 0.96 | 0.96 | 0.96 | 3.65 |
| RBF SVM | 0.95 | 0.95 | 0.95 | 0.95 | 0.92 |
| Random Forest | 0.92 | 0.92 | 0.92 | 0.92 | 1.71 |
| Decision Tree | 0.86 | 0.86 | 0.86 | 0.86 | 3.88 |
| K-Nearest Neighbor | 0.82 | 0.81 | 0.81 | 0.81 | 0.02 |

### 1.4 Baseline best model behavior
- Best overall baseline model: **Linear SVC (Accuracy 0.97)**.
- Baseline detailed class report confirms generally strong separability across activities.

---

## 2) Improved Workflow (Improved_nb.ipynb)

The improved notebook introduces staged feature filtering/selection and additional model families.

### 2.1 Improvement A: Near-zero variance filtering
- Method: `VarianceThreshold(threshold=0.01)`.
- Outcome:
  - Features removed: **37**
  - Features kept: **525**

### 2.2 Improvement B: ANOVA feature relevance analysis
- Method: `SelectKBest(f_classif, k='all')` for full ranking.
- Outcome:
  - Statistically significant features (`p < 0.05`): **519 / 525**
  - Top-ranked discriminative features are dominated by entropy/acceleration-derived signals.

### 2.3 Improvement C: K-sweep for feature count selection
- Method: sweep `k in [50,100,150,200,250,300,350,400,450,525]` using Linear SVC proxy.
- Outcome:
  - Best observed k: **525**
  - Best proxy accuracy: **0.9637**

### 2.4 Improvement D: Correlation pruning
- Method: greedy removal of features with `|r| > 0.95`.
- Outcome:
  - Input to pruning: 525 features
  - Removed for redundancy: **268**
  - Final selected feature space: **257 features**
  - Final matrix shape: **(7352, 257)**

### 2.5 Improvement E: Expanded model zoo
Added improved models in addition to retrained baselines:
- Extra Trees
- Hist Gradient Boosting
- Tuned Random Forest
- Tuned Logistic Regression
- Tuned RBF SVM
- XGBoost (when available)

---

## 3) Final Comparison on Selected Features (257 features)

### 3.1 Results from improved notebook final comparison

| Model | Type | Precision | Recall | F1-Score | Accuracy | Training Time (s) |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | Baseline (retrained) | 0.95 | 0.95 | 0.95 | **0.95** | 2.82 |
| Linear SVC | Baseline (retrained) | 0.95 | 0.95 | 0.95 | **0.95** | 0.84 |
| Extra Trees | Improved | 0.95 | 0.95 | 0.95 | **0.95** | 0.79 |
| Tuned Logistic Regression | Improved | 0.95 | 0.95 | 0.95 | **0.95** | 43.15 |
| Random Forest | Baseline (retrained) | 0.94 | 0.94 | 0.94 | 0.94 | 1.14 |
| RBF SVM | Baseline (retrained) | 0.94 | 0.94 | 0.94 | 0.94 | 0.52 |
| Hist Gradient Boosting | Improved | 0.94 | 0.94 | 0.94 | 0.94 | 11.14 |
| Tuned Random Forest | Improved | 0.94 | 0.94 | 0.94 | 0.94 | 5.31 |
| Tuned RBF SVM | Improved | 0.94 | 0.94 | 0.94 | 0.94 | 0.37 |
| XGBoost | Improved | 0.94 | 0.94 | 0.94 | 0.94 | 12.35 |
| Decision Tree | Baseline (retrained) | 0.84 | 0.84 | 0.84 | 0.84 | 2.25 |
| K-Nearest Neighbor | Baseline (retrained) | 0.79 | 0.77 | 0.77 | 0.77 | 0.01 |

---

## 4) How Metrics Differ from Baseline

### 4.1 Headline performance
- Best baseline notebook accuracy: **0.97 (Linear SVC)**.
- Best final improved notebook accuracy: **0.95** (tie among Logistic Regression, Linear SVC, Extra Trees, Tuned Logistic Regression).
- Net change in top accuracy: **-0.02**.

### 4.2 Baseline models before vs after feature-selection workflow

| Baseline Model | Accuracy (Baseline Notebook) | Accuracy (Improved Final) | Delta |
|---|---:|---:|---:|
| Linear SVC | 0.97 | 0.95 | -0.02 |
| Logistic Regression | 0.96 | 0.95 | -0.01 |
| RBF SVM | 0.95 | 0.94 | -0.01 |
| Random Forest | 0.92 | 0.94 | +0.02 |
| Decision Tree | 0.86 | 0.84 | -0.02 |
| K-Nearest Neighbor | 0.81 | 0.77 | -0.04 |

Interpretation:
- Feature selection + correlation pruning improved Random Forest accuracy but slightly reduced top linear/SVM peak.
- Precision/Recall/F1 track the same trend as accuracy (very close values due weighted averaging and balanced overall performance).

### 4.3 Improved models vs baseline best
- Improved models are competitive but do not exceed the original 0.97 peak in the current run.
- Extra Trees reaches 0.95 with relatively low training cost.
- Tuned Logistic Regression matches 0.95 but is much slower (43.15s), indicating weaker accuracy-time tradeoff.
- XGBoost reaches 0.94 in current configuration.

### 4.4 Ensemble results vs baseline and improved
- Best ensemble notebook result: **Stacking (Top models)** with Accuracy **0.9511**.
- Ensemble variants in latest run:
  - Stacking (Top models): 0.9511
  - Hard Voting (Top models): 0.9471
  - Soft Voting (Proba models): 0.9420
- Best single models inside ensemble study:
  - Extra Trees: 0.9505
  - XGBoost: 0.9406
  - Tuned RBF SVM: 0.9393
  - Hist Gradient Boosting: 0.9382
- Relative conclusion:
  - Stacking provides a small but consistent uplift over most individual improved models.
  - It is still below the original baseline best (Linear SVC at 0.97).

---

## 5) Ensemble Workflow (ensemble.ipynb)

### 5.1 Motivation
The ensemble notebook explicitly follows this rationale:
- Individual models have complementary strengths.
- Tree-based methods capture non-linear interactions.
- Linear/SVM methods perform well on high-dimensional sensor features.
- Combining them can improve robustness and generalization.

### 5.2 Preprocessing consistency
To maintain comparability, the ensemble notebook reuses the same improved preprocessing path:
- VarianceThreshold(0.01) -> 525 features.
- SelectKBest k-sweep -> best k = 525.
- Correlation pruning (|r| > 0.95) -> 257 final features.
- Final matrix shapes:
  - Train: (7352, 257)
  - Test: (2947, 257)

### 5.3 Systematic model selection (5-fold stratified CV)
Top-ranked models by CV accuracy mean in the ensemble notebook:
1. Hist Gradient Boosting: 0.9944 (+/- 0.0014)
2. XGBoost: 0.9924 (+/- 0.0016)
3. Tuned RBF SVM: 0.9882 (+/- 0.0021)
4. Extra Trees: 0.9844 (+/- 0.0031)

These top models were used to construct voting/stacking ensembles.

### 5.4 Ensemble performance on held-out test set
| Model | Type | Accuracy | F1-Weighted | Precision-Weighted | Recall-Weighted |
|---|---|---:|---:|---:|---:|
| Stacking (Top models) | Ensemble | **0.9511** | 0.9510 | 0.9523 | 0.9511 |
| Extra Trees | Individual | 0.9505 | 0.9501 | 0.9521 | 0.9505 |
| Hard Voting (Top models) | Ensemble | 0.9471 | 0.9469 | 0.9480 | 0.9471 |
| Soft Voting (Proba models) | Ensemble | 0.9420 | 0.9418 | 0.9430 | 0.9420 |
| XGBoost | Individual | 0.9406 | 0.9404 | 0.9417 | 0.9406 |
| Tuned RBF SVM | Individual | 0.9393 | 0.9391 | 0.9405 | 0.9393 |
| Hist Gradient Boosting | Individual | 0.9382 | 0.9381 | 0.9391 | 0.9382 |

### 5.5 Benchmark conclusion
- Ensemble benchmark target range in notebook: 0.95 to 0.97.
- Achieved: **0.9511** (target range reached).
- Global project best remains baseline Linear SVC at 0.97.

---

## 6) Research-Paper Ready Narrative

A concise narrative for the paper:
1. Start with a broad baseline on full engineered HAR features and benchmark multiple classifier families.
2. Introduce a staged feature-engineering pipeline:
   - remove low-information features,
   - rank by ANOVA discrimination strength,
   - tune feature count via k-sweep,
   - prune high-correlation redundancy.
3. Retrain both baseline and improved model families on the reduced representation.
4. Compare not only accuracy but also precision/recall/F1 and training time.
5. Conclude that reduced feature space improves compactness and benefits some tree ensembles, but may slightly reduce peak margin-based classifier accuracy under current settings.

---

## 7) Hyperparameter Optimization Workflow (hyper_parameter.ipynb)

### 7.1 Purpose
The hyperparameter notebook adds systematic tuning on top of the improved preprocessing so model comparisons are not limited to default settings.

### 7.2 Pipeline design
The notebook follows this sequence:
1. Load `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.
2. Apply the same improved preprocessing path used elsewhere:
  - VarianceThreshold(0.01)
  - SelectKBest (k selected by sweep)
  - Correlation pruning (`|r| > 0.95`)
3. Define compact candidate parameter sets for faster tuning of:
  - Logistic Regression
  - Linear SVC
  - Random Forest
  - Hist Gradient Boosting
  - XGBoost (if available)
4. Run fast model selection using stratified 3-fold cross-validation (`cross_val_score`) over the compact parameter candidates.
5. Evaluate best estimators on held-out test data.
6. Export:
  - `hyperparameter_tuning_cv.csv` (best CV scores + parameters)
  - `hyperparameter_tuned_test.csv` (held-out test metrics)
  - `hyperparameter_tuned_cv_validation.csv` (5-fold CV validation metrics)
  - `hyperparameter_tuned_test_validation.csv` (final held-out validation metrics)

### 7.3 Tuning results (executed)

Best cross-validation results (`hyperparameter_tuning_cv.csv`):

| Model | Best CV Accuracy | Selected Parameters |
|---|---:|---|
| Hist Gradient Boosting | **0.9946** | `learning_rate=0.08`, `max_depth=10`, `max_iter=300`, `min_samples_leaf=30`, `l2_regularization=0.01` |
| XGBoost | 0.9916 | `n_estimators=200`, `max_depth=5`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0` |
| Random Forest | 0.9793 | `n_estimators=200`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features=sqrt` |
| Linear SVC | 0.9781 | `C=0.3` |
| Logistic Regression | 0.9773 | `C=3.0`, `solver=saga` |

Held-out test results (`hyperparameter_tuned_test.csv`):

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1-score (weighted) |
|---|---:|---:|---:|---:|
| Linear SVC | **0.9525** | 0.9546 | 0.9525 | 0.9527 |
| Logistic Regression | 0.9501 | 0.9519 | 0.9501 | 0.9501 |
| XGBoost | 0.9423 | 0.9434 | 0.9423 | 0.9421 |
| Random Forest | 0.9410 | 0.9420 | 0.9410 | 0.9407 |
| Hist Gradient Boosting | 0.9369 | 0.9376 | 0.9369 | 0.9368 |

### 7.4 Simple parameter reasoning
- Logistic Regression (`C=3.0`, `solver=saga`): slightly weaker regularization (higher `C`) improves class separation on selected features; `saga` is robust for larger sparse/high-dimensional setups and supports flexible optimization behavior.
- Linear SVC (`C=0.3`): stronger regularization helps prevent overfitting and improved generalization on the held-out split.
- Random Forest (`n_estimators=200`, deep trees, `max_features=sqrt`): enough trees for stable voting without excessive runtime; `sqrt` feature subsampling increases tree diversity and helps generalization.
- Hist Gradient Boosting (`learning_rate=0.08`, `max_iter=300`, depth/leaf constraints): moderate learning rate with enough boosting rounds balances fit quality and stability; `min_samples_leaf` and `l2_regularization` reduce overfitting.
- XGBoost (`n_estimators=200`, `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`): medium-complex trees and row/column subsampling improve robustness; L2 regularization keeps model variance controlled.

### 7.5 Contribution to the study
- Reduces risk of under-reporting model capacity due to untuned defaults.
- Improves fairness in model-family comparison under the same preprocessing pipeline.
- Adds an explicit accuracy-vs-complexity perspective: best CV model (Hist Gradient Boosting) is not the best held-out model (Linear SVC), reinforcing the need for final test validation.

### 7.6 Final validation results (5-fold CV + held-out)

Validation cross-validation results (`hyperparameter_tuned_cv_validation.csv`):

| Model | CV Accuracy Mean | CV Accuracy Std | CV Precision Mean | CV Recall Mean | CV F1 Mean | CV Balanced Acc Mean |
|---|---:|---:|---:|---:|---:|---:|
| Hist Gradient Boosting | **0.9954** | 0.0017 | 0.9954 | 0.9954 | 0.9954 | 0.9956 |
| XGBoost | 0.9925 | 0.0011 | 0.9925 | 0.9925 | 0.9925 | 0.9928 |
| Random Forest | 0.9814 | 0.0032 | 0.9815 | 0.9814 | 0.9814 | 0.9820 |
| Linear SVC | 0.9785 | 0.0034 | 0.9786 | 0.9785 | 0.9785 | 0.9801 |
| Logistic Regression | 0.9782 | 0.0044 | 0.9784 | 0.9782 | 0.9782 | 0.9798 |

Held-out validation test results (`hyperparameter_tuned_test_validation.csv`):

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test Balanced Acc |
|---|---:|---:|---:|---:|---:|
| Linear SVC | **0.9525** | 0.9546 | 0.9525 | 0.9527 | 0.9530 |
| Logistic Regression | 0.9501 | 0.9519 | 0.9501 | 0.9501 | 0.9498 |
| XGBoost | 0.9423 | 0.9434 | 0.9423 | 0.9421 | 0.9410 |
| Random Forest | 0.9410 | 0.9420 | 0.9410 | 0.9407 | 0.9380 |
| Hist Gradient Boosting | 0.9369 | 0.9376 | 0.9369 | 0.9368 | 0.9360 |

### 7.7 Quick insights
- CV and test rankings are not identical: Hist Gradient Boosting leads in CV, but Linear SVC gives the strongest held-out generalization.
- Linear models (Linear SVC and Logistic Regression) show the best accuracy-time-generalization tradeoff in this tuned setup.
- Tree boosting models achieve very high CV scores, but their held-out drop suggests mild overfitting under current feature set and parameter candidates.
- Balanced accuracy tracks overall weighted metrics closely, indicating no severe class-level collapse in final tuned predictions.

---

## 8) Artifacts Produced
- Baseline metrics export: `baseline_results.csv`
- Post-selection metrics export: `result_after_feature_selection.csv`
- Ensemble metrics export: `ensemble_results.csv`
- Hyperparameter CV export: `hyperparameter_tuning_cv.csv` (generated after running notebook)
- Hyperparameter held-out test export: `hyperparameter_tuned_test.csv` (generated after running notebook)
- Hyperparameter 5-fold CV validation export: `hyperparameter_tuned_cv_validation.csv`
- Hyperparameter held-out validation export: `hyperparameter_tuned_test_validation.csv`
- Shared model/training logic: `models.py`
- Baseline workflow notebook: `baseline.ipynb`
- Improved workflow notebook: `Improved_nb.ipynb`
- Ensemble workflow notebook: `ensemble.ipynb`
- Hyperparameter workflow notebook: `hyper_parameter.ipynb`
