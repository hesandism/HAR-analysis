# UCI-HAR ML Workflow Summary (Baseline -> Improved)

## Project Objective
This project builds and evaluates a Human Activity Recognition (HAR) pipeline on the UCI-HAR dataset, then iteratively improves the workflow from a baseline notebook to a feature-engineered and model-extended notebook.

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

---

## 5) Research-Paper Ready Narrative

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

## 6) Artifacts Produced
- Baseline metrics export: `baseline_results.csv`
- Post-selection metrics export: `result_after_feature_selection.csv`
- Shared model/training logic: `models.py`
- Baseline workflow notebook: `baseline.ipynb`
- Improved workflow notebook: `Improved_nb.ipynb`
