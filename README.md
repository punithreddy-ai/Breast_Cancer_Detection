# Breast Cancer Detection

A machine learning project that classifies breast tumors as **malignant** or **benign** using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. This dataset is the same one hosted on the [UCI Machine Learning Repository (ID 17)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) and is also bundled directly with scikit-learn (`sklearn.datasets.load_breast_cancer`).

## Overview

Early and accurate detection of breast cancer significantly improves treatment outcomes. This project builds and compares several classification models on tumor characteristics — such as radius, texture, perimeter, area, smoothness, and concavity — computed from digitized images of fine needle aspirate (FNA) biopsies.

## Dataset

- **Samples:** 569 (357 benign, 212 malignant)
- **Features:** 30 numeric features per sample (mean, standard error, and "worst" values of 10 characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- **Target:** Diagnosis — Malignant (M) or Benign (B)

## Project Structure

```
.
├── breast_cancer_detection.py     # Main script: EDA, training, evaluation
├── outputs/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── feature_importance.png
│   ├── feature_importances.csv
│   ├── model_comparison.csv
│   ├── model_comparison_bar.png
│   ├── roc_curves.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_svm_rbf.png
│   ├── confusion_matrix_k-nearest_neighbors.png
│   └── best_model_report.txt
└── README.md
```

## Methodology

1. **Exploratory Data Analysis** — class balance, feature correlations, and distributions of key features by diagnosis.
2. **Preprocessing** — 80/20 stratified train/test split, followed by feature standardization (`StandardScaler`).
3. **Model Training** — four classifiers trained and evaluated:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (RBF kernel)
   - K-Nearest Neighbors
4. **Evaluation** — accuracy, precision, recall, F1-score, ROC-AUC, and 5-fold cross-validation for each model.
5. **Feature Importance** — extracted from the Random Forest model to identify the most predictive tumor characteristics.

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 98.2% | 98.6% | 98.6% | 98.6% | **0.995** |
| SVM (RBF) | 98.2% | 98.6% | 98.6% | 98.6% | 0.995 |
| Random Forest | 94.7% | 95.8% | 95.8% | 95.8% | 0.994 |
| K-Nearest Neighbors | 97.4% | 96.0% | 100% | 98.0% | 0.988 |

**Best model:** Logistic Regression, with only 2 misclassifications out of 114 test samples.

The most predictive features were the "worst" (largest) measurements of area, concave points, and radius — consistent with the clinical intuition that larger, more irregularly shaped masses are more likely to be malignant.

## How to Run

1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
2. Run the script:
   ```bash
   python breast_cancer_detection.py
   ```
3. All plots, metrics, and reports are saved to the `outputs/` folder.

## UI/UX (Proposed Interface)

While this project currently runs as a script producing plots and reports, a natural next step is a simple interactive interface so non-technical users (e.g. clinicians reviewing results) can explore predictions without touching code. Proposed design:

**Input screen**
- A clean form with sliders or numeric inputs for the key tumor features (radius, texture, perimeter, area, smoothness, concavity, concave points) — grouped under collapsible "mean / standard error / worst" sections so the full 30-feature input doesn't overwhelm the user.
- Sensible default values pre-filled (e.g. dataset averages) so the form is never empty.
- Inline tooltips explaining each feature in plain language (e.g. "Concavity — how indented the tumor's contour is").

**Results screen**
- A prominent, color-coded prediction badge: green "Benign" / red "Malignant", with the model's confidence (probability) shown as a percentage.
- A short explanation panel showing which input features pushed the prediction toward malignant or benign (e.g. a bar chart of feature contributions), so the result isn't a black box.
- A persistent disclaimer banner: *"For educational/demo purposes only — not a diagnostic tool."*

**Dashboard / model comparison view**
- The existing `outputs/` plots (ROC curves, confusion matrices, feature importance) surfaced as a tabbed dashboard rather than static files, letting a user switch between models to compare performance at a glance.

**Design principles**
- Accessibility first: high-contrast colors, readable font sizes, and no reliance on color alone to convey the diagnosis (use icons/text labels too).
- Minimal cognitive load: progressive disclosure (advanced/less-common features hidden by default).
- Trust and transparency: always show model confidence and a feature-contribution explanation alongside any prediction, never just a bare label.

This could be implemented as a lightweight Streamlit or Flask app, or a React frontend calling a small prediction API built on top of the trained model.

## Possible Extensions

- Hyperparameter tuning (`GridSearchCV`) for each model
- Threshold tuning to prioritize recall (minimizing false negatives, which is critical in a screening context)
- Model persistence (`joblib`/`pickle`) for deployment
- Build out the proposed UI/UX above as a Streamlit or Flask web app

## Disclaimer

This project is for educational and portfolio purposes only. It is **not** a certified medical diagnostic tool and should not be used for actual clinical decision-making.
