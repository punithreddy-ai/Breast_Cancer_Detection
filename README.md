<div align="center">

<img src="banner.svg" alt="Breast Cancer Detection banner" width="100%"/>

# 🎗️ Breast Cancer Detection

### Machine learning–powered diagnosis support using the Wisconsin Diagnostic Breast Cancer dataset

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

![Accuracy](https://img.shields.io/badge/Accuracy-98.2%25-2ECC71?style=flat-square)
![ROC AUC](https://img.shields.io/badge/ROC--AUC-0.995-2ECC71?style=flat-square)
![Samples](https://img.shields.io/badge/Samples-569-4C72B0?style=flat-square)
![Features](https://img.shields.io/badge/Features-30-4C72B0?style=flat-square)
![Models](https://img.shields.io/badge/Models%20Compared-4-4C72B0?style=flat-square)
![Status](https://img.shields.io/badge/Status-Educational%20Project-E8899C?style=flat-square)

</div>

---

A machine learning project that classifies breast tumors as **malignant** or **benign** using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset — the same dataset hosted on the [UCI Machine Learning Repository (ID 17)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) and bundled directly with scikit-learn (`sklearn.datasets.load_breast_cancer`).

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [UI/UX Design](#-uiux-design-proposed-diagnostic-assistant)
- [Possible Extensions](#-possible-extensions)
- [Disclaimer](#%EF%B8%8F-disclaimer)

## 🔬 Overview

Early and accurate detection of breast cancer significantly improves treatment outcomes. This project builds and compares several classification models on tumor characteristics — radius, texture, perimeter, area, smoothness, and concavity — computed from digitized images of fine needle aspirate (FNA) biopsies.

## 🧬 Dataset

| | |
|---|---|
| **Samples** | 569 (357 benign, 212 malignant) |
| **Features** | 30 numeric features — mean, standard error, and "worst" values of 10 characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension |
| **Target** | Diagnosis — Malignant (M) or Benign (B) |
| **Source** | [UCI ML Repository #17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) / `sklearn.datasets.load_breast_cancer` |

## 📁 Project Structure

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

## 🧪 Methodology

1. **Exploratory Data Analysis** — class balance, feature correlations, and distributions of key features by diagnosis.
2. **Preprocessing** — 80/20 stratified train/test split, followed by feature standardization (`StandardScaler`).
3. **Model Training** — four classifiers trained and evaluated:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (RBF kernel)
   - K-Nearest Neighbors
4. **Evaluation** — accuracy, precision, recall, F1-score, ROC-AUC, and 5-fold cross-validation for each model.
5. **Feature Importance** — extracted from the Random Forest model to identify the most predictive tumor characteristics.

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Logistic Regression** | **98.2%** | 98.6% | 98.6% | 98.6% | **0.995** |
| SVM (RBF) | 98.2% | 98.6% | 98.6% | 98.6% | 0.995 |
| Random Forest | 94.7% | 95.8% | 95.8% | 95.8% | 0.994 |
| K-Nearest Neighbors | 97.4% | 96.0% | 100% | 98.0% | 0.988 |

**Best model:** Logistic Regression, with only 2 misclassifications out of 114 test samples.

The most predictive features were the "worst" (largest) measurements of area, concave points, and radius — consistent with the clinical intuition that larger, more irregularly shaped masses are more likely to be malignant.

<div align="center">
<table>
<tr>
<td><img src="outputs/roc_curves.png" width="420"/></td>
<td><img src="outputs/model_comparison_bar.png" width="420"/></td>
</tr>
<tr>
<td align="center"><sub>ROC curves across all four models</sub></td>
<td align="center"><sub>Head-to-head metric comparison</sub></td>
</tr>
<tr>
<td><img src="outputs/feature_importance.png" width="420"/></td>
<td><img src="outputs/confusion_matrix_logistic_regression.png" width="420"/></td>
</tr>
<tr>
<td align="center"><sub>Top predictive features (Random Forest)</sub></td>
<td align="center"><sub>Confusion matrix — best model</sub></td>
</tr>
</table>
</div>

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
2. Run the script:
   ```bash
   python breast_cancer_detection.py
   ```
3. All plots, metrics, and reports are saved to the `outputs/` folder.

## 🖥️ UI/UX Design (Proposed Diagnostic Assistant)

While this project currently runs as a Python script, it's designed with a future-facing companion web interface in mind — a **Diagnostic Assistant** that lets a clinician or researcher enter tumor measurements and get an instant, explainable prediction.

### Concept

The interface is built around a single job: turn a handful of biopsy measurements into a clear, trustworthy read on malignancy risk — never a bare "yes/no," always shown with the model's confidence and the features that drove it.

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  Diagnostic Assistant                            [ ? ]  │
├─────────────────────────────────┬───────────────────────┤
│  Tumor Measurements              │   Result               │
│                                   │                       │
│  Radius (mean)      [ 14.2  ]    │   ● Benign             │
│  Texture (mean)     [ 19.8  ]    │     97% confidence     │
│  Perimeter (mean)   [ 91.3  ]    │                       │
│  Area (mean)        [ 654.1 ]    │   Top contributing     │
│  Smoothness (mean)  [ 0.096 ]    │   features:            │
│  Concavity (mean)   [ 0.079 ]    │   • Worst area   ▓▓▓▓░ │
│  Concave points     [ 0.052 ]    │   • Worst radius ▓▓▓░░ │
│                                   │   • Worst conc.  ▓▓░░░ │
│  [ Load sample case ▾ ]          │                       │
│  [        Predict        ]       │   [ Export report ]   │
└─────────────────────────────────┴───────────────────────┘
```

- **Left panel — input:** A clean form for the handful of features that matter most (rather than all 30), with sensible defaults and a "load sample case" dropdown for demos.
- **Right panel — result:** A single, unambiguous verdict (Benign / Malignant) with the model's confidence, plus a ranked list of the features that most influenced the prediction — so the tool explains itself rather than acting as a black box.
- **Tone:** Clinical, calm, and precise. No alarming colors or aggressive language — a soft green/amber/red indicator conveys risk level without being dramatic.

### Visual Direction

| Element | Choice | Why |
|---|---|---|
| 🎨 Palette | Muted clinical blue `#2C5F7C` primary, slate-gray background `#F5F7F8`, amber `#D98E04` and rose `#C1495F` reserved only for risk indicators | Keeps the tool feeling calm and trustworthy; color is used functionally, not decoratively |
| 🔤 Typography | A clean grotesk (e.g. Inter) for labels and data, slightly heavier weight for the result verdict | Data-dense screens need legibility over personality |
| 📐 Layout | Two-column split — inputs left, result right — so the "answer" is always visible while adjusting inputs | Encourages exploration without losing context |
| ✨ Signature element | A horizontal confidence bar next to the verdict, animated on prediction | Makes uncertainty visible at a glance, rather than hiding behind a single label |

### Key UX Principles

1. **Explainability first** — every prediction is shown with *why*, not just *what*.
2. **Non-diagnostic framing** — the UI visibly disclaims that this is a research/educational tool, not a certified medical device.
3. **Low friction for exploration** — sample cases and sliders make "what if" scenarios easy to explore.
4. **Accessible by default** — sufficient color contrast, no reliance on color alone to convey risk, full keyboard navigation.

*This section describes a proposed interface; it is not yet implemented. Contributions or mockups are welcome.*

## 🔭 Possible Extensions

- Hyperparameter tuning (`GridSearchCV`) for each model
- Threshold tuning to prioritize recall (minimizing false negatives — critical in a screening context)
- Model persistence (`joblib`/`pickle`) for deployment
- Build out the proposed UI/UX above as a Streamlit or Flask web app

## ⚠️ Disclaimer

This project is for educational and portfolio purposes only. It is **not** a certified medical diagnostic tool and should not be used for actual clinical decision-making.

---

<div align="center">
<sub>Built with 🩺 for early detection awareness</sub>
</div>
