"""
Breast Cancer Detection
========================
Predicts whether a breast tumor is malignant (M) or benign (B) using the
Wisconsin Diagnostic Breast Cancer (WDBC) dataset (same dataset as UCI ID 17,
bundled with scikit-learn as `load_breast_cancer`).

Pipeline:
1. Load & explore data
2. Preprocess (scale features)
3. Train multiple classifiers
4. Evaluate & compare (accuracy, precision, recall, F1, ROC-AUC)
5. Feature importance
6. Save all plots + a metrics summary
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

OUT = "outputs"
sns.set_style("whitegrid")
RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="diagnosis")  # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape}")
print(f"Class balance:\n{y.value_counts().rename({0:'malignant',1:'benign'})}")

# ---------------------------------------------------------------
# 2. EDA plots
# ---------------------------------------------------------------
# Class distribution
plt.figure(figsize=(5, 4))
sns.countplot(x=y.map({0: "Malignant", 1: "Benign"}), palette=["#d9534f", "#5cb85c"])
plt.title("Diagnosis Class Distribution")
plt.xlabel("")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT}/class_distribution.png", dpi=150)
plt.close()

# Correlation heatmap (mean features only, for readability)
mean_cols = [c for c in X.columns if not c.endswith("error") and "worst" not in c]
plt.figure(figsize=(10, 8))
sns.heatmap(X[mean_cols].corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Feature Correlation Heatmap (mean features)")
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=150)
plt.close()

# Distribution of a few key features by class
key_feats = ["mean radius", "mean texture", "mean perimeter", "mean area",
             "mean smoothness", "mean concavity"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat in zip(axes.ravel(), key_feats):
    sns.kdeplot(data=X.assign(diagnosis=y.map({0: "Malignant", 1: "Benign"})),
                x=feat, hue="diagnosis", fill=True, common_norm=False, ax=ax,
                palette=["#d9534f", "#5cb85c"])
    ax.set_title(feat)
plt.tight_layout()
plt.savefig(f"{OUT}/feature_distributions.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. Train / test split + scaling
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------
# 4. Train multiple models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
}

results = []
roc_data = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "CV Accuracy (mean)": cv_scores.mean(),
        "CV Accuracy (std)": cv_scores.std(),
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{OUT}/confusion_matrix_{safe_name}.png", dpi=150)
    plt.close()

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUT}/model_comparison.csv", index=False)

# ---------------------------------------------------------------
# 5. ROC curves — all models
# ---------------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/roc_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Model comparison bar chart
# ---------------------------------------------------------------
plt.figure(figsize=(9, 5))
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
plot_df = results_df.set_index("Model")[metrics_to_plot]
plot_df.plot(kind="bar", figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0.8, 1.0)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/model_comparison_bar.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. Feature importance (Random Forest)
# ---------------------------------------------------------------
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 8))
importances.head(15).sort_values().plot(kind="barh", color="#4c72b0")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
plt.close()
importances.to_csv(f"{OUT}/feature_importances.csv", header=["importance"])

# ---------------------------------------------------------------
# 8. Best model detailed report
# ---------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_s)

report_text = classification_report(
    y_test, y_pred_best, target_names=["Malignant", "Benign"]
)
with open(f"{OUT}/best_model_report.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n\n")
    f.write(report_text)

print(f"\nBest model: {best_model_name}")
print(report_text)
print("\nAll outputs saved to:", OUT)
