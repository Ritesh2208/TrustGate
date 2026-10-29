import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    accuracy_score,
    f1_score
)

INPUT = "outputs/trustgate_standardized_500_labeled.json"

with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)

X = np.array([r["fusion_features"] for r in data], dtype=np.float32)
y = np.array([r["label"] for r in data])

model = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=20, random_state=42)),
    ("clf", LinearSVC(
        class_weight="balanced",
        random_state=42
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

pred = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    n_jobs=-1
)

labels = ["RELIABLE", "UNCERTAIN", "UNRELIABLE"]

print("\n===== 500-SAMPLE LINEAR SVM =====")
print("Accuracy:", accuracy_score(y, pred))
print("Balanced Accuracy:", balanced_accuracy_score(y, pred))
print("Macro F1:", f1_score(y, pred, average="macro"))

print("\n===== CONFUSION MATRIX =====")
print("Rows = actual, columns = predicted")
print(pd.DataFrame(
    confusion_matrix(y, pred, labels=labels),
    index=labels,
    columns=labels
))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y,
    pred,
    labels=labels,
    digits=4,
    zero_division=0
))