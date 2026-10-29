import json
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


INPUT_FILE = "outputs/trustgate_standardized_500_labeled.json"


# --------------------------------------------------
# Load data
# --------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

X_hidden = np.array(
    [item["hidden_state"] for item in data],
    dtype=np.float32
)

confidence = np.array(
    [item["confidence"] for item in data],
    dtype=np.float32
).reshape(-1, 1)

entropy = np.array(
    [item["entropy"] for item in data],
    dtype=np.float32
).reshape(-1, 1)

labels = np.array(
    [item["label"] for item in data]
)

print("Samples:", len(data))
print("Hidden shape:", X_hidden.shape)
print("Confidence shape:", confidence.shape)
print("Entropy shape:", entropy.shape)


# --------------------------------------------------
# Build 22-D representation
# --------------------------------------------------
#
# Hidden: 896 → PCA → 20
# Confidence: raw scalar
# Entropy: raw scalar
#
# Final = 20 + 1 + 1 = 22
# --------------------------------------------------

hidden_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=20, random_state=42))
])

X_hidden_20 = hidden_pipeline.fit_transform(X_hidden)

X_22 = np.hstack([
    X_hidden_20,
    confidence,
    entropy
])

print("Final representation:", X_22.shape)


# --------------------------------------------------
# Classifier
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=10000
        )
    )
])


# --------------------------------------------------
# 5-fold stratified CV
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

predictions = cross_val_predict(
    model,
    X_22,
    labels,
    cv=cv
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(labels, predictions)
balanced = balanced_accuracy_score(labels, predictions)
macro_f1 = f1_score(
    labels,
    predictions,
    average="macro"
)

print("\n========================================")
print("22-D FUSION RESULTS")
print("========================================")

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        labels,
        predictions,
        labels=["RELIABLE", "UNCERTAIN", "UNRELIABLE"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        labels,
        predictions,
        labels=["RELIABLE", "UNCERTAIN", "UNRELIABLE"]
    )
)


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

output = []

for item, pred in zip(data, predictions):
    output.append({
        "id": item["id"],
        "true_label": item["label"],
        "predicted_label": str(pred)
    })

with open(
    "outputs/trustgate_500_22d_fusion_cv_predictions.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(output, f, indent=2)

print(
    "\nSaved:",
    "outputs/trustgate_500_22d_fusion_cv_predictions.json"
)