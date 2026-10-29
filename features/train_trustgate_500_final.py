import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

INPUT = "outputs/trustgate_standardized_500_labeled.json"
MODEL_OUT = "models/trustgate_500_linear_svm.joblib"

with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)

X = np.array(
    [r["fusion_features"] for r in data],
    dtype=np.float32
)

y = np.array([r["label"] for r in data])

print("Training data:", X.shape)
print("\nClass distribution:")
print(pd.Series(y).value_counts())

model = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=20, random_state=42)),
    ("clf", LinearSVC(
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X, y)

joblib.dump(model, MODEL_OUT)

print("\n===== TRUSTGATE 500 MODEL =====")
print("Model: Linear SVM")
print("Input dimension:", X.shape[1])
print("PCA components: 20")
print("Training samples:", len(X))
print("Saved:", MODEL_OUT)