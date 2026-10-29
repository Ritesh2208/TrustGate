import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score

INPUT = "outputs/trustgate_standardized_500_labeled.json"
OUTPUT = "outputs/trustgate_500_cv_results.csv"

with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)

X = np.array([r["fusion_features"] for r in data], dtype=np.float32)
y = np.array([r["label"] for r in data])

print("Dataset:", X.shape)
print("Labels:")
print(pd.Series(y).value_counts())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "LogisticRegression": Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=20, random_state=42)),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "LinearSVM": Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=20, random_state=42)),
        ("clf", LinearSVC(
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "RandomForest": Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=20, random_state=42)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "HistGradientBoosting": Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=20, random_state=42)),
        ("clf", HistGradientBoostingClassifier(
            max_iter=200,
            random_state=42
        ))
    ])
}

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "macro_f1": make_scorer(f1_score, average="macro")
}

results = []

for name, model in models.items():
    print("\nRunning:", name)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    row = {
        "model": name,
        "accuracy": scores["test_accuracy"].mean(),
        "balanced_accuracy": scores["test_balanced_accuracy"].mean(),
        "macro_f1": scores["test_macro_f1"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "balanced_accuracy_std": scores["test_balanced_accuracy"].std(),
        "macro_f1_std": scores["test_macro_f1"].std()
    }

    results.append(row)

    print(f"Accuracy:          {row['accuracy']:.4f}")
    print(f"Balanced Accuracy: {row['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {row['macro_f1']:.4f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    ["macro_f1", "balanced_accuracy", "accuracy"],
    ascending=False
)

results_df.to_csv(OUTPUT, index=False)

print("\n===== FINAL 500-SAMPLE CV RESULTS =====")
print(results_df.to_string(index=False))
print("\nSaved:", OUTPUT)