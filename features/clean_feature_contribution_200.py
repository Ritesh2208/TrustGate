
import json
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# TRUSTGATE FEATURE CONTRIBUTION STUDY — CLEAN 200 SAMPLE CV
#
# Purpose:
# Determine whether confidence, entropy, self-consistency,
# retrieval agreement, and verifier score add predictive value
# beyond the 896-dimensional hidden representation.
#
# IMPORTANT:
# - Uses only the 200 training/development samples.
# - All scaling/PCA is inside the CV pipeline.
# - Does NOT use the independent Next-100 for feature selection.
# ============================================================

ORIGINAL_LABEL_FILE = "outputs/trustgate_human_evaluation_completed.csv"
NEXT_LABEL_FILE = "outputs/trustgate_human_evaluation_next_100_completed.csv"

ORIGINAL_FEATURE_FILE = "outputs/truthfulqa_100_combined_features.json"
NEXT_FEATURE_FILE = "outputs/trustgate_next_100_master_901.json"

OUTPUT_CSV = "outputs/trustgate_clean_feature_contribution_200.csv"
OUTPUT_JSON = "outputs/trustgate_clean_feature_contribution_200.json"

SEED = 42

LABEL_MAP = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2,
}

random.seed(SEED)
np.random.seed(SEED)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm_label(x):
    return str(x).strip().upper()


def hidden(item):
    x = item.get("hidden_state", item.get("hidden_features"))
    if x is None:
        raise ValueError(f"Missing hidden representation: {item.get('id')}")
    x = np.asarray(x, dtype=np.float32)
    if x.size != 896:
        raise ValueError(f"Expected 896 hidden values, got {x.size}")
    return x


def signal(item, name):
    if name in item:
        return float(item[name])
    signals = item.get("signals", {})
    if isinstance(signals, dict):
        return float(signals.get(name, 0.0))
    return 0.0


def all_features(item):
    h = hidden(item)
    names = [
        "confidence",
        "entropy",
        "self_consistency",
        "retrieval_agreement",
        "verifier_score",
    ]
    s = np.asarray([signal(item, n) for n in names], dtype=np.float32)
    return h, s


def build(labels_df, fmap):
    hidden_rows = []
    signal_rows = []
    ys = []

    for _, row in labels_df.iterrows():
        item_id = str(row["id"])
        label = norm_label(row["human_label"])

        if label not in LABEL_MAP:
            raise ValueError(f"Unknown label {label} for {item_id}")
        if item_id not in fmap:
            raise ValueError(f"Missing feature record {item_id}")

        h, s = all_features(fmap[item_id])
        hidden_rows.append(h)
        signal_rows.append(s)
        ys.append(LABEL_MAP[label])

    return (
        np.asarray(hidden_rows, dtype=np.float32),
        np.asarray(signal_rows, dtype=np.float32),
        np.asarray(ys, dtype=np.int64),
    )


print("=" * 72)
print("TRUSTGATE CLEAN FEATURE CONTRIBUTION STUDY")
print("=" * 72)

orig_labels = pd.read_csv(ORIGINAL_LABEL_FILE)
next_labels = pd.read_csv(NEXT_LABEL_FILE)

orig_features = load_json(ORIGINAL_FEATURE_FILE)
next_features = load_json(NEXT_FEATURE_FILE)

orig_map = {str(x["id"]): x for x in orig_features}
next_map = {str(x["id"]): x for x in next_features}

H1, S1, y1 = build(orig_labels, orig_map)
H2, S2, y2 = build(next_labels, next_map)

H = np.concatenate([H1, H2], axis=0)
S = np.concatenate([S1, S2], axis=0)
y = np.concatenate([y1, y2], axis=0)

print("\nShapes:")
print("  Hidden:", H.shape)
print("  Signals:", S.shape)
print("  Combined:", (len(y), H.shape[1] + S.shape[1]))

signal_names = [
    "confidence",
    "entropy",
    "self_consistency",
    "retrieval_agreement",
    "verifier_score",
]


# ------------------------------------------------------------
# Candidate feature sets
# ------------------------------------------------------------

feature_sets = {
    "Hidden only": H,
    "Hidden + confidence": np.column_stack([H, S[:, 0]]),
    "Hidden + entropy": np.column_stack([H, S[:, 1]]),
    "Hidden + self_consistency": np.column_stack([H, S[:, 2]]),
    "Hidden + retrieval_agreement": np.column_stack([H, S[:, 3]]),
    "Hidden + verifier_score": np.column_stack([H, S[:, 4]]),
    "Hidden + confidence + entropy": np.column_stack([H, S[:, 0], S[:, 1]]),
    "Hidden + all 5 signals": np.column_stack([H, S]),
}


# ------------------------------------------------------------
# Clean CV
#
# PCA is deliberately fit inside each fold.
# Macro-F1 is the primary selection metric.
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

results = []

for name, X in feature_sets.items():

    print(f"\nTesting: {name} | dimensions={X.shape[1]}")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=20, random_state=SEED)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=1.0,
                    random_state=SEED,
                ),
            ),
        ]
    )

    pred = cross_val_predict(
        pipe,
        X,
        y,
        cv=cv,
        method="predict",
        n_jobs=None,
    )

    result = {
        "feature_set": name,
        "dimensions": int(X.shape[1]),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y, pred, average="weighted", zero_division=0)
        ),
    }

    results.append(result)

    print(
        f"  Accuracy={result['accuracy']:.4f} | "
        f"Balanced={result['balanced_accuracy']:.4f} | "
        f"Macro-F1={result['macro_f1']:.4f}"
    )


df = pd.DataFrame(results)

hidden_base = df.loc[df["feature_set"] == "Hidden only"].iloc[0]

df["delta_accuracy_vs_hidden"] = (
    df["accuracy"] - hidden_base["accuracy"]
)
df["delta_balanced_vs_hidden"] = (
    df["balanced_accuracy"] - hidden_base["balanced_accuracy"]
)
df["delta_macro_f1_vs_hidden"] = (
    df["macro_f1"] - hidden_base["macro_f1"]
)

df = df.sort_values(
    ["macro_f1", "balanced_accuracy", "accuracy"],
    ascending=False,
).reset_index(drop=True)

df.to_csv(OUTPUT_CSV, index=False)

summary = {
    "protocol": {
        "samples": 200,
        "cv": "5-fold stratified",
        "preprocessing": "StandardScaler and PCA20 fit inside each fold",
        "classifier": "LogisticRegression",
        "pca_components": 20,
        "primary_metric": "macro_f1",
    },
    "results": results,
    "ranked_results": df.to_dict(orient="records"),
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)


print("\n" + "=" * 72)
print("FEATURE CONTRIBUTION RESULTS")
print("=" * 72)

print(
    df[
        [
            "feature_set",
            "dimensions",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "delta_macro_f1_vs_hidden",
        ]
    ].to_string(index=False)
)

best = df.iloc[0]

print("\nBEST FEATURE SET:")
print(" ", best["feature_set"])
print(f"  Accuracy: {best['accuracy']:.4f}")
print(f"  Balanced Accuracy: {best['balanced_accuracy']:.4f}")
print(f"  Macro F1: {best['macro_f1']:.4f}")

print("\nSaved:")
print(" ", OUTPUT_CSV)
print(" ", OUTPUT_JSON)
