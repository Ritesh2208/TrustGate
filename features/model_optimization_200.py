import json
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# SETTINGS
# ============================================================

ORIGINAL_LABEL_FILE = "outputs/trustgate_human_evaluation_completed.csv"
NEXT_LABEL_FILE = "outputs/trustgate_human_evaluation_next_100_completed.csv"

ORIGINAL_FEATURE_FILE = "outputs/truthfulqa_100_combined_features.json"
NEXT_FEATURE_FILE = "outputs/trustgate_next_100_master_901.json"

OUTPUT_FILE = "outputs/model_optimization_200_results.csv"
OUTPUT_BEST = "outputs/model_optimization_200_best.json"

SEED = 42
FOLDS = 5

LABEL_MAP = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2,
}

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(x):
    return str(x).strip().upper()


def get_hidden(item):
    hidden = item.get("hidden_state", item.get("hidden_features"))
    if hidden is None:
        raise ValueError(f"No hidden representation found for {item.get('id')}")
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.size != 896:
        raise ValueError(
            f"Expected 896 hidden features for {item.get('id')}, got {hidden.size}"
        )
    return hidden


def get_signal(item, name):
    # Original JSON stores signals as top-level fields.
    if name in item:
        return float(item.get(name, 0.0))

    # Next JSON stores them inside "signals".
    signals = item.get("signals", {})
    return float(signals.get(name, 0.0))


def make_898(item):
    hidden = get_hidden(item)

    confidence = get_signal(item, "confidence")
    entropy = get_signal(item, "entropy")

    return np.concatenate(
        [hidden, np.array([confidence, entropy], dtype=np.float32)]
    ).astype(np.float32)


print("=" * 72)
print("TRUSTGATE 200-SAMPLE MODEL OPTIMIZATION")
print("=" * 72)

original_labels = pd.read_csv(ORIGINAL_LABEL_FILE)
next_labels = pd.read_csv(NEXT_LABEL_FILE)

original_features = load_json(ORIGINAL_FEATURE_FILE)
next_features = load_json(NEXT_FEATURE_FILE)

original_map = {str(x["id"]): x for x in original_features}
next_map = {str(x["id"]): x for x in next_features}

X_list = []
y_list = []
ids = []
dataset_source = []

def align(labels, fmap, source):
    for _, row in labels.iterrows():
        item_id = str(row["id"])
        label = normalize_label(row["human_label"])

        if label not in LABEL_MAP:
            raise ValueError(f"Invalid label: {label} for {item_id}")

        if item_id not in fmap:
            raise ValueError(f"Missing feature record: {item_id} ({source})")

        X_list.append(make_898(fmap[item_id]))
        y_list.append(LABEL_MAP[label])
        ids.append(item_id)
        dataset_source.append(source)

align(original_labels, original_map, "original_100")
align(next_labels, next_map, "next_100")

X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.int64)

print("\nDataset:")
print("  Samples:", len(X))
print("  Features:", X.shape[1])
print("  Original:", sum(s == "original_100" for s in dataset_source))
print("  Next:", sum(s == "next_100" for s in dataset_source))

print("\nLabels:")
for name, value in LABEL_MAP.items():
    print(f"  {name}: {int(np.sum(y == value))}")

if X.shape != (200, 898):
    raise ValueError(f"Expected X shape (200, 898), got {X.shape}")


# ============================================================
# MODELS
# ============================================================

def make_pipeline(model, pca_components):
    steps = [("scaler", StandardScaler())]

    if pca_components is not None:
        steps.append(
            ("pca", PCA(n_components=pca_components, random_state=SEED))
        )

    steps.append(("model", model))
    return Pipeline(steps)


models = {
    "LogisticRegression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=1.0,
        random_state=SEED,
    ),
    "LinearSVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=10000,
        random_state=SEED,
    ),
    "RBFSVM": SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        random_state=SEED,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=SEED,
    ),
}

pca_options = [None, 10, 20, 30, 50]


# ============================================================
# CROSS-VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=FOLDS,
    shuffle=True,
    random_state=SEED,
)

results = []

for model_name, model in models.items():
    for pca_components in pca_options:

        label = f"{model_name} + " + (
            "NoPCA" if pca_components is None else f"PCA{pca_components}"
        )

        print(f"\nTesting: {label}")

        fold_acc = []
        fold_bal = []
        fold_macro = []
        fold_weighted = []

        oof_true = []
        oof_pred = []

        for fold, (train_idx, test_idx) in enumerate(
            skf.split(X, y), start=1
        ):
            estimator = make_pipeline(
                clone(model),
                pca_components,
            )

            estimator.fit(X[train_idx], y[train_idx])
            pred = estimator.predict(X[test_idx])

            y_test = y[test_idx]

            fold_acc.append(accuracy_score(y_test, pred))
            fold_bal.append(balanced_accuracy_score(y_test, pred))
            fold_macro.append(
                f1_score(y_test, pred, average="macro", zero_division=0)
            )
            fold_weighted.append(
                f1_score(y_test, pred, average="weighted", zero_division=0)
            )

            oof_true.extend(y_test.tolist())
            oof_pred.extend(pred.tolist())

        result = {
            "model": model_name,
            "pca": "none" if pca_components is None else pca_components,
            "cv_accuracy": accuracy_score(oof_true, oof_pred),
            "cv_balanced_accuracy": balanced_accuracy_score(oof_true, oof_pred),
            "cv_macro_f1": f1_score(
                oof_true, oof_pred, average="macro", zero_division=0
            ),
            "cv_weighted_f1": f1_score(
                oof_true, oof_pred, average="weighted", zero_division=0
            ),
            "mean_fold_accuracy": float(np.mean(fold_acc)),
            "std_fold_accuracy": float(np.std(fold_acc)),
            "mean_fold_balanced_accuracy": float(np.mean(fold_bal)),
            "mean_fold_macro_f1": float(np.mean(fold_macro)),
            "mean_fold_weighted_f1": float(np.mean(fold_weighted)),
        }

        results.append(result)

        print(
            f"  Accuracy={result['cv_accuracy']:.4f} | "
            f"Balanced={result['cv_balanced_accuracy']:.4f} | "
            f"Macro-F1={result['cv_macro_f1']:.4f}"
        )


# ============================================================
# RANKING
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    ["cv_macro_f1", "cv_balanced_accuracy", "cv_accuracy"],
    ascending=False,
).reset_index(drop=True)

results_df.to_csv(OUTPUT_FILE, index=False)

best = results_df.iloc[0].to_dict()

with open(OUTPUT_BEST, "w", encoding="utf-8") as f:
    json.dump(
        {
            "selection_rule": "Highest 5-fold out-of-fold macro F1; ties resolved by balanced accuracy then accuracy.",
            "best_model": best,
            "data": {
                "samples": 200,
                "features": 898,
                "features_definition": "896 hidden + confidence + entropy",
                "labels": LABEL_MAP,
                "original_label_file": ORIGINAL_LABEL_FILE,
                "next_label_file": NEXT_LABEL_FILE,
            },
        },
        f,
        indent=4,
    )


print("\n" + "=" * 72)
print("OPTIMIZATION COMPLETE")
print("=" * 72)

print("\nTop 10 candidates:")
print(
    results_df[
        [
            "model",
            "pca",
            "cv_accuracy",
            "cv_balanced_accuracy",
            "cv_macro_f1",
            "cv_weighted_f1",
        ]
    ].head(10).to_string(index=False)
)

print("\nBEST CANDIDATE:")
print("  Model:", best["model"])
print("  PCA:", best["pca"])
print(f"  Accuracy: {best['cv_accuracy']:.4f}")
print(f"  Balanced Accuracy: {best['cv_balanced_accuracy']:.4f}")
print(f"  Macro F1: {best['cv_macro_f1']:.4f}")
print(f"  Weighted F1: {best['cv_weighted_f1']:.4f}")

print("\nSaved:")
print(" ", OUTPUT_FILE)
print(" ", OUTPUT_BEST)
