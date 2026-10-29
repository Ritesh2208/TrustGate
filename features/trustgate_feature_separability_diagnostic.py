
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# TRUSTGATE DIAGNOSTIC / SEPARABILITY ANALYSIS
# ============================================================

ORIGINAL_LABEL_FILE = "outputs/trustgate_human_evaluation_completed.csv"
NEXT_LABEL_FILE = "outputs/trustgate_human_evaluation_next_100_completed.csv"

ORIGINAL_FEATURE_FILE = "outputs/truthfulqa_100_combined_features.json"
NEXT_FEATURE_FILE = "outputs/trustgate_next_100_master_901.json"

OUTPUT_JSON = "outputs/trustgate_feature_separability_diagnostic.json"
OUTPUT_CSV = "outputs/trustgate_feature_separability_summary.csv"

LABEL_MAP = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2,
}

LABEL_NAMES = ["UNRELIABLE", "UNCERTAIN", "RELIABLE"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm_label(x):
    return str(x).strip().upper()


def get_hidden(item):
    x = item.get("hidden_state", item.get("hidden_features"))
    if x is None:
        raise ValueError(f"Missing hidden state for {item.get('id')}")
    x = np.asarray(x, dtype=np.float64)
    if len(x) != 896:
        raise ValueError(f"Expected 896 hidden features, got {len(x)}")
    return x


def get_signal(item, name):
    if name in item:
        return float(item[name])
    signals = item.get("signals", {})
    if isinstance(signals, dict):
        return float(signals.get(name, 0.0))
    return 0.0


def build(labels, feature_map):
    H, S, y, ids = [], [], [], []

    signal_names = [
        "confidence",
        "entropy",
        "self_consistency",
        "retrieval_agreement",
        "verifier_score",
    ]

    for _, row in labels.iterrows():
        item_id = str(row["id"])
        label = norm_label(row["human_label"])

        if item_id not in feature_map:
            raise ValueError(f"Missing feature ID: {item_id}")

        H.append(get_hidden(feature_map[item_id]))
        S.append([get_signal(feature_map[item_id], n) for n in signal_names])
        y.append(LABEL_MAP[label])
        ids.append(item_id)

    return (
        np.asarray(H),
        np.asarray(S),
        np.asarray(y),
        ids,
    )


print("=" * 72)
print("TRUSTGATE FEATURE SEPARABILITY DIAGNOSTIC")
print("=" * 72)

orig_labels = pd.read_csv(ORIGINAL_LABEL_FILE)
next_labels = pd.read_csv(NEXT_LABEL_FILE)

orig_features = load_json(ORIGINAL_FEATURE_FILE)
next_features = load_json(NEXT_FEATURE_FILE)

orig_map = {str(x["id"]): x for x in orig_features}
next_map = {str(x["id"]): x for x in next_features}

H1, S1, y1, ids1 = build(orig_labels, orig_map)
H2, S2, y2, ids2 = build(next_labels, next_map)

H = np.vstack([H1, H2])
S = np.vstack([S1, S2])
y = np.concatenate([y1, y2])

print("\nData:")
print("  Hidden:", H.shape)
print("  Signals:", S.shape)

# ============================================================
# 1. SIGNAL STATISTICS BY CLASS
# ============================================================

signal_names = [
    "confidence",
    "entropy",
    "self_consistency",
    "retrieval_agreement",
    "verifier_score",
]

signal_rows = []

for j, name in enumerate(signal_names):
    for cls, cls_name in enumerate(LABEL_NAMES):
        vals = S[y == cls, j]

        signal_rows.append(
            {
                "feature": name,
                "class": cls_name,
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        )

signal_df = pd.DataFrame(signal_rows)

print("\nSignal statistics:")
print(signal_df.to_string(index=False))


# ============================================================
# 2. ONE-WAY ANOVA / F SCORE FOR THE FIVE SIGNALS
# ============================================================

f_values, p_values = f_classif(S, y)

anova = []

for name, f, p in zip(signal_names, f_values, p_values):
    anova.append(
        {
            "feature": name,
            "anova_f": float(f),
            "anova_p": float(p),
        }
    )

anova_df = pd.DataFrame(anova).sort_values("anova_f", ascending=False)

print("\nSignal class-separation test:")
print(anova_df.to_string(index=False))


# ============================================================
# 3. HIDDEN-STATE PCA SEPARABILITY
# ============================================================

scaler = StandardScaler()
H_scaled = scaler.fit_transform(H)

pca = PCA(n_components=20, random_state=42)
H_pca = pca.fit_transform(H_scaled)

explained = pca.explained_variance_ratio_

print("\nHidden PCA:")
print("  PC1 variance:", round(float(explained[0]), 6))
print("  PC2 variance:", round(float(explained[1]), 6))
print("  PC1-20 variance:", round(float(np.sum(explained)), 6))


# ============================================================
# 4. LDA CLASS SEPARABILITY
# ============================================================

# LDA is used diagnostically, not as the TrustGate final model.
lda = LinearDiscriminantAnalysis(n_components=2)
H_lda = lda.fit_transform(H_scaled, y)

lda_explained = getattr(
    lda,
    "explained_variance_ratio_",
    np.array([]),
)

print("\nLDA:")
if len(lda_explained):
    print("  LD1 variance:", round(float(lda_explained[0]), 6))
    if len(lda_explained) > 1:
        print("  LD2 variance:", round(float(lda_explained[1]), 6))


# ============================================================
# 5. SILHOUETTE SCORE
# ============================================================

# Positive silhouette means better class clustering.
# Near zero means strong overlap.
# Negative means many points may be closer to another class.
silhouette_hidden = silhouette_score(
    H_pca,
    y,
    metric="euclidean",
)

print("\nSilhouette score:")
print("  Hidden PCA20:", round(float(silhouette_hidden), 6))


# ============================================================
# 6. CLASS CENTROID DISTANCES
# ============================================================

centroids = {}

for cls, cls_name in enumerate(LABEL_NAMES):
    centroids[cls_name] = np.mean(
        H_pca[y == cls],
        axis=0,
    )

def euclidean(a, b):
    return float(np.linalg.norm(a - b))


centroid_distances = {}

for i in range(3):
    for j in range(i + 1, 3):
        a = LABEL_NAMES[i]
        b = LABEL_NAMES[j]
        centroid_distances[f"{a}_vs_{b}"] = euclidean(
            centroids[a],
            centroids[b],
        )

print("\nCentroid distances in hidden PCA20:")
for k, v in centroid_distances.items():
    print(f"  {k}: {v:.6f}")


# ============================================================
# 7. ORIGINAL VS NEXT-100 FEATURE SHIFT
# ============================================================

shift_rows = []

for j, name in enumerate(signal_names):
    a = S1[:, j]
    b = S2[:, j]

    shift_rows.append(
        {
            "feature": name,
            "original_mean": float(np.mean(a)),
            "next_mean": float(np.mean(b)),
            "mean_difference": float(np.mean(b) - np.mean(a)),
            "original_std": float(np.std(a)),
            "next_std": float(np.std(b)),
        }
    )

shift_df = pd.DataFrame(shift_rows)

# Hidden-state shift using standardized combined space.
orig_centroid = np.mean(H_scaled[:100], axis=0)
next_centroid = np.mean(H_scaled[100:], axis=0)

hidden_centroid_shift = float(
    np.linalg.norm(next_centroid - orig_centroid)
)

print("\nOriginal vs Next feature shift:")
print(shift_df.to_string(index=False))
print(
    "\nHidden centroid shift (standardized 896-D):",
    round(hidden_centroid_shift, 6),
)


# ============================================================
# 8. CLASS CENTROIDS FOR SIGNALS
# ============================================================

signal_centroids = {}

for cls, cls_name in enumerate(LABEL_NAMES):
    signal_centroids[cls_name] = {
        signal_names[j]: float(np.mean(S[y == cls, j]))
        for j in range(len(signal_names))
    }


# ============================================================
# SAVE
# ============================================================

summary_rows = []

for row in anova:
    summary_rows.append(
        {
            "type": "signal_separation",
            "feature": row["feature"],
            "value_1": row["anova_f"],
            "value_2": row["anova_p"],
        }
    )

summary_rows.append(
    {
        "type": "hidden_silhouette",
        "feature": "hidden_PCA20",
        "value_1": float(silhouette_hidden),
        "value_2": np.nan,
    }
)

summary_rows.append(
    {
        "type": "hidden_centroid_shift",
        "feature": "original_vs_next",
        "value_1": hidden_centroid_shift,
        "value_2": np.nan,
    }
)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_CSV, index=False)

result = {
    "dataset": {
        "samples": int(len(y)),
        "hidden_dimensions": 896,
        "signal_dimensions": 5,
        "original_samples": 100,
        "next_samples": 100,
    },
    "label_distribution": {
        name: int(np.sum(y == value))
        for name, value in LABEL_MAP.items()
    },
    "signal_statistics": signal_df.to_dict(orient="records"),
    "signal_anova": anova,
    "hidden_pca": {
        "pc1_variance": float(explained[0]),
        "pc2_variance": float(explained[1]),
        "pc1_to_pc20_variance": float(np.sum(explained)),
    },
    "hidden_lda": {
        "explained_variance_ratio": lda_explained.tolist(),
    },
    "hidden_silhouette_pca20": float(silhouette_hidden),
    "hidden_centroid_distances_pca20": centroid_distances,
    "signal_centroids": signal_centroids,
    "original_next_signal_shift": shift_df.to_dict(orient="records"),
    "hidden_centroid_shift_standardized_896d": hidden_centroid_shift,
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)

print("\nSaved:")
print(" ", OUTPUT_CSV)
print(" ", OUTPUT_JSON)

print("\nDiagnostic complete.")
