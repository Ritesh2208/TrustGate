import json
import joblib
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = (
    "models/trustgate_final_fusion_pca20_logreg.joblib"
)

LABEL_ENCODER_FILE = (
    "models/trustgate_label_encoder.joblib"
)

INPUT_FILE = (
    "outputs/truthfulqa_next_100_features.json"
)

OUTPUT_FILE = (
    "outputs/trustgate_final_predictions.json"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUSTGATE FINAL MODEL INFERENCE")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading final TrustGate model...")

model = joblib.load(
    MODEL_FILE
)

label_encoder = joblib.load(
    LABEL_ENCODER_FILE
)

print("Model loaded successfully.")

print(
    "Classes:",
    label_encoder.classes_
)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading feature dataset...")

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


print(
    "Records loaded:",
    len(data)
)


# ============================================================
# BUILD FUSION FEATURES
# ============================================================

print("\nBuilding fusion feature matrix...")

feature_vectors = []


for record in data:

    hidden_state = record[
        "hidden_state"
    ]

    confidence = record[
        "confidence"
    ]

    entropy = record[
        "entropy"
    ]

    vector = (

        list(hidden_state)

        +

        [

            confidence,

            entropy

        ]

    )

    feature_vectors.append(
        vector
    )


X = np.array(

    feature_vectors,

    dtype=np.float32

)


print(
    "Feature matrix shape:",
    X.shape
)


if X.shape[1] != 898:

    raise ValueError(

        f"Expected 898 features, got {X.shape[1]}"

    )


# ============================================================
# PREDICT
# ============================================================

print("\nGenerating predictions...")

predicted_classes = model.predict(
    X
)

predicted_probabilities = model.predict_proba(
    X
)

predicted_labels = label_encoder.inverse_transform(
    predicted_classes
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = []


for index, record in enumerate(data):

    probabilities = predicted_probabilities[
        index
    ]

    prediction_confidence = float(
        np.max(
            probabilities
        )
    )


    result = {

        "id":
            record["id"],

        "question":
            record["question"],

        "answer":
            record["answer"],

        "reference_answer":
            record.get(
                "reference_answer",
                ""
            ),

        "predicted_label":
            predicted_labels[
                index
            ],

        "predicted_class":
            int(
                predicted_classes[
                    index
                ]
            ),

        "prediction_confidence":
            prediction_confidence,

        "probability_reliable":
            float(
                probabilities[
                    list(
                        label_encoder.classes_
                    ).index(
                        "RELIABLE"
                    )
                ]
            ),

        "probability_uncertain":
            float(
                probabilities[
                    list(
                        label_encoder.classes_
                    ).index(
                        "UNCERTAIN"
                    )
                ]
            ),

        "probability_unreliable":
            float(
                probabilities[
                    list(
                        label_encoder.classes_
                    ).index(
                        "UNRELIABLE"
                    )
                ]
            )

    }


    results.append(
        result
    )


# ============================================================
# SAVE JSON
# ============================================================

print("\nSaving predictions...")


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(

        results,

        f,

        indent=4,

        ensure_ascii=False

    )


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

labels = [

    x[
        "predicted_label"
    ]

    for x in results

]


print("\nPrediction distribution:")


for label in [

    "RELIABLE",

    "UNCERTAIN",

    "UNRELIABLE"

]:

    count = labels.count(
        label
    )

    percentage = (

        count /

        len(labels)

        *

        100

    )

    print(

        f"{label}: "

        f"{count} "

        f"({percentage:.1f}%)"

    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print(
    "FINAL TRUSTGATE INFERENCE COMPLETE!"
)

print("=" * 70)


print(
    "\nRecords predicted:",
    len(results)
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)