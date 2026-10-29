# ============================================================
# TRUSTGATE COMBINED 200 MODEL EVALUATION
#
# Matches the EXACT architecture used by the training script.
# Compatible with PyTorch 2.6+.
# ============================================================

import json
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# SETTINGS
# ============================================================

ORIGINAL_FEATURE_FILE = (
    "outputs/truthfulqa_100_combined_features.json"
)

NEXT_FEATURE_FILE = (
    "outputs/trustgate_next_100_master_901.json"
)

ORIGINAL_LABEL_FILE = (
    "outputs/trustgate_human_evaluation_completed.csv"
)

NEXT_LABEL_FILE = (
    "outputs/trustgate_ai_assisted_evaluation_next_100.csv"
)

MODEL_FILE = (
    "outputs/trustgate_fusion_model_combined_200.pt"
)

OUTPUT_JSON = (
    "outputs/trustgate_combined_200_evaluation.json"
)

OUTPUT_CSV = (
    "outputs/trustgate_combined_200_predictions.csv"
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


LABELS = [
    "UNRELIABLE",
    "UNCERTAIN",
    "RELIABLE",
]


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    print()
    print("Loading:", path)

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    print(
        "Records loaded:",
        len(data)
    )

    return data


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels(
    path,
    dataset_name
):

    print()
    print(
        "Loading labels:",
        dataset_name
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(path)

    print(
        "Records:",
        len(df)
    )

    if "id" not in df.columns:

        raise ValueError(
            f"{dataset_name}: Missing 'id' column"
        )

    if "human_label" not in df.columns:

        raise ValueError(
            f"{dataset_name}: Missing 'human_label' column"
        )

    df["human_label"] = (
        df["human_label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    valid_labels = set(LABELS)

    invalid = df[
        ~df["human_label"].isin(valid_labels)
    ]

    if len(invalid) > 0:

        print()
        print("Invalid labels found:")

        print(
            invalid[
                ["id", "human_label"]
            ].head(20)
        )

        raise ValueError(
            f"{dataset_name}: Invalid labels detected"
        )

    print(
        "Validated labels:",
        len(df)
    )

    return dict(

        zip(

            df["id"],

            df["human_label"]

        )

    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(item):

    item_id = item.get(
        "id",
        "UNKNOWN"
    )


    # ========================================================
    # FORMAT 1
    # fusion_features already exists
    # ========================================================

    if "fusion_features" in item:

        features = np.array(

            item["fusion_features"],

            dtype=np.float32

        ).flatten()


        if len(features) != 901:

            raise ValueError(

                f"ID {item_id}: fusion_features has "
                f"{len(features)} dimensions. Expected 901."

            )


        return features


    # ========================================================
    # FORMAT 2
    # hidden_features + signals dictionary
    # ========================================================

    if (

        "hidden_features" in item

        and

        "signals" in item

    ):

        hidden = np.array(

            item["hidden_features"],

            dtype=np.float32

        ).flatten()


        signals_dict = item["signals"]


        signal_order = [

            "confidence",

            "entropy",

            "self_consistency",

            "retrieval_agreement",

            "verifier_score",

        ]


        signals = np.array(

            [

                signals_dict.get(
                    key,
                    0.0
                )

                for key in signal_order

            ],

            dtype=np.float32

        )


        features = np.concatenate(

            [

                hidden,

                signals

            ]

        )


        if len(features) != 901:

            raise ValueError(

                f"ID {item_id}: generated "
                f"{len(features)} dimensions. Expected 901."

            )


        return features


    # ========================================================
    # FORMAT 3
    # hidden_state + individual signals
    # ========================================================

    if "hidden_state" in item:

        hidden = np.array(

            item["hidden_state"],

            dtype=np.float32

        ).flatten()


        if len(hidden) != 896:

            raise ValueError(

                f"ID {item_id}: hidden_state has "
                f"{len(hidden)} dimensions. Expected 896."

            )


        signal_names = [

            "confidence",

            "entropy",

            "self_consistency",

            "retrieval_agreement",

            "verifier_score",

        ]


        signal_values = []


        for name in signal_names:

            value = item.get(

                name,

                0.0

            )


            try:

                value = float(value)

            except (

                ValueError,

                TypeError

            ):

                value = 0.0


            signal_values.append(value)


        signals = np.array(

            signal_values,

            dtype=np.float32

        )


        features = np.concatenate(

            [

                hidden,

                signals

            ]

        )


        if len(features) != 901:

            raise ValueError(

                f"ID {item_id}: generated "
                f"{len(features)} dimensions. Expected 901."

            )


        return features


    # ========================================================
    # FORMAT 4
    # hidden_features + individual signals
    # ========================================================

    if "hidden_features" in item:

        hidden = np.array(

            item["hidden_features"],

            dtype=np.float32

        ).flatten()


        signal_names = [

            "confidence",

            "entropy",

            "self_consistency",

            "retrieval_agreement",

            "verifier_score",

        ]


        signals = []


        for name in signal_names:

            value = item.get(

                name,

                0.0

            )


            try:

                value = float(value)

            except (

                ValueError,

                TypeError

            ):

                value = 0.0


            signals.append(value)


        signals = np.array(

            signals,

            dtype=np.float32

        )


        features = np.concatenate(

            [

                hidden,

                signals

            ]

        )


        if len(features) != 901:

            raise ValueError(

                f"ID {item_id}: generated "
                f"{len(features)} dimensions. Expected 901."

            )


        return features


    raise ValueError(

        f"Unknown feature format for ID {item_id}. "
        f"Available keys: {list(item.keys())}"

    )


# ============================================================
# NORMALIZE FEATURES
#
# MUST MATCH TRAINING SCRIPT
# ============================================================

def normalize_features(

    features,

    checkpoint

):

    features = np.array(

        features,

        dtype=np.float32

    ).flatten()


    if len(features) != 901:

        raise ValueError(

            f"Expected 901 features, got {len(features)}"

        )


    hidden = features[:896]

    signals = features[896:]


    # ========================================================
    # Hidden normalization
    # ========================================================

    hidden_mean = np.array(

        checkpoint["hidden_mean"],

        dtype=np.float32

    ).flatten()


    hidden_std = np.array(

        checkpoint["hidden_std"],

        dtype=np.float32

    ).flatten()


    if len(hidden_mean) != 896:

        raise ValueError(

            f"hidden_mean dimension mismatch: "
            f"{len(hidden_mean)}"

        )


    if len(hidden_std) != 896:

        raise ValueError(

            f"hidden_std dimension mismatch: "
            f"{len(hidden_std)}"

        )


    hidden_std = np.where(

        hidden_std == 0,

        1.0,

        hidden_std

    )


    hidden_normalized = (

        hidden
        -
        hidden_mean

    ) / hidden_std


    # ========================================================
    # Signal normalization
    # ========================================================

    signal_mean = np.array(

        checkpoint["signal_mean"],

        dtype=np.float32

    ).flatten()


    signal_std = np.array(

        checkpoint["signal_std"],

        dtype=np.float32

    ).flatten()


    if len(signal_mean) != 5:

        raise ValueError(

            f"signal_mean dimension mismatch: "
            f"{len(signal_mean)}"

        )


    if len(signal_std) != 5:

        raise ValueError(

            f"signal_std dimension mismatch: "
            f"{len(signal_std)}"

        )


    signal_std = np.where(

        signal_std == 0,

        1.0,

        signal_std

    )


    signal_normalized = (

        signals
        -
        signal_mean

    ) / signal_std


    normalized = np.concatenate(

        [

            hidden_normalized,

            signal_normalized

        ]

    ).astype(

        np.float32

    )


    return normalized


# ============================================================
# EXACT MODEL ARCHITECTURE
#
# THIS MATCHES THE TRAINING SCRIPT EXACTLY
# ============================================================

class TrustGateFusionNet(

    nn.Module

):

    def __init__(

        self,

        hidden_dimension=896,

        signal_dimension=5

    ):

        super().__init__()


        # ====================================================
        # HIDDEN ENCODER
        #
        # Linear(896 -> 128)
        # BatchNorm1d(128)
        # ReLU
        # Dropout(0.20)
        # ====================================================

        self.hidden_encoder = nn.Sequential(

            nn.Linear(

                hidden_dimension,

                128

            ),

            nn.BatchNorm1d(

                128

            ),

            nn.ReLU(),

            nn.Dropout(

                0.20

            )

        )


        # ====================================================
        # SIGNAL ENCODER
        #
        # Linear(5 -> 32)
        # BatchNorm1d(32)
        # ReLU
        # Dropout(0.10)
        # ====================================================

        self.signal_encoder = nn.Sequential(

            nn.Linear(

                signal_dimension,

                32

            ),

            nn.BatchNorm1d(

                32

            ),

            nn.ReLU(),

            nn.Dropout(

                0.10

            )

        )


        # ====================================================
        # FUSION
        #
        # 128 + 32 = 160
        #
        # Linear(160 -> 128)
        # BatchNorm1d(128)
        # ReLU
        # Dropout(0.20)
        # ====================================================

        self.fusion = nn.Sequential(

            nn.Linear(

                160,

                128

            ),

            nn.BatchNorm1d(

                128

            ),

            nn.ReLU(),

            nn.Dropout(

                0.20

            )

        )


        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Linear(

            128,

            3

        )


        # ====================================================
        # TRUST HEAD
        # ====================================================

        self.trust_head = nn.Sequential(

            nn.Linear(

                128,

                1

            )

        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(

        self,

        x

    ):


        hidden_input = x[

            :,

            :896

        ]


        signal_input = x[

            :,

            896:

        ]


        hidden_encoded = (

            self.hidden_encoder(

                hidden_input

            )

        )


        signal_encoded = (

            self.signal_encoder(

                signal_input

            )

        )


        fused = torch.cat(

            [

                hidden_encoded,

                signal_encoded

            ],

            dim=1

        )


        fused = (

            self.fusion(

                fused

            )

        )


        logits = (

            self.classifier(

                fused

            )

        )


        trust_score = torch.sigmoid(

            self.trust_head(

                fused

            )

        )


        return (

            logits,

            trust_score

        )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint():

    print()

    print("=" * 70)

    print(
        "LOADING COMBINED 200 TRUSTGATE MODEL"
    )

    print("=" * 70)


    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(

            f"Model file not found:\n{MODEL_FILE}"

        )


    # ========================================================
    # PyTorch 2.6 compatibility
    # ========================================================

    try:

        checkpoint = torch.load(

            MODEL_FILE,

            map_location=DEVICE,

            weights_only=False

        )


    except TypeError:

        checkpoint = torch.load(

            MODEL_FILE,

            map_location=DEVICE

        )


    print()

    print(
        "Checkpoint loaded successfully."
    )


    if not isinstance(

        checkpoint,

        dict

    ):

        raise ValueError(

            "Checkpoint is not a dictionary."

        )


    print()

    print(

        "Checkpoint keys:"

    )


    print(

        list(

            checkpoint.keys()

        )

    )


    required_keys = [

        "model_state_dict",

        "hidden_mean",

        "hidden_std",

        "signal_mean",

        "signal_std",

    ]


    for key in required_keys:

        if key not in checkpoint:

            raise KeyError(

                f"Checkpoint missing required key: {key}"

            )


    print()

    print("Model information:")


    print(

        "Feature dimension:",

        checkpoint.get(

            "feature_dimension",

            "UNKNOWN"

        )

    )


    print(

        "Hidden dimension:",

        checkpoint.get(

            "hidden_dimension",

            "UNKNOWN"

        )

    )


    print(

        "Signal dimension:",

        checkpoint.get(

            "signal_dimension",

            "UNKNOWN"

        )

    )


    print(

        "Label map:",

        checkpoint.get(

            "label_map",

            "UNKNOWN"

        )

    )


    return checkpoint


# ============================================================
# LOAD MODEL
#
# STRICT LOADING
# ============================================================

def load_model():

    checkpoint = load_checkpoint()


    print()

    print("=" * 70)

    print(
        "BUILDING EXACT TRUSTGATE MODEL ARCHITECTURE"
    )

    print("=" * 70)


    # ========================================================
    # EXACT ARCHITECTURE FROM TRAINING SCRIPT
    # ========================================================

    model = TrustGateFusionNet(

        hidden_dimension=896,

        signal_dimension=5

    )


    state_dict = checkpoint[

        "model_state_dict"

    ]


    print()

    print(

        "Checkpoint model parameters:"

    )


    for key, value in state_dict.items():

        if hasattr(

            value,

            "shape"

        ):

            print(

                f"{key}: {tuple(value.shape)}"

            )


    print()

    print(

        "Loading checkpoint weights strictly..."

    )


    # ========================================================
    # STRICT LOAD
    #
    # This should succeed because architecture now matches
    # your training script exactly.
    # ========================================================

    model.load_state_dict(

        state_dict,

        strict=True

    )


    model.to(

        DEVICE

    )


    model.eval()


    print()

    print("=" * 70)

    print(

        "MODEL LOADED SUCCESSFULLY"

    )

    print("=" * 70)


    print()

    print(

        "Architecture:"

    )


    print(

        "Hidden Encoder: 896 -> 128"

    )


    print(

        "Signal Encoder: 5 -> 32"

    )


    print(

        "Fusion: 160 -> 128"

    )


    print(

        "Classifier: 128 -> 3"

    )


    print(

        "Trust Head: 128 -> 1"

    )


    return (

        model,

        checkpoint

    )


# ============================================================
# ALIGN DATASET
# ============================================================

def align_dataset(

    features,

    label_map,

    dataset_name

):

    print()

    print(

        "Aligning",

        dataset_name,

        "dataset..."

    )


    aligned = []

    missing = []


    for item in features:


        item_id = item.get(

            "id"

        )


        if item_id not in label_map:

            missing.append(

                item_id

            )

            continue


        feature_vector = (

            extract_features(

                item

            )

        )


        aligned.append(

            {

                "id":

                    item_id,


                "question":

                    item.get(

                        "question",

                        ""

                    ),


                "answer":

                    item.get(

                        "answer",

                        item.get(

                            "qwen_answer",

                            ""

                        )

                    ),


                "reference_answer":

                    item.get(

                        "reference_answer",

                        ""

                    ),


                "features":

                    feature_vector,


                "human_label":

                    label_map[

                        item_id

                    ]

            }

        )


    if missing:

        print()

        print(

            "WARNING: Missing labels for IDs:"

        )


        print(

            missing[:20]

        )


    print(

        dataset_name,

        "aligned:",

        len(aligned)

    )


    return aligned


# ============================================================
# PREDICT
# ============================================================

def predict_records(

    model,

    records,

    checkpoint

):

    results = []


    print()

    print("=" * 70)

    print(

        "RUNNING COMBINED 200 EVALUATION"

    )

    print("=" * 70)


    with torch.no_grad():

        for index, record in enumerate(

            records

        ):


            normalized_features = (

                normalize_features(

                    record["features"],

                    checkpoint

                )

            )


            x = torch.tensor(

                normalized_features,

                dtype=torch.float32

            ).unsqueeze(

                0

            ).to(

                DEVICE

            )


            logits, trust_score = model(

                x

            )


            probabilities = torch.softmax(

                logits,

                dim=1

            )[0]


            predicted_class = int(

                torch.argmax(

                    probabilities

                ).item()

            )


            predicted_label = LABELS[

                predicted_class

            ]


            prediction_confidence = float(

                probabilities[

                    predicted_class

                ].item()

            )


            trust_value = float(

                trust_score.item()

            )


            result = {

                "id":

                    record["id"],


                "question":

                    record["question"],


                "answer":

                    record["answer"],


                "reference_answer":

                    record["reference_answer"],


                "human_label":

                    record["human_label"],


                "predicted_label":

                    predicted_label,


                "predicted_class":

                    predicted_class,


                "prediction_confidence":

                    prediction_confidence,


                "trust_score":

                    trust_value,


                "probability_unreliable":

                    float(

                        probabilities[0].item()

                    ),


                "probability_uncertain":

                    float(

                        probabilities[1].item()

                    ),


                "probability_reliable":

                    float(

                        probabilities[2].item()

                    ),

            }


            results.append(

                result

            )


            if (

                (index + 1) % 10 == 0

            ):

                print(

                    f"Processed "

                    f"{index + 1}"

                    f"/"

                    f"{len(records)}"

                )


    return results


# ============================================================
# SPLIT METRICS
# ============================================================

def calculate_split_metrics(

    records

):

    if len(records) == 0:

        return {

            "records": 0,

            "accuracy": None,

            "balanced_accuracy": None,

            "macro_f1": None,

            "weighted_f1": None,

        }


    true_labels = [

        x["human_label"]

        for x in records

    ]


    predicted_labels = [

        x["predicted_label"]

        for x in records

    ]


    return {

        "records":

            len(records),


        "accuracy":

            float(

                accuracy_score(

                    true_labels,

                    predicted_labels

                )

            ),


        "balanced_accuracy":

            float(

                balanced_accuracy_score(

                    true_labels,

                    predicted_labels

                )

            ),


        "macro_f1":

            float(

                f1_score(

                    true_labels,

                    predicted_labels,

                    labels=LABELS,

                    average="macro",

                    zero_division=0

                )

            ),


        "weighted_f1":

            float(

                f1_score(

                    true_labels,

                    predicted_labels,

                    labels=LABELS,

                    average="weighted",

                    zero_division=0

                )

            )

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(

        "TRUSTGATE COMBINED 200 MODEL EVALUATION"

    )

    print("=" * 70)


    # ========================================================
    # LOAD FEATURE DATASETS
    # ========================================================

    print()

    print(

        "Loading feature datasets..."

    )


    original_features = load_json(

        ORIGINAL_FEATURE_FILE

    )


    next_features = load_json(

        NEXT_FEATURE_FILE

    )


    # ========================================================
    # LOAD LABELS
    # ========================================================

    print()

    print(

        "Loading labels..."

    )


    original_labels = load_labels(

        ORIGINAL_LABEL_FILE,

        "Original labels"

    )


    next_labels = load_labels(

        NEXT_LABEL_FILE,

        "Next labels"

    )


    # ========================================================
    # ALIGN DATASETS
    # ========================================================

    original_records = align_dataset(

        original_features,

        original_labels,

        "Original"

    )


    next_records = align_dataset(

        next_features,

        next_labels,

        "Next"

    )


    combined_records = (

        original_records

        +

        next_records

    )


    print()

    print(

        "Combined records:",

        len(combined_records)

    )


    if len(combined_records) != 200:

        print()

        print(

            "WARNING: Expected exactly 200 records."

        )


    if len(combined_records) == 0:

        raise ValueError(

            "No records available for evaluation."

        )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, checkpoint = load_model()


    # ========================================================
    # PREDICT
    # ========================================================

    predictions = predict_records(

        model,

        combined_records,

        checkpoint

    )


    # ========================================================
    # PREPARE LABELS
    # ========================================================

    y_true = [

        item["human_label"]

        for item in predictions

    ]


    y_pred = [

        item["predicted_label"]

        for item in predictions

    ]


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(

        y_true,

        y_pred

    )


    balanced_accuracy = (

        balanced_accuracy_score(

            y_true,

            y_pred

        )

    )


    macro_f1 = f1_score(

        y_true,

        y_pred,

        labels=LABELS,

        average="macro",

        zero_division=0

    )


    weighted_f1 = f1_score(

        y_true,

        y_pred,

        labels=LABELS,

        average="weighted",

        zero_division=0

    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print("=" * 70)

    print(

        "COMBINED 200 EVALUATION RESULTS"

    )

    print("=" * 70)


    print()

    print(

        "Accuracy:",

        round(

            accuracy,

            4

        )

    )


    print(

        "Balanced Accuracy:",

        round(

            balanced_accuracy,

            4

        )

    )


    print(

        "Macro F1:",

        round(

            macro_f1,

            4

        )

    )


    print(

        "Weighted F1:",

        round(

            weighted_f1,

            4

        )

    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()

    print("=" * 70)

    print(

        "CLASSIFICATION REPORT"

    )

    print("=" * 70)


    report = classification_report(

        y_true,

        y_pred,

        labels=LABELS,

        zero_division=0

    )


    print()

    print(

        report

    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=LABELS

    )


    print()

    print("=" * 70)

    print(

        "CONFUSION MATRIX"

    )

    print("=" * 70)


    print()

    print(

        "Rows = Human Labels"

    )


    print(

        "Columns = TrustGate Predictions"

    )


    print()

    print(

        "Labels:",

        LABELS

    )


    print()

    print(

        cm

    )


    # ========================================================
    # DATASET SPLIT METRICS
    #
    # Use dataset position rather than ID assumptions.
    # ========================================================

    original_count = len(

        original_records

    )


    original_predictions = predictions[

        :original_count

    ]


    next_predictions = predictions[

        original_count:

    ]


    original_metrics = (

        calculate_split_metrics(

            original_predictions

        )

    )


    next_metrics = (

        calculate_split_metrics(

            next_predictions

        )

    )


    print()

    print("=" * 70)

    print(

        "ORIGINAL DATASET METRICS"

    )

    print("=" * 70)


    print()

    for key, value in original_metrics.items():

        print(

            f"{key}: {value}"

        )


    print()

    print("=" * 70)

    print(

        "NEXT DATASET METRICS"

    )

    print("=" * 70)


    print()

    for key, value in next_metrics.items():

        print(

            f"{key}: {value}"

        )


    # ========================================================
    # TRUST SCORE ANALYSIS
    # ========================================================

    trust_scores = [

        x["trust_score"]

        for x in predictions

    ]


    correct_scores = [

        x["trust_score"]

        for x in predictions

        if (

            x["human_label"]

            ==

            x["predicted_label"]

        )

    ]


    incorrect_scores = [

        x["trust_score"]

        for x in predictions

        if (

            x["human_label"]

            !=

            x["predicted_label"]

        )

    ]


    trust_score_summary = {

        "minimum":

            float(

                min(

                    trust_scores

                )

            ),


        "maximum":

            float(

                max(

                    trust_scores

                )

            ),


        "average":

            float(

                np.mean(

                    trust_scores

                )

            ),


        "correct_prediction_average":

            float(

                np.mean(

                    correct_scores

                )

            )

            if correct_scores

            else None,


        "incorrect_prediction_average":

            float(

                np.mean(

                    incorrect_scores

                )

            )

            if incorrect_scores

            else None,

    }


    # ========================================================
    # SAVE JSON
    # ========================================================

    output_data = {

        "records":

            len(predictions),


        "metrics": {

            "accuracy":

                float(

                    accuracy

                ),


            "balanced_accuracy":

                float(

                    balanced_accuracy

                ),


            "macro_f1":

                float(

                    macro_f1

                ),


            "weighted_f1":

                float(

                    weighted_f1

                )

        },


        "original_dataset_metrics":

            original_metrics,


        "next_dataset_metrics":

            next_metrics,


        "confusion_matrix":

            cm.tolist(),


        "labels":

            LABELS,


        "trust_score_summary":

            trust_score_summary,


        "classification_report":

            report,

    }


    with open(

        OUTPUT_JSON,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            output_data,

            f,

            indent=4

        )


    # ========================================================
    # SAVE CSV
    # ========================================================

    df = pd.DataFrame(

        predictions

    )


    df.to_csv(

        OUTPUT_CSV,

        index=False,

        encoding="utf-8"

    )


    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print()

    print("=" * 70)

    print(

        "OUTPUT FILES"

    )

    print("=" * 70)


    print()

    print(

        "JSON evaluation saved to:"

    )


    print(

        OUTPUT_JSON

    )


    print()

    print(

        "CSV predictions saved to:"

    )


    print(

        OUTPUT_CSV

    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("=" * 70)

    print(

        "COMBINED 200 MODEL EVALUATION COMPLETE!"

    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()