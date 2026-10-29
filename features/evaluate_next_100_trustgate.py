import json
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# SETTINGS
# ============================================================

CHECKPOINT_FILE = (
    "outputs/trustgate_fusion_model.pt"
)

FEATURE_FILE = (
    "outputs/trustgate_next_100_master_901.json"
)

OUTPUT_FILE = (
    "outputs/trustgate_next_100_predictions.json"
)

CSV_OUTPUT_FILE = (
    "outputs/trustgate_next_100_predictions.csv"
)


# ============================================================
# TRUSTGATE FUSION MODEL
# ============================================================

class TrustGateFusionNet(nn.Module):

    def __init__(
        self,
        hidden_dim=896,
        signal_dim=5,
        num_classes=3
    ):

        super().__init__()


        # ----------------------------------------------------
        # HIDDEN STATE ENCODER
        # ----------------------------------------------------

        self.hidden_encoder = nn.Sequential(

            nn.Linear(
                hidden_dim,
                128
            ),

            nn.LayerNorm(
                128
            ),

            nn.ReLU()

        )


        # ----------------------------------------------------
        # TRUST SIGNAL ENCODER
        # ----------------------------------------------------

        self.signal_encoder = nn.Sequential(

            nn.Linear(
                signal_dim,
                32
            ),

            nn.LayerNorm(
                32
            ),

            nn.ReLU()

        )


        # ----------------------------------------------------
        # FUSION NETWORK
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                128 + 32,
                128
            ),

            nn.LayerNorm(
                128
            ),

            nn.ReLU()

        )


        # ----------------------------------------------------
        # CLASSIFICATION HEAD
        # ----------------------------------------------------

        self.classifier = nn.Linear(

            128,
            num_classes

        )


        # ----------------------------------------------------
        # TRUST SCORE HEAD
        # ----------------------------------------------------

        self.trust_head = nn.Sequential(

            nn.Linear(
                128,
                1
            )

        )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(

        self,
        hidden_features,
        signals

    ):


        # Encode hidden states

        hidden_encoded = (

            self.hidden_encoder(

                hidden_features

            )

        )


        # Encode trust signals

        signal_encoded = (

            self.signal_encoder(

                signals

            )

        )


        # Combine representations

        combined = torch.cat(

            [

                hidden_encoded,

                signal_encoded

            ],

            dim=1

        )


        # Fusion

        fused = (

            self.fusion(

                combined

            )

        )


        # Classification output

        logits = (

            self.classifier(

                fused

            )

        )


        # Trust score output

        trust_score = (

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

print()

print("=" * 60)

print(
    "TRUSTGATE NEXT 100 INDEPENDENT EVALUATION"
)

print("=" * 60)


print()

print(
    "Loading TrustGate model checkpoint..."
)


# PyTorch 2.6 compatibility
# weights_only=False is appropriate here because this is
# your own locally generated checkpoint.

checkpoint = torch.load(

    CHECKPOINT_FILE,

    map_location="cpu",

    weights_only=False

)


print(
    "Checkpoint loaded successfully."
)


print()

print(
    "Model information:"
)


print(

    "Feature dimension:",

    checkpoint[
        "feature_dimension"
    ]

)


print(

    "Hidden dimension:",

    checkpoint[
        "hidden_dimension"
    ]

)


print(

    "Signal dimension:",

    checkpoint[
        "signal_dimension"
    ]

)


print(

    "Label map:",

    checkpoint[
        "label_map"
    ]

)


# ============================================================
# MODEL INFORMATION
# ============================================================

HIDDEN_DIM = (

    checkpoint[
        "hidden_dimension"
    ]

)


SIGNAL_DIM = (

    checkpoint[
        "signal_dimension"
    ]

)


LABEL_MAP = (

    checkpoint[
        "label_map"
    ]

)


NUM_CLASSES = (

    len(

        LABEL_MAP

    )

)


# ============================================================
# CREATE REVERSE LABEL MAP
# ============================================================

REVERSE_LABEL_MAP = {

    value: key

    for key, value

    in LABEL_MAP.items()

}


# ============================================================
# CREATE MODEL
# ============================================================

print()

print(
    "Creating TrustGate Fusion model..."
)


model = TrustGateFusionNet(

    hidden_dim=HIDDEN_DIM,

    signal_dim=SIGNAL_DIM,

    num_classes=NUM_CLASSES

)


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

print()

print(
    "Loading model weights..."
)


model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]

)


model.eval()


print(
    "Model loaded successfully!"
)


# ============================================================
# LOAD NORMALIZATION VALUES
# ============================================================

hidden_mean = (

    checkpoint[
        "hidden_mean"
    ]

)


hidden_std = (

    checkpoint[
        "hidden_std"
    ]

)


signal_mean = (

    checkpoint[
        "signal_mean"
    ]

)


signal_std = (

    checkpoint[
        "signal_std"
    ]

)


# Prevent division by zero

hidden_std = np.where(

    hidden_std == 0,

    1.0,

    hidden_std

)


signal_std = np.where(

    signal_std == 0,

    1.0,

    signal_std

)


# ============================================================
# LOAD NEXT 100 FEATURES
# ============================================================

print()

print("=" * 60)

print(
    "LOADING NEXT 100 TRUSTGATE FEATURES"
)

print("=" * 60)


with open(

    FEATURE_FILE,

    "r",

    encoding="utf-8"

) as f:

    data = json.load(

        f

    )


print()

print(

    "Records loaded:",

    len(data)

)


# ============================================================
# CHECK FEATURE STRUCTURE
# ============================================================

if len(data) == 0:

    raise ValueError(

        "Feature dataset is empty."

    )


first_record = data[0]


print()

print(
    "Feature keys:"
)


print(

    first_record.keys()

)


# ============================================================
# SIGNAL ORDER
# ============================================================

SIGNAL_NAMES = [

    "confidence",

    "entropy",

    "self_consistency",

    "retrieval_agreement",

    "verifier_score"

]


print()

print(
    "Signal order:"
)


for index, name in enumerate(

    SIGNAL_NAMES

):

    print(

        f"{index + 1}. {name}"

    )


# ============================================================
# EVALUATION
# ============================================================

print()

print("=" * 60)

print(
    "RUNNING TRUSTGATE PREDICTIONS"
)

print("=" * 60)


results = []


# Disable gradients during inference

with torch.no_grad():


    for position, item in enumerate(

        data

    ):


        item_id = (

            item.get(

                "id"

            )

        )


        question = (

            item.get(

                "question",

                ""

            )

        )


        answer = (

            item.get(

                "answer",

                ""

            )

        )


        reference_answer = (

            item.get(

                "reference_answer",

                ""

            )

        )


        # ====================================================
        # EXTRACT HIDDEN FEATURES
        # ====================================================

        hidden_features = (

            np.array(

                item[
                    "hidden_features"
                ],

                dtype=np.float32

            )

        )


        # ====================================================
        # EXTRACT TRUST SIGNALS
        # ====================================================

        signals_dict = (

            item[
                "signals"
            ]

        )


        signals = (

            np.array(

                [

                    signals_dict.get(

                        signal_name,

                        0.0

                    )

                    for signal_name

                    in SIGNAL_NAMES

                ],

                dtype=np.float32

            )

        )


        # ====================================================
        # DIMENSION CHECKS
        # ====================================================

        if len(hidden_features) != HIDDEN_DIM:

            raise ValueError(

                f"ID {item_id}: "

                f"Expected hidden dimension "

                f"{HIDDEN_DIM}, "

                f"but received "

                f"{len(hidden_features)}"

            )


        if len(signals) != SIGNAL_DIM:

            raise ValueError(

                f"ID {item_id}: "

                f"Expected signal dimension "

                f"{SIGNAL_DIM}, "

                f"but received "

                f"{len(signals)}"

            )


        # ====================================================
        # NORMALIZE FEATURES
        # ====================================================

        hidden_normalized = (

            hidden_features

            -

            hidden_mean

        )

        hidden_normalized = (

            hidden_normalized

            /

            hidden_std

        )


        signals_normalized = (

            signals

            -

            signal_mean

        )

        signals_normalized = (

            signals_normalized

            /

            signal_std

        )


        # ====================================================
        # CONVERT TO PYTORCH TENSORS
        # ====================================================

        hidden_tensor = torch.tensor(

            hidden_normalized,

            dtype=torch.float32

        ).unsqueeze(

            0

        )


        signals_tensor = torch.tensor(

            signals_normalized,

            dtype=torch.float32

        ).unsqueeze(

            0

        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        logits, trust_score = model(

            hidden_tensor,

            signals_tensor

        )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        probabilities = torch.softmax(

            logits,

            dim=1

        ).squeeze(

            0

        ).cpu().numpy()


        predicted_class = int(

            np.argmax(

                probabilities

            )

        )


        predicted_label = (

            REVERSE_LABEL_MAP[
                predicted_class
            ]

        )


        # ====================================================
        # TRUST SCORE
        # ====================================================

        # The original checkpoint's trust head does not have
        # a sigmoid layer, so convert its output here.

        trust_probability = float(

            torch.sigmoid(

                trust_score

            ).item()

        )


        # ====================================================
        # PREDICTION CONFIDENCE
        # ====================================================

        prediction_confidence = float(

            np.max(

                probabilities

            )

        )


        # ====================================================
        # DISPLAY PROGRESS
        # ====================================================

        print()

        print(

            f"Question "

            f"{position + 1}"

            f"/"

            f"{len(data)}"

        )


        print(

            "ID:",

            item_id

        )


        print(

            "Prediction:",

            predicted_label

        )


        print(

            "Classification confidence:",

            round(

                prediction_confidence,

                4

            )

        )


        print(

            "Trust score:",

            round(

                trust_probability,

                4

            )

        )


        # ====================================================
        # STORE RESULT
        # ====================================================

        result = {


            "id":

                item_id,


            "question":

                question,


            "answer":

                answer,


            "reference_answer":

                reference_answer,


            "predicted_label":

                predicted_label,


            "predicted_class":

                predicted_class,


            "prediction_confidence":

                prediction_confidence,


            "trust_score":

                trust_probability,


            "probability_unreliable":

                float(

                    probabilities[
                        LABEL_MAP[
                            "UNRELIABLE"
                        ]
                    ]

                ),


            "probability_uncertain":

                float(

                    probabilities[
                        LABEL_MAP[
                            "UNCERTAIN"
                        ]
                    ]

                ),


            "probability_reliable":

                float(

                    probabilities[
                        LABEL_MAP[
                            "RELIABLE"
                        ]
                    ]

                ),


            "signals":

                {

                    name:

                        float(

                            signals_dict.get(

                                name,

                                0.0

                            )

                        )

                    for name

                    in SIGNAL_NAMES

                }

        }


        results.append(

            result

        )


        # ====================================================
        # SAVE PROGRESS
        # ====================================================

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


        print(

            "Progress saved!"

        )


# ============================================================
# CREATE CSV OUTPUT
# ============================================================

print()

print(
    "Creating CSV output..."
)


csv_rows = []


for result in results:


    csv_rows.append(

        {

            "id":

                result[
                    "id"
                ],


            "question":

                result[
                    "question"
                ],


            "predicted_label":

                result[
                    "predicted_label"
                ],


            "prediction_confidence":

                result[
                    "prediction_confidence"
                ],


            "trust_score":

                result[
                    "trust_score"
                ],


            "probability_unreliable":

                result[
                    "probability_unreliable"
                ],


            "probability_uncertain":

                result[
                    "probability_uncertain"
                ],


            "probability_reliable":

                result[
                    "probability_reliable"
                ]

        }

    )


df = pd.DataFrame(

    csv_rows

)


df.to_csv(

    CSV_OUTPUT_FILE,

    index=False,

    encoding="utf-8"

)


# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 60)

print(
    "TRUSTGATE NEXT 100 EVALUATION COMPLETE!"
)

print("=" * 60)


print()

print(

    "Records evaluated:",

    len(results)

)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()

print(
    "PREDICTED LABEL DISTRIBUTION"
)

print()


label_counts = {}


for result in results:


    label = (

        result[
            "predicted_label"
        ]

    )


    label_counts[label] = (

        label_counts.get(

            label,

            0

        )

        +

        1

    )


for label in [

    "UNRELIABLE",

    "UNCERTAIN",

    "RELIABLE"

]:


    count = (

        label_counts.get(

            label,

            0

        )

    )


    percentage = (

        count

        /

        len(results)

        *

        100

    )


    print(

        f"{label}: "

        f"{count} "

        f"({percentage:.1f}%)"

    )


# ============================================================
# TRUST SCORE SUMMARY
# ============================================================

trust_scores = [

    result[
        "trust_score"
    ]

    for result

    in results

]


print()

print(
    "TRUST SCORE SUMMARY"
)

print()

print(

    "Minimum:",

    round(

        min(

            trust_scores

        ),

        4

    )

)


print(

    "Maximum:",

    round(

        max(

            trust_scores

        ),

        4

    )

)


print(

    "Average:",

    round(

        sum(

            trust_scores

        )

        /

        len(

            trust_scores

        ),

        4

    )

)


# ============================================================
# CONFIDENCE SUMMARY
# ============================================================

prediction_confidences = [

    result[
        "prediction_confidence"
    ]

    for result

    in results

]


print()

print(
    "CLASSIFICATION CONFIDENCE SUMMARY"
)

print()

print(

    "Minimum:",

    round(

        min(

            prediction_confidences

        ),

        4

    )

)


print(

    "Maximum:",

    round(

        max(

            prediction_confidences

        ),

        4

    )

)


print(

    "Average:",

    round(

        sum(

            prediction_confidences

        )

        /

        len(

            prediction_confidences

        ),

        4

    )

)


# ============================================================
# FINAL FILE LOCATIONS
# ============================================================

print()

print("=" * 60)

print(
    "OUTPUT FILES"
)

print("=" * 60)


print()

print(

    "JSON predictions saved to:"

)


print(

    OUTPUT_FILE

)


print()

print(

    "CSV predictions saved to:"

)


print(

    CSV_OUTPUT_FILE

)


print()

print(
    "Evaluation finished successfully."
)