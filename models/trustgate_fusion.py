import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "outputs/trustgate_100_labeled_v3.json"
)

MODEL_OUTPUT = (
    "outputs/trustgate_fusion_model.pt"
)

SCORES_OUTPUT = (
    "outputs/trustgate_fusion_scores.json"
)

SEED = 42

EPOCHS = 150

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

HIDDEN_DIM = 128

DROPOUT = 0.20


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING TRUSTGATE DATA")
print("=" * 60)


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


print(
    "Records:",
    len(data)
)


# ============================================================
# LABELS
# ============================================================

LABEL_MAP = {

    "UNRELIABLE": 0,

    "UNCERTAIN": 1,

    "RELIABLE": 2
}


REVERSE_LABEL = {

    0: "UNRELIABLE",

    1: "UNCERTAIN",

    2: "RELIABLE"
}


# ============================================================
# BUILD DATA MATRICES
# ============================================================

hidden_states = []

trust_signals = []

labels = []

ids = []

questions = []


for item in data:

    hidden = np.asarray(
        item["hidden_state"],
        dtype=np.float32
    )


    if hidden.shape[0] != 896:

        raise ValueError(
            f"ID {item['id']} "
            f"has hidden dimension "
            f"{hidden.shape[0]}"
        )


    # --------------------------------------------------------
    # Explicit trust signals
    # --------------------------------------------------------

    signals = np.array(
        [

            item["confidence"],

            item["entropy"],

            item["self_consistency"],

            item["retrieval_agreement"],

            item["verifier_score"]

        ],
        dtype=np.float32
    )


    hidden_states.append(
        hidden
    )

    trust_signals.append(
        signals
    )

    labels.append(
        LABEL_MAP[
            item[
                "reliability_label"
            ]
        ]
    )

    ids.append(
        item["id"]
    )

    questions.append(
        item["question"]
    )


hidden_states = np.asarray(
    hidden_states,
    dtype=np.float32
)


trust_signals = np.asarray(
    trust_signals,
    dtype=np.float32
)


labels = np.asarray(
    labels,
    dtype=np.int64
)


print(
    "Hidden states:",
    hidden_states.shape
)

print(
    "Trust signals:",
    trust_signals.shape
)


# ============================================================
# NORMALIZATION
# ============================================================

# Normalize hidden states across the dataset.

hidden_mean = (
    hidden_states.mean(
        axis=0
    )
)

hidden_std = (
    hidden_states.std(
        axis=0
    ) + 1e-6
)


hidden_states = (
    hidden_states
    - hidden_mean
) / hidden_std


# Normalize explicit signals.

signal_mean = (
    trust_signals.mean(
        axis=0
    )
)

signal_std = (
    trust_signals.std(
        axis=0
    ) + 1e-6
)


trust_signals = (
    trust_signals
    - signal_mean
) / signal_std


# ============================================================
# MODEL
# ============================================================

class TrustGateFusion(
    nn.Module
):

    def __init__(
        self,
        hidden_dim=896,
        signal_dim=5,
        fusion_dim=128,
        num_classes=3,
        dropout=0.2
    ):

        super().__init__()


        # ----------------------------------------------------
        # Internal-state encoder
        # ----------------------------------------------------

        self.hidden_encoder = nn.Sequential(

            nn.Linear(
                hidden_dim,
                fusion_dim
            ),

            nn.LayerNorm(
                fusion_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )


        # ----------------------------------------------------
        # Explicit trust-signal encoder
        # ----------------------------------------------------

        self.signal_encoder = nn.Sequential(

            nn.Linear(
                signal_dim,
                32
            ),

            nn.LayerNorm(
                32
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )


        # ----------------------------------------------------
        # Fusion layer
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                fusion_dim + 32,
                fusion_dim
            ),

            nn.LayerNorm(
                fusion_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )


        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            fusion_dim,
            num_classes
        )


        # ----------------------------------------------------
        # Trust-score head
        # ----------------------------------------------------

        self.trust_head = nn.Sequential(

            nn.Linear(
                fusion_dim,
                1
            ),

            nn.Sigmoid()
        )


    def forward(
        self,
        hidden,
        signals
    ):

        hidden_features = (
            self.hidden_encoder(
                hidden
            )
        )


        signal_features = (
            self.signal_encoder(
                signals
            )
        )


        combined = torch.cat(
            [
                hidden_features,
                signal_features
            ],
            dim=1
        )


        fused = self.fusion(
            combined
        )


        logits = self.classifier(
            fused
        )


        trust_score = self.trust_head(
            fused
        )


        return (
            logits,
            trust_score,
            fused
        )


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    labels,
    minlength=3
)


class_weights = (
    len(labels)
    /
    (
        3
        *
        np.maximum(
            class_counts,
            1
        )
    )
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)


print(
    "\nClass counts:"
)

for class_id in range(3):

    print(
        REVERSE_LABEL[
            class_id
        ],
        ":",
        int(
            class_counts[
                class_id
            ]
        )
    )


print(
    "\nClass weights:",
    class_weights.numpy()
)


# ============================================================
# CROSS-VALIDATION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TRUSTGATE FUSION CROSS-VALIDATION"
)

print(
    "=" * 60
)


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)


all_predictions = np.zeros(
    len(labels),
    dtype=np.int64
)


all_trust_scores = np.zeros(
    len(labels),
    dtype=np.float32
)


for fold, (
    train_idx,
    test_idx
) in enumerate(
    skf.split(
        hidden_states,
        labels
    )
):

    print(
        f"\nFold {fold + 1}/5"
    )


    Xh_train = torch.tensor(
        hidden_states[
            train_idx
        ],
        dtype=torch.float32
    )


    Xs_train = torch.tensor(
        trust_signals[
            train_idx
        ],
        dtype=torch.float32
    )


    y_train = torch.tensor(
        labels[
            train_idx
        ],
        dtype=torch.long
    )


    Xh_test = torch.tensor(
        hidden_states[
            test_idx
        ],
        dtype=torch.float32
    )


    Xs_test = torch.tensor(
        trust_signals[
            test_idx
        ],
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = TrustGateFusion(
        hidden_dim=896,
        signal_dim=5,
        fusion_dim=HIDDEN_DIM,
        dropout=DROPOUT
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    classification_loss = (
        nn.CrossEntropyLoss(
            weight=class_weights
        )
    )


    regression_loss = (
        nn.MSELoss()
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()


    for epoch in range(
        EPOCHS
    ):

        optimizer.zero_grad()


        logits, trust_score, _ = model(
            Xh_train,
            Xs_train
        )


        # Main classification objective.

        loss_class = (
            classification_loss(
                logits,
                y_train
            )
        )


        # ----------------------------------------------------
        # Auxiliary trust target
        #
        # We use the silver trust score already generated by
        # the V3 labeling process.
        # ----------------------------------------------------

        target_scores = torch.tensor(
            [
                data[i][
                    "silver_trust_score"
                ]
                for i in train_idx
            ],
            dtype=torch.float32
        ).reshape(
            -1,
            1
        )


        loss_trust = (
            regression_loss(
                trust_score,
                target_scores
            )
        )


        # Combined objective.

        loss = (
            loss_class
            +
            0.30 * loss_trust
        )


        loss.backward()


        optimizer.step()


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    model.eval()


    with torch.no_grad():

        logits, trust_score, _ = model(
            Xh_test,
            Xs_test
        )


        predictions = (
            torch.argmax(
                logits,
                dim=1
            )
            .cpu()
            .numpy()
        )


        scores = (
            trust_score
            .reshape(-1)
            .cpu()
            .numpy()
        )


    all_predictions[
        test_idx
    ] = predictions


    all_trust_scores[
        test_idx
    ] = scores


# ============================================================
# CROSS-VALIDATION RESULTS
# ============================================================

accuracy = accuracy_score(
    labels,
    all_predictions
)


balanced_accuracy = (
    balanced_accuracy_score(
        labels,
        all_predictions
    )
)


print(
    "\n" + "=" * 60
)

print(
    "TRUSTGATE FUSION RESULTS"
)

print(
    "=" * 60
)


print(
    "Accuracy:",
    round(
        accuracy,
        4
    )
)


print(
    "Balanced accuracy:",
    round(
        balanced_accuracy,
        4
    )
)


print(
    "\nClassification report:"
)


print(
    classification_report(
        labels,
        all_predictions,
        labels=[
            0,
            1,
            2
        ],
        target_names=[
            "UNRELIABLE",
            "UNCERTAIN",
            "RELIABLE"
        ],
        zero_division=0
    )
)


print(
    "\nConfusion matrix:"
)

print(
    confusion_matrix(
        labels,
        all_predictions,
        labels=[
            0,
            1,
            2
        ]
    )
)


# ============================================================
# TRUST SCORE DISTRIBUTION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "CROSS-VALIDATED TRUST SCORE"
)

print(
    "=" * 60
)


print(
    "Minimum:",
    round(
        float(
            all_trust_scores.min()
        ),
        4
    )
)


print(
    "25%:",
    round(
        float(
            np.quantile(
                all_trust_scores,
                0.25
            )
        ),
        4
    )
)


print(
    "Median:",
    round(
        float(
            np.median(
                all_trust_scores
            )
        ),
        4
    )
)


print(
    "75%:",
    round(
        float(
            np.quantile(
                all_trust_scores,
                0.75
            )
        ),
        4
    )
)


print(
    "Maximum:",
    round(
        float(
            all_trust_scores.max()
        ),
        4
    )
)


# ============================================================
# SAVE CROSS-VALIDATED SCORES
# ============================================================

score_records = []


for i in range(
    len(data)
):

    score_records.append(
        {

            "id":
                ids[i],

            "question":
                questions[i],

            "silver_label":
                REVERSE_LABEL[
                    int(
                        labels[i]
                    )
                ],

            "predicted_label":
                REVERSE_LABEL[
                    int(
                        all_predictions[i]
                    )
                ],

            "trust_score":
                float(
                    all_trust_scores[i]
                )
        }
    )


with open(
    SCORES_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        score_records,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TRAINING FINAL TRUSTGATE FUSION MODEL"
)

print(
    "=" * 60
)


Xh = torch.tensor(
    hidden_states,
    dtype=torch.float32
)


Xs = torch.tensor(
    trust_signals,
    dtype=torch.float32
)


y = torch.tensor(
    labels,
    dtype=torch.long
)


final_model = TrustGateFusion(
    hidden_dim=896,
    signal_dim=5,
    fusion_dim=HIDDEN_DIM,
    dropout=DROPOUT
)


optimizer = torch.optim.AdamW(
    final_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


classification_loss = (
    nn.CrossEntropyLoss(
        weight=class_weights
    )
)


regression_loss = (
    nn.MSELoss()
)


target_scores = torch.tensor(
    [
        item[
            "silver_trust_score"
        ]
        for item in data
    ],
    dtype=torch.float32
).reshape(
    -1,
    1
)


final_model.train()


for epoch in range(
    EPOCHS
):

    optimizer.zero_grad()


    logits, trust_score, _ = final_model(
        Xh,
        Xs
    )


    loss_class = (
        classification_loss(
            logits,
            y
        )
    )


    loss_trust = (
        regression_loss(
            trust_score,
            target_scores
        )
    )


    loss = (
        loss_class
        +
        0.30 * loss_trust
    )


    loss.backward()


    optimizer.step()


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            final_model.state_dict(),

        "hidden_mean":
            hidden_mean,

        "hidden_std":
            hidden_std,

        "signal_mean":
            signal_mean,

        "signal_std":
            signal_std,

        "feature_dimension":
            901,

        "hidden_dimension":
            896,

        "signal_dimension":
            5,

        "label_map":
            LABEL_MAP
    },
    MODEL_OUTPUT
)


print(
    "\n" + "=" * 60
)

print(
    "TRUSTGATE FUSION MODEL COMPLETE!"
)

print(
    "=" * 60
)

print(
    "Model saved to:"
)

print(
    MODEL_OUTPUT
)

print(
    "\nCross-validated scores saved to:"
)

print(
    SCORES_OUTPUT
)