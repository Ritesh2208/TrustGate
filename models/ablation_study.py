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
    f1_score,
)


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "outputs/trustgate_100_labeled_v3.json"
)

OUTPUT_FILE = (
    "outputs/trustgate_ablation_results.json"
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
print("LOADING DATA FOR ABLATION STUDY")
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


# ============================================================
# BUILD MATRICES
# ============================================================

hidden_states = []

signals = []

labels = []


for item in data:

    hidden = np.asarray(
        item["hidden_state"],
        dtype=np.float32
    )

    trust = np.asarray(
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

    signals.append(
        trust
    )

    labels.append(
        LABEL_MAP[
            item["reliability_label"]
        ]
    )


hidden_states = np.asarray(
    hidden_states,
    dtype=np.float32
)

signals = np.asarray(
    signals,
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
    "Signals:",
    signals.shape
)


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class SignalOnlyModel(
    nn.Module
):

    def __init__(
        self,
        input_dim=5,
        hidden_dim=64,
        num_classes=3
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                hidden_dim,
                32
            ),

            nn.GELU(),

            nn.Linear(
                32,
                num_classes
            )
        )


    def forward(
        self,
        x
    ):

        return self.network(
            x
        )


class HiddenOnlyModel(
    nn.Module
):

    def __init__(
        self,
        input_dim=896,
        hidden_dim=128,
        num_classes=3
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                hidden_dim,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(
        self,
        x
    ):

        return self.network(
            x
        )


class TrustGateFusion(
    nn.Module
):

    def __init__(
        self,
        hidden_dim=896,
        signal_dim=5,
        fusion_dim=128,
        num_classes=3
    ):

        super().__init__()


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
                DROPOUT
            )
        )


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
                DROPOUT
            )
        )


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
                DROPOUT
            )
        )


        self.classifier = nn.Linear(
            fusion_dim,
            num_classes
        )


    def forward(
        self,
        hidden,
        trust
    ):

        hidden_features = (
            self.hidden_encoder(
                hidden
            )
        )


        signal_features = (
            self.signal_encoder(
                trust
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


        return self.classifier(
            fused
        )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_and_evaluate(
    model,
    train_inputs,
    train_labels,
    test_inputs,
    test_labels,
    class_weights
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    model.train()


    for epoch in range(
        EPOCHS
    ):

        optimizer.zero_grad()


        logits = model(
            *train_inputs
        ) if isinstance(
            train_inputs,
            tuple
        ) else model(
            train_inputs
        )


        loss = criterion(
            logits,
            train_labels
        )


        loss.backward()

        optimizer.step()


    model.eval()


    with torch.no_grad():

        logits = model(
            *test_inputs
        ) if isinstance(
            test_inputs,
            tuple
        ) else model(
            test_inputs
        )


        predictions = (
            torch.argmax(
                logits,
                dim=1
            )
            .cpu()
            .numpy()
        )


    return predictions


# ============================================================
# RUN ABLATION
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)


results = {}


# ============================================================
# 1. SIGNAL ONLY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "ABLATION 1: SIGNALS ONLY"
)

print(
    "=" * 60
)


signal_predictions = np.zeros(
    len(labels),
    dtype=np.int64
)


for fold, (
    train_idx,
    test_idx
) in enumerate(
    skf.split(
        signals,
        labels
    )
):

    X_train = torch.tensor(
        signals[
            train_idx
        ],
        dtype=torch.float32
    )

    X_test = torch.tensor(
        signals[
            test_idx
        ],
        dtype=torch.float32
    )

    y_train = torch.tensor(
        labels[
            train_idx
        ],
        dtype=torch.long
    )

    y_test = torch.tensor(
        labels[
            test_idx
        ],
        dtype=torch.long
    )


    counts = np.bincount(
        labels[
            train_idx
        ],
        minlength=3
    )


    weights = torch.tensor(
        len(train_idx)
        /
        (
            3
            *
            np.maximum(
                counts,
                1
            )
        ),
        dtype=torch.float32
    )


    model = SignalOnlyModel()


    predictions = train_and_evaluate(

        model,

        X_train,

        y_train,

        X_test,

        y_test,

        weights
    )


    signal_predictions[
        test_idx
    ] = predictions


signal_accuracy = accuracy_score(
    labels,
    signal_predictions
)

signal_balanced = (
    balanced_accuracy_score(
        labels,
        signal_predictions
    )
)

signal_macro_f1 = (
    f1_score(
        labels,
        signal_predictions,
        average="macro"
    )
)


results["signal_only"] = {

    "accuracy":
        float(signal_accuracy),

    "balanced_accuracy":
        float(signal_balanced),

    "macro_f1":
        float(signal_macro_f1)
}


print(
    "Accuracy:",
    round(
        signal_accuracy,
        4
    )
)

print(
    "Balanced accuracy:",
    round(
        signal_balanced,
        4
    )
)

print(
    "Macro F1:",
    round(
        signal_macro_f1,
        4
    )
)


# ============================================================
# 2. HIDDEN ONLY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "ABLATION 2: HIDDEN STATE ONLY"
)

print(
    "=" * 60
)


hidden_predictions = np.zeros(
    len(labels),
    dtype=np.int64
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

    X_train = torch.tensor(
        hidden_states[
            train_idx
        ],
        dtype=torch.float32
    )

    X_test = torch.tensor(
        hidden_states[
            test_idx
        ],
        dtype=torch.float32
    )

    y_train = torch.tensor(
        labels[
            train_idx
        ],
        dtype=torch.long
    )

    y_test = torch.tensor(
        labels[
            test_idx
        ],
        dtype=torch.long
    )


    counts = np.bincount(
        labels[
            train_idx
        ],
        minlength=3
    )


    weights = torch.tensor(
        len(train_idx)
        /
        (
            3
            *
            np.maximum(
                counts,
                1
            )
        ),
        dtype=torch.float32
    )


    model = HiddenOnlyModel()


    predictions = train_and_evaluate(

        model,

        X_train,

        y_train,

        X_test,

        y_test,

        weights
    )


    hidden_predictions[
        test_idx
    ] = predictions


hidden_accuracy = accuracy_score(
    labels,
    hidden_predictions
)

hidden_balanced = (
    balanced_accuracy_score(
        labels,
        hidden_predictions
    )
)

hidden_macro_f1 = (
    f1_score(
        labels,
        hidden_predictions,
        average="macro"
    )
)


results["hidden_only"] = {

    "accuracy":
        float(hidden_accuracy),

    "balanced_accuracy":
        float(hidden_balanced),

    "macro_f1":
        float(hidden_macro_f1)
}


print(
    "Accuracy:",
    round(
        hidden_accuracy,
        4
    )
)

print(
    "Balanced accuracy:",
    round(
        hidden_balanced,
        4
    )
)

print(
    "Macro F1:",
    round(
        hidden_macro_f1,
        4
    )
)


# ============================================================
# 3. TRUSTGATE FUSION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "ABLATION 3: TRUSTGATE FUSION"
)

print(
    "=" * 60
)


fusion_predictions = np.zeros(
    len(labels),
    dtype=np.int64
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

    H_train = torch.tensor(
        hidden_states[
            train_idx
        ],
        dtype=torch.float32
    )

    H_test = torch.tensor(
        hidden_states[
            test_idx
        ],
        dtype=torch.float32
    )

    S_train = torch.tensor(
        signals[
            train_idx
        ],
        dtype=torch.float32
    )

    S_test = torch.tensor(
        signals[
            test_idx
        ],
        dtype=torch.float32
    )

    y_train = torch.tensor(
        labels[
            train_idx
        ],
        dtype=torch.long
    )

    y_test = torch.tensor(
        labels[
            test_idx
        ],
        dtype=torch.long
    )


    counts = np.bincount(
        labels[
            train_idx
        ],
        minlength=3
    )


    weights = torch.tensor(
        len(train_idx)
        /
        (
            3
            *
            np.maximum(
                counts,
                1
            )
        ),
        dtype=torch.float32
    )


    model = TrustGateFusion()


    predictions = train_and_evaluate(

        model,

        (
            H_train,
            S_train
        ),

        y_train,

        (
            H_test,
            S_test
        ),

        y_test,

        weights
    )


    fusion_predictions[
        test_idx
    ] = predictions


fusion_accuracy = accuracy_score(
    labels,
    fusion_predictions
)

fusion_balanced = (
    balanced_accuracy_score(
        labels,
        fusion_predictions
    )
)

fusion_macro_f1 = (
    f1_score(
        labels,
        fusion_predictions,
        average="macro"
    )
)


results["trustgate_fusion"] = {

    "accuracy":
        float(fusion_accuracy),

    "balanced_accuracy":
        float(fusion_balanced),

    "macro_f1":
        float(fusion_macro_f1)
}


print(
    "Accuracy:",
    round(
        fusion_accuracy,
        4
    )
)

print(
    "Balanced accuracy:",
    round(
        fusion_balanced,
        4
    )
)

print(
    "Macro F1:",
    round(
        fusion_macro_f1,
        4
    )
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TRUSTGATE ABLATION SUMMARY"
)

print(
    "=" * 60
)


print(
    "\nModel                         "
    "Accuracy   Balanced Acc.   Macro F1"
)

print(
    "-" * 60
)


for name, result in results.items():

    display_name = {

        "signal_only":
            "Signal-only",

        "hidden_only":
            "Hidden-only",

        "trustgate_fusion":
            "TrustGate Fusion"

    }[
        name
    ]


    print(
        f"{display_name:<28}"
        f"{result['accuracy']:.4f}      "
        f"{result['balanced_accuracy']:.4f}        "
        f"{result['macro_f1']:.4f}"
    )


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(
    "\n" + "=" * 60
)

print(
    "ABLATION STUDY COMPLETE!"
)

print(
    "=" * 60
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)