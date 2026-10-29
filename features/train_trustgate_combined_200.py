import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)


# ============================================================
# SETTINGS
# ============================================================

ORIGINAL_LABEL_FILE = (
    "outputs/trustgate_human_evaluation_completed.csv"
)

NEXT_LABEL_FILE = (
    "outputs/trustgate_ai_assisted_evaluation_next_100.csv"
)

ORIGINAL_FEATURE_FILE = (
    "outputs/truthfulqa_100_combined_features.json"
)

NEXT_FEATURE_FILE = (
    "outputs/trustgate_next_100_master_901.json"
)

OUTPUT_MODEL = (
    "outputs/trustgate_fusion_model_combined_200.pt"
)

OUTPUT_RESULTS = (
    "outputs/trustgate_combined_200_training_results.json"
)


RANDOM_SEED = 42

NUM_FOLDS = 5

EPOCHS = 100

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001


LABEL_MAP = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


set_seed(
    RANDOM_SEED
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    print(
        f"\nLoading: {path}"
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
# LOAD LABEL DATA
# ============================================================

print("=" * 70)
print("TRUSTGATE COMBINED 200 TRAINING")
print("=" * 70)


print(
    "\nLoading human labels..."
)


original_labels = pd.read_csv(
    ORIGINAL_LABEL_FILE
)

next_labels = pd.read_csv(
    NEXT_LABEL_FILE
)


print(
    "Original labels:",
    len(original_labels)
)

print(
    "Next labels:",
    len(next_labels)
)


# ============================================================
# VALIDATE LABELS
# ============================================================

valid_labels = set(
    LABEL_MAP.keys()
)


def validate_labels(df, name):

    df["human_label"] = (
        df["human_label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid = df[
        ~df["human_label"].isin(
            valid_labels
        )
    ]

    if len(invalid) > 0:

        print(
            f"\nInvalid labels in {name}:"
        )

        print(
            invalid[
                ["id", "human_label"]
            ]
        )

        raise ValueError(
            "Invalid labels found."
        )

    print(
        f"{name} labels validated."
    )


validate_labels(
    original_labels,
    "Original"
)

validate_labels(
    next_labels,
    "Next 100"
)


# ============================================================
# LOAD FEATURES
# ============================================================

original_features = load_json(
    ORIGINAL_FEATURE_FILE
)

next_features = load_json(
    NEXT_FEATURE_FILE
)


# ============================================================
# CREATE FEATURE MAPPINGS
# ============================================================

print(
    "\nCreating feature mappings..."
)


original_feature_map = {

    item["id"]: item

    for item in original_features

}


next_feature_map = {

    item["id"]: item

    for item in next_features

}


# ============================================================
# EXTRACT FEATURE VECTOR
# ============================================================

def extract_features(item):

    # --------------------------------------------------------
    # CASE 1:
    # Already contains fusion_features
    # --------------------------------------------------------

    if "fusion_features" in item:

        features = item[
            "fusion_features"
        ]

        return np.array(
            features,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # CASE 2:
    # hidden_features + signals
    # --------------------------------------------------------

    if (
        "hidden_features" in item
        and
        "signals" in item
    ):

        hidden = np.array(
            item["hidden_features"],
            dtype=np.float32
        )


        signals_dict = item[
            "signals"
        ]


        signal_order = [

            "confidence",

            "entropy",

            "self_consistency",

            "retrieval_agreement",

            "verifier_score"

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


        return np.concatenate(

            [
                hidden,
                signals
            ]

        )


    # --------------------------------------------------------
    # CASE 3:
    # hidden_state + individual signals
    # --------------------------------------------------------

    hidden = item.get(
        "hidden_state",
        item.get(
            "hidden_features",
            None
        )
    )


    if hidden is None:

        raise ValueError(
            "Could not find hidden features."
        )


    hidden = np.array(
        hidden,
        dtype=np.float32
    )


    signals = np.array(

        [

            item.get(
                "confidence",
                0.0
            ),

            item.get(
                "entropy",
                0.0
            ),

            item.get(
                "self_consistency",
                0.0
            ),

            item.get(
                "retrieval_agreement",
                0.0
            ),

            item.get(
                "verifier_score",
                0.0
            )

        ],

        dtype=np.float32

    )


    return np.concatenate(

        [
            hidden,
            signals
        ]

    )


# ============================================================
# ALIGN ORIGINAL DATASET
# ============================================================

print(
    "\nAligning original 100 samples..."
)


X_list = []

y_list = []

dataset_ids = []


missing_original = []


for _, row in original_labels.iterrows():

    item_id = row["id"]


    if item_id not in original_feature_map:

        missing_original.append(
            item_id
        )

        continue


    feature_item = original_feature_map[
        item_id
    ]


    vector = extract_features(
        feature_item
    )


    X_list.append(
        vector
    )


    y_list.append(

        LABEL_MAP[
            row["human_label"]
        ]

    )


    dataset_ids.append(
        item_id
    )


print(
    "Original aligned:",
    len(X_list)
)


if missing_original:

    print(
        "Missing original IDs:",
        missing_original
    )


# ============================================================
# ALIGN NEXT 100 DATASET
# ============================================================

print(
    "\nAligning next 100 samples..."
)


next_aligned = 0

missing_next = []


for _, row in next_labels.iterrows():

    item_id = row["id"]


    if item_id not in next_feature_map:

        missing_next.append(
            item_id
        )

        continue


    feature_item = next_feature_map[
        item_id
    ]


    vector = extract_features(
        feature_item
    )


    X_list.append(
        vector
    )


    y_list.append(

        LABEL_MAP[
            row["human_label"]
        ]

    )


    dataset_ids.append(
        item_id
    )


    next_aligned += 1


print(
    "Next aligned:",
    next_aligned
)


if missing_next:

    print(
        "Missing next IDs:",
        missing_next
    )


# ============================================================
# CREATE NUMPY ARRAYS
# ============================================================

X = np.array(
    X_list,
    dtype=np.float32
)

y = np.array(
    y_list,
    dtype=np.int64
)


print(
    "\n" + "=" * 70
)

print(
    "COMBINED DATASET"
)

print(
    "=" * 70
)


print(
    "Total samples:",
    len(X)
)


print(
    "Feature dimension:",
    X.shape[1]
)


print(
    "\nLabel distribution:"
)


for label_name, label_value in LABEL_MAP.items():

    count = int(
        np.sum(
            y == label_value
        )
    )

    print(
        f"{label_name}: {count}"
    )


# ============================================================
# FEATURE DIMENSION CHECK
# ============================================================

if X.shape[1] != 901:

    raise ValueError(

        f"Expected 901 features, got {X.shape[1]}"

    )


# ============================================================
# NORMALIZATION
# ============================================================

hidden_dimension = 896

signal_dimension = 5


hidden_X = X[
    :,
    :hidden_dimension
]


signal_X = X[
    :,
    hidden_dimension:
]


hidden_mean = np.mean(

    hidden_X,
    axis=0

)


hidden_std = np.std(

    hidden_X,
    axis=0

)


signal_mean = np.mean(

    signal_X,
    axis=0

)


signal_std = np.std(

    signal_X,
    axis=0

)


# Avoid division by zero

hidden_std[
    hidden_std == 0
] = 1.0


signal_std[
    signal_std == 0
] = 1.0


X_hidden_normalized = (

    hidden_X
    -
    hidden_mean

) / hidden_std


X_signal_normalized = (

    signal_X
    -
    signal_mean

) / signal_std


X_normalized = np.concatenate(

    [

        X_hidden_normalized,

        X_signal_normalized

    ],

    axis=1

).astype(
    np.float32
)


# ============================================================
# MODEL
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


        self.classifier = nn.Linear(

            128,
            3

        )


        self.trust_head = nn.Sequential(

            nn.Linear(
                128,
                1
            )

        )


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
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y,
    minlength=3
)


class_weights = (

    len(y)
    /
    (
        3
        *
        class_counts
    )

)


print(
    "\nClass weights:"
)


for index, weight in enumerate(
    class_weights
):

    print(
        index,
        round(
            float(weight),
            4
        )
    )


# ============================================================
# CROSS VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "5-FOLD STRATIFIED CROSS VALIDATION"
)

print(
    "=" * 70
)


skf = StratifiedKFold(

    n_splits=NUM_FOLDS,

    shuffle=True,

    random_state=RANDOM_SEED

)


fold_results = []


all_true = []

all_pred = []


for fold_number, (

    train_index,

    test_index

) in enumerate(

    skf.split(
        X_normalized,
        y
    ),

    start=1

):


    print(
        f"\nFold {fold_number}/{NUM_FOLDS}"
    )


    X_train = torch.tensor(

        X_normalized[
            train_index
        ],

        dtype=torch.float32

    )


    y_train = torch.tensor(

        y[
            train_index
        ],

        dtype=torch.long

    )


    X_test = torch.tensor(

        X_normalized[
            test_index
        ],

        dtype=torch.float32

    )


    y_test = y[
        test_index
    ]


    model = TrustGateFusionNet()


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    criterion = nn.CrossEntropyLoss(

        weight=torch.tensor(

            class_weights,

            dtype=torch.float32

        )

    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()


    for epoch in range(

        EPOCHS

    ):


        optimizer.zero_grad()


        logits, trust_score = model(

            X_train

        )


        loss = criterion(

            logits,

            y_train

        )


        loss.backward()


        optimizer.step()


    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    model.eval()


    with torch.no_grad():


        logits, trust_score = model(

            X_test

        )


        predictions = torch.argmax(

            logits,

            dim=1

        ).cpu().numpy()


    accuracy = accuracy_score(

        y_test,

        predictions

    )


    balanced_accuracy = (

        balanced_accuracy_score(

            y_test,

            predictions

        )

    )


    macro_f1 = f1_score(

        y_test,

        predictions,

        average="macro",

        zero_division=0

    )


    weighted_f1 = f1_score(

        y_test,

        predictions,

        average="weighted",

        zero_division=0

    )


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


    fold_results.append(

        {

            "fold":
                fold_number,

            "accuracy":
                float(accuracy),

            "balanced_accuracy":
                float(balanced_accuracy),

            "macro_f1":
                float(macro_f1),

            "weighted_f1":
                float(weighted_f1)

        }

    )


    all_true.extend(

        y_test.tolist()

    )


    all_pred.extend(

        predictions.tolist()

    )


# ============================================================
# CROSS VALIDATION RESULTS
# ============================================================

cv_accuracy = accuracy_score(

    all_true,

    all_pred

)


cv_balanced_accuracy = (

    balanced_accuracy_score(

        all_true,

        all_pred

    )

)


cv_macro_f1 = f1_score(

    all_true,

    all_pred,

    average="macro",

    zero_division=0

)


cv_weighted_f1 = f1_score(

    all_true,

    all_pred,

    average="weighted",

    zero_division=0

)


print(
    "\n" + "=" * 70
)

print(
    "COMBINED CROSS VALIDATION RESULTS"
)

print(
    "=" * 70
)


print(
    "Accuracy:",
    round(
        cv_accuracy,
        4
    )
)


print(
    "Balanced Accuracy:",
    round(
        cv_balanced_accuracy,
        4
    )
)


print(
    "Macro F1:",
    round(
        cv_macro_f1,
        4
    )
)


print(
    "Weighted F1:",
    round(
        cv_weighted_f1,
        4
    )
)


# ============================================================
# TRAIN FINAL MODEL ON ALL 200 SAMPLES
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TRAINING FINAL TRUSTGATE MODEL ON ALL 200 SAMPLES"
)

print(
    "=" * 70
)


X_tensor = torch.tensor(

    X_normalized,

    dtype=torch.float32

)


y_tensor = torch.tensor(

    y,

    dtype=torch.long

)


final_model = TrustGateFusionNet()


optimizer = torch.optim.AdamW(

    final_model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


criterion = nn.CrossEntropyLoss(

    weight=torch.tensor(

        class_weights,

        dtype=torch.float32

    )

)


final_model.train()


for epoch in range(

    EPOCHS

):


    optimizer.zero_grad()


    logits, trust_score = final_model(

        X_tensor

    )


    loss = criterion(

        logits,

        y_tensor

    )


    loss.backward()


    optimizer.step()


    if (

        (epoch + 1)
        %
        10
        ==
        0

    ):


        print(

            f"Epoch {epoch + 1}/{EPOCHS}",

            "| Loss:",

            round(
                float(
                    loss.item()
                ),
                4
            )

        )


# ============================================================
# SAVE MODEL
# ============================================================

print(
    "\nSaving final model..."
)


checkpoint = {

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

}


torch.save(

    checkpoint,

    OUTPUT_MODEL

)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "experiment":
        "TrustGate Combined 200 Training",

    "records":
        int(
            len(X)
        ),

    "label_distribution":

        {

            label:

                int(
                    np.sum(
                        y == value
                    )
                )

            for label, value in LABEL_MAP.items()

        },


    "cross_validation_folds":
        NUM_FOLDS,


    "cross_validation_results":

        {

            "accuracy":
                float(
                    cv_accuracy
                ),

            "balanced_accuracy":
                float(
                    cv_balanced_accuracy
                ),

            "macro_f1":
                float(
                    cv_macro_f1
                ),

            "weighted_f1":
                float(
                    cv_weighted_f1
                )

        },


    "fold_results":
        fold_results,


    "model_architecture":

        {

            "hidden_dimension":
                896,

            "signal_dimension":
                5,

            "fusion_dimension":
                901

        }

}


with open(

    OUTPUT_RESULTS,

    "w",

    encoding="utf-8"

) as f:


    json.dump(

        results,

        f,

        indent=4

    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TRUSTGATE COMBINED 200 TRAINING COMPLETE!"
)

print(
    "=" * 70
)


print(
    "\nRecords:",
    len(X)
)


print(
    "\nCross-validation results:"
)


print(
    "Accuracy:",
    round(
        cv_accuracy,
        4
    )
)


print(
    "Balanced Accuracy:",
    round(
        cv_balanced_accuracy,
        4
    )
)


print(
    "Macro F1:",
    round(
        cv_macro_f1,
        4
    )
)


print(
    "\nModel saved to:"
)

print(
    OUTPUT_MODEL
)


print(
    "\nResults saved to:"
)

print(
    OUTPUT_RESULTS
)