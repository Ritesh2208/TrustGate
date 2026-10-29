import json
import os
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

FEATURE_FILE = (
    "outputs/truthfulqa_next_100_features.json"
)

CONSISTENCY_FILE = (
    "outputs/truthfulqa_next_100_consistency.json"
)

RETRIEVAL_FILE = (
    "outputs/truthfulqa_next_100_retrieval.json"
)

VERIFIER_FILE = (
    "outputs/truthfulqa_next_100_verified.json"
)

OUTPUT_FILE = (
    "outputs/trustgate_next_100_master_901.json"
)


# ============================================================
# LOAD JSON FILE
# ============================================================

def load_json(path):

    print(f"\nLoading: {path}")

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
# START
# ============================================================

print("=" * 60)
print("TRUSTGATE NEXT 100 FEATURE COMBINATION")
print("=" * 60)


features_data = load_json(
    FEATURE_FILE
)

consistency_data = load_json(
    CONSISTENCY_FILE
)

retrieval_data = load_json(
    RETRIEVAL_FILE
)

verifier_data = load_json(
    VERIFIER_FILE
)


# ============================================================
# CREATE ID LOOKUPS
# ============================================================

print("\nCreating ID mappings...")


consistency_map = {

    item["id"]: item

    for item in consistency_data

}


retrieval_map = {

    item["id"]: item

    for item in retrieval_data

}


verifier_map = {

    item["id"]: item

    for item in verifier_data

}


# ============================================================
# DETECT FEATURE FIELD
# ============================================================

print("\nChecking feature structure...")


print(
    "Feature record keys:"
)

print(
    features_data[0].keys()
)


# ============================================================
# EXTRACT VALUE SAFELY
# ============================================================

def get_value(item, possible_keys, default=0.0):

    for key in possible_keys:

        if key in item:

            value = item.get(key)

            if value is not None:

                return float(value)

    return float(default)


# ============================================================
# FIND HIDDEN STATE
# ============================================================

def get_hidden_state(item):

    possible_keys = [

        "hidden_state",
        "hidden_states",
        "hidden_features",
        "features",
        "representation"

    ]


    for key in possible_keys:

        if key in item:

            value = item[key]

            if isinstance(value, list):

                return value


    raise ValueError(

        f"Could not find hidden state for ID {item.get('id')}.\n"
        f"Available keys: {item.keys()}"

    )


# ============================================================
# COMBINE FEATURES
# ============================================================

results = []


print("\n" + "=" * 60)
print("BUILDING 901-D TRUSTGATE REPRESENTATIONS")
print("=" * 60)


for position, item in enumerate(

    features_data,

    start=1

):

    item_id = item.get(
        "id"
    )


    print(

        f"\nProcessing {position}/{len(features_data)} "
        f"(ID: {item_id})"

    )


    # --------------------------------------------------------
    # CHECK ALIGNMENT
    # --------------------------------------------------------

    if item_id not in consistency_map:

        raise ValueError(

            f"Missing consistency record for ID {item_id}"

        )


    if item_id not in retrieval_map:

        raise ValueError(

            f"Missing retrieval record for ID {item_id}"

        )


    if item_id not in verifier_map:

        raise ValueError(

            f"Missing verifier record for ID {item_id}"

        )


    consistency_item = consistency_map[
        item_id
    ]

    retrieval_item = retrieval_map[
        item_id
    ]

    verifier_item = verifier_map[
        item_id
    ]


    # --------------------------------------------------------
    # HIDDEN FEATURES
    # --------------------------------------------------------

    hidden_state = get_hidden_state(
        item
    )


    hidden_state = [

        float(x)

        for x in hidden_state

    ]


    if len(hidden_state) != 896:

        raise ValueError(

            f"ID {item_id}: Expected 896 hidden dimensions, "
            f"got {len(hidden_state)}"

        )


    # --------------------------------------------------------
    # TRUST SIGNAL 1: CONFIDENCE
    # --------------------------------------------------------

    confidence = get_value(

        item,

        [

            "confidence",
            "answer_confidence",
            "mean_confidence"

        ],

        default=0.0

    )


    # --------------------------------------------------------
    # TRUST SIGNAL 2: ENTROPY
    # --------------------------------------------------------

    entropy = get_value(

        item,

        [

            "entropy",
            "answer_entropy",
            "mean_entropy"

        ],

        default=0.0

    )


    # --------------------------------------------------------
    # TRUST SIGNAL 3: SELF-CONSISTENCY
    # --------------------------------------------------------

    self_consistency = get_value(

        consistency_item,

        [

            "self_consistency",
            "consistency_score",
            "consistency"

        ],

        default=0.0

    )


    # --------------------------------------------------------
    # TRUST SIGNAL 4: RETRIEVAL AGREEMENT
    # --------------------------------------------------------

    retrieval_agreement = get_value(

        retrieval_item,

        [

            "retrieval_agreement"

        ],

        default=0.0

    )


    # --------------------------------------------------------
    # TRUST SIGNAL 5: VERIFIER SCORE
    # --------------------------------------------------------

    verifier_score = get_value(

        verifier_item,

        [

            "verifier_score",
            "verification_score"

        ],

        default=0.0

    )


    # --------------------------------------------------------
    # CREATE SIGNAL VECTOR
    # --------------------------------------------------------

    signals = [

        confidence,

        entropy,

        self_consistency,

        retrieval_agreement,

        verifier_score

    ]


    # --------------------------------------------------------
    # CREATE 901-D FUSION VECTOR
    # --------------------------------------------------------

    fusion_features = (

        hidden_state

        +

        signals

    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if len(signals) != 5:

        raise ValueError(

            f"ID {item_id}: Signal dimension is not 5"

        )


    if len(fusion_features) != 901:

        raise ValueError(

            f"ID {item_id}: Expected 901 dimensions, "
            f"got {len(fusion_features)}"

        )


    # --------------------------------------------------------
    # SAVE RECORD
    # --------------------------------------------------------

    results.append(

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
                    ""
                ),

            "reference_answer":
                item.get(
                    "reference_answer",
                    ""
                ),

            "hidden_features":
                hidden_state,

            "signals":

                {

                    "confidence":
                        confidence,

                    "entropy":
                        entropy,

                    "self_consistency":
                        self_consistency,

                    "retrieval_agreement":
                        retrieval_agreement,

                    "verifier_score":
                        verifier_score

                },

            "fusion_features":
                fusion_features,

            "hidden_dimension":
                len(hidden_state),

            "signal_dimension":
                len(signals),

            "fusion_dimension":
                len(fusion_features)

        }

    )


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(

    "outputs",

    exist_ok=True

)


with open(

    OUTPUT_FILE,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        results,

        f,

        indent=2,

        ensure_ascii=False

    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NEXT 100 FEATURE COMBINATION COMPLETE!")
print("=" * 60)


print(
    "Records processed:",
    len(results)
)


print(
    "\nRepresentation dimensions:"
)


print(
    "Hidden features:",
    results[0]["hidden_dimension"]
)


print(
    "Trust signals:",
    results[0]["signal_dimension"]
)


print(
    "TrustGate fusion:",
    results[0]["fusion_dimension"]
)


print("\nCONFIRMED:")

print(
    "896 hidden dimensions + 5 signals = 901 dimensions"
)


print("\nSaved to:")

print(
    OUTPUT_FILE
)


print("\nFirst ID:")

print(
    results[0]["id"]
)


print("Last ID:")

print(
    results[-1]["id"]
)