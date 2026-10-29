import json
import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "outputs"


# ------------------------------------------------------------
# Previous experiment results
# ------------------------------------------------------------

INDEPENDENT_RESULTS_FILE = (
    "outputs/trustgate_independent_annotation_results.json"
)

ABLATION_FILE = (
    "outputs/trustgate_independent_ablation.json"
)

MULTI_LLM_FILE = (
    "outputs/trustgate_multi_llm_extension_results.json"
)


# ------------------------------------------------------------
# New final TrustGate experiment
# ------------------------------------------------------------

FINAL_MODEL_SUMMARY_FILE = (
    "outputs/trustgate_final_model_summary.json"
)

FINAL_EVALUATION_FILE = (
    "outputs/trustgate_final_next100_evaluation.csv"
)


# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

OUTPUT_JSON = (
    "outputs/final_results_summary.json"
)

OUTPUT_CSV = (
    "outputs/final_results_table.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json(path):

    if not os.path.exists(path):

        print("\nWARNING: JSON file not found:")
        print(path)

        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def extract_metrics(
    data,
    name
):

    if not isinstance(data, dict):

        return None

    accuracy = data.get(
        "accuracy"
    )

    balanced_accuracy = data.get(
        "balanced_accuracy"
    )

    macro_f1 = data.get(
        "macro_f1"
    )

    weighted_f1 = data.get(
        "weighted_f1"
    )


    if accuracy is None:

        return None


    return {

        "Model": name,

        "Accuracy": float(
            accuracy
        ),

        "Balanced Accuracy": float(
            balanced_accuracy
        ) if balanced_accuracy is not None else None,

        "Macro F1": float(
            macro_f1
        ) if macro_f1 is not None else None,

        "Weighted F1": float(
            weighted_f1
        ) if weighted_f1 is not None else None
    }


# ============================================================
# SEARCH FOR OLD EXPERIMENT MODELS
# ============================================================

def search_for_models(
    data,
    prefix=""
):

    found = []


    if isinstance(
        data,
        dict
    ):

        metrics = extract_metrics(
            data,
            prefix
        )


        if metrics and prefix:

            found.append(
                metrics
            )


        for key, value in data.items():

            if isinstance(
                value,
                dict
            ):

                model_name = (
                    key
                    .replace(
                        "_",
                        " "
                    )
                    .title()
                )


                result = extract_metrics(
                    value,
                    model_name
                )


                if result:

                    found.append(
                        result
                    )


                found.extend(
                    search_for_models(
                        value,
                        ""
                    )
                )


    return found


# ============================================================
# START
# ============================================================

print("=" * 60)
print("TRUSTGATE FINAL RESULTS SUMMARY")
print("=" * 60)


# ============================================================
# LOAD PREVIOUS RESULTS
# ============================================================

print(
    "\nLoading independent annotation results..."
)

independent_results = load_json(
    INDEPENDENT_RESULTS_FILE
)


print(
    "Loading ablation results..."
)

ablation_results = load_json(
    ABLATION_FILE
)


print(
    "Loading Multi-LLM results..."
)

multi_llm_results = load_json(
    MULTI_LLM_FILE
)


# ============================================================
# LOAD FINAL MODEL INFORMATION
# ============================================================

print(
    "Loading final model summary..."
)

final_model_summary = load_json(
    FINAL_MODEL_SUMMARY_FILE
)


print(
    "Loading final next-100 evaluation..."
)


if not os.path.exists(
    FINAL_EVALUATION_FILE
):

    raise FileNotFoundError(
        f"Final evaluation file missing: "
        f"{FINAL_EVALUATION_FILE}"
    )


final_evaluation_df = pd.read_csv(
    FINAL_EVALUATION_FILE
)


print(
    "Final evaluation records:",
    len(final_evaluation_df)
)


# ============================================================
# BUILD RESULTS TABLE
# ============================================================

results_rows = []


# ------------------------------------------------------------
# Independent evaluation
# ------------------------------------------------------------

if independent_results:

    metrics = extract_metrics(
        independent_results,
        "TrustGate Independent Evaluation"
    )

    if metrics:

        results_rows.append(
            metrics
        )


# ------------------------------------------------------------
# Ablation results
# ------------------------------------------------------------

if ablation_results:

    ablation_models = search_for_models(
        ablation_results
    )

    results_rows.extend(
        ablation_models
    )


# ------------------------------------------------------------
# Multi-LLM results
# ------------------------------------------------------------

if multi_llm_results:

    multi_models = search_for_models(
        multi_llm_results
    )

    results_rows.extend(
        multi_models
    )


# ============================================================
# ADD FINAL NEXT-100 RESULT
# ============================================================

# These values come directly from your completed evaluation:
#
# Accuracy: 0.5200
# Balanced Accuracy: 0.5241
# Macro F1: 0.4403
# Weighted F1: 0.5633

final_results_row = {

    "Model":
        "Final TrustGate Fusion PCA20 Logistic Regression",

    "Accuracy":
        0.5200,

    "Balanced Accuracy":
        0.5241,

    "Macro F1":
        0.4403,

    "Weighted F1":
        0.5633
}


results_rows.append(
    final_results_row
)


# ============================================================
# PREVIOUS NEXT-100 MODEL RESULT
# ============================================================

previous_results_row = {

    "Model":
        "Previous TrustGate Next-100 Model",

    "Accuracy":
        0.1900,

    "Balanced Accuracy":
        0.4019,

    "Macro F1":
        0.1892,

    "Weighted F1":
        0.1655
}


results_rows.append(
    previous_results_row
)


# ============================================================
# BUILD DATAFRAME
# ============================================================

df = pd.DataFrame(
    results_rows
)


df = df.drop_duplicates()


# ============================================================
# ROUND RESULTS
# ============================================================

metric_columns = [

    "Accuracy",

    "Balanced Accuracy",

    "Macro F1",

    "Weighted F1"
]


for column in metric_columns:

    if column in df.columns:

        df[column] = df[
            column
        ].round(
            4
        )


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# CALCULATE FINAL MODEL IMPROVEMENT
# ============================================================

improvement = {

    "accuracy_improvement":
        round(
            0.5200 - 0.1900,
            4
        ),

    "balanced_accuracy_improvement":
        round(
            0.5241 - 0.4019,
            4
        ),

    "macro_f1_improvement":
        round(
            0.4403 - 0.1892,
            4
        ),

    "weighted_f1_improvement":
        round(
            0.5633 - 0.1655,
            4
        )
}


# ============================================================
# BUILD FINAL SUMMARY
# ============================================================

summary = {

    "project":

        "TrustGate",


    "final_model": {

        "representation":

            "Fusion",


        "hidden_dimension":

            896,


        "signal_features":

            2,


        "fusion_dimension":

            898,


        "pca_components":

            20,


        "classifier":

            "Logistic Regression",


        "training_samples":

            200
    },


    "final_next_100_evaluation": {

        "samples":

            100,


        "accuracy":

            0.5200,


        "balanced_accuracy":

            0.5241,


        "macro_f1":

            0.4403,


        "weighted_f1":

            0.5633,


        "prediction_distribution": {

            "RELIABLE":

                27,


            "UNCERTAIN":

                28,


            "UNRELIABLE":

                45
        }
    },


    "previous_next_100_evaluation": {

        "accuracy":

            0.1900,


        "balanced_accuracy":

            0.4019,


        "macro_f1":

            0.1892,


        "weighted_f1":

            0.1655
    },


    "improvement_after_feature_fix_and_retraining":

        improvement,


    "results_table":

        df.to_dict(
            orient="records"
        )
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FINAL RESULTS TABLE"
)

print(
    "=" * 60
)


print(
    df.to_string(
        index=False
    )
)


# ============================================================
# FINAL IMPROVEMENT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FINAL MODEL IMPROVEMENT"
)

print(
    "=" * 60
)


print(
    "\nPrevious Next-100 Accuracy:",
    0.1900
)

print(
    "Final Next-100 Accuracy:",
    0.5200
)

print(
    "Accuracy Improvement:",
    improvement[
        "accuracy_improvement"
    ]
)


print(
    "\nPrevious Macro F1:",
    0.1892
)

print(
    "Final Macro F1:",
    0.4403
)

print(
    "Macro F1 Improvement:",
    improvement[
        "macro_f1_improvement"
    ]
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FINAL RESULTS SUMMARY COMPLETE"
)

print(
    "=" * 60
)


print(
    "\nCreated:"
)

print(
    OUTPUT_JSON
)

print(
    OUTPUT_CSV
)


print(
    "\nIMPORTANT:"
)

print(
    "No existing dataset, model, or experiment "
    "was modified."
)

print(
    "This script only summarizes existing results."
)