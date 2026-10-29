import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "outputs/trustgate_publishable_results.csv"
)

OUTPUT_DIRECTORY = (
    "outputs/publication_analysis"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(

    OUTPUT_DIRECTORY,

    exist_ok=True

)


# ============================================================
# LOAD RESULTS
# ============================================================

print()

print("=" * 70)

print("TRUSTGATE PUBLICATION RESULTS ANALYSIS")

print("=" * 70)

print()

print("Loading:")

print(INPUT_FILE)

df = pd.read_csv(

    INPUT_FILE

)


print()

print(

    "Total results:",

    len(df)

)


print()

print(

    "Columns:"

)

print(

    df.columns.tolist()

)


# ============================================================
# ROUND NUMERICAL VALUES
# ============================================================

metric_columns = [

    "accuracy",

    "balanced_accuracy",

    "macro_f1",

    "weighted_f1"

]


for column in metric_columns:

    df[column] = pd.to_numeric(

        df[column],

        errors="coerce"

    )


# ============================================================
# SAVE COMPLETE CLEAN RESULTS
# ============================================================

complete_output = os.path.join(

    OUTPUT_DIRECTORY,

    "trustgate_complete_results.csv"

)


df.to_csv(

    complete_output,

    index=False

)


# ============================================================
# BEST MODELS BY EVALUATION
# ============================================================

print()

print("=" * 70)

print("BEST MODELS BY MACRO F1")

print("=" * 70)


best_macro = (

    df

    .sort_values(

        [

            "evaluation",

            "macro_f1"

        ],

        ascending=[

            True,

            False

        ]

    )

    .groupby(

        "evaluation",

        as_index=False

    )

    .first()

)


print()

print(

    best_macro[

        [

            "evaluation",

            "experiment",

            "accuracy",

            "balanced_accuracy",

            "macro_f1"

        ]

    ].to_string(

        index=False

    )

)


best_macro.to_csv(

    os.path.join(

        OUTPUT_DIRECTORY,

        "best_models_macro_f1.csv"

    ),

    index=False

)


# ============================================================
# BEST MODELS BY BALANCED ACCURACY
# ============================================================

print()

print("=" * 70)

print("BEST MODELS BY BALANCED ACCURACY")

print("=" * 70)


best_balanced = (

    df

    .sort_values(

        [

            "evaluation",

            "balanced_accuracy"

        ],

        ascending=[

            True,

            False

        ]

    )

    .groupby(

        "evaluation",

        as_index=False

    )

    .first()

)


print()

print(

    best_balanced[

        [

            "evaluation",

            "experiment",

            "accuracy",

            "balanced_accuracy",

            "macro_f1"

        ]

    ].to_string(

        index=False

    )

)


best_balanced.to_csv(

    os.path.join(

        OUTPUT_DIRECTORY,

        "best_models_balanced_accuracy.csv"

    ),

    index=False

)


# ============================================================
# COMBINED 200 ANALYSIS
# ============================================================

print()

print("=" * 70)

print("COMBINED 200 CROSS VALIDATION")

print("=" * 70)


combined = df[

    df["evaluation"]

    ==

    "combined_200_cv"

].copy()


combined = combined.sort_values(

    "macro_f1",

    ascending=False

)


print()

print(

    combined[

        [

            "experiment",

            "accuracy",

            "balanced_accuracy",

            "macro_f1",

            "weighted_f1"

        ]

    ].to_string(

        index=False

    )

)


combined.to_csv(

    os.path.join(

        OUTPUT_DIRECTORY,

        "combined_200_results.csv"

    ),

    index=False

)


# ============================================================
# BEST COMBINED MODEL
# ============================================================

best_combined_macro = (

    combined.iloc[0]

)


best_combined_balanced = (

    combined

    .sort_values(

        "balanced_accuracy",

        ascending=False

    )

    .iloc[0]

)


print()

print(

    "BEST COMBINED MODEL BY MACRO F1:"

)

print(

    best_combined_macro["experiment"]

)

print(

    "Macro F1:",

    round(

        best_combined_macro["macro_f1"],

        4

    )

)


print()

print(

    "BEST COMBINED MODEL BY BALANCED ACCURACY:"

)

print(

    best_combined_balanced["experiment"]

)

print(

    "Balanced Accuracy:",

    round(

        best_combined_balanced["balanced_accuracy"],

        4

    )

)


# ============================================================
# REPRESENTATION ANALYSIS
# ============================================================

print()

print("=" * 70)

print("REPRESENTATION ANALYSIS")

print("=" * 70)


representation_summary = (

    combined

    .groupby(

        "representation"

    )[

        [

            "accuracy",

            "balanced_accuracy",

            "macro_f1"

        ]

    ]

    .mean()

    .reset_index()

)


representation_summary = (

    representation_summary

    .sort_values(

        "macro_f1",

        ascending=False

    )

)


print()

print(

    representation_summary.to_string(

        index=False

    )

)


representation_summary.to_csv(

    os.path.join(

        OUTPUT_DIRECTORY,

        "representation_summary.csv"

    ),

    index=False

)


# ============================================================
# MODEL TYPE ANALYSIS
# ============================================================

print()

print("=" * 70)

print("MODEL TYPE ANALYSIS")

print("=" * 70)


model_summary = (

    combined

    .groupby(

        "model"

    )[

        [

            "accuracy",

            "balanced_accuracy",

            "macro_f1"

        ]

    ]

    .mean()

    .reset_index()

)


model_summary = (

    model_summary

    .sort_values(

        "macro_f1",

        ascending=False

    )

)


print()

print(

    model_summary.to_string(

        index=False

    )

)


model_summary.to_csv(

    os.path.join(

        OUTPUT_DIRECTORY,

        "model_summary.csv"

    ),

    index=False

)


# ============================================================
# CROSS DATASET GENERALIZATION
# ============================================================

print()

print("=" * 70)

print("CROSS-DATASET GENERALIZATION")

print("=" * 70)


cross_dataset = df[

    df["evaluation"].isin(

        [

            "original_to_next",

            "next_to_original"

        ]

    )

].copy()


cross_dataset = (

    cross_dataset

    .sort_values(

        [

            "evaluation",

            "macro_f1"

        ],

        ascending=[

            True,

            False

        ]

    )

)


print()

print(

    cross_dataset[

        [

            "evaluation",

            "experiment",

            "accuracy",

            "balanced_accuracy",

            "macro_f1"

        ]

    ].to_string(

        index=False

    )

)


cross_dataset.to_csv(

    os.path.join(

        OUTPUT_DIRECTORY,

        "cross_dataset_results.csv"

    ),

    index=False

)


# ============================================================
# CROSS DATASET COMPARISON
# ============================================================

pivot_cross = (

    cross_dataset

    .pivot(

        index="experiment",

        columns="evaluation",

        values="macro_f1"

    )

    .reset_index()

)


if (

    "original_to_next"

    in

    pivot_cross.columns

    and

    "next_to_original"

    in

    pivot_cross.columns

):

    pivot_cross[

        "generalization_difference"

    ] = (

        pivot_cross[

            "original_to_next"

        ]

        -

        pivot_cross[

            "next_to_original"

        ]

    )


print()

print("=" * 70)

print("GENERALIZATION ASYMMETRY")

print("=" * 70)

print()

print(

    pivot_cross.to_string(

        index=False

    )

)


pivot_cross.to_csv(

    os.path.join(

        OUTPUT_DIRECTORY,

        "generalization_asymmetry.csv"

    ),

    index=False

)


# ============================================================
# CREATE MAIN PUBLICATION TABLE
# ============================================================

publication_table = (

    combined[

        [

            "experiment",

            "accuracy",

            "balanced_accuracy",

            "macro_f1",

            "weighted_f1",

            "accuracy_ci_lower",

            "accuracy_ci_upper",

            "balanced_accuracy_ci_lower",

            "balanced_accuracy_ci_upper",

            "macro_f1_ci_lower",

            "macro_f1_ci_upper"

        ]

    ]

    .copy()

)


publication_table = (

    publication_table

    .sort_values(

        "macro_f1",

        ascending=False

    )

)


publication_table.to_csv(

    os.path.join(

        OUTPUT_DIRECTORY,

        "publication_table_combined_200.csv"

    ),

    index=False

)


print()

print("=" * 70)

print("PUBLICATION TABLE")

print("=" * 70)

print()

print(

    publication_table.to_string(

        index=False

    )

)


# ============================================================
# PLOT 1
# COMBINED 200 MACRO F1
# ============================================================

plt.figure(

    figsize=(12, 7)

)


plot_data = (

    combined

    .sort_values(

        "macro_f1"

    )

)


plt.barh(

    plot_data["experiment"],

    plot_data["macro_f1"]

)


plt.xlabel(

    "Macro F1 Score"

)


plt.ylabel(

    "Experiment"

)


plt.title(

    "TrustGate: Combined 200 Cross-Validation Macro F1"

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIRECTORY,

        "combined_200_macro_f1.png"

    ),

    dpi=300

)


plt.close()


# ============================================================
# PLOT 2
# BALANCED ACCURACY
# ============================================================

plt.figure(

    figsize=(12, 7)

)


plot_data = (

    combined

    .sort_values(

        "balanced_accuracy"

    )

)


plt.barh(

    plot_data["experiment"],

    plot_data["balanced_accuracy"]

)


plt.xlabel(

    "Balanced Accuracy"

)


plt.ylabel(

    "Experiment"

)


plt.title(

    "TrustGate: Combined 200 Balanced Accuracy"

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIRECTORY,

        "combined_200_balanced_accuracy.png"

    ),

    dpi=300

)


plt.close()


# ============================================================
# PLOT 3
# CROSS DATASET MACRO F1
# ============================================================

pivot_plot = (

    cross_dataset

    .pivot(

        index="experiment",

        columns="evaluation",

        values="macro_f1"

    )

)


pivot_plot.plot(

    kind="bar",

    figsize=(13, 7)

)


plt.xlabel(

    "Experiment"

)


plt.ylabel(

    "Macro F1 Score"

)


plt.title(

    "TrustGate Cross-Dataset Generalization"

)


plt.xticks(

    rotation=45,

    ha="right"

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIRECTORY,

        "cross_dataset_macro_f1.png"

    ),

    dpi=300

)


plt.close()


# ============================================================
# PLOT 4
# REPRESENTATION COMPARISON
# ============================================================

plt.figure(

    figsize=(9, 6)

)


plt.bar(

    representation_summary["representation"],

    representation_summary["macro_f1"]

)


plt.xlabel(

    "Representation"

)


plt.ylabel(

    "Mean Macro F1"

)


plt.title(

    "TrustGate Representation Comparison"

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIRECTORY,

        "representation_macro_f1.png"

    ),

    dpi=300

)


plt.close()


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_file = os.path.join(

    OUTPUT_DIRECTORY,

    "publication_analysis_report.txt"

)


with open(

    report_file,

    "w",

    encoding="utf-8"

) as f:


    f.write(

        "TRUSTGATE PUBLICATION RESULTS ANALYSIS\n"

    )


    f.write(

        "=" * 70

    )


    f.write(

        "\n\n"

    )


    f.write(

        "BEST COMBINED MODEL BY MACRO F1\n"

    )


    f.write(

        "-" * 70

    )


    f.write(

        "\n"

    )


    f.write(

        str(

            best_combined_macro["experiment"]

        )

    )


    f.write(

        "\nMacro F1: "

    )


    f.write(

        str(

            best_combined_macro["macro_f1"]

        )

    )


    f.write(

        "\nBalanced Accuracy: "

    )


    f.write(

        str(

            best_combined_macro["balanced_accuracy"]

        )

    )


    f.write(

        "\nAccuracy: "

    )


    f.write(

        str(

            best_combined_macro["accuracy"]

        )

    )


    f.write(

        "\n\n"

    )


    f.write(

        "BEST COMBINED MODEL BY BALANCED ACCURACY\n"

    )


    f.write(

        "-" * 70

    )


    f.write(

        "\n"

    )


    f.write(

        str(

            best_combined_balanced["experiment"]

        )

    )


    f.write(

        "\nBalanced Accuracy: "

    )


    f.write(

        str(

            best_combined_balanced["balanced_accuracy"]

        )

    )


    f.write(

        "\nMacro F1: "

    )


    f.write(

        str(

            best_combined_balanced["macro_f1"]

        )

    )


    f.write(

        "\n\n"

    )


    f.write(

        "COMBINED 200 RESULTS\n"

    )


    f.write(

        "-" * 70

    )


    f.write(

        "\n"

    )


    f.write(

        combined.to_string(

            index=False

        )

    )


    f.write(

        "\n\n"

    )


    f.write(

        "REPRESENTATION SUMMARY\n"

    )


    f.write(

        "-" * 70

    )


    f.write(

        "\n"

    )


    f.write(

        representation_summary.to_string(

            index=False

        )

    )


    f.write(

        "\n\n"

    )


    f.write(

        "CROSS-DATASET GENERALIZATION\n"

    )


    f.write(

        "-" * 70

    )


    f.write(

        "\n"

    )


    f.write(

        cross_dataset.to_string(

            index=False

        )

    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()

print("=" * 70)

print("PUBLICATION ANALYSIS COMPLETE")

print("=" * 70)

print()

print(

    "Results directory:"

)

print(

    OUTPUT_DIRECTORY

)

print()

print(

    "Generated files:"

)

for filename in sorted(

    os.listdir(

        OUTPUT_DIRECTORY

    )

):

    print(

        "-", filename

    )


print()

print("=" * 70)