import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_JSON = Path("outputs/trustgate_publishable_evaluation.json")
OUTPUT_DIR = Path("outputs/diagnostic_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def safe_filename(text):
    """
    Convert experiment names into safe filenames.
    """
    replacements = {
        " ": "_",
        "+": "plus",
        "/": "_",
        "\\": "_",
        ":": "_",
        ";": "_",
        ",": "_",
        "(": "",
        ")": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.lower()


def extract_confusion_matrix(entry):
    """
    Extract confusion matrix from an evaluation entry.
    """
    try:
        cm = entry["metrics"]["confusion_matrix"]
        return np.array(cm, dtype=int)
    except (KeyError, TypeError):
        return None


def reconstruct_labels_from_confusion_matrix(cm):
    """
    Reconstruct y_true and y_pred from a confusion matrix.

    This allows us to calculate:
    - precision
    - recall
    - F1
    - predicted class distribution

    without requiring the original predictions.
    """

    y_true = []
    y_pred = []

    for true_class in range(cm.shape[0]):
        for predicted_class in range(cm.shape[1]):

            count = cm[true_class, predicted_class]

            if count <= 0:
                continue

            y_true.extend([true_class] * count)
            y_pred.extend([predicted_class] * count)

    return np.array(y_true), np.array(y_pred)


def get_class_names(cm):
    """
    Generate generic class names.

    If your TrustGate dataset has known semantic class names,
    these can later be replaced.
    """

    return [f"Class {i}" for i in range(cm.shape[0])]


def analyze_confusion_matrix(cm):
    """
    Perform detailed analysis of a confusion matrix.
    """

    y_true, y_pred = reconstruct_labels_from_confusion_matrix(cm)

    class_names = get_class_names(cm)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    # True class distribution
    true_counts = cm.sum(axis=1)

    # Predicted class distribution
    predicted_counts = cm.sum(axis=0)

    total_samples = cm.sum()

    true_percentages = (
        true_counts / total_samples * 100
        if total_samples > 0
        else np.zeros_like(true_counts, dtype=float)
    )

    predicted_percentages = (
        predicted_counts / total_samples * 100
        if total_samples > 0
        else np.zeros_like(predicted_counts, dtype=float)
    )

    per_class_df = pd.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
            "true_count": true_counts,
            "true_percentage": true_percentages,
            "predicted_count": predicted_counts,
            "predicted_percentage": predicted_percentages,
        }
    )

    return per_class_df


def calculate_prediction_collapse(cm):
    """
    Detect whether a model is collapsing toward one predicted class.
    """

    predicted_counts = cm.sum(axis=0)

    total = predicted_counts.sum()

    if total == 0:
        return {
            "dominant_class": None,
            "dominant_predictions": 0,
            "dominant_percentage": 0.0,
            "collapse_warning": False,
        }

    dominant_class = int(np.argmax(predicted_counts))

    dominant_predictions = int(predicted_counts[dominant_class])

    dominant_percentage = dominant_predictions / total * 100

    # Flag strong prediction collapse.
    collapse_warning = dominant_percentage >= 70

    return {
        "dominant_class": dominant_class,
        "dominant_predictions": dominant_predictions,
        "dominant_percentage": dominant_percentage,
        "collapse_warning": collapse_warning,
    }


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_confusion_matrix(cm, class_names, title, output_path):
    """
    Save a publication-quality confusion matrix.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    ax.set_title(title)

    # Add values inside matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_class_distribution(
    per_class_df,
    title,
    output_path,
):
    """
    Compare true vs predicted class distributions.
    """

    classes = per_class_df["class"].tolist()

    true_percentages = per_class_df["true_percentage"].tolist()

    predicted_percentages = (
        per_class_df["predicted_percentage"].tolist()
    )

    x = np.arange(len(classes))

    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        true_percentages,
        width,
        label="True Distribution",
    )

    ax.bar(
        x + width / 2,
        predicted_percentages,
        width,
        label="Predicted Distribution",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(classes)

    ax.set_ylabel("Percentage")

    ax.set_xlabel("Class")

    ax.set_title(title)

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_per_class_f1(
    per_class_df,
    title,
    output_path,
):
    """
    Plot per-class F1 scores.
    """

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        per_class_df["class"],
        per_class_df["f1_score"],
    )

    ax.set_ylabel("F1 Score")

    ax.set_xlabel("Class")

    ax.set_ylim(0, 1)

    ax.set_title(title)

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# JSON SECTION DISCOVERY
# ============================================================

def find_evaluation_sections(data):
    """
    Find JSON sections containing experiment entries with
    confusion matrices.

    This makes the script flexible even if section names differ.
    """

    sections = {}

    for section_name, section_data in data.items():

        if not isinstance(section_data, dict):
            continue

        found_entries = {}

        for experiment_name, entry in section_data.items():

            if not isinstance(entry, dict):
                continue

            cm = extract_confusion_matrix(entry)

            if cm is not None:

                found_entries[experiment_name] = entry

        if found_entries:

            sections[section_name] = found_entries

    return sections


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print_header(
        "TRUSTGATE DIAGNOSTIC ANALYSIS"
    )

    # --------------------------------------------------------
    # LOAD JSON
    # --------------------------------------------------------

    print("Loading evaluation file:")
    print(INPUT_JSON)

    if not INPUT_JSON.exists():

        raise FileNotFoundError(
            f"Evaluation file not found:\n{INPUT_JSON}"
        )

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print()
    print("JSON loaded successfully.")

    # --------------------------------------------------------
    # FIND EVALUATION SECTIONS
    # --------------------------------------------------------

    print_header(
        "DISCOVERING EVALUATION SECTIONS"
    )

    sections = find_evaluation_sections(data)

    if not sections:

        raise ValueError(
            "No evaluation sections containing confusion matrices "
            "were found."
        )

    print("Detected sections:")

    for section_name, experiments in sections.items():

        print(
            f"- {section_name}: "
            f"{len(experiments)} experiments"
        )

    # --------------------------------------------------------
    # COLLECT DIAGNOSTIC RESULTS
    # --------------------------------------------------------

    all_experiment_diagnostics = []

    all_per_class_results = []

    # --------------------------------------------------------
    # PROCESS EACH SECTION
    # --------------------------------------------------------

    for section_name, experiments in sections.items():

        print_header(
            f"PROCESSING SECTION: {section_name}"
        )

        section_output = OUTPUT_DIR / safe_filename(
            section_name
        )

        section_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        for experiment_name, entry in experiments.items():

            print(
                f"Analyzing: {experiment_name}"
            )

            cm = extract_confusion_matrix(entry)

            if cm is None:

                print(
                    "  WARNING: No confusion matrix found."
                )

                continue

            # ------------------------------------------------
            # METADATA
            # ------------------------------------------------

            representation = entry.get(
                "representation",
                "unknown",
            )

            model = entry.get(
                "model",
                "unknown",
            )

            pca_used = entry.get(
                "pca_used",
                False,
            )

            pca_components = entry.get(
                "pca_components",
                None,
            )

            metrics = entry.get(
                "metrics",
                {},
            )

            # ------------------------------------------------
            # PER-CLASS ANALYSIS
            # ------------------------------------------------

            per_class_df = analyze_confusion_matrix(
                cm
            )

            per_class_df.insert(
                0,
                "experiment",
                experiment_name,
            )

            per_class_df.insert(
                1,
                "evaluation_section",
                section_name,
            )

            per_class_df.insert(
                2,
                "representation",
                representation,
            )

            per_class_df.insert(
                3,
                "model",
                model,
            )

            per_class_df.insert(
                4,
                "pca_used",
                pca_used,
            )

            per_class_df.insert(
                5,
                "pca_components",
                pca_components,
            )

            all_per_class_results.append(
                per_class_df
            )

            # ------------------------------------------------
            # COLLAPSE ANALYSIS
            # ------------------------------------------------

            collapse_info = (
                calculate_prediction_collapse(cm)
            )

            total_samples = int(cm.sum())

            diagnostic_row = {

                "evaluation_section":
                    section_name,

                "experiment":
                    experiment_name,

                "representation":
                    representation,

                "model":
                    model,

                "pca_used":
                    pca_used,

                "pca_components":
                    pca_components,

                "total_samples":
                    total_samples,

                "accuracy":
                    metrics.get(
                        "accuracy",
                        np.nan,
                    ),

                "balanced_accuracy":
                    metrics.get(
                        "balanced_accuracy",
                        np.nan,
                    ),

                "macro_f1":
                    metrics.get(
                        "macro_f1",
                        np.nan,
                    ),

                "weighted_f1":
                    metrics.get(
                        "weighted_f1",
                        np.nan,
                    ),

                "dominant_predicted_class":
                    collapse_info[
                        "dominant_class"
                    ],

                "dominant_predictions":
                    collapse_info[
                        "dominant_predictions"
                    ],

                "dominant_prediction_percentage":
                    collapse_info[
                        "dominant_percentage"
                    ],

                "prediction_collapse_warning":
                    collapse_info[
                        "collapse_warning"
                    ],
            }

            all_experiment_diagnostics.append(
                diagnostic_row
            )

            # ------------------------------------------------
            # FILE PREFIX
            # ------------------------------------------------

            experiment_file_name = safe_filename(
                experiment_name
            )

            # ------------------------------------------------
            # SAVE CONFUSION MATRIX
            # ------------------------------------------------

            class_names = get_class_names(cm)

            plot_confusion_matrix(

                cm=cm,

                class_names=class_names,

                title=(
                    f"{section_name}\n"
                    f"{experiment_name}"
                ),

                output_path=(
                    section_output
                    /
                    f"{experiment_file_name}"
                    "_confusion_matrix.png"
                ),
            )

            # ------------------------------------------------
            # SAVE CLASS DISTRIBUTION
            # ------------------------------------------------

            plot_class_distribution(

                per_class_df=per_class_df,

                title=(
                    f"True vs Predicted Distribution\n"
                    f"{experiment_name}"
                ),

                output_path=(
                    section_output
                    /
                    f"{experiment_file_name}"
                    "_class_distribution.png"
                ),
            )

            # ------------------------------------------------
            # SAVE PER-CLASS F1
            # ------------------------------------------------

            plot_per_class_f1(

                per_class_df=per_class_df,

                title=(
                    f"Per-Class F1 Scores\n"
                    f"{experiment_name}"
                ),

                output_path=(
                    section_output
                    /
                    f"{experiment_file_name}"
                    "_per_class_f1.png"
                ),
            )

            # ------------------------------------------------
            # SAVE INDIVIDUAL PER-CLASS CSV
            # ------------------------------------------------

            per_class_df.to_csv(

                section_output
                /
                f"{experiment_file_name}"
                "_per_class_metrics.csv",

                index=False,
            )

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    print_header(
        "SAVING COMBINED DIAGNOSTIC RESULTS"
    )

    diagnostics_df = pd.DataFrame(
        all_experiment_diagnostics
    )

    per_class_results_df = pd.concat(
        all_per_class_results,
        ignore_index=True,
    )

    diagnostics_path = (
        OUTPUT_DIR
        /
        "experiment_diagnostics.csv"
    )

    diagnostics_df.to_csv(
        diagnostics_path,
        index=False,
    )

    per_class_path = (
        OUTPUT_DIR
        /
        "per_class_metrics_all_experiments.csv"
    )

    per_class_results_df.to_csv(
        per_class_path,
        index=False,
    )

    # ========================================================
    # PREDICTION COLLAPSE REPORT
    # ========================================================

    collapse_df = diagnostics_df[
        diagnostics_df[
            "prediction_collapse_warning"
        ]
    ].copy()

    collapse_path = (
        OUTPUT_DIR
        /
        "prediction_collapse_warnings.csv"
    )

    collapse_df.to_csv(
        collapse_path,
        index=False,
    )

    # ========================================================
    # BEST MODELS
    # ========================================================

    print_header(
        "BEST MODELS DIAGNOSTIC SUMMARY"
    )

    if not diagnostics_df.empty:

        best_macro_f1 = diagnostics_df.sort_values(
            "macro_f1",
            ascending=False,
        ).head(10)

        print(
            "TOP 10 MODELS BY MACRO F1"
        )

        print()

        print(
            best_macro_f1[
                [
                    "evaluation_section",
                    "experiment",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "dominant_prediction_percentage",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # COLLAPSE WARNINGS
    # ========================================================

    print_header(
        "PREDICTION COLLAPSE ANALYSIS"
    )

    if collapse_df.empty:

        print(
            "No severe prediction collapse detected."
        )

    else:

        print(
            "WARNING: Potential prediction collapse detected."
        )

        print()

        print(
            collapse_df[
                [
                    "evaluation_section",
                    "experiment",
                    "dominant_predicted_class",
                    "dominant_predictions",
                    "dominant_prediction_percentage",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # CREATE SUMMARY REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        /
        "diagnostic_analysis_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "=" * 70 + "\n"
        )

        report.write(
            "TRUSTGATE DIAGNOSTIC ANALYSIS REPORT\n"
        )

        report.write(
            "=" * 70 + "\n\n"
        )

        report.write(
            f"Input file: {INPUT_JSON}\n"
        )

        report.write(
            f"Output directory: {OUTPUT_DIR}\n\n"
        )

        report.write(
            "EVALUATION SECTIONS\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        for section_name, experiments in sections.items():

            report.write(
                f"{section_name}: "
                f"{len(experiments)} experiments\n"
            )

        report.write("\n")

        # ----------------------------------------------------

        report.write(
            "TOP MODELS BY MACRO F1\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        report.write(
            diagnostics_df.sort_values(
                "macro_f1",
                ascending=False,
            ).head(15).to_string(
                index=False
            )
        )

        report.write(
            "\n\n"
        )

        # ----------------------------------------------------

        report.write(
            "PREDICTION COLLAPSE WARNINGS\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        if collapse_df.empty:

            report.write(
                "No severe prediction collapse detected.\n"
            )

        else:

            report.write(
                collapse_df.to_string(
                    index=False
                )
            )

            report.write("\n")

        # ----------------------------------------------------

        report.write(
            "\nPER-CLASS PERFORMANCE SUMMARY\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        per_class_summary = (
            per_class_results_df
            .groupby("class")
            [
                [
                    "precision",
                    "recall",
                    "f1_score",
                ]
            ]
            .mean()
        )

        report.write(
            per_class_summary.to_string()
        )

        report.write(
            "\n"
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print_header(
        "TRUSTGATE DIAGNOSTIC ANALYSIS COMPLETE"
    )

    print(
        "Results directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Generated files:"
    )

    print(
        "- experiment_diagnostics.csv"
    )

    print(
        "- per_class_metrics_all_experiments.csv"
    )

    print(
        "- prediction_collapse_warnings.csv"
    )

    print(
        "- diagnostic_analysis_report.txt"
    )

    print()

    print(
        "Additional confusion matrices, class distributions,"
    )

    print(
        "per-class F1 charts, and CSV files are organized"
    )

    print(
        "inside folders corresponding to each evaluation section."
    )

    print()

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()