"""
TRUSTGATE DATASET SHIFT & LABEL INTEGRITY ANALYSIS

Purpose:
    Diagnose why cross-dataset evaluation produces prediction collapse.

Checks:
    1. Dataset discovery
    2. Label distributions
    3. Label consistency
    4. Feature dimensions
    5. Missing / NaN / infinite values
    6. Feature distribution statistics
    7. Original vs Next feature shift
    8. Statistical distribution differences
    9. Feature magnitude comparison
   10. Saves publication-quality diagnostic outputs

Outputs:
    outputs/dataset_shift_analysis/
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, wasserstein_distance


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_shift_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def print_subheader(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)
    print()


def safe_array(value):
    """
    Convert lists / numpy arrays to a clean numpy array.
    """

    if value is None:
        return None

    try:
        array = np.asarray(value, dtype=float)
        return array
    except Exception:
        return None


def find_files():
    """
    Search project directories for likely dataset files.
    """

    print_header("SEARCHING FOR DATASET FILES")

    possible_extensions = {
        ".csv",
        ".json",
        ".npy",
        ".npz",
        ".pkl",
        ".pickle"
    }

    search_directories = [
        PROJECT_ROOT,
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "datasets",
        PROJECT_ROOT / "outputs"
    ]

    files = []

    for directory in search_directories:

        if not directory.exists():
            continue

        for file_path in directory.rglob("*"):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() in possible_extensions:

                if "dataset_shift_analysis" in str(file_path):
                    continue

                files.append(file_path)

    files = sorted(set(files))

    print(f"Files discovered: {len(files)}")
    print()

    for file_path in files:

        try:
            relative_path = file_path.relative_to(PROJECT_ROOT)
        except Exception:
            relative_path = file_path

        print(relative_path)

    return files


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset_file(file_path):
    """
    Attempt to load a dataset from common formats.
    """

    suffix = file_path.suffix.lower()

    try:

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        if suffix == ".csv":

            df = pd.read_csv(file_path)

            return {
                "type": "dataframe",
                "data": df
            }


        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        elif suffix == ".json":

            with open(file_path, "r", encoding="utf-8") as f:

                data = json.load(f)

            return {
                "type": "json",
                "data": data
            }


        # ----------------------------------------------------
        # NPY
        # ----------------------------------------------------

        elif suffix == ".npy":

            data = np.load(
                file_path,
                allow_pickle=True
            )

            return {
                "type": "array",
                "data": data
            }


        # ----------------------------------------------------
        # NPZ
        # ----------------------------------------------------

        elif suffix == ".npz":

            data = np.load(
                file_path,
                allow_pickle=True
            )

            extracted = {}

            for key in data.files:

                extracted[key] = data[key]

            return {
                "type": "npz",
                "data": extracted
            }


    except Exception as error:

        print(
            f"WARNING: Could not load "
            f"{file_path.name}: {error}"
        )

        return None

    return None


# ============================================================
# DATASET IDENTIFICATION
# ============================================================

def classify_dataset_name(file_path):
    """
    Attempt to determine whether a file belongs to:
        original
        next
        unknown
    """

    name = file_path.name.lower()

    original_keywords = [
        "original",
        "train",
        "dataset1",
        "data1"
    ]

    next_keywords = [
        "next",
        "test",
        "dataset2",
        "data2"
    ]

    for keyword in original_keywords:

        if keyword in name:

            return "original"

    for keyword in next_keywords:

        if keyword in name:

            return "next"

    return "unknown"


# ============================================================
# DATAFRAME ANALYSIS
# ============================================================

def identify_label_column(df):
    """
    Try to identify a likely label column.
    """

    candidate_names = [

        "label",
        "labels",
        "target",
        "targets",
        "class",
        "classes",
        "y"
    ]

    lower_columns = {
        column.lower(): column
        for column in df.columns
    }

    for candidate in candidate_names:

        if candidate in lower_columns:

            return lower_columns[candidate]

    return None


def dataframe_dataset_summary(
    df,
    dataset_name,
    source_file
):
    """
    Analyze a dataframe dataset.
    """

    print_subheader(
        f"DATASET SUMMARY: {dataset_name.upper()}"
    )

    print(
        f"Source file: {source_file.name}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    label_column = identify_label_column(df)

    print(
        f"Detected label column: {label_column}"
    )

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    print(
        f"Numeric columns: {numeric_df.shape[1]}"
    )

    print(
        f"Missing values: "
        f"{int(df.isna().sum().sum())}"
    )

    summary = {

        "dataset": dataset_name,

        "source_file": str(source_file),

        "records": int(len(df)),

        "columns": int(len(df.columns)),

        "numeric_columns": int(
            numeric_df.shape[1]
        ),

        "label_column": label_column,

        "missing_values": int(
            df.isna().sum().sum()
        )
    }

    label_distribution = None

    if label_column is not None:

        counts = (
            df[label_column]
            .value_counts(
                dropna=False
            )
            .sort_index()
        )

        print()

        print(
            "LABEL DISTRIBUTION:"
        )

        print(counts)

        label_distribution = (
            counts
            .reset_index()
        )

        label_distribution.columns = [
            "class",
            "count"
        ]

        label_distribution[
            "percentage"
        ] = (

            label_distribution[
                "count"
            ]

            /

            len(df)

            *

            100
        )

        label_distribution[
            "dataset"
        ] = dataset_name

    return {

        "summary": summary,

        "label_distribution": (
            label_distribution
        ),

        "numeric_data": numeric_df
    }


# ============================================================
# FEATURE MATRIX ANALYSIS
# ============================================================

def feature_matrix_summary(
    X,
    dataset_name,
    representation_name
):
    """
    Analyze a feature matrix.
    """

    X = safe_array(X)

    if X is None:

        return None

    if X.ndim == 1:

        X = X.reshape(
            -1,
            1
        )

    if X.ndim != 2:

        return None

    records = X.shape[0]

    features = X.shape[1]

    nan_count = int(
        np.isnan(X).sum()
    )

    inf_count = int(
        np.isinf(X).sum()
    )

    finite_values = X[
        np.isfinite(X)
    ]

    if len(finite_values) > 0:

        global_mean = float(
            np.mean(finite_values)
        )

        global_std = float(
            np.std(finite_values)
        )

        global_min = float(
            np.min(finite_values)
        )

        global_max = float(
            np.max(finite_values)
        )

        global_median = float(
            np.median(finite_values)
        )

    else:

        global_mean = np.nan

        global_std = np.nan

        global_min = np.nan

        global_max = np.nan

        global_median = np.nan

    summary = {

        "dataset": dataset_name,

        "representation": (
            representation_name
        ),

        "records": int(records),

        "features": int(features),

        "nan_values": nan_count,

        "inf_values": inf_count,

        "mean": global_mean,

        "std": global_std,

        "median": global_median,

        "min": global_min,

        "max": global_max
    }

    return {

        "matrix": X,

        "summary": summary
    }


# ============================================================
# NPZ FEATURE DISCOVERY
# ============================================================

def analyze_npz_file(
    npz_data,
    dataset_name,
    source_file
):
    """
    Analyze arrays stored inside NPZ files.
    """

    results = []

    print_subheader(
        f"NPZ ANALYSIS: {source_file.name}"
    )

    for key, value in npz_data.items():

        array = safe_array(value)

        if array is None:

            continue

        print(
            f"Array: {key}"
        )

        print(
            f"Shape: {array.shape}"
        )

        if array.ndim >= 1:

            result = feature_matrix_summary(

                array,

                dataset_name,

                key
            )

            if result is not None:

                results.append(result)

    return results


# ============================================================
# ARRAY ANALYSIS
# ============================================================

def analyze_array_file(
    array,
    dataset_name,
    source_file
):
    """
    Analyze standalone numpy arrays.
    """

    print_subheader(
        f"ARRAY ANALYSIS: {source_file.name}"
    )

    result = feature_matrix_summary(

        array,

        dataset_name,

        source_file.stem
    )

    return [result] if result else []


# ============================================================
# FEATURE SHIFT ANALYSIS
# ============================================================

def compare_feature_matrices(
    original_result,
    next_result
):
    """
    Compare Original and Next feature matrices.
    """

    original_X = original_result["matrix"]

    next_X = next_result["matrix"]

    original_name = (
        original_result["summary"]
        ["representation"]
    )

    next_name = (
        next_result["summary"]
        ["representation"]
    )

    if (
        original_X.shape[1]
        !=
        next_X.shape[1]
    ):

        return {

            "representation_original":
                original_name,

            "representation_next":
                next_name,

            "dimension_match":
                False,

            "original_features":
                original_X.shape[1],

            "next_features":
                next_X.shape[1]
        }

    features = original_X.shape[1]

    # --------------------------------------------
    # Per-feature means
    # --------------------------------------------

    original_means = np.nanmean(
        original_X,
        axis=0
    )

    next_means = np.nanmean(
        next_X,
        axis=0
    )

    mean_absolute_difference = float(

        np.mean(

            np.abs(
                original_means
                -
                next_means
            )

        )

    )

    mean_difference = float(

        np.mean(

            original_means
            -
            next_means
        )

    )

    # --------------------------------------------
    # Standard deviations
    # --------------------------------------------

    original_stds = np.nanstd(

        original_X,
        axis=0
    )

    next_stds = np.nanstd(

        next_X,
        axis=0
    )

    std_absolute_difference = float(

        np.mean(

            np.abs(

                original_stds
                -
                next_stds

            )

        )

    )

    # --------------------------------------------
    # KS tests
    # --------------------------------------------

    max_features_for_test = min(
        features,
        200
    )

    feature_indices = np.linspace(

        0,

        features - 1,

        max_features_for_test,

        dtype=int
    )

    ks_statistics = []

    ks_pvalues = []

    wasserstein_values = []

    for feature_index in feature_indices:

        original_feature = original_X[
            :,
            feature_index
        ]

        next_feature = next_X[
            :,
            feature_index
        ]

        original_feature = original_feature[

            np.isfinite(
                original_feature
            )

        ]

        next_feature = next_feature[

            np.isfinite(
                next_feature
            )

        ]

        if (

            len(original_feature) < 2

            or

            len(next_feature) < 2

        ):

            continue

        try:

            ks_result = ks_2samp(

                original_feature,

                next_feature
            )

            ks_statistics.append(

                ks_result.statistic
            )

            ks_pvalues.append(

                ks_result.pvalue
            )

            wasserstein = (

                wasserstein_distance(

                    original_feature,

                    next_feature

                )

            )

            wasserstein_values.append(

                wasserstein
            )

        except Exception:

            continue

    # --------------------------------------------
    # Significant KS shifts
    # --------------------------------------------

    if len(ks_pvalues) > 0:

        significant_features = int(

            np.sum(

                np.array(
                    ks_pvalues
                )

                <

                0.05

            )

        )

        significant_percentage = float(

            significant_features

            /

            len(ks_pvalues)

            *

            100

        )

        average_ks = float(

            np.mean(
                ks_statistics
            )

        )

        average_wasserstein = float(

            np.mean(
                wasserstein_values
            )

        )

    else:

        significant_features = 0

        significant_percentage = np.nan

        average_ks = np.nan

        average_wasserstein = np.nan

    return {

        "representation_original":
            original_name,

        "representation_next":
            next_name,

        "dimension_match":
            True,

        "original_features":
            int(original_X.shape[1]),

        "next_features":
            int(next_X.shape[1]),

        "mean_absolute_difference":
            mean_absolute_difference,

        "mean_difference":
            mean_difference,

        "std_absolute_difference":
            std_absolute_difference,

        "features_tested":
            int(len(ks_pvalues)),

        "significant_shift_features":
            significant_features,

        "significant_shift_percentage":
            significant_percentage,

        "average_ks_statistic":
            average_ks,

        "average_wasserstein_distance":
            average_wasserstein
    }


# ============================================================
# DATAFRAME FEATURE SHIFT
# ============================================================

def compare_numeric_dataframes(
    original_df,
    next_df
):
    """
    Compare numeric columns between two dataframes.
    """

    original_columns = set(
        original_df.columns
    )

    next_columns = set(
        next_df.columns
    )

    common_columns = sorted(

        original_columns

        &

        next_columns

    )

    rows = []

    for column in common_columns:

        original_values = pd.to_numeric(

            original_df[column],

            errors="coerce"

        ).dropna()

        next_values = pd.to_numeric(

            next_df[column],

            errors="coerce"

        ).dropna()

        if (

            len(original_values) < 2

            or

            len(next_values) < 2

        ):

            continue

        try:

            ks_result = ks_2samp(

                original_values,

                next_values

            )

            wasserstein = wasserstein_distance(

                original_values,

                next_values

            )

            rows.append({

                "feature": column,

                "original_mean":
                    original_values.mean(),

                "next_mean":
                    next_values.mean(),

                "mean_difference":

                    original_values.mean()

                    -

                    next_values.mean(),

                "original_std":
                    original_values.std(),

                "next_std":
                    next_values.std(),

                "ks_statistic":
                    ks_result.statistic,

                "ks_pvalue":
                    ks_result.pvalue,

                "wasserstein_distance":
                    wasserstein,

                "significant_shift":
                    ks_result.pvalue < 0.05
            })

        except Exception:

            continue

    return pd.DataFrame(rows)


# ============================================================
# LABEL DISTRIBUTION COMPARISON
# ============================================================

def compare_label_distributions(
    original_distribution,
    next_distribution
):
    """
    Compare class distributions.
    """

    if (

        original_distribution is None

        or

        next_distribution is None

    ):

        return None

    original = original_distribution[

        [
            "class",
            "count",
            "percentage"
        ]

    ].copy()

    next_data = next_distribution[

        [
            "class",
            "count",
            "percentage"
        ]

    ].copy()

    original.columns = [

        "class",

        "original_count",

        "original_percentage"
    ]

    next_data.columns = [

        "class",

        "next_count",

        "next_percentage"
    ]

    comparison = pd.merge(

        original,

        next_data,

        on="class",

        how="outer"

    ).fillna(0)

    comparison[

        "percentage_difference"

    ] = (

        comparison[
            "original_percentage"
        ]

        -

        comparison[
            "next_percentage"
        ]

    )

    return comparison


# ============================================================
# VISUALIZATION
# ============================================================

def plot_label_distribution(
    comparison
):
    """
    Create label distribution chart.
    """

    if comparison is None:

        return

    if len(comparison) == 0:

        return

    classes = comparison[
        "class"
    ].astype(str)

    positions = np.arange(
        len(classes)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(

        positions - width / 2,

        comparison[
            "original_percentage"
        ],

        width,

        label="Original"
    )

    plt.bar(

        positions + width / 2,

        comparison[
            "next_percentage"
        ],

        width,

        label="Next"
    )

    plt.xticks(

        positions,

        classes
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Percentage"
    )

    plt.title(
        "Original vs Next Label Distribution"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (

        OUTPUT_DIR

        /

        "label_distribution_comparison.png"

    )

    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()


def plot_feature_shift_summary(
    shift_df
):
    """
    Plot significant feature shift percentages.
    """

    if shift_df is None:

        return

    if len(shift_df) == 0:

        return

    valid = shift_df.dropna(

        subset=[
            "significant_shift_percentage"
        ]

    )

    if len(valid) == 0:

        return

    labels = [

        f"{row['representation_original']} → "
        f"{row['representation_next']}"

        for _, row in valid.iterrows()
    ]

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(

        labels,

        valid[
            "significant_shift_percentage"
        ]
    )

    plt.xticks(

        rotation=45,

        ha="right"
    )

    plt.ylabel(
        "Significantly Shifted Features (%)"
    )

    plt.title(
        "Feature Distribution Shift"
    )

    plt.tight_layout()

    output_path = (

        OUTPUT_DIR

        /

        "feature_shift_summary.png"

    )

    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(
    dataset_summaries,
    label_comparison,
    feature_shift_df,
    dataframe_shift_df
):
    """
    Generate a readable diagnostic report.
    """

    report_path = (

        OUTPUT_DIR

        /

        "dataset_shift_report.txt"

    )

    lines = []

    lines.append(
        "=" * 70
    )

    lines.append(
        "TRUSTGATE DATASET SHIFT & LABEL INTEGRITY REPORT"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    # --------------------------------------------------------
    # Dataset summaries
    # --------------------------------------------------------

    lines.append(
        "DATASET SUMMARIES"
    )

    lines.append(
        "-" * 70
    )

    for summary in dataset_summaries:

        lines.append(

            str(summary)

        )

    lines.append("")

    # --------------------------------------------------------
    # Label comparison
    # --------------------------------------------------------

    lines.append(
        "LABEL DISTRIBUTION COMPARISON"
    )

    lines.append(
        "-" * 70
    )

    if label_comparison is not None:

        lines.append(

            label_comparison.to_string(
                index=False
            )

        )

    else:

        lines.append(

            "No label comparison available."
        )

    lines.append("")

    # --------------------------------------------------------
    # Feature shifts
    # --------------------------------------------------------

    lines.append(
        "FEATURE MATRIX SHIFT SUMMARY"
    )

    lines.append(
        "-" * 70
    )

    if (

        feature_shift_df is not None

        and

        len(feature_shift_df) > 0

    ):

        lines.append(

            feature_shift_df.to_string(
                index=False
            )

        )

    else:

        lines.append(

            "No feature matrix comparison available."
        )

    lines.append("")

    # --------------------------------------------------------
    # Dataframe shifts
    # --------------------------------------------------------

    lines.append(
        "NUMERIC DATAFRAME FEATURE SHIFT"
    )

    lines.append(
        "-" * 70
    )

    if (

        dataframe_shift_df is not None

        and

        len(dataframe_shift_df) > 0

    ):

        lines.append(

            dataframe_shift_df.to_string(
                index=False
            )

        )

    else:

        lines.append(

            "No dataframe feature comparison available."
        )

    lines.append("")

    # --------------------------------------------------------
    # Conclusions
    # --------------------------------------------------------

    lines.append(
        "AUTOMATIC DIAGNOSTIC INTERPRETATION"
    )

    lines.append(
        "-" * 70
    )

    if label_comparison is not None:

        maximum_difference = (

            label_comparison[
                "percentage_difference"
            ]

            .abs()

            .max()

        )

        lines.append(

            "Maximum class distribution difference: "
            f"{maximum_difference:.2f}%"
        )

        if maximum_difference > 20:

            lines.append(

                "WARNING: Large label distribution "
                "difference detected."
            )

    if (

        feature_shift_df is not None

        and

        len(feature_shift_df) > 0

    ):

        valid = feature_shift_df.dropna(

            subset=[
                "significant_shift_percentage"
            ]

        )

        if len(valid) > 0:

            maximum_shift = valid[

                "significant_shift_percentage"

            ].max()

            lines.append(

                "Maximum significant feature shift: "
                f"{maximum_shift:.2f}%"
            )

            if maximum_shift > 50:

                lines.append(

                    "WARNING: Major feature "
                    "distribution shift detected."
                )

    lines.append("")

    lines.append(
        "=" * 70
    )

    with open(

        report_path,

        "w",

        encoding="utf-8"

    ) as report_file:

        report_file.write(

            "\n".join(lines)

        )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "TRUSTGATE DATASET SHIFT & LABEL INTEGRITY ANALYSIS"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    files = find_files()

    if len(files) == 0:

        print()

        print(
            "ERROR: No dataset files discovered."
        )

        return

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    dataframe_datasets = {}

    dataframe_results = {}

    feature_results = {

        "original": [],

        "next": []

    }

    dataset_summaries = []

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    print_header(
        "LOADING AND ANALYZING DATASETS"
    )

    for file_path in files:

        dataset_name = classify_dataset_name(
            file_path
        )

        if dataset_name == "unknown":

            continue

        print()

        print(
            f"Processing: {file_path.name}"
        )

        print(
            f"Dataset classification: "
            f"{dataset_name}"
        )

        loaded = load_dataset_file(
            file_path
        )

        if loaded is None:

            continue

        # ----------------------------------------------------
        # DataFrame
        # ----------------------------------------------------

        if loaded["type"] == "dataframe":

            df = loaded["data"]

            result = dataframe_dataset_summary(

                df,

                dataset_name,

                file_path
            )

            dataframe_datasets[
                dataset_name
            ] = df

            dataframe_results[
                dataset_name
            ] = result

            dataset_summaries.append(

                result["summary"]

            )

        # ----------------------------------------------------
        # NPZ
        # ----------------------------------------------------

        elif loaded["type"] == "npz":

            results = analyze_npz_file(

                loaded["data"],

                dataset_name,

                file_path
            )

            for result in results:

                feature_results[
                    dataset_name
                ].append(result)

                dataset_summaries.append(

                    result["summary"]

                )

        # ----------------------------------------------------
        # Array
        # ----------------------------------------------------

        elif loaded["type"] == "array":

            results = analyze_array_file(

                loaded["data"],

                dataset_name,

                file_path
            )

            for result in results:

                feature_results[
                    dataset_name
                ].append(result)

                dataset_summaries.append(

                    result["summary"]

                )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print_header(
        "DATASET INTEGRITY SUMMARY"
    )

    summary_df = pd.DataFrame(
        dataset_summaries
    )

    if len(summary_df) > 0:

        print(

            summary_df.to_string(
                index=False
            )

        )

        summary_df.to_csv(

            OUTPUT_DIR

            /

            "dataset_integrity_summary.csv",

            index=False
        )

    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    print_header(
        "LABEL DISTRIBUTION ANALYSIS"
    )

    label_comparison = None

    if (

        "original" in dataframe_results

        and

        "next" in dataframe_results

    ):

        original_labels = dataframe_results[
            "original"
        ][
            "label_distribution"
        ]

        next_labels = dataframe_results[
            "next"
        ][
            "label_distribution"
        ]

        label_comparison = (

            compare_label_distributions(

                original_labels,

                next_labels
            )

        )

        if label_comparison is not None:

            print(

                label_comparison.to_string(
                    index=False
                )

            )

            label_comparison.to_csv(

                OUTPUT_DIR

                /

                "label_distribution_comparison.csv",

                index=False
            )

            plot_label_distribution(
                label_comparison
            )

    else:

        print(

            "Could not automatically identify both "
            "Original and Next dataframe datasets."
        )

    # ========================================================
    # DATAFRAME FEATURE SHIFT
    # ========================================================

    print_header(
        "NUMERIC FEATURE SHIFT ANALYSIS"
    )

    dataframe_shift_df = None

    if (

        "original" in dataframe_datasets

        and

        "next" in dataframe_datasets

    ):

        original_numeric = (

            dataframe_datasets[
                "original"
            ]

            .select_dtypes(
                include=[np.number]
            )

        )

        next_numeric = (

            dataframe_datasets[
                "next"
            ]

            .select_dtypes(
                include=[np.number]
            )

        )

        dataframe_shift_df = (

            compare_numeric_dataframes(

                original_numeric,

                next_numeric
            )

        )

        if len(dataframe_shift_df) > 0:

            dataframe_shift_df.to_csv(

                OUTPUT_DIR

                /

                "numeric_feature_shift.csv",

                index=False
            )

            significant = dataframe_shift_df[

                dataframe_shift_df[
                    "significant_shift"
                ]

            ]

            print(

                f"Common numeric features analyzed: "
                f"{len(dataframe_shift_df)}"
            )

            print(

                f"Significantly shifted features: "
                f"{len(significant)}"
            )

            print()

            print(

                dataframe_shift_df

                .sort_values(

                    "ks_statistic",

                    ascending=False

                )

                .head(20)

                .to_string(
                    index=False
                )

            )

        else:

            print(
                "No common numeric features available."
            )

    # ========================================================
    # FEATURE MATRIX SHIFT
    # ========================================================

    print_header(
        "FEATURE MATRIX DISTRIBUTION SHIFT"
    )

    feature_shift_results = []

    original_features = feature_results[
        "original"
    ]

    next_features = feature_results[
        "next"
    ]

    for original_result in original_features:

        for next_result in next_features:

            comparison = (

                compare_feature_matrices(

                    original_result,

                    next_result
                )

            )

            feature_shift_results.append(
                comparison
            )

    feature_shift_df = pd.DataFrame(
        feature_shift_results
    )

    if len(feature_shift_df) > 0:

        print(

            feature_shift_df.to_string(
                index=False
            )

        )

        feature_shift_df.to_csv(

            OUTPUT_DIR

            /

            "feature_matrix_shift_summary.csv",

            index=False
        )

        plot_feature_shift_summary(
            feature_shift_df
        )

    else:

        print(

            "No matching feature matrices were "
            "automatically discovered."
        )

    # ========================================================
    # REPORT
    # ========================================================

    print_header(
        "GENERATING DIAGNOSTIC REPORT"
    )

    report_path = generate_report(

        dataset_summaries,

        label_comparison,

        feature_shift_df,

        dataframe_shift_df
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print_header(
        "TRUSTGATE DATASET SHIFT ANALYSIS COMPLETE"
    )

    print(
        f"Results directory:\n{OUTPUT_DIR}"
    )

    print()

    print(
        "Generated files:"
    )

    generated_files = sorted(

        OUTPUT_DIR.glob("*")

    )

    for file_path in generated_files:

        print(
            f"- {file_path.name}"
        )

    print()

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        import traceback

        print()

        print("=" * 70)

        print(
            "ERROR DURING DATASET SHIFT ANALYSIS"
        )

        print("=" * 70)

        print()

        traceback.print_exc()

        print()

        raise