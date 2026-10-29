import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "outputs/final_results_table.csv"

FIGURE_DIR = "outputs/figures"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 60)
print("TRUSTGATE PUBLICATION FIGURE GENERATOR")
print("=" * 60)

print("\nLoading final results...")

df = pd.read_csv(
    INPUT_FILE
)

print("\nResults loaded:")
print(df.to_string(index=False))


# ============================================================
# FIGURE 1
# ALL MODEL COMPARISON
# ============================================================

print("\nCreating Figure 1: Model Comparison...")

plt.figure(figsize=(12, 7))

x = range(len(df))

plt.bar(
    [i - 0.25 for i in x],
    df["Accuracy"],
    width=0.25,
    label="Accuracy"
)

plt.bar(
    x,
    df["Balanced Accuracy"],
    width=0.25,
    label="Balanced Accuracy"
)

plt.bar(
    [i + 0.25 for i in x],
    df["Macro F1"],
    width=0.25,
    label="Macro F1"
)

plt.xticks(
    x,
    df["Model"],
    rotation=45,
    ha="right"
)

plt.ylabel("Score")

plt.title(
    "TrustGate Experimental Results"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.tight_layout()

figure_path = (
    os.path.join(
        FIGURE_DIR,
        "figure_1_all_models_comparison.png"
    )
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    figure_path
)


# ============================================================
# FIGURE 2
# ABLATION STUDY
# ============================================================

print("\nCreating Figure 2: Ablation Study...")

ablation_models = [

    "Signals Only",

    "Hidden Only",

    "Trustgate Fusion"
]


ablation_df = df[
    df["Model"].isin(
        ablation_models
    )
].copy()


plt.figure(figsize=(10, 6))

x = range(
    len(ablation_df)
)


plt.bar(
    [i - 0.25 for i in x],
    ablation_df["Accuracy"],
    width=0.25,
    label="Accuracy"
)

plt.bar(
    x,
    ablation_df["Balanced Accuracy"],
    width=0.25,
    label="Balanced Accuracy"
)

plt.bar(
    [i + 0.25 for i in x],
    ablation_df["Macro F1"],
    width=0.25,
    label="Macro F1"
)


plt.xticks(
    x,
    ablation_df["Model"]
)

plt.ylabel(
    "Score"
)

plt.title(
    "TrustGate Ablation Study"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.tight_layout()


figure_path = (
    os.path.join(
        FIGURE_DIR,
        "figure_2_ablation_study.png"
    )
)


plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    figure_path
)


# ============================================================
# FIGURE 3
# MULTI-LLM CONTRIBUTION
# ============================================================

print(
    "\nCreating Figure 3: Multi-LLM Contribution..."
)


multi_models = [

    "Signals Only 5D",

    "Signals Plus Multi Llm 6D",

    "Original Trustgate 901D",

    "Extended Trustgate 902D"
]


multi_df = df[
    df["Model"].isin(
        multi_models
    )
].copy()


plt.figure(
    figsize=(11, 6)
)


x = range(
    len(multi_df)
)


plt.bar(
    [i - 0.25 for i in x],
    multi_df["Accuracy"],
    width=0.25,
    label="Accuracy"
)

plt.bar(
    x,
    multi_df["Balanced Accuracy"],
    width=0.25,
    label="Balanced Accuracy"
)

plt.bar(
    [i + 0.25 for i in x],
    multi_df["Macro F1"],
    width=0.25,
    label="Macro F1"
)


plt.xticks(
    x,
    multi_df["Model"],
    rotation=20,
    ha="right"
)


plt.ylabel(
    "Score"
)

plt.title(
    "Effect of Multi-LLM Agreement"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.tight_layout()


figure_path = (
    os.path.join(
        FIGURE_DIR,
        "figure_3_multi_llm_contribution.png"
    )
)


plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    figure_path
)


# ============================================================
# FIGURE 4
# SIGNAL-ONLY MULTI-LLM IMPROVEMENT
# ============================================================

print(
    "\nCreating Figure 4: Signal Improvement..."
)


signal_models = [

    "Signals Only 5D",

    "Signals Plus Multi Llm 6D"
]


signal_df = df[
    df["Model"].isin(
        signal_models
    )
].copy()


plt.figure(
    figsize=(8, 6)
)


x = range(
    len(signal_df)
)


plt.bar(
    [i - 0.2 for i in x],
    signal_df["Balanced Accuracy"],
    width=0.4,
    label="Balanced Accuracy"
)


plt.bar(
    [i + 0.2 for i in x],
    signal_df["Macro F1"],
    width=0.4,
    label="Macro F1"
)


plt.xticks(
    x,
    [
        "Signals Only",
        "Signals + Multi-LLM"
    ]
)


plt.ylabel(
    "Score"
)


plt.title(
    "Multi-LLM Agreement Improves Signal-Based Trust Prediction"
)


plt.ylim(
    0,
    1
)


plt.legend()

plt.tight_layout()


figure_path = (
    os.path.join(
        FIGURE_DIR,
        "figure_4_signal_multi_llm_improvement.png"
    )
)


plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    figure_path
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)

print(
    "PUBLICATION FIGURES COMPLETE!"
)

print("=" * 60)


print(
    "\nFigures saved in:"
)

print(
    FIGURE_DIR
)


print(
    "\nGenerated figures:"
)

print(
    "1. figure_1_all_models_comparison.png"
)

print(
    "2. figure_2_ablation_study.png"
)

print(
    "3. figure_3_multi_llm_contribution.png"
)

print(
    "4. figure_4_signal_multi_llm_improvement.png"
)

print("\nNo existing data or models were modified.")