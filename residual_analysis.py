import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
import os

DATA_FILE = "data/natural_stories_phase3.csv"
RESULTS_DIR = "results"


def load_and_clean_data(file_path):
    """Loads CSV and removes infinite values or rows with missing critical metrics."""
    df = pd.read_csv(file_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['RT', 'word_len', 'llama_base_surprisal']
    )
    return df


def apply_length_correction(df):
    """Removes linear effect of word length from RT using linear regression residuals."""
    slope, intercept, _, _, _ = linregress(df['word_len'], df['RT'])
    df['RT_corrected'] = df['RT'] - (df['word_len'] * slope + intercept)
    print(f"Removed linear effect of length. (Slope: {slope:.2f} ms/char)")
    return df


def calculate_correlations(df, models):
    """Calculates Pearson correlation between corrected RT and model surprisals."""
    print("\n--- Corrected Correlations (Length Removed) ---")
    results = []
    for name, col in models.items():
        r_val, _ = pearsonr(df['RT_corrected'], df[col])
        results.append({"Model": name, "Corrected_r": r_val})
        print(f"{name}: r = {r_val:.4f}")
    return pd.DataFrame(results)


def plot_model_comparison(results_df, output_dir):
    """Generates a bar plot comparing Pearson r values across models."""
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=results_df, x="Model", y="Corrected_r", palette="viridis")

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', label_type='edge', padding=-20,
                     color='black', weight='bold')

    plt.axhline(0, color='black', linewidth=1)
    min_val = results_df["Corrected_r"].min()
    max_val = results_df["Corrected_r"].max()
    buffer = 0.05
    plt.ylim(min(0, min_val - buffer), max_val + buffer)

    plt.title("Figure 3: Pearson Correlation (r) Between Model Surprisal and Corrected Human RT")
    plt.ylabel("Correlation with Human RT (Pearson r)")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "model_comparison_bar_corrected.png")
    plt.savefig(save_path)
    print(f"\nSaved comparison plot to {save_path}")


def plot_binned_surprisal(df, models, output_dir):
    """Visualizes the relationship between surprisal deciles and corrected RT."""
    plt.figure(figsize=(10, 6))
    for name, col in models.items():
        if name == "N-gram":
            continue

        df[f'{col}_bin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
        binned_data = df.groupby(f'{col}_bin')['RT_corrected'].mean().reset_index()
        sns.lineplot(x=binned_data.index, y=binned_data['RT_corrected'], label=name, marker='o')

    plt.title("Figure 4: Surprisal Effect after Removing Word Length")
    plt.xlabel("Model Surprisal Decile")
    plt.ylabel("Corrected Human RT (ms)")
    plt.xticks(range(10))
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)

    save_path = os.path.join(output_dir, "sanity_check_corrected_gap.png")
    plt.savefig(save_path)
    print(f"Saved corrected plot to {save_path}")


if __name__ == "__main__":
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    models_to_analyze = {
        "N-gram": "ngram_surprisal",
        "Llama Base": "llama_base_surprisal",
        "Llama Instruct": "llama_instruct_surprisal"
    }

    data = load_and_clean_data(DATA_FILE)
    data = apply_length_correction(data)

    correlation_results = calculate_correlations(data, models_to_analyze)

    plot_model_comparison(correlation_results, RESULTS_DIR)
    plot_binned_surprisal(data, models_to_analyze, RESULTS_DIR)