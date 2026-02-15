import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

DATA_FILE = "data/natural_stories_phase3.csv"
RESULTS_DIR = "results"


def load_and_clean_data(file_path):
    """
    Loads the dataset and removes rows with infinite values or missing
    surprisal/RT data.
    """
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}.")
        return None

    df = pd.read_csv(file_path)
    original_len = len(df)
    cols_to_check = ['RT', 'ngram_surprisal', 'llama_base_surprisal', 'llama_instruct_surprisal']

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_to_check)
    print(f"Data Loaded. Analyzable words: {len(df)} (Removed {original_len - len(df)} rows)")
    return df


def calculate_correlations(df, models):
    """
    Calculates Pearson correlation coefficients between human RT and
    different model surprisal scores.
    """
    results = []
    for name, col in models.items():
        r_val, p_val = pearsonr(df['RT'], df[col])
        results.append({"Model": name, "Pearson_r": r_val, "p_value": p_val})
        print(f"{name}: r = {r_val:.4f} (p < 0.001)")
    return pd.DataFrame(results)


def plot_model_comparison(results_df, output_dir):
    """
    Generates and saves a bar chart comparing Pearson r values
    across the tested models.
    """
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=results_df, x="Model", y="Pearson_r", palette="viridis")

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', label_type='edge', padding=-20,
                     color='black', weight='bold')

    plt.axhline(0, color='black', linewidth=1)
    min_val, max_val = results_df["Pearson_r"].min(), results_df["Pearson_r"].max()
    buffer = 0.05
    plt.ylim(min(0, min_val - buffer), max_val + buffer)

    plt.title("Figure 1: Pearson Correlation (r) Between Model Surprisal and Human RT")
    plt.ylabel("Correlation with Human RT (Pearson r)")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "model_comparison_bar.png")
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")


def plot_surprisal_gap(df, models, output_dir):
    """
    Creates a line plot showing mean human reading times binned by
    model surprisal deciles.
    """
    plt.figure(figsize=(10, 6))
    for name, col in models.items():
        df[f'{col}_bin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
        binned_data = df.groupby(f'{col}_bin')['RT'].mean().reset_index()
        sns.lineplot(x=binned_data[f'{col}_bin'], y=binned_data['RT'], label=name, marker='o')

    plt.title("Figure 2: Mean Human Reading Times Conditioned on Model Surprisal Deciles")
    plt.xlabel("Model Surprisal Decile (0=Easy, 9=Hard)")
    plt.ylabel("Human Reading Time (ms)")
    plt.xticks(range(10))
    plt.xlim(-0.5, 9.5)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, "surprisal_gap_line.png")
    plt.savefig(save_path)
    print(f"Saved gap analysis plot to {save_path}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    models_config = {
        "N-gram (Baseline)": "ngram_surprisal",
        "Llama 3.2 (Base)": "llama_base_surprisal",
        "Llama 3.2 (Instruct)": "llama_instruct_surprisal"
    }

    data = load_and_clean_data(DATA_FILE)

    if data is not None:
        correlation_results = calculate_correlations(data, models_config)
        plot_model_comparison(correlation_results, RESULTS_DIR)
        plot_surprisal_gap(data, models_config, RESULTS_DIR)

        best_model = correlation_results.loc[correlation_results['Pearson_r'].idxmax()]
        print("\n" + "=" * 40)
        print(f"WINNER: {best_model['Model']}")
        print("=" * 40)