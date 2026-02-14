import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# --- CONFIGURATION ---
DATA_FILE = "data/natural_stories_phase3.csv"
RESULTS_DIR = "results"
import os

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Load and Clean Data
# ---------------------------------------------------------
print("Loading data...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print("Error: Could not find the Phase 3 CSV. Make sure you ran run_llama.py!")
    exit()

# Filter out bad data (NaNs or infinite values from log(0))
original_len = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=['RT', 'ngram_surprisal', 'llama_base_surprisal', 'llama_instruct_surprisal'])
print(f"Data Loaded. Analyzable words: {len(df)} (Removed {original_len - len(df)} bad rows)")

# ---------------------------------------------------------
# 2. The Main Test: Correlations
# ---------------------------------------------------------
print("\n--- 📊 Main Result: Correlations with Human RT ---")
models = {
    "N-gram (Baseline)": "ngram_surprisal",
    "Llama 3.2 (Base)": "llama_base_surprisal",
    "Llama 3.2 (Instruct)": "llama_instruct_surprisal"
}

results = []
for name, col in models.items():
    # Pearson (Linear relationship)
    r_val, p_val = pearsonr(df['RT'], df[col])
    results.append({"Model": name, "Pearson_r": r_val, "p_value": p_val})
    print(f"{name}: r = {r_val:.4f} (p < 0.001)")

results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# 3. Visualization: Correlation Comparison
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Model", y="Pearson_r", palette="viridis")
plt.title("Which Model Predicts Human Reading Best?")
plt.ylabel("Correlation with Human RT (Pearson r)")
plt.ylim(0, max(results_df["Pearson_r"]) + 0.05)  # Add some headroom
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/model_comparison_bar.png")
print(f"\n✅ Saved comparison plot to {RESULTS_DIR}/model_comparison_bar.png")

# ---------------------------------------------------------
# 4. Visualization: The "Surprisal Gap" (Binned Means)
# ---------------------------------------------------------
# This plot shows HOW the models fail.
# We group words by difficulty (Model Surprisal) and calculate the average Human RT.
print("\n--- 📉 Generating 'Surprisal Gap' Plot ---")

plt.figure(figsize=(10, 6))

# Create bins for surprisal
for name, col in models.items():
    # Bin the data into 10 quantiles (deciles)
    df[f'{col}_bin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')

    # Calculate mean RT for each bin
    binned_data = df.groupby(f'{col}_bin')['RT'].mean().reset_index()

    # Plot
    sns.lineplot(x=binned_data.index, y=binned_data['RT'], label=name, marker='o')

plt.title("The Surprisal Gap: Do models predict the hardest words?")
plt.xlabel("Model Surprisal Decile (0=Easy, 9=Hard)")
plt.ylabel("Human Reading Time (ms)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{RESULTS_DIR}/surprisal_gap_line.png")
print(f"✅ Saved gap analysis plot to {RESULTS_DIR}/surprisal_gap_line.png")

# ---------------------------------------------------------
# 5. Interpretation
# ---------------------------------------------------------
best_model = results_df.loc[results_df['Pearson_r'].idxmax()]
print("\n" + "=" * 40)
print(f"🏆 WINNER: {best_model['Model']}")
print("=" * 40)
print("Interpretation Guide:")
print("1. If Instruct > Base: 'Alignment' helps predict human processing.")
print("2. If Base > Instruct: Raw statistical probability is a better predictor.")
print("3. Check 'surprisal_gap_line.png': Does the line go FLAT at the top?")
print("   If yes, that is the 'Underestimation' your TA talked about.")
print("   (The model thinks it's hard, but humans find it MUCH harder).")