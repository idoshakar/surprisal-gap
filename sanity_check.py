import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

# --- CONFIGURATION ---
DATA_FILE = "data/natural_stories_phase3.csv"
RESULTS_DIR = "results"

print("--- 🧠 Running Phase 5: Length Control Sanity Check ---")

# 1. Load Data
df = pd.read_csv(DATA_FILE)
original_len = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['RT', 'word_len', 'llama_base_surprisal'])

# 2. Residualize RT against Word Length
# We fit a simple line: RT = a * Length + b
# Then we subtract that line to get "Residual RT" (how much slower/faster than expected for this length)
slope, intercept, _, _, _ = linregress(df['word_len'], df['RT'])
df['RT_corrected'] = df['RT'] - (df['word_len'] * slope + intercept)

print(f"Removed linear effect of length. (Slope: {slope:.2f} ms/char)")

# 3. New Correlations
models = {
    "N-gram": "ngram_surprisal",
    "Llama Base": "llama_base_surprisal",
    "Llama Instruct": "llama_instruct_surprisal"
}

print("\n--- Corrected Correlations (Length Removed) ---")
results = []
for name, col in models.items():
    r_val, p_val = pearsonr(df['RT_corrected'], df[col])
    results.append({"Model": name, "Corrected_r": r_val})
    print(f"{name}: r = {r_val:.4f}")

# 4. Plot the "Corrected" Gap
plt.figure(figsize=(10, 6))
for name, col in models.items():
    if name == "N-gram": continue  # Skip N-gram as it's too noisy for this plot

    # Bin by Model Surprisal
    df[f'{col}_bin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')

    # Plot against Corrected RT
    binned_data = df.groupby(f'{col}_bin')['RT_corrected'].mean().reset_index()
    sns.lineplot(x=binned_data.index, y=binned_data['RT_corrected'], label=name, marker='o')

plt.title("Surprisal Effect after Removing Word Length")
plt.xlabel("Model Surprisal Decile")
plt.ylabel("Corrected Human RT (ms)")
plt.axhline(0, color='black', linestyle='--', alpha=0.3)  # The "expected" speed line
plt.savefig(f"{RESULTS_DIR}/sanity_check_corrected_gap.png")
print(f"\nSaved corrected plot to {RESULTS_DIR}/sanity_check_corrected_gap.png")