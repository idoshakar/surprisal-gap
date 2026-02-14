import os
import pandas as pd
from datasets import load_dataset
import requests
import io

# 1. Setup Directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
print(f"Created '{DATA_DIR}' directory.")

# ---------------------------------------------------------
# 2. Download Natural Stories (Corrected TSV)
# ---------------------------------------------------------
print("\nDownloading Natural Stories (Corrected TSV)...")

# The Raw URL for the file you found
url = "https://raw.githubusercontent.com/languageMIT/naturalstories/master/naturalstories_RTS/processed_RTs.tsv"

try:
    response = requests.get(url)
    response.raise_for_status()

    # Read TSV (sep='\t')
    df_human = pd.read_csv(io.BytesIO(response.content), sep='\t')

    # FILTERING:
    # 1. Remove outliers (Standard practice: 50ms < RT < 3000ms)
    original_count = len(df_human)
    df_human = df_human[(df_human['RT'] > 50) & (df_human['RT'] < 3000)]

    # 2. Create WORD LENGTH column (Our Baseline Feature)
    # The column might be named 'word' or 'Word'. We ensure it's string.
    if 'word' in df_human.columns:
        df_human['word_len'] = df_human['word'].astype(str).apply(len)
    elif 'Word' in df_human.columns:
        df_human['word'] = df_human['Word']  # Standardize to lowercase 'word'
        df_human['word_len'] = df_human['word'].astype(str).apply(len)

    save_path = os.path.join(DATA_DIR, "natural_stories_cleaned.csv")
    df_human.to_csv(save_path, index=False)
    print(f"Saved cleaned human data to: {save_path}")
    print(f"   (Filtered {original_count - len(df_human)} outlier rows)")
    print(f"   (Columns found: {list(df_human.columns)})")

except Exception as e:
    print(f"Error downloading Natural Stories: {e}")

# ---------------------------------------------------------
# 3. Download WikiText-2 (For N-gram Training)
# ---------------------------------------------------------
print("\nDownloading WikiText-2 (for N-gram training)...")
try:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    wiki_path = os.path.join(DATA_DIR, "wikitext_train.txt")
    with open(wiki_path, "w", encoding="utf-8") as f:
        for line in dataset["text"]:
            if line.strip():
                f.write(line)

    print(f"Saved WikiText training data to: {wiki_path}")
    print("\nPhase 1 Complete. You are ready for the models.")

except Exception as e:
    print(f"Error downloading WikiText: {e}")