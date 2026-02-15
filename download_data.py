import os
import pandas as pd
from datasets import load_dataset
import requests
import io

DATA_DIR = "data"

def setup_directory(directory):
    """Creates the target directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)
    print(f"Created '{directory}' directory.")

def download_natural_stories(directory):
    """
    Downloads, filters, and cleans the Natural Stories dataset.
    Removes RT outliers and calculates word lengths.
    """
    print("\nDownloading Natural Stories (Corrected TSV)...")
    url = "https://raw.githubusercontent.com/languageMIT/naturalstories/master/naturalstories_RTS/processed_RTs.tsv"

    try:
        response = requests.get(url)
        response.raise_for_status()

        df_human = pd.read_csv(io.BytesIO(response.content), sep='\t')
        original_count = len(df_human)
        df_human = df_human[(df_human['RT'] > 50) & (df_human['RT'] < 3000)]

        if 'word' in df_human.columns:
            df_human['word_len'] = df_human['word'].astype(str).apply(len)
        elif 'Word' in df_human.columns:
            df_human['word'] = df_human['Word']
            df_human['word_len'] = df_human['word'].astype(str).apply(len)

        save_path = os.path.join(directory, "natural_stories_cleaned.csv")
        df_human.to_csv(save_path, index=False)
        print(f"Saved cleaned human data to: {save_path}")
        print(f"   (Filtered {original_count - len(df_human)} outlier rows)")
        print(f"   (Columns found: {list(df_human.columns)})")

    except Exception as e:
        print(f"Error downloading Natural Stories: {e}")

def download_wikitext(directory):
    """
    Downloads the WikiText-2 dataset and saves it as a raw text file
    for language model training.
    """
    print("\nDownloading WikiText-2 (for N-gram training)...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        wiki_path = os.path.join(directory, "wikitext_train.txt")
        with open(wiki_path, "w", encoding="utf-8") as f:
            for line in dataset["text"]:
                if line.strip():
                    f.write(line)

        print(f"Saved WikiText training data to: {wiki_path}")

    except Exception as e:
        print(f"Error downloading WikiText: {e}")

if __name__ == "__main__":
    setup_directory(DATA_DIR)
    download_natural_stories(DATA_DIR)
    download_wikitext(DATA_DIR)
