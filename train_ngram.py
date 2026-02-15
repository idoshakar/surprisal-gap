import os
import pandas as pd
import numpy as np
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk.tokenize import word_tokenize


def setup_nltk_resources():
    """Verify and download required NLTK tokenization resources."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')


def train_ngram_model(data_path, order):
    """
    Train a Laplace-smoothed N-gram model on text data.
    Returns the trained language model and the processed vocabulary.
    """
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            text_data = f.read()
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        exit()

    sentences = nltk.sent_tokenize(text_data)
    tokenized_text = [word_tokenize(sent.lower()) for sent in sentences]
    train_data, vocab = padded_everygram_pipeline(order, tokenized_text)

    lm = Laplace(order)
    lm.fit(train_data, vocab)
    return lm


def calculate_surprisal(df, lm, order):
    """
    Calculate word-level surprisal scores based on the language model.
    Resets context when a new item (story) is encountered.
    """
    surprisals = []
    context = []
    current_item = -1

    for index, row in df.iterrows():
        if row['item'] != current_item:
            context = []
            current_item = row['item']

        word = str(row['word']).lower()
        relevant_context = tuple(context[-(order - 1):])
        score = lm.score(word, relevant_context)

        # Convert probability to surprisal in bits
        if score > 0:
            surprisal = -np.log2(score)
        else:
            surprisal = 20.0

        surprisals.append(surprisal)
        context.append(word)

        if index % 1000 == 0:
            print(f"Processed {index}/{len(df)} words...", end='\r')

    return surprisals


if __name__ == "__main__":
    setup_nltk_resources()

    DATA_DIR = "data"
    NGRAM_ORDER = 5
    TRAIN_FILE = os.path.join(DATA_DIR, "wikitext_train.txt")
    TARGET_FILE = os.path.join(DATA_DIR, "natural_stories_cleaned.csv")

    print(f"Training {NGRAM_ORDER}-gram model...")
    language_model = train_ngram_model(TRAIN_FILE, NGRAM_ORDER)

    print("Calculating Surprisal...")
    try:
        df_stories = pd.read_csv(TARGET_FILE)
    except FileNotFoundError:
        print(f"Error: {TARGET_FILE} not found.")
        exit()

    df_stories['ngram_surprisal'] = calculate_surprisal(df_stories, language_model, NGRAM_ORDER)

    save_path = os.path.join(DATA_DIR, "natural_stories_phase2.csv")
    df_stories.to_csv(save_path, index=False)
    print(f"\nPhase 2 Complete. Results saved to: {save_path}")