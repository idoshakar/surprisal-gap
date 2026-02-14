import os
import pandas as pd
import numpy as np
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk.tokenize import word_tokenize

# 1. Setup NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

DATA_DIR = "data"
NGRAM_ORDER = 5  # 5-gram = predicts word based on previous 4

# ---------------------------------------------------------
# Step 1: Train the N-gram Model
# ---------------------------------------------------------
print(f"Training {NGRAM_ORDER}-gram model on WikiText-2...")

try:
    with open(os.path.join(DATA_DIR, "wikitext_train.txt"), "r", encoding="utf-8") as f:
        text_data = f.read()
except FileNotFoundError:
    print("Error: 'wikitext_train.txt' not found.")
    exit()

# Tokenize the text into sentences, then words
sentences = nltk.sent_tokenize(text_data)
tokenized_text = [word_tokenize(sent.lower()) for sent in sentences]

# Preprocess for N-gram training
train_data, vocab = padded_everygram_pipeline(NGRAM_ORDER, tokenized_text)

# Initialize and Train
lm = Laplace(NGRAM_ORDER)
lm.fit(train_data, vocab)

print(f"Model Trained. Vocab size: {len(lm.vocab)}")

# ---------------------------------------------------------
# Step 2: Calculate Surprisal on Natural Stories
# ---------------------------------------------------------
print("Calculating Surprisal on Natural Stories...")

try:
    df = pd.read_csv(os.path.join(DATA_DIR, "natural_stories_cleaned.csv"))
except FileNotFoundError:
    print("Error: 'natural_stories_cleaned.csv' not found.")
    exit()

surprisals = []
context = []

# Reset context when story changes
current_item = -1

for index, row in df.iterrows():
    # Check if we moved to a new story
    if row['item'] != current_item:
        context = []
        current_item = row['item']

    word = str(row['word']).lower()

    # 1. Ask the model
    relevant_context = tuple(context[-(NGRAM_ORDER - 1):])
    score = lm.score(word, relevant_context)

    # 2. Convert to Surprisal (bits): -log2(probability)
    if score > 0:
        surprisal = -np.log2(score)
    else:
        surprisal = 20.0

    surprisals.append(surprisal)

    # 3. Update context
    context.append(word)

    if index % 1000 == 0:
        print(f"   Processed {index}/{len(df)} words...", end='\r')

# ---------------------------------------------------------
# Step 3: Save Results
# ---------------------------------------------------------
df['ngram_surprisal'] = surprisals

save_path = os.path.join(DATA_DIR, "natural_stories_phase2.csv")
df.to_csv(save_path, index=False)

print("\nPhase 2 Complete.")
print(f"Saved new dataset with N-gram scores to: {save_path}")