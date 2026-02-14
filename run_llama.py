import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---
HF_TOKEN = "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"
DATA_DIR = "data"
INPUT_FILE = "natural_stories_phase2.csv"
OUTPUT_FILE = "natural_stories_phase3.csv"
WINDOW_SIZE = 512  # Limits memory usage by processing 512 tokens at a time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")


# --- HELPER: SLIDING WINDOW SURPRISAL ---
def calculate_surprisal_for_story(model, tokenizer, word_list):
    full_text = " ".join([str(w) for w in word_list])

    # Tokenize the whole story
    encodings = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True).to(device)
    input_ids = encodings["input_ids"]
    offsets = encodings["offset_mapping"][0].cpu().numpy()

    seq_len = input_ids.size(1)
    all_token_surprisals = np.zeros(seq_len)

    # Process in overlapping windows to ensure every token has context
    # We move through the sequence, calculating surprisal for the new tokens in each window
    for i in range(0, seq_len, WINDOW_SIZE // 2):
        begin_loc = max(i + WINDOW_SIZE // 2 - WINDOW_SIZE, 0)
        end_loc = min(i + WINDOW_SIZE // 2, seq_len)
        trg_len = end_loc - i

        input_window = input_ids[:, begin_loc:end_loc].to(device)
        target_window = input_window.clone()

        # We only want to calculate loss for the 'new' tokens in this window
        target_window[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_window, labels=target_window)

            # Extract log-likelihoods
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = target_window[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Map losses back to our global array
            loss_vals = token_losses.view(-1).cpu().numpy()
            start_idx = end_loc - len(loss_vals)
            # Only fill if the slot is empty or we have better context
            for j, loss_val in enumerate(loss_vals):
                actual_pos = start_idx + j
                if all_token_surprisals[actual_pos] == 0:
                    all_token_surprisals[actual_pos] = loss_val

    # Align back to words using character map
    char_surprisal_map = np.zeros(len(full_text))
    for idx, (start, end) in enumerate(offsets):
        if start == end: continue
        if idx < len(all_token_surprisals):
            char_surprisal_map[start:end] = all_token_surprisals[idx] / (end - start)

    aligned_surprisals = []
    current_pos = 0
    for word in word_list:
        word = str(word)
        start_idx = full_text.find(word, current_pos)
        if start_idx == -1:
            aligned_surprisals.append(0.0)
            continue
        end_idx = start_idx + len(word)
        aligned_surprisals.append(np.sum(char_surprisal_map[start_idx:end_idx]))
        current_pos = end_idx

    return aligned_surprisals


# --- MAIN PIPELINE ---
def run_pipeline():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE))
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    story_ids = df['item'].unique()
    df['llama_base_surprisal'] = 0.0
    df['llama_instruct_surprisal'] = 0.0

    models_config = [
        ("meta-llama/Llama-3.2-1B", "llama_base_surprisal"),
        ("meta-llama/Llama-3.2-1B-Instruct", "llama_instruct_surprisal")
    ]

    for model_name, col_name in models_config:
        print(f"\nLoading Model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            torch_dtype=torch.float16,  # Vital for 6GB VRAM
            device_map="auto"
        )
        model.eval()

        print(f"Calculating {col_name}...")
        all_results = {}
        for item_id in story_ids:
            story_mask = df['item'] == item_id
            story_df = df[story_mask]
            word_list = story_df['word'].tolist()

            surprisals = calculate_surprisal_for_story(model, tokenizer, word_list)
            for idx, val in zip(story_df.index, surprisals):
                all_results[idx] = val
            print(f"   Finished Story {item_id}", end='\r')

        df[col_name] = df.index.map(all_results)
        del model
        del tokenizer
        torch.cuda.empty_cache()

    df.to_csv(os.path.join(DATA_DIR, OUTPUT_FILE), index=False)
    print(f"\nPhase 3 Complete.")


if __name__ == "__main__":
    run_pipeline()