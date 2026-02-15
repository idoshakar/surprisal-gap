import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_and_preprocess(file_path):
    """
    Loads the processed dataset and ensures critical metrics are numeric.
    Groups data by item, zone, and word to get mean values.
    """
    df = pd.read_csv(file_path)
    cols = ['RT', 'llama_base_surprisal', 'llama_instruct_surprisal']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df.groupby(['item', 'zone', 'word'])[cols].mean().reset_index()


def extract_case_study_data(word_data, item_id, target_word):
    """
    Filters data for a specific story and locates the target word context.
    Returns the target row and a window of surrounding words for plotting.
    """
    story_df = word_data[word_data['item'] == item_id].copy()
    story_df = story_df.sort_values('zone').reset_index()

    indices = story_df.index[story_df['word'].str.lower() == target_word.lower()].tolist()

    if not indices:
        return None, None, None

    green_idx = indices[0]
    target_idx = green_idx + 1
    target_row = story_df.iloc[target_idx]

    plot_start = max(0, target_idx - 5)
    plot_end = min(len(story_df), target_idx + 2)
    context_df = story_df.iloc[plot_start:plot_end].copy()

    return target_row, context_df, story_df.iloc[green_idx]['word']


def generate_case_study_plot(context_df, target_row, anchor_word, output_path):
    """
    Creates a dual-axis visualization comparing human RT (bars)
    with model surprisal (lines).
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    sns.barplot(data=context_df, x='word', y='RT', color='skyblue', alpha=0.8, ax=ax1, label='Human RT')
    ax1.set_ylabel('Human Reading Time (ms)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlabel('Sentence Segment', fontsize=12)
    ax1.set_xticklabels(context_df['word'], rotation=30, ha='right')

    ax2 = ax1.twinx()
    sns.lineplot(data=context_df, x='word', y='llama_base_surprisal', color='orange', marker='o', linewidth=3,
                 label='Base', ax=ax2)
    sns.lineplot(data=context_df, x='word', y='llama_instruct_surprisal', color='green', marker='o', linewidth=3,
                 label='Instruct', ax=ax2)

    ax2.set_ylabel('Model Surprisal (bits)', color='darkgreen', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkgreen')

    plt.title("Figure 5: Case Study - Token Prediction vs. World Knowledge Integration", fontsize=14)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path)

    full_sentence = " ".join(context_df['word'].astype(str).tolist())
    print(f"Phrase found: '{anchor_word} {target_row['word']}'")
    print(f"Context: \"...{full_sentence}...\"")
    print(f"Human RT on '{target_row['word']}': {target_row['RT']:.0f} ms")
    print(f"Llama Instruct Surprisal: {target_row['llama_instruct_surprisal']:.2f} bits")
    print(f"\nPlot saved as '{output_path}'")


if __name__ == "__main__":
    DATA_PATH = 'data/natural_stories_phase3.csv'
    OUTPUT_FILE = 'results/case_study_elvis.png'

    if not os.path.exists('results'):
        os.makedirs('results')

    processed_data = load_and_preprocess(DATA_PATH)
    target, context, anchor = extract_case_study_data(processed_data, item_id=5, target_word='green')

    if target is not None:
        generate_case_study_plot(context, target, anchor, OUTPUT_FILE)
    else:
        print("Error: Could not find the target word in the specified story.")