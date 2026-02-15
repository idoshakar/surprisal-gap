# Surprisal Analysis Pipeline: N-gram vs. Llama 3.2

This repository contains a research pipeline designed to compare human linguistic processing (measured via reading times) with computational surprisal metrics. It evaluates a traditional N-gram baseline against modern transformer models, specifically **Llama 3.2 1B** (Base and Instruct versions).

---

## Hardware and Performance Note

The transformer models used in this pipeline are **computationally heavy**. This project was developed and tested using an **NVIDIA RTX 3050 GPU**.

* **GPU Acceleration:** CUDA is required for reasonable inference speeds with the Llama models.
* **VRAM Management:** The pipeline utilizes `torch.float16` precision and automated device mapping to ensure the models fit within the 4GB/6GB VRAM constraints of mid-range hardware like the RTX 3050.
* **Expectation:** Running Phase 3 (`run_llama.py`) on a CPU is not recommended as it will be significantly slower.

---

## Prerequisites

### Hugging Face Access

To use the Llama 3.2 models, you must have a Hugging Face account and request access to the **Meta Llama 3.2** model family. Once access is granted, generate a **User Access Token** with read permissions from your Hugging Face settings.

### Local Environment

1. **Python 3.8+** is required.
2. Install the necessary dependencies:
```bash
pip install -r requirements.txt

```


3. Create a `.env` file in the root directory to store your credentials:
```text
TOKEN=your_huggingface_token_here

```



---

## Data and Directories

### Data Storage

The project relies on two main datasets:

* **Natural Stories Corpus:** Human self-paced reading times from the [MIT Natural Stories repository](https://github.com/languageMIT/naturalstories).
* **WikiText-2:** Used for training the N-gram baseline.

The `data/` directory is not included in this repository due to its size. The `download_data.py` script will automatically fetch and clean these files. If you encounter issues acquiring the data via the script, please contact **idoshakar@gmail.com**.

### Directory Structure

* **`data/`**: Stores raw and processed datasets (generated automatically).
* **`results/`**: Stores all generated plots and statistical summaries (generated automatically).

---

## Execution Guide

To run the complete analysis from data acquisition to final visualization, execute the master script:

```bash
python main.py

```

### Pipeline Overview

The pipeline consists of six distinct phases:

1. **Phase 1: download_data.py** – Fetches the Natural Stories TSV and WikiText-2, performs outlier filtering (RTs between 50ms and 3000ms), and calculates word lengths.
2. **Phase 2: train_ngram.py** – Trains a 5-gram Laplace-smoothed language model to provide a traditional baseline for surprisal.
3. **Phase 3: run_llama.py** – Conducts inference using Llama 3.2 models. It uses a sliding window approach and character-offset mapping to align token-level surprisal with the original word-level reading times.
4. **Phase 4: analyze_results.py** – Generates primary correlation plots (Pearson r) between model surprisal and raw human reading times.
5. **Phase 5: residual_analysis.py** – Performs length correction by removing the linear effect of word length from RTs and recalculating correlations on the residuals.
6. **Phase 6: case_study.py** – Produces a qualitative visualization of a specific segment (Story 5) to observe how models handle world-knowledge integration compared to humans.

---

## Security Note

The `.env` file contains sensitive API tokens. **Do not** commit this file to version control. A `.gitignore` is recommended to exclude both `.env` and the `data/` folder.