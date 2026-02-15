import subprocess
import sys
import os

def run_phase(script_name):
    """
    Executes a script as a separate process and monitors for successful completion.
    """
    print(f"Starting Phase: {script_name}")
    try:
        # Run the script and wait for it to finish
        result = subprocess.run([sys.executable, script_name], check=True)
        if result.returncode == 0:
            print(f"Successfully completed {script_name}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error: {script_name} failed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {script_name} not found in the current directory.")
        sys.exit(1)

def main():
    """
    Orchestrates the full pipeline from data acquisition to final case study.
    """
    # Define the sequence of scripts
    pipeline_steps = [
        "download_data.py",      # Phase 1: Setup and data download
        "train_ngram.py",        # Phase 2: N-gram model training
        "run_llama.py",          # Phase 3: Llama model surprisals
        "analyze_results.py",    # Phase 4: Primary correlation analysis
        "residual_analysis.py",  # Phase 5: Length-corrected analysis
        "case_study.py"          # Phase 6: Specific visualization
    ]

    # Check for .env file if run_llama.py is in the pipeline
    if not os.path.exists(".env"):
        print("Warning: .env file not found. run_llama.py may fail without a token.")

    print("--- Starting Full Surprisal Analysis Pipeline ---\n")

    for script in pipeline_steps:
        run_phase(script)

    print("--- All Phases Complete ---")
    print("Results are available in the 'data' and 'results' directories.")

if __name__ == "__main__":
    main()