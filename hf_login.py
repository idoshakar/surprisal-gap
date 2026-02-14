from huggingface_hub import login

# Replace with your actual Write token from Hugging Face settings
HF_TOKEN = "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"

try:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("Login successful.")
except Exception as e:
    print(f"Login failed: {e}")