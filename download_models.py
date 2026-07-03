import os
import subprocess
import sys

from huggingface_hub import snapshot_download, hf_hub_download


def install_spacy_model():
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("✓ spaCy model already installed.")
    except Exception:
        print("Downloading spaCy model...")
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        )


def download_florence():
    print("\nDownloading Florence-2-base...")

    snapshot_download(
        repo_id="microsoft/Florence-2-base",
        local_dir="models/Florence-2-base",
        local_dir_use_symlinks=False,
        resume_download=True
    )

    print("✓ Florence downloaded.")


def download_llama():
    print("\nDownloading Llama-3.2-1B GGUF...")

    os.makedirs("models/llm", exist_ok=True)

    hf_hub_download(
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        local_dir="models/llm"
    )

    print("✓ Llama downloaded.")


if __name__ == "__main__":

    download_florence()
    download_llama()
    install_spacy_model()

    print("\n======================================")
    print("All models downloaded successfully.")
    print("======================================")