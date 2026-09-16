import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PACKAGES_DIR = BASE_DIR / "packages"
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
RUNTIME_DIRS = [
    PACKAGES_DIR,
    CACHE_DIR / "huggingface",
    CACHE_DIR / "torch",
    CACHE_DIR / "pip",
    MODELS_DIR,
    BASE_DIR / "data",
    BASE_DIR / "chats",
    BASE_DIR / "logs",
    BASE_DIR / "temp",
]


def load_config():
    with (BASE_DIR / "config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_directories():
    for path in RUNTIME_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def prepare_environment():
    os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "huggingface" / "hub")
    os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
    os.environ["PIP_CACHE_DIR"] = str(CACHE_DIR / "pip")
    os.environ["TEMP"] = str(BASE_DIR / "temp")
    os.environ["TMP"] = str(BASE_DIR / "temp")


def install_packages():
    requirements = BASE_DIR / "requirements.txt"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(PACKAGES_DIR),
        "-r",
        str(requirements),
    ]

    print("\nInstalling Python dependencies locally...")
    print(f"Target: {PACKAGES_DIR}")
    subprocess.run(command, check=True, cwd=BASE_DIR)


def download_model(config):
    sys.path.insert(0, str(PACKAGES_DIR))

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub could not be imported from packages/. "
            "Run bootstrap without --skip-packages first."
        ) from exc

    model_config = config["model"]
    model_id = model_config["id"]
    model_dir = BASE_DIR / model_config["local_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\nDownloading model...")
    print(f"Source: {model_id}")
    print(f"Target: {model_dir}")

    snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,
    )

    print("Model download complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a repository-local Python runtime and download the configured model."
        )
    )
    parser.add_argument(
        "--skip-packages",
        action="store_true",
        help="Do not install/update packages/.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Do not download/update model files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11 or newer is required.")

    print("local_llm_stack bootstrap")
    print("=========================")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Repository: {BASE_DIR}")

    prepare_directories()
    prepare_environment()
    config = load_config()

    if not args.skip_packages:
        install_packages()

    if not args.skip_model:
        download_model(config)

    print("\nBootstrap complete.")
    print("Start the application with:")
    print("  python app.py")


if __name__ == "__main__":
    main()
