from pathlib import Path
import os
import sys


BASE_DIR = Path(__file__).resolve().parent.parent
PACKAGES_DIR = BASE_DIR / "packages"
CACHE_DIR = BASE_DIR / "cache"
TEMP_DIR = BASE_DIR / "temp"


def ensure_runtime_directories():
    directories = [
        PACKAGES_DIR,
        CACHE_DIR / "huggingface",
        CACHE_DIR / "torch",
        CACHE_DIR / "pip",
        BASE_DIR / "models",
        BASE_DIR / "data",
        BASE_DIR / "chats",
        BASE_DIR / "logs",
        TEMP_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def configure_local_environment(offline=True):
    """Configure imports and caches so runtime state stays inside the repository."""
    ensure_runtime_directories()

    packages = str(PACKAGES_DIR)
    if packages not in sys.path:
        sys.path.insert(0, packages)

    os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "huggingface" / "hub")
    os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
    os.environ["PIP_CACHE_DIR"] = str(CACHE_DIR / "pip")
    os.environ["TEMP"] = str(TEMP_DIR)
    os.environ["TMP"] = str(TEMP_DIR)

    if offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"


def resolve_repo_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path
