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
CONFIG_PATH = BASE_DIR / "config.json"
PROFILES_PATH = BASE_DIR / "model_profiles.json"
TORCH_SPEC = "torch>=2.6,<3.0"
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


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_config():
    return load_json(CONFIG_PATH)


def load_profiles():
    profiles = load_json(PROFILES_PATH)
    if not isinstance(profiles, dict) or not profiles:
        raise RuntimeError("model_profiles.json is empty or invalid.")
    return profiles


def resolve_model_config(config, profiles):
    model = dict(config.get("model", {}))
    profile_name = model.get("profile")

    if profile_name:
        if profile_name not in profiles:
            available = ", ".join(sorted(profiles))
            raise RuntimeError(
                f"Unknown profile '{profile_name}'. Available: {available}"
            )
        resolved = dict(profiles[profile_name])
        for key, value in model.items():
            if key != "profile":
                resolved[key] = value
        resolved["profile"] = profile_name
    else:
        resolved = model

    for key in ("id", "local_dir"):
        if key not in resolved:
            raise RuntimeError(f"Resolved model configuration is missing '{key}'.")

    resolved.setdefault("device", "auto")
    resolved.setdefault("dtype", "auto")
    resolved.setdefault("quantization", "none")
    return resolved


def select_profile(config, profile_name, profiles):
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise SystemExit(
            f"Unknown profile '{profile_name}'. Available profiles: {available}"
        )

    enable_thinking = bool(config.get("model", {}).get("enable_thinking", False))
    config["model"] = {
        "profile": profile_name,
        "enable_thinking": enable_thinking,
    }
    save_json(CONFIG_PATH, config)
    print(f"Selected profile written to config.json: {profile_name}")


def list_profiles(profiles):
    print("Available model profiles")
    print("========================")
    for name, profile in profiles.items():
        print(f"\n{name}")
        print(f"  {profile.get('label', '')}")
        print(f"  Model: {profile.get('id')}")
        print(f"  Device: {profile.get('device', 'auto')}")
        print(f"  dtype: {profile.get('dtype', 'auto')}")
        print(f"  Quantization: {profile.get('quantization', 'none')}")
        print(f"  Licence: {profile.get('license', 'check upstream')}")
        print(f"  Gated: {profile.get('gated', 'check upstream')}")
        note = profile.get("recommended_for")
        if note:
            print(f"  Recommended for: {note}")


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


def run_pip(arguments):
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(PACKAGES_DIR),
        *arguments,
    ]
    subprocess.run(command, check=True, cwd=BASE_DIR)


def install_base_packages():
    requirements = BASE_DIR / "requirements.txt"
    print("\nInstalling base Python dependencies locally...")
    print(f"Target: {PACKAGES_DIR}")
    run_pip(["-r", str(requirements)])


def install_torch(index_url=None):
    print("\nInstalling PyTorch locally...")
    args = [TORCH_SPEC]
    if index_url:
        print(f"PyTorch index: {index_url}")
        args = ["--index-url", index_url, TORCH_SPEC]
    run_pip(args)


def install_quantization_packages():
    requirements = BASE_DIR / "requirements-quantization.txt"
    print("\nInstalling optional quantization dependencies...")
    run_pip(["-r", str(requirements)])


def verify_torch_for_profile(model_config):
    sys.path.insert(0, str(PACKAGES_DIR))
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch could not be imported from packages/. "
            "Run bootstrap without --skip-torch."
        ) from exc

    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    requested_device = str(model_config.get("device", "auto")).lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "The selected profile requires CUDA, but the installed PyTorch build "
            "does not provide CUDA. Re-run bootstrap with --torch-index-url set "
            "to the CUDA wheel index recommended by https://pytorch.org/get-started/locally/ "
            "for this machine, or install a CUDA-enabled PyTorch build into packages/."
        )


def download_model(model_config):
    sys.path.insert(0, str(PACKAGES_DIR))

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub could not be imported from packages/. "
            "Run bootstrap without --skip-packages first."
        ) from exc

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
            "Prepare a repository-local Python runtime and download the selected model."
        )
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List built-in model profiles and exit.",
    )
    parser.add_argument(
        "--profile",
        help="Select a model profile and write it to config.json before bootstrap.",
    )
    parser.add_argument(
        "--torch-index-url",
        help=(
            "Optional PyTorch wheel index URL, typically used for a CUDA-enabled "
            "PyTorch build. Use the URL recommended by pytorch.org for the machine."
        ),
    )
    parser.add_argument(
        "--skip-packages",
        action="store_true",
        help="Do not install/update base packages/.",
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Do not install/update PyTorch in packages/.",
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

    profiles = load_profiles()
    if args.list_profiles:
        list_profiles(profiles)
        return

    print("local_llm_stack bootstrap")
    print("=========================")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Repository: {BASE_DIR}")

    prepare_directories()
    prepare_environment()
    config = load_config()

    if args.profile:
        select_profile(config, args.profile, profiles)
        config = load_config()

    model_config = resolve_model_config(config, profiles)
    print(f"Profile: {model_config.get('profile', 'custom')}")
    print(f"Model: {model_config['id']}")
    print(f"Device: {model_config.get('device', 'auto')}")
    print(f"Quantization: {model_config.get('quantization', 'none')}")

    if not args.skip_packages:
        install_base_packages()

    if not args.skip_torch:
        install_torch(args.torch_index_url)

    quantization = str(model_config.get("quantization", "none")).lower()
    if quantization in {"4bit", "8bit"} and not args.skip_packages:
        install_quantization_packages()

    verify_torch_for_profile(model_config)

    if not args.skip_model:
        download_model(model_config)

    print("\nBootstrap complete.")
    print("Run a smoke test with:")
    print("  python smoke_test.py")
    print("Run a benchmark with:")
    print("  python benchmark.py")
    print("Start the application with:")
    print("  python app.py")


if __name__ == "__main__":
    main()
