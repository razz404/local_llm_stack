# Installation

`local_llm_stack` keeps its runtime inside the cloned repository. A Python virtual environment is optional, not required.

## Requirements

- Python 3.11 or newer
- enough free disk space for Python packages and the selected model
- network access during initial package/model download
- for CUDA profiles: a supported NVIDIA GPU, driver and CUDA-enabled PyTorch build

## CPU setup

```cmd
git clone https://github.com/razz404/local_llm_stack.git
cd local_llm_stack
python bootstrap.py
```

The default `qwen3-0.6b-cpu` profile is intended to work on CPU-only machines.

Test:

```cmd
python smoke_test.py
python benchmark.py
```

Start the UI:

```cmd
python app.py
```

## Choosing a profile

List profiles without installing anything:

```cmd
python bootstrap.py --list-profiles
```

Select one:

```cmd
python bootstrap.py --profile qwen3-1.7b-cpu
```

The selection is written to `config.json`, so later runs of `app.py`, `smoke_test.py` and `benchmark.py` use the same model.

## CUDA setup

CUDA profiles require a PyTorch build that actually exposes CUDA.

Because the correct PyTorch wheel channel depends on the current PyTorch release, supported CUDA versions and the target driver, the repository does not guess or pin a CUDA channel.

1. Use the official PyTorch installation selector for the target machine.
2. Copy the appropriate wheel index URL.
3. Bootstrap with that URL.

Example:

```cmd
python bootstrap.py ^
  --profile qwen3-4b-cuda-bf16 ^
  --torch-index-url <PYTORCH_CUDA_INDEX_URL>
```

For 4/8-bit:

```cmd
python bootstrap.py ^
  --profile qwen3-8b-cuda-4bit ^
  --torch-index-url <PYTORCH_CUDA_INDEX_URL>
```

Bootstrap verifies `torch.cuda.is_available()` when the selected profile explicitly requires CUDA and stops with an actionable error if the installed build is CPU-only.

## Quantization packages

`bitsandbytes` is optional. It is installed from `requirements-quantization.txt` only when the selected profile uses `4bit` or `8bit`.

The project's supported v0.2 quantized path is CUDA. Upstream bitsandbytes supports additional backends, but those are not yet tested/supported here.

## Repository-local directories

Bootstrap creates:

```text
packages/
models/
cache/
data/
chats/
logs/
temp/
```

They are all local to the repository and runtime directories are excluded from version control as appropriate.

## Group Policy / WinError 1260

A managed Windows environment may block:

```cmd
python -m venv .venv
```

The default bootstrap does not require venv. It uses:

```text
pip install --target packages/
```

and the application adds that folder to `sys.path` before importing third-party packages.

## Gated Hugging Face models

The built-in profiles are intended to be zero-friction official Qwen profiles. If you create a profile for a gated model, bootstrap may fail until you authenticate and accept the upstream terms.

Typical Hugging Face authentication:

```cmd
hf auth login
```

Never put tokens in `config.json`, `model_profiles.json`, batch files or Git.

## Advanced bootstrap switches

```text
--list-profiles
--profile NAME
--torch-index-url URL
--skip-packages
--skip-torch
--skip-model
```

`--skip-*` switches are useful when iterating on an already prepared repository, but they assume the skipped component is already correct.
