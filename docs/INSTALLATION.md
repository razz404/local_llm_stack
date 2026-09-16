# Installation

## Prerequisites

The default setup needs:

- Git
- Python 3.11 or newer
- pip available through `python -m pip`
- enough free disk space for Python packages, caches and the selected model
- internet access during the initial bootstrap

The project was initially developed on Windows with Python 3.13 and a CPU-only PyTorch installation. Other environments may work, but should be treated as separate test targets.

## Windows: recommended repository-local setup

Clone the repository:

```cmd
git clone https://github.com/razz404/local_llm_stack.git
cd local_llm_stack
```

Run:

```cmd
python bootstrap.py
```

or:

```cmd
scripts\bootstrap.cmd
```

The bootstrap script creates and uses these local directories:

```text
packages/
models/
cache/
data/
chats/
logs/
temp/
```

It then runs pip with `--target packages`, so no venv is needed.

### Why not require venv?

Some managed Windows systems allow Python and pip but block creation or execution of a virtual environment through Group Policy. A typical symptom is:

```text
Error: [WinError 1260] This program is blocked by group policy
```

The repository-local `packages/` approach avoids creating another Python executable and still keeps third-party modules inside the project directory.

## Start the application

```cmd
python app.py
```

or:

```cmd
scripts\run.cmd
```

By default the UI listens only on:

```text
127.0.0.1:7860
```

## Verify with a terminal-only test

```cmd
python smoke_test.py
```

This is useful for separating model/runtime problems from Gradio/UI problems.

## Bootstrap options

Install/update packages but do not download the model:

```cmd
python bootstrap.py --skip-model
```

Download/update the configured model without reinstalling packages:

```cmd
python bootstrap.py --skip-packages
```

## Optional traditional virtual environment

On an unmanaged computer, a normal venv can also be used:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python bootstrap.py --skip-packages
python app.py
```

The application still supports the repository-local `packages/` directory; if it is empty, normal interpreter/venv packages remain available through the rest of `sys.path`.

## Model selection

The default model is configured in `config.json`:

```json
"model": {
  "id": "Qwen/Qwen3-0.6B",
  "local_dir": "models/qwen3-0.6b",
  "device": "auto",
  "enable_thinking": false
}
```

To use another Transformers-compatible causal language model:

1. change `model.id`
2. change `model.local_dir`
3. run `python bootstrap.py --skip-packages`
4. start `python app.py`

Different models can require different tokenizer/template options or significantly more RAM/VRAM. The current stack is deliberately optimized for the small Qwen3 reference model, not every model on Hugging Face.

## CPU versus GPU

`device: "auto"` checks available acceleration at runtime:

1. CUDA
2. Apple MPS
3. CPU

The default `pip` installation installs the PyTorch build provided for your platform by the configured package index. If you need a particular NVIDIA/CUDA build, install the appropriate PyTorch package into `packages/` or a venv according to PyTorch's platform-specific instructions, then verify:

```cmd
python -c "import sys; sys.path.insert(0, 'packages'); import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Moving the clone

Project paths are relative to the repository root. You can therefore move or clone the project into another directory without editing hard-coded `D:\...` paths.

For example both of these are valid:

```text
D:\projekt\local-ai\
C:\Users\alice\source\local_llm_stack\
```

The only exception is an explicitly absolute `model.local_dir` value in `config.json`.
