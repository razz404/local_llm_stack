# local_llm_stack

A small, Python-first stack for running a local language model with **PyTorch + Hugging Face Transformers + Gradio**.

The project is intentionally simple:

- no Ollama requirement
- no llama.cpp server requirement
- no Docker requirement
- model inference is called directly from Python
- model files, Python packages, caches and runtime data can all live inside the cloned repository directory
- after bootstrap, the application can run offline
- the web UI listens on `127.0.0.1` only by default

The initial reference model is **Qwen3-0.6B**. It is small enough for experimenting on CPU-only Windows machines while still providing a useful local chat interface.

> This repository does **not** include model weights. `bootstrap.py` downloads them from the upstream model repository on first setup.

## Quick start

### 1. Clone

```cmd
git clone https://github.com/razz404/local_llm_stack.git
cd local_llm_stack
```

### 2. Bootstrap the local environment

```cmd
python bootstrap.py
```

The bootstrap script installs Python dependencies into the repository-local `packages/` directory and downloads the configured model into `models/`.

No virtual environment is required. This mode is useful on managed Windows computers where creating a venv may be blocked by Group Policy.

### 3. Start Local AI

```cmd
python app.py
```

or on Windows:

```cmd
scripts\run.cmd
```

The browser should open at:

```text
http://127.0.0.1:7860
```

## Repository layout

```text
local_llm_stack/
├── app.py                  # Gradio chat application
├── bootstrap.py            # Installs dependencies + downloads model
├── smoke_test.py           # Minimal terminal inference test
├── config.json             # Model, generation and UI settings
├── requirements.txt        # Python dependencies
├── local_ai/
│   ├── __init__.py
│   ├── runtime.py           # Local paths, caches and offline environment
│   ├── settings.py          # Config loader
│   └── engine.py            # Model loading and generation
├── scripts/
│   ├── bootstrap.cmd
│   └── run.cmd
├── docs/
│   ├── ARCHITECTURE.md
│   ├── INSTALLATION.md
│   ├── OFFLINE.md
│   └── TROUBLESHOOTING.md
├── packages/               # Created locally; ignored by Git
├── models/                 # Created locally; ignored by Git
├── cache/                  # Created locally; ignored by Git
├── chats/                  # Reserved for future local chat persistence
├── data/                   # Reserved for future local documents/RAG
├── logs/                   # Local logs
└── temp/                   # Local temporary files
```

## How local is it?

After `python bootstrap.py` has completed, the runtime is configured to use local directories for Hugging Face, Torch, pip and temporary data. `app.py` also enables Hugging Face/Transformers offline mode.

At runtime the intended data flow is:

```text
Browser on localhost
        |
        v
Gradio UI
        |
        v
Python application
        |
        v
Transformers + PyTorch
        |
        v
models/qwen3-0.6b/
```

There is no external inference API in the default configuration.

See [docs/OFFLINE.md](docs/OFFLINE.md) for the exact boundary and verification steps.

## Configuration

Edit `config.json` to change the local model path, generation settings, system prompt or UI port.

Example:

```json
{
  "model": {
    "id": "Qwen/Qwen3-0.6B",
    "local_dir": "models/qwen3-0.6b",
    "device": "auto",
    "enable_thinking": false
  },
  "generation": {
    "max_new_tokens": 300,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": true
  }
}
```

If you change `model.id` or `model.local_dir`, run `python bootstrap.py` again to download that model.

## CPU and GPU

`device: "auto"` selects CUDA when PyTorch reports CUDA as available; otherwise it uses CPU.

The first development system used a CPU-only PyTorch build and successfully ran Qwen3-0.6B. CPU model loading and generation can be slow, especially on managed business laptops. Streaming makes the UI responsive as soon as tokens are generated, but does not make the model itself compute faster.

## Verify the installation

Run the terminal smoke test:

```cmd
python smoke_test.py
```

It loads the configured model strictly from local files and asks it for a short Swedish response.

## Managed Windows / Group Policy environments

If this fails:

```cmd
python -m venv .venv
```

with an error such as `WinError 1260` or a Group Policy restriction, you can still use this project. The default bootstrap intentionally avoids venv and installs dependencies with pip's `--target` option into:

```text
packages/
```

The application prepends that directory to `sys.path` before importing third-party packages.

See [docs/INSTALLATION.md](docs/INSTALLATION.md) and [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Security defaults

The default UI configuration is deliberately conservative:

- host: `127.0.0.1`
- Gradio public sharing: disabled
- Transformers/Hugging Face offline mode: enabled during runtime
- no model weights or runtime data committed to Git

This is a convenience boundary, not a sandbox. A Python process and downloaded model files should still be treated according to the trust level of their source.

See [SECURITY.md](SECURITY.md).

## Current scope

Version 0.1 provides:

- repository-local Python dependencies
- repository-local model storage
- repository-local caches
- offline model loading after bootstrap
- Qwen3-0.6B reference configuration
- CPU/CUDA auto-selection
- Gradio chat UI
- streaming output
- configurable system prompt and generation settings
- terminal smoke test

Planned areas include chat persistence, model selection, document ingestion/RAG and optional local tools. These are intentionally not part of the first minimal stack.

## Documentation

- [Installation](docs/INSTALLATION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Offline operation](docs/OFFLINE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Security](SECURITY.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## License

No project license has been selected yet. Model files are not part of this repository and remain subject to the terms of their upstream publisher.