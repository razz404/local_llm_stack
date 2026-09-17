# local_llm_stack

A small, Python-first stack for running local language models with **PyTorch + Hugging Face Transformers + Gradio**.

The project is intentionally simple:

- no Ollama requirement
- no llama.cpp server requirement
- no Docker requirement
- model inference is called directly from Python
- model files, Python packages, caches and runtime data can all live inside the cloned repository directory
- after bootstrap, the application can run offline
- the web UI listens on `127.0.0.1` only by default
- v0.2 adds model profiles, optional 4/8-bit quantization and a local benchmark

The default profile is **Qwen3-0.6B**. It is deliberately small so the project can be tested on CPU-only Windows laptops with 16 GB RAM.

> Model weights are not included in this repository. `bootstrap.py` downloads the selected model from its upstream Hugging Face repository.

## Quick start

### CPU / default profile

```cmd
git clone https://github.com/razz404/local_llm_stack.git
cd local_llm_stack

python bootstrap.py
python smoke_test.py
python benchmark.py
python app.py
```

No virtual environment is required. Dependencies are installed into the repository-local `packages/` directory. This is useful on managed Windows computers where creating a venv may be blocked by Group Policy.

### See available model profiles

```cmd
python bootstrap.py --list-profiles
```

Built-in v0.2 profiles include:

| Profile | Model | Intended use |
| --- | --- | --- |
| `qwen3-0.6b-cpu` | Qwen3-0.6B | CPU baseline and development |
| `qwen3-1.7b-cpu` | Qwen3-1.7B | Better CPU quality, slower |
| `qwen3-4b-cuda-bf16` | Qwen3-4B | NVIDIA GPU BF16 baseline |
| `qwen3-8b-cuda-8bit` | Qwen3-8B | NVIDIA GPU, bitsandbytes 8-bit |
| `qwen3-8b-cuda-4bit` | Qwen3-8B | NVIDIA GPU, bitsandbytes 4-bit |

Select a profile during bootstrap:

```cmd
python bootstrap.py --profile qwen3-1.7b-cpu
```

For CUDA profiles, use a CUDA-enabled PyTorch build suitable for the machine. The project deliberately does not hard-code one CUDA wheel channel because PyTorch/CUDA compatibility changes over time:

```cmd
python bootstrap.py ^
  --profile qwen3-8b-cuda-4bit ^
  --torch-index-url <PYTORCH_CUDA_INDEX_URL>
```

Get the current wheel command/index for the target machine from the official PyTorch installation selector.

## Quantization

The 4-bit and 8-bit profiles use Hugging Face Transformers with `BitsAndBytesConfig`.

v0.2 intentionally treats quantization as an **optional CUDA profile**, not as the default path. The bootstrap installs `bitsandbytes` only when the selected profile requires it.

4-bit makes larger models practical in limited VRAM, but it is not free:

- quality can change compared with BF16/FP16
- real memory use is higher than the raw bits-per-parameter estimate
- long context consumes additional VRAM through KV cache
- backend and hardware compatibility matter
- community-produced quantized checkpoints are separate supply-chain artefacts
- quantization does not remove the original model licence or terms

See [Choosing a local LLM](docs/MODEL_SELECTION.md).

## Benchmark

Run:

```cmd
python benchmark.py
```

The benchmark reports the selected profile, model, device, dtype, quantization, model load time, time to first streamed text, prompt/output token counts, approximate output tokens per second, process RAM and CUDA peak memory when available.

Write a machine-readable result:

```cmd
python benchmark.py --json benchmarks\my-machine.json
```

Benchmark numbers are meaningful only when the **same prompt, generation settings, profile and hardware conditions** are compared.

See [Benchmarking](docs/BENCHMARKING.md).

## Repository layout

```text
local_llm_stack/
├── app.py
├── bootstrap.py
├── benchmark.py
├── smoke_test.py
├── config.json
├── model_profiles.json
├── requirements.txt
├── requirements-quantization.txt
├── local_ai/
│   ├── __init__.py
│   ├── runtime.py
│   ├── profiles.py
│   ├── settings.py
│   └── engine.py
├── scripts/
│   ├── bootstrap.cmd
│   └── run.cmd
├── docs/
│   ├── ARCHITECTURE.md
│   ├── INSTALLATION.md
│   ├── MODEL_SELECTION.md
│   ├── BENCHMARKING.md
│   ├── OFFLINE.md
│   └── TROUBLESHOOTING.md
├── packages/
├── models/
├── cache/
├── chats/
├── data/
├── logs/
└── temp/
```

## How local is it?

After bootstrap, runtime paths are redirected into the repository and Hugging Face/Transformers offline mode is enabled when `app.py`, `smoke_test.py` or `benchmark.py` starts.

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
local models/
```

There is no external inference API in the default configuration.

Bootstrap itself requires network access when packages or model files must be downloaded. Gated models may also require a Hugging Face account and authentication.

See [Offline operation](docs/OFFLINE.md).

## Configuration

`config.json` selects a model profile:

```json
{
  "model": {
    "profile": "qwen3-0.6b-cpu",
    "enable_thinking": false
  }
}
```

Profile details live in `model_profiles.json`. Explicit fields added under `config.json -> model` override profile defaults, which is useful for experiments.

Example:

```json
{
  "model": {
    "profile": "qwen3-4b-cuda-bf16",
    "dtype": "float16",
    "enable_thinking": false
  }
}
```

Prefer adding a reusable profile to `model_profiles.json` instead of accumulating machine-specific overrides.

## Model licences and Hugging Face access

The repository code is MIT licensed. Downloaded models are not.

Before selecting another model, check:

- the upstream model licence
- whether commercial or redistribution rights meet your use case
- whether the model is gated
- whether an access agreement or account is required
- whether a community checkpoint preserves upstream notices and provenance
- whether custom model code is required

The built-in Qwen3 profiles use official upstream repositories and record their licence metadata for convenience, but the upstream model card remains authoritative.

See [Choosing a local LLM](docs/MODEL_SELECTION.md).

## Managed Windows / Group Policy environments

If:

```cmd
python -m venv .venv
```

fails with `WinError 1260` or a Group Policy restriction, that does not block this project. `bootstrap.py` uses pip `--target` and keeps dependencies in `packages/`.

See [Installation](docs/INSTALLATION.md) and [Troubleshooting](docs/TROUBLESHOOTING.md).

## Security defaults

- UI host: `127.0.0.1`
- Gradio public sharing: disabled
- offline mode at runtime
- model weights and runtime data are excluded from Git
- no Hugging Face token is stored in project configuration
- `trust_remote_code=True` is not enabled

This is a convenience boundary, not a sandbox. Models and Python dependencies remain software supply-chain inputs.

See [SECURITY.md](SECURITY.md).

## Documentation

- [Installation](docs/INSTALLATION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Choosing a local LLM](docs/MODEL_SELECTION.md)
- [Benchmarking](docs/BENCHMARKING.md)
- [Offline operation](docs/OFFLINE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Security](SECURITY.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## License

This project is released under the [MIT License](LICENSE).

You are free to use, copy, modify, merge, publish, distribute, sublicense and sell copies of this project's code under the terms of the MIT License.

Model weights and third-party dependencies are not relicensed by this repository. They remain subject to the licences and terms of their respective upstream publishers.
