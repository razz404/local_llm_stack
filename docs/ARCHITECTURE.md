# Architecture

## Design goals

`local_llm_stack` is intentionally built around a few constraints:

1. Python is the application layer and invokes the model directly.
2. A separate Ollama or llama.cpp server is not required.
3. Dependencies, model files, caches and runtime data can remain inside the cloned repository.
4. Runtime inference works offline after bootstrap.
5. Model/hardware experiments should be explicit, repeatable and documented.
6. The code should stay small enough to understand and modify.

## Components

```text
                     config.json
                         |
                         v
                model_profiles.json
                         |
                         v
+----------------+   +----------------------+   +----------------------+
| Browser/CLI    |-->| app.py / benchmark.py|-->| local_ai/settings.py |
+----------------+   +----------------------+   +----------+-----------+
                                                         |
                                                         v
                                               +----------------------+
                                               | local_ai/engine.py   |
                                               | LocalLLM             |
                                               +----------+-----------+
                                                          |
                              +---------------------------+--------------------+
                              |                                                |
                              v                                                v
                    PyTorch + Transformers                           BitsAndBytesConfig
                              |                                      (optional 4/8-bit)
                              +---------------------------+--------------------+
                                                          |
                                                          v
                                                  local models/
```

## Runtime initialization

`app.py`, `smoke_test.py` and `benchmark.py` call:

```python
configure_local_environment(offline=True)
```

before importing third-party runtime packages.

`local_ai/runtime.py`:

- creates repository-local runtime directories
- prepends `packages/` to `sys.path`
- redirects Hugging Face, Torch and pip caches
- redirects temporary files
- enables `TRANSFORMERS_OFFLINE`
- enables `HF_HUB_OFFLINE`

## Model profiles

`model_profiles.json` contains reusable hardware/model combinations.

A profile can define model id, local model directory, device, dtype, quantization, bitsandbytes parameters, licence/access metadata and a human-readable recommendation.

`config.json` selects a profile by name. Explicit keys placed under `config.json -> model` override profile defaults.

`local_ai/profiles.py` resolves this into the concrete model configuration used by the rest of the application.

## Bootstrap

`bootstrap.py` handles the online preparation phase:

```text
base Python packages
        |
        v
packages/

PyTorch
  |
  +--> default wheel source
  |
  +--> optional --torch-index-url for CUDA wheel source

optional quantization profile
        |
        v
bitsandbytes

selected upstream model
        |
        v
snapshot_download()
        |
        v
models/<model>/
```

The correct CUDA-enabled PyTorch wheel is deliberately not hard-coded because PyTorch/CUDA compatibility changes over time.

## Model engine

`local_ai/engine.py` owns device validation, dtype selection, tokenizer loading, unquantized/quantized model loading, chat-template construction, input placement, history normalization, generation and benchmark helpers.

For `device: auto` the order is CUDA -> MPS -> CPU. Named CPU profiles set CPU explicitly for predictable benchmark comparisons.

For `dtype: auto`: CUDA uses BF16 when supported, otherwise FP16; MPS uses FP16; CPU uses FP32.

The v0.2 built-in quantized profiles use Hugging Face `BitsAndBytesConfig`. The project's supported quantized path is CUDA.

## Benchmark path

`benchmark.py` uses the same settings resolver and `LocalLLM` engine as the UI. It measures load time, time to first streamed text, generation time, token counts, process RSS, model memory footprint and CUDA peaks when available.

## Network boundary

The default Gradio UI binds to `127.0.0.1` with `share=False`. No external inference API is used by default. Bootstrap is the separate online phase for downloads.

## Why PyTorch/Transformers instead of llama.cpp?

The project is specifically a Python-first local LLM experiment:

```text
Python -> Transformers -> PyTorch -> model
```

rather than requiring a separate llama.cpp/Ollama inference service. Native compiled components still exist inside PyTorch/Transformers/quantization libraries.

## Supply-chain boundary

The repository MIT licence applies to project code. Python dependencies, model weights, model terms, gated access, community checkpoints and quantization libraries remain separate trust/licence inputs.

The project does not enable `trust_remote_code=True` by default.

## Future extension points

- local chat persistence
- more model families
- tested non-CUDA quantization backends
- RAG/document ingestion
- richer benchmark suites
- local tool calling
- desktop packaging
