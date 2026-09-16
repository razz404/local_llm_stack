# Architecture

## Design goals

The stack is intentionally built around a few constraints:

1. Python is the application layer and the model is invoked directly from Python.
2. A separate Ollama or llama.cpp server is not required.
3. A clone can keep its dependencies, model files, caches and runtime data inside its own directory.
4. Runtime model loading works without internet access after bootstrap.
5. The first implementation should stay small enough to understand and modify.

## Components

```text
+-------------------------+
| Browser                 |
| http://127.0.0.1:7860   |
+------------+------------+
             |
             v
+-------------------------+
| Gradio ChatInterface    |
| app.py                  |
+------------+------------+
             |
             v
+-------------------------+
| LocalLLM                |
| local_ai/engine.py      |
+------------+------------+
             |
             v
+-------------------------+
| Transformers + PyTorch  |
| packages/               |
+------------+------------+
             |
             v
+-------------------------+
| Local model files       |
| models/qwen3-0.6b/      |
+-------------------------+
```

## Runtime initialization

`app.py` first imports only `local_ai.runtime`, which uses Python's standard library. It then calls:

```python
configure_local_environment(offline=True)
```

This happens before Gradio, Torch or Transformers are imported.

The runtime function:

- creates local runtime directories
- prepends `packages/` to `sys.path`
- points Hugging Face cache to `cache/huggingface/`
- points Torch cache to `cache/torch/`
- points pip cache to `cache/pip/`
- points temporary file variables to `temp/`
- sets `TRANSFORMERS_OFFLINE=1`
- sets `HF_HUB_OFFLINE=1`

## Bootstrap phase

`bootstrap.py` is the only normal phase that requires network access.

It performs two tasks:

```text
requirements.txt
      |
      v
pip --target packages/

Qwen/Qwen3-0.6B
      |
      v
snapshot_download()
      |
      v
models/qwen3-0.6b/
```

Because `packages/`, `models/` and `cache/` are ignored by Git, a clone remains small while each machine keeps its own runtime state.

## Model engine

`local_ai/engine.py` owns:

- device selection
- tokenizer loading
- model loading
- chat-template construction
- history normalization
- one-shot generation
- streaming generation

The reference model is loaded with `local_files_only=True`.

For chat streaming, the engine uses `TextIteratorStreamer` and runs `model.generate()` on a background thread. The Gradio generator yields progressively larger response strings as text arrives.

## Device selection

With `device: "auto"` the current order is:

```text
CUDA -> MPS -> CPU
```

The current dtype policy is:

```text
CUDA -> float16
CPU/MPS -> float32
```

This policy favors simplicity and compatibility for the first version rather than maximum performance.

## Configuration boundary

`config.json` contains values a user is expected to change without editing Python source:

- upstream model ID
- local model directory
- device preference
- thinking mode flag
- generation parameters
- system prompt
- UI host/port/title

## Network boundary

The default runtime binds Gradio to:

```text
127.0.0.1
```

and sets:

```python
share=False
```

That means the application is not intentionally exposed to the LAN and does not request a Gradio public sharing tunnel.

See `OFFLINE.md` and `SECURITY.md` for limitations and verification.

## Why PyTorch/Transformers instead of llama.cpp?

The purpose of this repository is specifically to explore a Python-first local LLM stack. PyTorch and Transformers still contain compiled/native components internally, but the application does not require the user to write, compile or operate a C++ inference server.

This distinction is deliberate:

```text
This project:
Python -> Transformers -> PyTorch -> model

Not required:
Python -> external llama.cpp/Ollama service -> model
```

## Future extension points

The current structure leaves clear places for later additions without mixing them into the first working implementation:

- `chats/` for local conversation persistence
- `data/` for documents
- a RAG/indexing module
- multiple model profiles
- configurable quantization/backends
- local tool calling
- desktop packaging

Those features should be added only when the minimal runtime remains understandable and testable.
