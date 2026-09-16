# Troubleshooting

## `WinError 1260` when creating a virtual environment

Example:

```text
Error: [WinError 1260] This program is blocked by group policy
```

This usually means Windows application control or Group Policy blocks part of venv creation/execution. It is not necessarily a PowerShell problem.

Use the repository-local installation instead:

```cmd
python bootstrap.py
```

The bootstrap script installs dependencies into `packages/` and does not require a new `python.exe` inside `.venv/`.

## `ModuleNotFoundError`

First confirm that bootstrap completed:

```cmd
python bootstrap.py --skip-model
```

Then verify imports explicitly:

```cmd
python -c "import sys; sys.path.insert(0, 'packages'); import torch, transformers, gradio; print(torch.__version__); print(transformers.__version__); print(gradio.__version__)"
```

If packages are partially installed, rerun bootstrap. It uses pip with `--upgrade --target packages`.

## Model directory not found

Error:

```text
Model directory not found ... Run 'python bootstrap.py' first.
```

Run:

```cmd
python bootstrap.py --skip-packages
```

Also confirm that `config.json` points to the same model directory that bootstrap downloads.

## Model loading is very slow

CPU-only model loading can take a noticeable amount of time. On the initial test laptop Qwen3-0.6B loaded successfully on CPU, but model loading was much slower than on a GPU system.

Check the active device shown during startup:

```text
Device: cpu
```

You can also verify PyTorch directly:

```cmd
python -c "import sys; sys.path.insert(0, 'packages'); import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

If CUDA prints `False`, the current PyTorch runtime will use CPU.

## `CUDA: False` even though the computer has an NVIDIA GPU

The installed PyTorch build may not include the CUDA runtime required for that system. GPU-enabled PyTorch installation is platform-specific.

Install the appropriate PyTorch build into `packages/` or a venv, then rerun the check above. The application's `device: auto` setting can only use CUDA when `torch.cuda.is_available()` is true.

## Gradio page does not open

Look at the terminal output for the local address. The default is:

```text
http://127.0.0.1:7860
```

Try opening that address manually.

If the port is already used, change:

```json
"port": 7860
```

in `config.json`, for example to `7861`.

## I can reach the UI only on the same computer

That is intentional. The default host is:

```text
127.0.0.1
```

which is loopback-only.

Do not change it to `0.0.0.0` unless you explicitly intend to expose the application to other network interfaces and understand the security implications.

## The application tries to access the internet at runtime

The default runtime sets:

```text
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
```

and uses `local_files_only=True` when loading the model and tokenizer.

If you need a hard guarantee, enforce outbound network blocking with the operating system/firewall as described in `OFFLINE.md`.

## Strange or low-quality answers

Qwen3-0.6B is intentionally small. It is a development/reference model for this project, not a replacement for a large hosted model.

Try adjusting in `config.json`:

```json
"max_new_tokens": 300,
"temperature": 0.7,
"top_p": 0.9
```

For deterministic testing, set:

```json
"do_sample": false
```

## Garbled chat history after changing Gradio versions

The application explicitly uses Gradio's message format and also contains compatibility handling for older tuple/list history. If a future Gradio version changes the API, test with `smoke_test.py` first. If the smoke test works but the UI fails, the problem is likely isolated to the Gradio layer.

## Reinstall from scratch

Because runtime state is local, a clean rebuild is simple:

1. close the application
2. remove `packages/` and optionally `cache/`
3. keep or remove `models/` depending on whether you want to redownload model weights
4. run `python bootstrap.py`

Do not delete `models/` unless you are prepared to download the model again.
