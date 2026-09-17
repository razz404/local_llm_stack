# Troubleshooting

## `WinError 1260` when creating a virtual environment

On managed Windows systems this can be caused by Group Policy.

You do not need a venv for this project. Use:

```cmd
python bootstrap.py
```

The bootstrap installs packages into the local `packages/` directory with pip `--target`.

## PyTorch shows `+cpu` / `CUDA: False`

Check:

```cmd
python -c "import sys; sys.path.insert(0, r'packages'); import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

A CUDA profile requires `torch.cuda.is_available()` to be `True`.

If it is `False` on a machine with an NVIDIA GPU, use the official PyTorch installation selector and re-run bootstrap with the recommended CUDA wheel index:

```cmd
python bootstrap.py ^
  --profile qwen3-4b-cuda-bf16 ^
  --torch-index-url <PYTORCH_CUDA_INDEX_URL>
```

Do not assume the correct CUDA wheel URL from another machine or an old guide.

## CUDA profile fails during bootstrap

`bootstrap.py` intentionally validates CUDA before downloading a large GPU-targeted model. Verify the NVIDIA driver, GPU visibility and CUDA-enabled PyTorch build, then re-run bootstrap.

## `bitsandbytes` import or load failure

The built-in 4-bit and 8-bit profiles require the optional quantization packages. Re-run bootstrap with the quantized profile and the correct PyTorch CUDA index for the machine.

v0.2 supports the built-in bitsandbytes profiles on CUDA only.

## Out of memory on GPU

Runtime memory also includes KV cache, non-quantized layers, quantization metadata/scales, temporary tensors, CUDA allocator reservations and other GPU processes.

Try a smaller profile, 8-bit, then 4-bit, reduce context/history and close other GPU workloads. Use `python benchmark.py` to inspect CUDA peaks.

## BF16 profile rejected

If the GPU does not report BF16 support, override the profile with `"dtype": "float16"` in `config.json` or choose another profile.

## Model directory not found

Run:

```cmd
python bootstrap.py --skip-packages --skip-torch
```

This downloads the model selected by `config.json` without reinstalling local packages.

## Hugging Face returns 401/403 or asks for access

The model may be gated or require accepted terms. Check the model page and, if appropriate:

```cmd
hf auth login
```

Do not commit tokens to the repository.

## Model is extremely slow on CPU

Use `python benchmark.py` to establish a baseline. For a 16 GB CPU laptop, start with `qwen3-0.6b-cpu` and then try `qwen3-1.7b-cpu` only if latency is acceptable.

## Duplicate target-directory warnings from pip

Repeated `pip --target packages/` runs may report existing directories. Bootstrap uses `--upgrade`. If the local tree becomes inconsistent, delete only the project `packages/` directory and bootstrap again.

## Port 7860 already in use

Change the `ui.port` value in `config.json`.

## Runtime unexpectedly accesses the network

`app.py`, `smoke_test.py` and `benchmark.py` enable offline mode before loading model libraries, and model loading uses `local_files_only=True`. Bootstrap is intentionally online when downloads are required.
