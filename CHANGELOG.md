# Changelog

All notable project changes are documented here.

## 0.2.0 - 2026-09-17

### Added

- reusable `model_profiles.json`
- built-in Qwen3 0.6B, 1.7B, 4B BF16, 8B 8-bit and 8B 4-bit profiles
- experimental `qwen3-1.7b-cpu-int8-dynamic` TorchAO profile
- experimental `qwen3-1.7b-cpu-int8-weightonly` TorchAO profile
- optional `requirements-torchao.txt`
- Hugging Face `TorchAoConfig` integration for CPU INT8 experiments
- `bootstrap.py --list-profiles`
- `bootstrap.py --profile NAME`
- optional `--torch-index-url` for CUDA-enabled PyTorch wheels
- automatic profile resolution in runtime settings
- dtype selection (`auto`, FP32, FP16, BF16)
- optional bitsandbytes 4-bit and 8-bit loading
- NF4/double-quantization settings for the Qwen3-8B 4-bit profile
- explicit CUDA validation for GPU profiles
- `benchmark.py` with load time, streamed latency, token throughput, RAM and CUDA memory metrics
- JSON benchmark output
- cold and warm generation passes in `benchmark.py`
- warm-vs-cold throughput and latency comparison metrics
- per-pass process RSS and CUDA peak-memory measurements
- quantization backend version recording for TorchAO/bitsandbytes benchmarks
- verified Dell Latitude 5340 Qwen3-1.7B reference results for FP32, TorchAO INT8 weight-only and TorchAO dynamic INT8
- committed raw benchmark JSON under `benchmarks/reference/`
- `docs/VERIFIED_BENCHMARKS.md` for measured hardware-specific results and interpretation
- model-profile, quantization, licensing and benchmarking documentation

### Changed

- PyTorch installation is handled explicitly by `bootstrap.py`
- quantization dependencies are optional and selected per profile
- TorchAO is installed with `--no-deps` after PyTorch to avoid replacing the selected PyTorch build
- default `config.json` now selects a named model profile
- model runtime prints profile, device, dtype and quantization mode
- benchmark JSON schema 2 stores separate `cold_run`, `warm_run` and `warm_vs_cold` objects while retaining the original top-level generation fields as cold-pass compatibility aliases
- model-memory reporting now degrades gracefully to `null` if a quantized tensor subclass cannot expose the usual footprint API
- CPU profile descriptions now distinguish the practical FP32 baseline from memory-saving or research-only TorchAO experiments based on verified Dell Latitude 5340 measurements
- static checks now validate committed benchmark reference JSON

### Fixed

- runtime profile validation now accepts `torchao-int8-dynamic` and `torchao-int8-weightonly`
- CI now resolves every declared profile through the same runtime profile resolver used by `app.py` and `benchmark.py`, preventing bootstrap/runtime validation drift

### Support boundary

- built-in bitsandbytes profiles are supported on CUDA only in v0.2
- TorchAO CPU INT8 profiles are experimental and intended for benchmark-driven validation on the target machine
- on the tested Dell Latitude 5340 Windows CPU configuration, TorchAO dynamic INT8 is not recommended for interactive use because measured throughput and TTFT were substantially worse than FP32
- the first TorchAO comparison intentionally does not auto-enable `torch.compile`
- other upstream TorchAO/bitsandbytes/XPU/MPS quantized backends are not yet claimed as tested configurations

## 0.1.0 - 2026-09-16

Initial documented public version.

### Added

- Python-first local LLM runtime
- Qwen3-0.6B reference configuration
- direct PyTorch + Transformers inference
- Gradio localhost chat UI
- streaming responses
- repository-local `packages/` installation without requiring venv
- repository-local model, cache and temporary directories
- bootstrap script for dependencies and model download
- CPU/CUDA/MPS device auto-selection
- terminal smoke test
- offline runtime configuration
- Windows `.cmd` launcher scripts
- installation, architecture, offline, troubleshooting and security documentation
