# Changelog

All notable project changes are documented here.

## Unreleased

### Added

- model-selection guide covering hardware sizing, context and memory headroom
- guidance for FP32, FP16/BF16, 8-bit and 4-bit model loading
- quantization caveats covering quality, dependencies, provenance and redistribution
- Hugging Face gated-model and authentication guidance
- notes on model-specific licences and terms, including the distinction between the project's MIT licence and downloaded model weights
- practical Qwen3 and Gemma examples for local testing

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
