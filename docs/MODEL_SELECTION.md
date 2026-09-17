# Choosing a local LLM

This guide explains how to choose a model for `local_llm_stack`, with particular attention to hardware limits, quantization, Hugging Face access controls, and model-specific licence or terms-of-use obligations.

> This is practical engineering guidance, not legal advice. Always read the current model card, licence and any additional terms before downloading, redistributing, fine-tuning or deploying a model.

## Start with the workload, not the parameter count

A larger model is not automatically the best local model. Choose based on:

- available system RAM and GPU VRAM
- CPU-only vs CUDA GPU inference
- acceptable latency and model-load time
- context length you actually need
- whether you need text-only or multimodal input
- whether the model must be fully redistributable
- whether you need a permissive licence for commercial or public use
- whether the model requires Hugging Face authentication or acceptance of separate terms

For this repository, a model that starts reliably and fits with memory headroom is a better baseline than the largest model that can barely be loaded.

## Built-in v0.2 profiles

List them with:

```cmd
python bootstrap.py --list-profiles
```

| Profile | Upstream model | Precision / quantization | Intended hardware |
| --- | --- | --- | --- |
| `qwen3-0.6b-cpu` | `Qwen/Qwen3-0.6B` | unquantized / auto dtype | CPU baseline, 16 GB RAM |
| `qwen3-1.7b-cpu` | `Qwen/Qwen3-1.7B` | FP32 on CPU | CPU quality step-up |
| `qwen3-1.7b-cpu-int8-dynamic` | `Qwen/Qwen3-1.7B` | TorchAO INT8 activations + weights | experimental CPU benchmark |
| `qwen3-1.7b-cpu-int8-weightonly` | `Qwen/Qwen3-1.7B` | TorchAO INT8 weights | experimental CPU benchmark |
| `qwen3-4b-cuda-bf16` | `Qwen/Qwen3-4B` | BF16 | NVIDIA GPU around 10-12 GB VRAM |
| `qwen3-8b-cuda-8bit` | `Qwen/Qwen3-8B` | bitsandbytes 8-bit | NVIDIA GPU around 12 GB VRAM |
| `qwen3-8b-cuda-4bit` | `Qwen/Qwen3-8B` | bitsandbytes NF4 4-bit | NVIDIA GPU around 12 GB VRAM |

These are test profiles, not guarantees. Actual fit depends on model revision, runtime overhead, context length, KV cache, CPU/GPU architecture and other processes.

## Practical size guide

Raw weight memory estimates are useful for orientation, not capacity planning.

| Model class | 16-bit weights, rough | 4-bit raw weights, rough |
| --- | ---: | ---: |
| 0.5-0.6B | ~1-1.2 GB | ~0.25-0.3 GB |
| 1-2B | ~2-4 GB | ~0.5-1 GB |
| 4B | ~8 GB | ~2 GB |
| 7-8B | ~14-16 GB | ~3.5-4 GB |
| 12-14B | ~24-28 GB | ~6-7 GB |

Real memory use is higher because of caches, scales/metadata, non-quantized layers, temporary tensors and framework overhead.

## Precision

### FP32

Roughly four bytes per parameter. Broadly compatible, but expensive for local LLMs.

### FP16 / BF16

Roughly two bytes per parameter. These are the natural unquantized formats for modern GPUs.

BF16 is attractive on hardware that supports it because it retains a wider numerical range than FP16. The `qwen3-4b-cuda-bf16` profile deliberately uses BF16 as a clean GPU baseline before quantization.

## CPU INT8 with TorchAO

The two experimental Qwen3-1.7B CPU profiles use Hugging Face `TorchAoConfig` and current TorchAO configuration objects:

- `Int8DynamicActivationInt8WeightConfig()`
- `Int8WeightOnlyConfig()`

The dynamic profile quantizes activations at runtime and stores the targeted linear weights in INT8. This is the more interesting experiment when matrix multiplication is compute-bound.

The weight-only profile reduces weight precision but keeps activation computation at higher precision. It can be more useful when model execution is limited by memory traffic rather than arithmetic throughput.

Important limitations:

- TorchAO is an additional dependency with its own PyTorch compatibility requirements
- CPU acceleration depends on processor instruction support, operating system and available kernels
- lower model memory does not guarantee higher tokens/second
- quantization itself can add model-load cost
- output quality may differ from the FP32 baseline
- `torch.compile` can materially change results, but this project deliberately leaves it disabled for the first CPU INT8 comparison

The right way to evaluate these profiles is therefore to compare the exact same deterministic cold/warm benchmark against `qwen3-1.7b-cpu`.

## CUDA 8-bit quantization

The v0.2 CUDA 8-bit profile uses Hugging Face `BitsAndBytesConfig(load_in_8bit=True)`.

8-bit can reduce model-weight memory substantially relative to 16-bit and is a useful intermediate step when a model is close to fitting.

Limitations:

- `bitsandbytes` is an extra native/runtime dependency
- backend and hardware compatibility matter
- some components remain at higher precision
- speed is not guaranteed to improve just because memory use drops
- quality can change slightly

## CUDA 4-bit quantization

The v0.2 4-bit profile uses bitsandbytes with:

- `load_in_4bit=True`
- NF4
- double quantization
- BF16 compute where the selected GPU supports it

4-bit is attractive because it can make an 8B-class model practical on a 12 GB GPU, leaving more headroom than 16-bit.

Important limitations:

- 4-bit is more aggressive than 8-bit
- output quality can differ from BF16/FP16
- long context still consumes substantial KV-cache memory
- raw "0.5 byte per parameter" estimates understate real runtime memory
- GPTQ, AWQ, bitsandbytes and other methods are not interchangeable
- a community quantized checkpoint is a separate supply-chain artefact

The project prefers **on-load quantization of the official upstream model** for its built-in profiles. That keeps provenance simpler than silently switching to an arbitrary community checkpoint.

## Current project support boundary

`local_llm_stack` currently treats quantization in two distinct groups:

- **TorchAO CPU INT8** — experimental built-in profiles for Qwen3-1.7B
- **bitsandbytes CUDA 4/8-bit** — built-in GPU profiles

Other TorchAO, bitsandbytes, XPU, MPS, GPTQ, AWQ and community-specific quantization paths may work upstream but are not claimed as tested project configurations.

This boundary is intentional: benchmark and document one path before expanding the compatibility matrix.

## CUDA-enabled PyTorch is separate from "having an NVIDIA GPU"

An NVIDIA GPU in Device Manager does not mean the Python runtime can use it.

Check:

```python
import torch
print(torch.cuda.is_available())
```

CUDA profiles stop early if the installed PyTorch build is CPU-only.

The correct CUDA-enabled PyTorch wheel depends on the current PyTorch release and supported CUDA channels. `bootstrap.py` therefore accepts:

```cmd
--torch-index-url <URL>
```

rather than hard-coding a wheel index that will become stale.

## Model licences and Hugging Face terms

Hugging Face is a hosting platform. Models hosted there do not share one universal licence.

A repository can be:

- openly downloadable under Apache-2.0, MIT or another permissive licence
- openly downloadable under a custom model licence
- gated and require a Hugging Face account
- gated and require additional terms/fields
- manually approved by the publisher
- subject to other access restrictions

A successful download does not by itself grant every right you may need.

## Quantization does not change the upstream licence

Quantizing model weights does not erase the original model obligations.

Before using or redistributing a quantized model, check:

1. the upstream model licence/terms
2. the quantized repository's metadata if using a third-party checkpoint
3. whether modified-weight redistribution is allowed
4. attribution/notice requirements
5. acceptable-use or field-of-use restrictions

The MIT licence in this Git repository applies to this project's code, not downloaded model weights.

## Gated models

Some Hugging Face repositories require account authentication and explicit approval/acceptance.

Typical authentication:

```cmd
hf auth login
```

Never put a Hugging Face token in:

- `config.json`
- `model_profiles.json`
- shell/batch scripts
- Git commits

For employer-owned systems, also check organisational policy for third-party software, model weights, account use and local processing.

## Qwen versus Gemma as project defaults

The built-in Qwen3 profiles point to official Qwen repositories and record Apache-2.0 metadata. This makes Qwen convenient for a public clone-and-run example.

Gemma models are technically interesting, including multimodal variants, but use Google's Gemma terms rather than this repository's MIT licence and may involve access acceptance. They are therefore documented as optional choices rather than built-in zero-friction defaults.

Always verify the current upstream model card before use; licence metadata and access settings can change.

## Upstream versus community checkpoints

Prefer the original publisher where practical.

Questions to ask for community checkpoints:

- who produced the weights?
- what conversion/quantization was performed?
- can it be reproduced?
- were upstream notices preserved?
- does it require custom Python?
- are hashes/releases provided?

`local_llm_stack` does not enable `trust_remote_code=True` by default. Enabling it means repository-supplied Python can execute inside the local process and should be treated as a code-execution trust decision.

## Context length is also a hardware choice

A model card may advertise a large context window, but maximum context is not always sensible locally.

Long context increases:

- KV-cache memory
- prompt processing time
- latency

Start small and increase context only for a real workload.

## Decision process

Use this order:

1. **Licence and access**
2. **Task/modality**
3. **RAM/VRAM**
4. **Unquantized precision if it fits**
5. **Quantization only when it solves a measured constraint**
6. **Context headroom**
7. **Benchmark on the actual machine**
8. **Quality-test with real prompts**

Use:

```cmd
python benchmark.py
```

to compare configurations consistently.

## Primary upstream references

- Hugging Face gated models: https://huggingface.co/docs/hub/models-gated
- Transformers quantization: https://huggingface.co/docs/transformers/main/quantization
- Transformers TorchAO: https://huggingface.co/docs/transformers/main/quantization/torchao
- TorchAO inference workflows: https://docs.pytorch.org/ao/stable/workflows/inference.html
- Transformers bitsandbytes: https://huggingface.co/docs/transformers/main/quantization/bitsandbytes
- bitsandbytes docs: https://huggingface.co/docs/bitsandbytes/
- PyTorch local install selector: https://pytorch.org/get-started/locally/
- Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
- Qwen3-1.7B: https://huggingface.co/Qwen/Qwen3-1.7B
- Qwen3-4B: https://huggingface.co/Qwen/Qwen3-4B
- Qwen3-8B: https://huggingface.co/Qwen/Qwen3-8B
