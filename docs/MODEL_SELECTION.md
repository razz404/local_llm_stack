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

For this repository, a small model that starts reliably and fits comfortably in memory is usually a better baseline than the largest model that can barely be loaded.

## Practical size guide

The table below is deliberately approximate. Actual memory use depends on architecture, precision, context length, KV cache, runtime, tokenizer, temporary tensors and other processes on the machine.

| Model class | Typical use in this project | Unquantized 16-bit weight size, rough estimate | 4-bit weight size, rough estimate |
| --- | --- | ---: | ---: |
| 0.5-0.6B | Development, CPU smoke tests | ~1-1.2 GB | ~0.25-0.3 GB |
| 1-2B | Small CPU assistants | ~2-4 GB | ~0.5-1 GB |
| 4B | Good small GPU baseline | ~8 GB | ~2 GB |
| 7-8B | Strong local assistant on a discrete GPU | ~14-16 GB | ~3.5-4 GB |
| 12-14B | Larger local model; usually quantized | ~24-28 GB | ~6-7 GB |

These figures describe model weights only. Leave headroom for the operating system, Python, PyTorch, the KV cache and intermediate tensors.

## Reference choices

### Qwen3 family

Qwen3 is a convenient default for this repository because the official Hugging Face repositories for Qwen3-0.6B, 1.7B, 4B and 8B are published under Apache-2.0 and work directly with Transformers.

Suggested starting points:

| Hardware | Suggested first test |
| --- | --- |
| CPU-only laptop, 16 GB RAM | Qwen3-0.6B |
| CPU-only laptop, 16 GB RAM, willing to accept slower inference | Qwen3-1.7B |
| NVIDIA GPU with around 8-12 GB VRAM | Qwen3-4B in BF16/FP16 where it fits |
| NVIDIA GPU with around 12 GB VRAM | Qwen3-8B in 4-bit is an interesting target |
| 12 GB VRAM, experimental upper limit | 12-14B class in 4-bit, with limited context and careful memory testing |

Do not treat the table as a guarantee. Test the exact model, runtime and context length on the target machine.

### Gemma family

Gemma models are also technically interesting, and larger Gemma 3 variants add multimodal image-and-text capability. However, Gemma uses Google's Gemma licence/terms rather than Apache-2.0 or MIT.

That is not inherently a problem, but it means the user must review and accept the applicable terms before using or redistributing the model. Some Hugging Face models can also require an authenticated account and an access request before their files can be downloaded.

For that reason, Gemma is better treated as an optional model in this project rather than the zero-friction default.

## Precision and quantization

### FP32

FP32 uses roughly four bytes per parameter for the model weights.

It is broadly compatible and numerically conservative, but memory use is high. A 4B model is roughly 16 GB in FP32 before runtime overhead, which makes FP32 a poor choice for larger local models on ordinary laptops.

### FP16 / BF16

16-bit formats use roughly two bytes per parameter.

They are a natural choice on modern GPUs when supported. BF16 has a wider exponent range than FP16 and is often attractive for inference on hardware that supports it well.

A 4B model is roughly 8 GB of weights at 16-bit precision. This can make a 4B model realistic on a 12 GB GPU, provided enough VRAM remains for the KV cache and runtime overhead.

### 8-bit quantization

8-bit quantization typically reduces model-weight memory substantially compared with 16-bit loading. Hugging Face Transformers integrates with `bitsandbytes` for 8-bit model loading.

8-bit can be a useful compromise when a model almost fits in VRAM in 16-bit, but it is not free:

- extra runtime dependencies are required
- hardware/backend support matters
- some operations remain at higher precision
- performance gains are workload and hardware dependent
- quantization can introduce small quality changes

### 4-bit quantization

4-bit quantization allows much larger models to fit in limited VRAM. A rough first-order estimate is about 0.5 byte per parameter for raw 4-bit weights, but real memory usage is higher because of quantization metadata, scales, caches and non-quantized components.

This is why an 8B model can be a practical target on a 12 GB GPU when loaded in 4-bit, while the same model would normally exceed 12 GB in 16-bit form.

Important limitations:

- 4-bit is more aggressive than 8-bit and can affect output quality
- the amount of degradation depends on the model and quantization method
- long context can consume substantial extra VRAM through the KV cache
- `bitsandbytes`, GPTQ, AWQ and other quantization approaches are not interchangeable
- a quantized checkpoint may have a different licence or provenance from the original model repository
- community quantizations should be treated as a separate software/model supply-chain artefact and verified accordingly

For this repository, prefer quantizing an official upstream model yourself or using a quantized release from a source you explicitly trust.

## Quantization does not change the model licence

Quantizing a model does not normally remove or replace the obligations attached to the original model.

Before using or redistributing a quantized model, check:

1. the licence and terms of the original upstream model
2. the licence and metadata of the quantized repository
3. whether redistribution of modified weights is permitted
4. whether attribution, notices or use restrictions must be preserved
5. whether the model repository adds separate conditions or acceptable-use terms

The MIT licence of `local_llm_stack` covers the code in this repository. It does **not** relicense model weights downloaded from Hugging Face.

## Hugging Face licences, gated models and access agreements

Hugging Face is a hosting platform. Models hosted there do not all share one licence.

A repository can be:

- openly downloadable with a permissive licence such as Apache-2.0 or MIT
- openly downloadable but subject to a custom model licence
- gated, requiring a Hugging Face account and an access request
- gated with additional fields or terms that must be accepted
- manually approved by the model author
- subject to geographic or other access restrictions

For gated models, Hugging Face states that access is granted to individual users. The user may need to share account information with the model author, accept additional conditions, and authenticate when downloading model files.

Therefore `bootstrap.py` should not be expected to download every Hugging Face model anonymously.

### Authentication

If a selected model is gated, users may need to authenticate first, for example with:

```text
hf auth login
```

Do not commit Hugging Face tokens to this repository, `config.json`, shell scripts or documentation examples.

### Corporate environments

Before downloading a model on an employer-owned system, check both:

- the model's legal terms and licence
- your organisation's policy for third-party software, model weights, data processing and cloud/account use

A model being downloadable from Hugging Face does not by itself mean it is approved for business use.

## Upstream versus community checkpoints

Prefer the original model publisher when possible.

For example:

```text
Qwen/Qwen3-8B
```

is easier to reason about than an arbitrary community repository containing a modified or quantized copy.

Community checkpoints can be perfectly legitimate and useful, but add questions around:

- who produced the weights
- what conversion or quantization was performed
- whether the original licence and notices were preserved
- whether custom Python code is required
- whether the files can be reproduced from the upstream model
- whether hashes or signed releases are available

Do not enable `trust_remote_code=True` casually. It permits code from the model repository to execute in your local Python process. Prefer architectures already supported natively by Transformers.

## Context length is a hardware decision too

A model card may advertise a large maximum context window, but that does not mean using the maximum is sensible on local hardware.

Longer prompts increase:

- KV-cache memory use
- prompt-processing time
- total latency

When testing a new model, start with a modest context and increase it only when the workload requires it.

## A simple decision process

Use this order:

1. **Licence and access** — are the model terms acceptable for your use?
2. **Modality** — text only, or image + text?
3. **Hardware** — CPU RAM and/or GPU VRAM available?
4. **Precision** — can the model fit in BF16/FP16 without quantization?
5. **Quantization** — if not, would 8-bit or 4-bit make it practical?
6. **Context** — how much context can you afford while retaining memory headroom?
7. **Benchmark** — measure load time, time to first token, tokens/second and memory usage on the actual machine.
8. **Quality test** — compare models on your real prompts rather than parameter count alone.

## Recommended project policy

For `local_llm_stack`:

- keep a small, permissively licensed, non-gated model as the default
- do not bundle model weights in the Git repository
- keep model downloads explicit and local
- display or document the upstream model ID
- never imply that the project's MIT licence applies to downloaded model weights
- document gated/authenticated models as optional
- treat quantized community checkpoints as separate supply-chain artefacts
- benchmark each model/hardware combination instead of promising that a model will fit

## Current examples

As of September 2026, the official Hugging Face model pages identify:

- `Qwen/Qwen3-0.6B` — Apache-2.0
- `Qwen/Qwen3-1.7B` — Apache-2.0
- `Qwen/Qwen3-4B` — Apache-2.0
- `Qwen/Qwen3-8B` — Apache-2.0
- `google/gemma-3-1b-it` — Gemma licence
- `google/gemma-3-12b-it` — Gemma licence; multimodal image/text model

Always verify the current model card before use because licences, access settings, model revisions and supported runtimes can change.

## Primary references

- Hugging Face gated models: https://huggingface.co/docs/hub/models-gated
- Transformers quantization overview: https://huggingface.co/docs/transformers/main/quantization
- Transformers bitsandbytes guide: https://huggingface.co/docs/transformers/main/quantization/bitsandbytes
- bitsandbytes documentation: https://huggingface.co/docs/bitsandbytes/
- Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
- Qwen3-1.7B: https://huggingface.co/Qwen/Qwen3-1.7B
- Qwen3-4B: https://huggingface.co/Qwen/Qwen3-4B
- Qwen3-8B: https://huggingface.co/Qwen/Qwen3-8B
- Gemma 3 1B IT: https://huggingface.co/google/gemma-3-1b-it
- Gemma 3 12B IT: https://huggingface.co/google/gemma-3-12b-it
