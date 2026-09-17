# Benchmarking

`benchmark.py` provides a small repeatable benchmark for the currently selected model profile.

It is intended for comparing local machines and model configurations, not for producing a universal model ranking.

## Basic run

```cmd
python benchmark.py
```

The benchmark:

1. loads the selected model
2. records model-load time
3. sends a fixed Swedish prompt
4. measures time until the first streamed text is received
5. measures total generation time
6. estimates output tokens per second
7. records process RSS
8. records PyTorch model memory footprint when available
9. records CUDA peak allocated/reserved memory when CUDA is available

## JSON output

```cmd
python benchmark.py --json benchmarks\latitude-5340-qwen06.json
```

The output includes the response so quality and performance can be reviewed together.

## Custom prompt

```cmd
python benchmark.py --prompt "Sammanfatta skillnaden mellan NIS2 och CER." --max-new-tokens 200
```

## Compare fairly

Keep these constant when comparing results:

- prompt
- `max_new_tokens`
- sampling setting
- model revision
- model profile
- quantization method
- context/history
- Python/PyTorch/Transformers versions
- power mode
- other system load

Laptop results can vary significantly with thermal throttling and power policy.

## Interpreting the metrics

### Load time

Time from creating `LocalLLM` until the model is ready. Disk speed, model size, dtype, quantization and driver initialization all influence this.

### Time to first text

Time from starting generation until the streamer yields the first text fragment. This is a user-perceived latency metric, not a mathematically exact "first token" measurement.

### Output tokens/second

Output token count divided by total generation time. Tokenization is model-specific, so tokens/second should primarily be compared for the same tokenizer/model family.

### Process RSS

Resident memory of the Python process. It does not necessarily represent every OS-level allocation and should not be confused with total system memory use.

### CUDA peak allocated/reserved

Values reported by PyTorch's CUDA allocator. They are useful for relative testing but do not equal the total VRAM shown by every system monitoring tool.

## Suggested test matrix

For a 16 GB CPU laptop:

```text
qwen3-0.6b-cpu
qwen3-1.7b-cpu
```

For an NVIDIA workstation laptop with around 12 GB VRAM:

```text
qwen3-4b-cuda-bf16
qwen3-8b-cuda-8bit
qwen3-8b-cuda-4bit
```

The goal is to find the best balance of quality, latency, memory and context headroom for the actual workload.
