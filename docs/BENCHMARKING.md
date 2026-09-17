# Benchmarking

`benchmark.py` provides a small repeatable benchmark for the currently selected model profile.

It is intended for comparing local machines and model configurations, not for producing a universal model ranking.

## Basic run

```cmd
python benchmark.py
```

The benchmark performs two generation passes after loading the model:

1. **Cold pass** — the first real inference after model load
2. **Warm pass** — the same deterministic prompt is generated again immediately with the same model process

This helps distinguish model-compute performance from first-use effects such as memory paging, file-system cache, lazy allocation and backend/kernel initialization.

The benchmark records:

1. model-load time
2. fixed prompt token count
3. cold time to first streamed text
4. cold total generation time
5. cold output tokens/second
6. warm time to first streamed text
7. warm total generation time
8. warm output tokens/second
9. warm/cold throughput ratio
10. warm/cold latency improvement percentages
11. process RSS before load, after load, after cold generation and after warm generation
12. PyTorch model memory footprint when available
13. CUDA peak allocated/reserved memory separately for each pass when CUDA is available
14. whether the deterministic cold and warm responses match
15. `torchao` or `bitsandbytes` version for quantized profiles when available

The benchmark forces `do_sample = false` so the two passes should normally produce the same text.

## Why cold vs warm matters

A model can appear much slower on the first generation than on later requests even when the model object has already been constructed.

On CPU systems this can happen because model pages are not fully resident in physical RAM until the first inference touches them. Operating-system page cache and lazy memory mapping can also affect the first pass.

On GPU systems, first-use overhead can include CUDA context work, allocator behavior and kernel initialization.

A large difference between cold and warm results therefore does **not** automatically mean that the model itself computes that much faster after the first request. The warm pass is best treated as an estimate of steady-state interactive behavior inside an already-running process.

## JSON output

```cmd
python benchmark.py --json benchmarks\latitude-5340-qwen17.json
```

Schema 2 JSON contains dedicated objects:

```json
{
  "cold_run": {
    "time_to_first_text_seconds": 7.353,
    "generation_seconds": 43.864,
    "output_tokens_per_second": 2.918
  },
  "warm_run": {
    "time_to_first_text_seconds": 1.234,
    "generation_seconds": 39.500,
    "output_tokens_per_second": 3.241
  },
  "warm_vs_cold": {
    "warm_throughput_speedup_x": 1.111,
    "warm_ttft_improvement_percent": 83.2,
    "warm_generation_time_improvement_percent": 10.0,
    "responses_match": true
  }
}
```

The numbers above are illustrative only.

For compatibility with JSON produced by the first v0.2 benchmark, the original top-level generation fields are retained and refer to the **cold pass**:

- `time_to_first_text_seconds`
- `generation_seconds`
- `output_tokens`
- `output_tokens_per_second`
- `process_rss_after_generation_gib`

New consumers should prefer `cold_run` and `warm_run`.

## Custom prompt

```cmd
python benchmark.py --prompt "Sammanfatta skillnaden mellan NIS2 och CER." --max-new-tokens 200
```

Both cold and warm passes use the same prompt and maximum token setting.

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
- quantization backend version
- power mode
- other system load

Laptop results can vary significantly with thermal throttling, battery state and power policy.

When comparing different machines, record enough hardware context to explain the result: CPU model, RAM, GPU/VRAM if present, operating system and whether the machine was on AC power.

## Interpreting the metrics

### Load time

Time from creating `LocalLLM` until the model is ready. Disk speed, model size, dtype, quantization and driver initialization all influence this. On-load quantization can significantly increase this value.

### Time to first text

Time from starting one benchmark pass until the streamer yields the first text fragment. This is a user-perceived latency metric, not a mathematically exact first-token measurement.

### Cold time to first text

The first prompt after model loading. This can include first-touch paging and lazy initialization costs.

### Warm time to first text

The same prompt immediately repeated in the same process. This is often closer to what repeated interactive use feels like after the model is already active.

### Output tokens/second

Output token count divided by total generation time. Tokenization is model-specific, so tokens/second should primarily be compared for the same tokenizer/model family.

### Warm throughput speedup

`warm tokens/s / cold tokens/s`.

A value of `1.0` means no throughput difference. `1.2` means the warm pass generated output about 20% faster by this measurement.

### Process RSS

Resident memory of the Python process. It does not necessarily represent every OS-level allocation and should not be confused with total system memory use.

A low RSS immediately after model load followed by a large increase after the cold pass can indicate memory-mapped/lazily resident weights rather than a faulty memory measurement.

### Model memory footprint

Reported by the loaded Transformers/PyTorch model when available. It is useful for understanding parameter/storage footprint but is not identical to process RSS or total machine memory consumption. Some quantized tensor subclasses may not expose a usable footprint through the same API; in that case the benchmark records `null` rather than failing the run.

For the verified Dell Latitude 5340 TorchAO comparison, `model_memory_footprint_gib` remained 6.41 GiB for FP32 and both INT8 variants even though warm process RSS ranged from 6.497 GiB down to 2.668 GiB. For that backend/configuration, process RSS was therefore the more informative memory comparison.

### CUDA peak allocated/reserved

Values reported by PyTorch's CUDA allocator. v0.2 records these separately for cold and warm passes. They are useful for relative testing but do not equal the total VRAM shown by every system monitoring tool.

## Verified reference measurements

The repository now keeps selected real-hardware results under `benchmarks/reference/` and summarizes them in [Verified benchmark results](VERIFIED_BENCHMARKS.md).

The first recorded CPU comparison used Qwen3-1.7B on a Dell Latitude 5340 with 16 GB RAM, Python 3.13.14 and PyTorch 2.14.0+cpu. TorchAO runs used TorchAO 0.18.0.

Warm-pass results from that machine were:

| Profile | Warm TTFT | Warm tok/s | Warm RSS |
| --- | ---: | ---: | ---: |
| FP32 | **2.120 s** | **3.078** | 6.497 GiB |
| TorchAO INT8 weight-only | 3.711 s | 1.230 | 2.801 GiB |
| TorchAO INT8 dynamic | 87.904 s | 0.432 | **2.668 GiB** |

These values are observations from one tested environment, not general claims about TorchAO performance on every CPU or operating system.

## Suggested test matrix

For a 16 GB CPU laptop:

```text
qwen3-0.6b-cpu
qwen3-1.7b-cpu
qwen3-1.7b-cpu-int8-dynamic
qwen3-1.7b-cpu-int8-weightonly
```

The most useful comparison for CPU INT8 is:

```text
same Qwen3-1.7B weights
same prompt
same 128-token cap
same machine / power mode

FP32 baseline
    vs
TorchAO INT8 dynamic
    vs
TorchAO INT8 weight-only
```

Compare at least:

- model-load time
- cold TTFT
- warm TTFT
- warm tokens/second
- process RSS after warm generation
- model footprint when available
- whether the deterministic answer remains the same or materially changes

For an NVIDIA workstation laptop with around 12 GB VRAM:

```text
qwen3-4b-cuda-bf16
qwen3-8b-cuda-8bit
qwen3-8b-cuda-4bit
```

The goal is to find the best balance of quality, cold-start latency, warm interactive performance, memory and context headroom for the actual workload.
