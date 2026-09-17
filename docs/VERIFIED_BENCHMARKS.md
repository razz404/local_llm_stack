# Verified benchmark results

This document records benchmark runs that have been executed on real target hardware using this repository. The purpose is not to create a universal ranking, but to preserve measured reference points for future changes.

## Dell Latitude 5340 / 16 GB RAM / Windows CPU

Date: 2026-09-17

Common runtime for the Qwen3-1.7B comparison:

- Python 3.13.14
- PyTorch 2.14.0+cpu
- CPU device
- Qwen/Qwen3-1.7B
- 81 prompt tokens
- 128 maximum new tokens
- deterministic benchmark (`do_sample=false`)
- cold and warm passes in the same process
- TorchAO 0.18.0 for INT8 runs

The exact CPU model was not recorded with these runs, so these results should be treated as machine-specific Dell Latitude 5340 measurements rather than a processor-family benchmark.

### Qwen3-1.7B comparison

| Profile | Quantization | Load | Cold TTFT | Warm TTFT | Cold tok/s | Warm tok/s | Warm RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3-1.7b-cpu` | none / FP32 | 23.673 s | 4.686 s | **2.120 s** | **3.242** | **3.078** | 6.497 GiB |
| `qwen3-1.7b-cpu-int8-weightonly` | TorchAO INT8 weight-only | 26.803 s | **3.340 s** | 3.711 s | 1.193 | 1.230 | 2.801 GiB |
| `qwen3-1.7b-cpu-int8-dynamic` | TorchAO dynamic INT8 activations + weights | **18.442 s** | 77.721 s | 87.904 s | 0.475 | 0.432 | **2.668 GiB** |

Raw JSON is stored under `benchmarks/reference/`.

### Observations from this machine

The FP32 profile produced the highest sustained generation throughput and the lowest warm time to first text. Its cost was substantially higher resident memory use.

TorchAO INT8 weight-only reduced warm process RSS from 6.497 GiB to 2.801 GiB, roughly a 57% reduction, but warm throughput fell from 3.078 to 1.230 tokens/second, roughly a 60% reduction. Warm TTFT increased from 2.120 to 3.711 seconds.

TorchAO dynamic INT8 reduced warm RSS further to 2.668 GiB, but performance on this Windows CPU test was not suitable for interactive use: warm throughput was 0.432 tokens/second and warm TTFT was 87.904 seconds.

The reported `model_memory_footprint_gib` remained 6.41 GiB for all three runs even though process RSS differed substantially. For these TorchAO tests, process RSS is therefore a more useful comparison metric than the model footprint value reported through the generic Transformers/PyTorch API.

### Current project interpretation

For this tested Dell Latitude 5340 environment:

- `qwen3-1.7b-cpu` is the practical Qwen3-1.7B interactive baseline.
- `qwen3-1.7b-cpu-int8-weightonly` remains useful as a memory-saving experiment when RAM pressure matters more than latency or throughput.
- `qwen3-1.7b-cpu-int8-dynamic` remains available for research and regression testing, but is not recommended for interactive use on this tested Windows CPU configuration.

These conclusions are intentionally scoped to the measured environment. Different CPUs, operating systems, Torch/TorchAO versions, compiler settings, instruction sets and kernels can produce materially different results.

## Reference files

- `benchmarks/reference/dell-5340-qwen3-1.7b-fp32.json`
- `benchmarks/reference/dell-5340-qwen3-1.7b-int8-weightonly.json`
- `benchmarks/reference/dell-5340-qwen3-1.7b-int8-dynamic.json`
