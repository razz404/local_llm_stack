import argparse
import json
import platform
import time
from pathlib import Path

from local_ai.runtime import BASE_DIR, configure_local_environment

configure_local_environment(offline=True)

import psutil
import torch

from local_ai.engine import LocalLLM
from local_ai.settings import load_settings


DEFAULT_PROMPT = (
    "Förklara kort vad en lokal språkmodell är och nämn två fördelar med att "
    "köra den lokalt."
)


def gib(value):
    return value / (1024 ** 3)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the currently selected local LLM profile."
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for the benchmark.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens for each benchmark pass.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Optional path for a JSON result file, relative to the repository.",
    )
    return parser.parse_args()


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_generation(engine, prompt, process):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cuda_sync()
    first_piece_seconds = None
    final_text = ""
    generation_start = time.perf_counter()

    for chunk in engine.stream_chat(prompt, []):
        if first_piece_seconds is None:
            first_piece_seconds = time.perf_counter() - generation_start
        final_text = chunk

    cuda_sync()
    generation_seconds = time.perf_counter() - generation_start

    output_tokens = engine.count_text_tokens(final_text)
    tokens_per_second = (
        output_tokens / generation_seconds if generation_seconds > 0 else None
    )

    run = {
        "time_to_first_text_seconds": (
            round(first_piece_seconds, 3)
            if first_piece_seconds is not None
            else None
        ),
        "generation_seconds": round(generation_seconds, 3),
        "output_tokens": output_tokens,
        "output_tokens_per_second": (
            round(tokens_per_second, 3)
            if tokens_per_second is not None
            else None
        ),
        "process_rss_after_gib": round(gib(process.memory_info().rss), 3),
    }

    if torch.cuda.is_available():
        run["cuda_peak_allocated_gib"] = round(
            gib(torch.cuda.max_memory_allocated()), 3
        )
        run["cuda_peak_reserved_gib"] = round(
            gib(torch.cuda.max_memory_reserved()), 3
        )

    return run, final_text


def safe_ratio(numerator, denominator):
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def safe_improvement_percent(cold, warm):
    if cold in {None, 0} or warm is None:
        return None
    return ((cold - warm) / cold) * 100


def print_run(title, run):
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in run.items():
        print(f"{key}: {value}")


def optional_backend_versions(engine):
    versions = {}

    if engine.quantization.startswith("torchao-"):
        try:
            import torchao

            versions["torchao"] = getattr(torchao, "__version__", "unknown")
        except ImportError:
            versions["torchao"] = "unavailable"

    if engine.quantization in {"4bit", "8bit"}:
        try:
            import bitsandbytes

            versions["bitsandbytes"] = getattr(
                bitsandbytes,
                "__version__",
                "unknown",
            )
        except ImportError:
            versions["bitsandbytes"] = "unavailable"

    return versions


def main():
    args = parse_args()
    settings = load_settings()
    settings["generation"] = dict(settings["generation"])
    settings["generation"]["max_new_tokens"] = args.max_new_tokens
    settings["generation"]["do_sample"] = False

    process = psutil.Process()
    rss_before = process.memory_info().rss

    print("Loading model for benchmark...")
    load_start = time.perf_counter()
    engine = LocalLLM(settings)
    cuda_sync()
    load_seconds = time.perf_counter() - load_start
    rss_after_load = process.memory_info().rss

    prompt_tokens = engine.count_prompt_tokens(args.prompt)
    model_memory = engine.model_memory_bytes()
    backend_versions = optional_backend_versions(engine)

    print("\nRunning cold generation pass...")
    cold_run, cold_text = benchmark_generation(engine, args.prompt, process)

    print("Running warm generation pass...")
    warm_run, warm_text = benchmark_generation(engine, args.prompt, process)

    throughput_speedup = safe_ratio(
        warm_run["output_tokens_per_second"],
        cold_run["output_tokens_per_second"],
    )
    ttft_improvement = safe_improvement_percent(
        cold_run["time_to_first_text_seconds"],
        warm_run["time_to_first_text_seconds"],
    )
    generation_improvement = safe_improvement_percent(
        cold_run["generation_seconds"],
        warm_run["generation_seconds"],
    )

    comparison = {
        "warm_throughput_speedup_x": (
            round(throughput_speedup, 3)
            if throughput_speedup is not None
            else None
        ),
        "warm_ttft_improvement_percent": (
            round(ttft_improvement, 1)
            if ttft_improvement is not None
            else None
        ),
        "warm_generation_time_improvement_percent": (
            round(generation_improvement, 1)
            if generation_improvement is not None
            else None
        ),
        "responses_match": cold_text == warm_text,
    }

    result = {
        "benchmark_schema": 2,
        "profile": engine.model_config.get("profile", "custom"),
        "model_id": engine.model_config["id"],
        "device": str(engine.device),
        "dtype": str(engine.dtype),
        "quantization": engine.quantization,
        "python": platform.python_version(),
        "torch": torch.__version__,
        **backend_versions,
        "load_seconds": round(load_seconds, 3),
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "process_rss_before_gib": round(gib(rss_before), 3),
        "process_rss_after_load_gib": round(gib(rss_after_load), 3),
        "model_memory_footprint_gib": (
            round(gib(model_memory), 3) if model_memory is not None else None
        ),
        "cold_run": cold_run,
        "warm_run": warm_run,
        "warm_vs_cold": comparison,
        "response": cold_text,
        "warm_response": warm_text,
        # Backwards-compatible aliases for benchmark JSON produced before schema 2.
        # They intentionally refer to the cold pass.
        "time_to_first_text_seconds": cold_run["time_to_first_text_seconds"],
        "generation_seconds": cold_run["generation_seconds"],
        "output_tokens": cold_run["output_tokens"],
        "output_tokens_per_second": cold_run["output_tokens_per_second"],
        "process_rss_after_generation_gib": cold_run["process_rss_after_gib"],
    }

    if torch.cuda.is_available():
        result["cuda_device_name"] = torch.cuda.get_device_name(0)

    print("\nBenchmark result")
    print("================")
    common_keys = [
        "profile",
        "model_id",
        "device",
        "dtype",
        "quantization",
        "python",
        "torch",
    ]
    common_keys.extend(backend_versions)
    common_keys.extend(
        [
            "load_seconds",
            "prompt_tokens",
            "max_new_tokens",
            "process_rss_before_gib",
            "process_rss_after_load_gib",
            "model_memory_footprint_gib",
        ]
    )
    if "cuda_device_name" in result:
        common_keys.append("cuda_device_name")

    for key in common_keys:
        print(f"{key}: {result[key]}")

    print_run("Cold generation", cold_run)
    print_run("Warm generation", warm_run)
    print_run("Warm vs cold", comparison)

    print("\nResponse (cold):")
    print(cold_text)
    if warm_text != cold_text:
        print("\nResponse (warm):")
        print(warm_text)

    if args.json_path:
        path = Path(args.json_path)
        if not path.is_absolute():
            path = BASE_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"\nJSON written to: {path}")


if __name__ == "__main__":
    main()
