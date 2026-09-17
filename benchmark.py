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
        help="Maximum generated tokens for the benchmark run.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Optional path for a JSON result file, relative to the repository.",
    )
    return parser.parse_args()


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
    load_seconds = time.perf_counter() - load_start
    rss_after_load = process.memory_info().rss

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prompt_tokens = engine.count_prompt_tokens(args.prompt)
    first_piece_seconds = None
    final_text = ""

    generation_start = time.perf_counter()
    for chunk in engine.stream_chat(args.prompt, []):
        if first_piece_seconds is None:
            first_piece_seconds = time.perf_counter() - generation_start
        final_text = chunk
    generation_seconds = time.perf_counter() - generation_start

    output_tokens = engine.count_text_tokens(final_text)
    tokens_per_second = (
        output_tokens / generation_seconds if generation_seconds > 0 else None
    )
    rss_after_generation = process.memory_info().rss

    model_memory = engine.model_memory_bytes()

    result = {
        "profile": engine.model_config.get("profile", "custom"),
        "model_id": engine.model_config["id"],
        "device": str(engine.device),
        "dtype": str(engine.dtype),
        "quantization": engine.quantization,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "load_seconds": round(load_seconds, 3),
        "time_to_first_text_seconds": (
            round(first_piece_seconds, 3)
            if first_piece_seconds is not None
            else None
        ),
        "generation_seconds": round(generation_seconds, 3),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "output_tokens_per_second": (
            round(tokens_per_second, 3)
            if tokens_per_second is not None
            else None
        ),
        "process_rss_before_gib": round(gib(rss_before), 3),
        "process_rss_after_load_gib": round(gib(rss_after_load), 3),
        "process_rss_after_generation_gib": round(gib(rss_after_generation), 3),
        "model_memory_footprint_gib": (
            round(gib(model_memory), 3) if model_memory is not None else None
        ),
        "response": final_text,
    }

    if torch.cuda.is_available():
        result["cuda_device_name"] = torch.cuda.get_device_name(0)
        result["cuda_peak_allocated_gib"] = round(
            gib(torch.cuda.max_memory_allocated()), 3
        )
        result["cuda_peak_reserved_gib"] = round(
            gib(torch.cuda.max_memory_reserved()), 3
        )

    print("\nBenchmark result")
    print("================")
    for key, value in result.items():
        if key != "response":
            print(f"{key}: {value}")
    print("\nResponse:")
    print(final_text)

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
