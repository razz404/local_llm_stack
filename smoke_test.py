from local_ai.runtime import configure_local_environment

# Force local imports/caches and disable network access for model loading.
configure_local_environment(offline=True)

from local_ai.engine import LocalLLM
from local_ai.settings import load_settings


def main():
    settings = load_settings()
    engine = LocalLLM(settings)

    print("\nGenerating test response...")
    answer = engine.generate_once(
        "Svara kort på svenska: vad är du och var körs du?"
    )

    print("\nAI:")
    print(answer)


if __name__ == "__main__":
    main()
