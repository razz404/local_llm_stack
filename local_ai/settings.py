import json

from .runtime import BASE_DIR


CONFIG_PATH = BASE_DIR / "config.json"


def load_settings():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        settings = json.load(handle)

    required_sections = {"model", "generation", "assistant", "ui"}
    missing = required_sections.difference(settings)
    if missing:
        raise ValueError(
            "config.json is missing required section(s): " + ", ".join(sorted(missing))
        )

    return settings
