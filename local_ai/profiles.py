import json

from .runtime import BASE_DIR


PROFILES_PATH = BASE_DIR / "model_profiles.json"


def load_profiles():
    with PROFILES_PATH.open("r", encoding="utf-8") as handle:
        profiles = json.load(handle)

    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("model_profiles.json must contain at least one profile.")

    return profiles


def resolve_model_config(settings, profiles=None):
    model = dict(settings.get("model", {}))
    profile_name = model.get("profile")

    if profile_name:
        profiles = profiles or load_profiles()
        if profile_name not in profiles:
            available = ", ".join(sorted(profiles))
            raise ValueError(
                f"Unknown model profile '{profile_name}'. Available profiles: {available}"
            )

        resolved = dict(profiles[profile_name])
        # Explicit config.json keys override profile defaults.
        for key, value in model.items():
            if key != "profile":
                resolved[key] = value
        resolved["profile"] = profile_name
    else:
        resolved = model

    required = {"id", "local_dir"}
    missing = required.difference(resolved)
    if missing:
        raise ValueError(
            "Resolved model configuration is missing required field(s): "
            + ", ".join(sorted(missing))
        )

    resolved.setdefault("device", "auto")
    resolved.setdefault("dtype", "auto")
    resolved.setdefault("quantization", "none")
    resolved.setdefault("enable_thinking", False)

    quantization = str(resolved["quantization"]).lower()
    if quantization not in {"none", "4bit", "8bit"}:
        raise ValueError(
            "model.quantization must be one of: none, 4bit, 8bit."
        )
    resolved["quantization"] = quantization

    return resolved
