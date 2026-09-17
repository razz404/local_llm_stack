from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    TorchAoConfig,
)

from .runtime import resolve_repo_path


class LocalLLM:
    def __init__(self, settings):
        self.settings = settings
        self.model_config = settings["model"]
        self.generation_config = settings["generation"]
        self.system_prompt = settings["assistant"]["system_prompt"]
        self.model_dir = resolve_repo_path(self.model_config["local_dir"])

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                "Run 'python bootstrap.py' first."
            )

        self.device = self._select_device(self.model_config.get("device", "auto"))
        self.dtype = self._select_dtype(
            self.model_config.get("dtype", "auto"),
            self.device,
        )
        self.quantization = str(
            self.model_config.get("quantization", "none")
        ).lower()

        print("Local AI")
        print("========")
        print(f"Profile: {self.model_config.get('profile', 'custom')}")
        print(f"Model: {self.model_dir}")
        print(f"Device: {self.device}")
        print(f"dtype: {self.dtype}")
        print(f"Quantization: {self.quantization}")
        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )

        print("Loading model...")
        load_kwargs = {
            "local_files_only": True,
            "dtype": self.dtype,
        }

        quantization_config = self._build_quantization_config()
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = self._quantized_device_map()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            **load_kwargs,
        )

        if quantization_config is None:
            self.model.to(self.device)

        self.model.eval()
        self.input_device = self._resolve_input_device()
        print(f"Input device: {self.input_device}")
        print("Model loaded.")

    @staticmethod
    def _select_device(preference):
        preference = str(preference).lower()

        if preference == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if preference == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "The selected model profile requires CUDA, but this PyTorch "
                "installation does not report CUDA as available. Install a "
                "CUDA-enabled PyTorch build and run the bootstrap again."
            )

        if preference == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise RuntimeError("config.json requests MPS, but MPS is not available.")

        if preference not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device setting: {preference}")

        return preference

    @staticmethod
    def _select_dtype(preference, device):
        preference = str(preference).lower()

        if preference == "auto":
            if device == "cuda":
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            if device == "mps":
                return torch.float16
            return torch.float32

        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if preference not in dtype_map:
            raise ValueError(
                "Unsupported dtype. Use auto, float32, float16 or bfloat16."
            )

        dtype = dtype_map[preference]
        if (
            device == "cuda"
            and dtype is torch.bfloat16
            and hasattr(torch.cuda, "is_bf16_supported")
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "The selected profile requests bfloat16, but this CUDA device "
                "does not report BF16 support. Choose float16 or another profile."
            )
        return dtype

    def _quantized_device_map(self):
        if self.quantization.startswith("torchao-"):
            return "cpu"
        return "auto"

    def _build_quantization_config(self):
        if self.quantization == "none":
            return None

        if self.quantization in {"4bit", "8bit"}:
            if self.device != "cuda":
                raise RuntimeError(
                    "The built-in bitsandbytes 4/8-bit profiles are supported "
                    "on CUDA only in this project."
                )

            if self.quantization == "8bit":
                return BitsAndBytesConfig(load_in_8bit=True)

            compute_dtype = self._select_dtype(
                self.model_config.get(
                    "bnb_4bit_compute_dtype",
                    self.model_config.get("dtype", "auto"),
                ),
                self.device,
            )

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.model_config.get(
                    "bnb_4bit_quant_type", "nf4"
                ),
                bnb_4bit_use_double_quant=bool(
                    self.model_config.get("bnb_4bit_use_double_quant", True)
                ),
            )

        if self.quantization in {
            "torchao-int8-dynamic",
            "torchao-int8-weightonly",
        }:
            if self.device != "cpu":
                raise RuntimeError(
                    "The experimental TorchAO INT8 profiles are currently "
                    "supported on CPU only in this project."
                )

            try:
                from torchao.quantization import (
                    Int8DynamicActivationInt8WeightConfig,
                    Int8WeightOnlyConfig,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "TorchAO is required for the selected CPU INT8 profile. "
                    "Run bootstrap for that profile before starting the app or benchmark."
                ) from exc

            if self.quantization == "torchao-int8-dynamic":
                ao_config = Int8DynamicActivationInt8WeightConfig()
            else:
                ao_config = Int8WeightOnlyConfig()

            return TorchAoConfig(quant_type=ao_config)

        raise ValueError(
            "Unsupported quantization mode: "
            f"{self.quantization}. Use none, 4bit, 8bit, "
            "torchao-int8-dynamic or torchao-int8-weightonly."
        )

    def _resolve_input_device(self):
        model_device = getattr(self.model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            return model_device

        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for value in hf_device_map.values():
                if value not in {"cpu", "disk"}:
                    if isinstance(value, int):
                        return torch.device(f"cuda:{value}")
                    return torch.device(value)
            if any(value == "cpu" for value in hf_device_map.values()):
                return torch.device("cpu")

        return torch.device(self.device)

    @staticmethod
    def _normalise_history(history):
        messages = []

        for item in history or []:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role in {"user", "assistant"} and isinstance(content, str):
                    messages.append({"role": role, "content": content})
                continue

            if isinstance(item, (list, tuple)) and len(item) == 2:
                user_text, assistant_text = item
                if isinstance(user_text, str) and user_text:
                    messages.append({"role": "user", "content": user_text})
                if isinstance(assistant_text, str) and assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})

        return messages

    def _apply_chat_template(self, messages):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        enable_thinking = self.model_config.get("enable_thinking")
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)

        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _prepare_inputs(self, messages):
        prompt = self._apply_chat_template(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return {
            name: tensor.to(self.input_device)
            for name, tensor in inputs.items()
        }

    def stream_chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._normalise_history(history))
        messages.append({"role": "user", "content": message})

        inputs = self._prepare_inputs(messages)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": int(self.generation_config.get("max_new_tokens", 300)),
            "do_sample": bool(self.generation_config.get("do_sample", True)),
            "pad_token_id": self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
        }

        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = float(
                self.generation_config.get("temperature", 0.7)
            )
            generation_kwargs["top_p"] = float(
                self.generation_config.get("top_p", 0.9)
            )

        def generate():
            with torch.inference_mode():
                self.model.generate(**generation_kwargs)

        thread = Thread(target=generate, daemon=True)
        thread.start()

        response = ""
        for text_piece in streamer:
            response += text_piece
            yield response

        thread.join()

    def generate_once(self, user_text):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]

        inputs = self._prepare_inputs(messages)

        kwargs = {
            **inputs,
            "max_new_tokens": int(self.generation_config.get("max_new_tokens", 300)),
            "do_sample": bool(self.generation_config.get("do_sample", True)),
            "pad_token_id": self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
        }

        if kwargs["do_sample"]:
            kwargs["temperature"] = float(
                self.generation_config.get("temperature", 0.7)
            )
            kwargs["top_p"] = float(
                self.generation_config.get("top_p", 0.9)
            )

        with torch.inference_mode():
            output = self.model.generate(**kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def count_prompt_tokens(self, user_text):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        prompt = self._apply_chat_template(messages)
        return len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def count_text_tokens(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def model_memory_bytes(self):
        getter = getattr(self.model, "get_memory_footprint", None)
        if getter is None:
            return None
        try:
            return int(getter())
        except Exception:
            return None
