from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

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
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print("Local AI")
        print("========")
        print(f"Model: {self.model_dir}")
        print(f"Device: {self.device}")
        print(f"dtype: {self.dtype}")
        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            local_files_only=True,
            dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
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
            raise RuntimeError("config.json requests CUDA, but CUDA is not available.")

        if preference == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise RuntimeError("config.json requests MPS, but MPS is not available.")

        if preference not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device setting: {preference}")

        return preference

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

            # Compatibility with older Gradio tuple/list history formats.
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
            # Some non-Qwen tokenizers do not understand enable_thinking.
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def stream_chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._normalise_history(history))
        messages.append({"role": "user", "content": message})

        prompt = self._apply_chat_template(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

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
            generation_kwargs["top_p"] = float(self.generation_config.get("top_p", 0.9))

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

        prompt = self._apply_chat_template(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        kwargs = {
            **inputs,
            "max_new_tokens": int(self.generation_config.get("max_new_tokens", 300)),
            "do_sample": bool(self.generation_config.get("do_sample", True)),
            "pad_token_id": self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
        }

        if kwargs["do_sample"]:
            kwargs["temperature"] = float(self.generation_config.get("temperature", 0.7))
            kwargs["top_p"] = float(self.generation_config.get("top_p", 0.9))

        with torch.inference_mode():
            output = self.model.generate(**kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
