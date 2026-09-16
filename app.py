from local_ai.runtime import configure_local_environment

# Must run before importing third-party packages.
configure_local_environment(offline=True)

import gradio as gr

from local_ai.engine import LocalLLM
from local_ai.settings import load_settings


settings = load_settings()
engine = LocalLLM(settings)


def chat(message, history):
    yield from engine.stream_chat(message, history)


def main():
    ui = settings["ui"]

    demo = gr.ChatInterface(
        fn=chat,
        type="messages",
        title=ui.get("title", "Local AI"),
        description=ui.get(
            "description",
            "Local language model running directly through Python, Transformers and PyTorch.",
        ),
    )

    demo.launch(
        server_name=ui.get("host", "127.0.0.1"),
        server_port=int(ui.get("port", 7860)),
        inbrowser=bool(ui.get("open_browser", True)),
        share=False,
    )


if __name__ == "__main__":
    main()
