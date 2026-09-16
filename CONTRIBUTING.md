# Contributing

Contributions should preserve the project's main design goal: a small, understandable, Python-first local LLM stack.

## Principles

Prefer changes that:

- keep the default setup easy to clone and run
- avoid mandatory external inference services
- keep runtime files local to the repository by default
- preserve offline-capable inference after bootstrap
- work on CPU-only systems unless the feature is explicitly GPU-specific
- add documentation for new configuration or operational behavior

Avoid introducing Docker, Ollama, llama.cpp or another service as a mandatory dependency for the default path. Optional backends may be considered later if they do not replace the Python-first reference implementation.

## Development setup

```cmd
git clone https://github.com/razz404/local_llm_stack.git
cd local_llm_stack
python bootstrap.py
python smoke_test.py
python app.py
```

## Before a pull request

At minimum:

1. run `python -m compileall app.py bootstrap.py smoke_test.py local_ai`
2. run `python smoke_test.py` when model/runtime code changed
3. start `python app.py` when UI code changed
4. update documentation when behavior or configuration changed
5. do not commit `models/`, `packages/`, caches, local documents or chat data

## Commit scope

Keep commits focused and explain why the change is useful. For larger features, document the design before introducing a large dependency stack.
