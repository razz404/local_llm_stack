# Security

## Default exposure

The default configuration is designed for local experimentation:

- Gradio binds to `127.0.0.1`
- `share=False` disables Gradio public sharing
- Hugging Face/Transformers runtime is configured for offline mode
- model and tokenizer loading uses local files only
- local model weights, packages, caches and user data are ignored by Git

## Important limitation

This project is **not a sandbox**. Running Python code, third-party Python packages or downloaded model artifacts still executes within the permissions of the current operating-system user.

Offline environment variables reduce accidental network use by the Transformers/Hugging Face layer, but they are not a firewall. Use operating-system or network controls when a strong isolation boundary is required.

## Supply-chain considerations

Bootstrap downloads:

1. Python packages from the configured pip package index
2. model files from the model repository identified in `config.json`

Review and pin dependencies/model revisions if the stack is used in a controlled or production-like environment. The current repository is an experimental local-AI stack, not a hardened deployment baseline.

## Network exposure

Changing the UI host from:

```text
127.0.0.1
```

to:

```text
0.0.0.0
```

can make the service reachable from other interfaces. Do not do this casually, especially on a corporate or untrusted network. The default project does not implement authentication or authorization for the Gradio UI.

## Sensitive documents

Future RAG/document features may process confidential or personal data. Before such features are introduced, the project should define explicit rules for:

- document storage
- embeddings/vector indexes
- chat persistence
- deletion
- access control
- backup
- logging

The current version does not implement document ingestion or persistent chat storage.

## Reporting a security issue

For now, open a GitHub issue only for non-sensitive security observations. Do not publish credentials, private data, exploit material tied to a real environment, or confidential corporate information in a public issue.
