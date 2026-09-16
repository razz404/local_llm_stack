# Offline operation

## What "offline" means in this project

The normal lifecycle has two phases.

### Phase 1: bootstrap (online)

Internet access is expected while running:

```cmd
python bootstrap.py
```

The bootstrap process can contact:

- the configured Python package index through pip
- Hugging Face Hub to download the configured model

### Phase 2: runtime (offline-capable)

After packages and model files exist locally, `app.py` configures:

```text
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
```

and model/tokenizer loading uses:

```python
local_files_only=True
```

The default Gradio interface binds to:

```text
127.0.0.1:7860
```

with public sharing disabled.

## Local runtime locations

The project redirects common runtime locations into the clone:

```text
packages/               Python dependencies
models/                 model files
cache/huggingface/      Hugging Face cache
cache/torch/            Torch cache
cache/pip/              pip cache
temp/                   TEMP/TMP
chats/                  reserved local chat data
data/                   reserved local document data
logs/                   local logs
```

## Verifying offline behavior

A simple practical test is:

1. complete `python bootstrap.py`
2. disconnect the machine from the network or use an isolated test network
3. run `python smoke_test.py`
4. run `python app.py`
5. chat through `http://127.0.0.1:7860`

If both the smoke test and UI can generate responses, inference is using the local model files.

## Strict environments

Environment variables and `local_files_only=True` prevent normal Transformers/Hugging Face network access, but they are not a security sandbox or a host firewall.

For environments that require hard network isolation, enforce it outside the application as well, for example with:

- host firewall policy
- application allow/deny rules
- an isolated VLAN/network
- a disconnected machine

That is the appropriate way to create a strong network boundary.

## Updating while remaining controlled

To update only dependencies:

```cmd
python bootstrap.py --skip-model
```

To update/download only the configured model:

```cmd
python bootstrap.py --skip-packages
```

After the update completes, runtime can return to offline operation.

## Files intentionally not stored in Git

`.gitignore` excludes model weights, local dependencies, caches and user/runtime data. Cloning the repository therefore restores code and configuration, not a machine's downloaded model or local conversations.
