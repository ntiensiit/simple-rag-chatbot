# Simple RAG Chatbot

CLI chatbot that answers from local `.txt`/`.md` files using Ollama.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Run

```bash
.venv\Scripts\activate
python main.py
```

Add documents to `data/`, then use `/reload` to re-index.

## Commands

- `/help` — list commands
- `/reload` — rebuild index from `data/`
- `/quit` — exit

## How it works

Documents are chunked, embedded with `nomic-embed-text`, and retrieved by cosine similarity. `llama3.2` answers from the top matches. Intent is classified by prompt before routing to chat or retrieval.
