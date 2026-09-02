from pathlib import Path

import numpy as np

from rag import answer, build_vectors, chunk_text, warmup

DATA_DIR = Path("data")

def list_topics(folder):
    names = []
    for path in sorted(folder.glob("**/*")):
        if path.suffix.lower() in {".txt", ".md"} and path.is_file():
            names.append(path.stem.replace("-", " "))
    return names

def load_documents(folder):
    docs = []
    for path in sorted(folder.glob("**/*")):
        if path.suffix.lower() in {".txt", ".md"} and path.is_file():
            docs.append((path.stem.replace("-", " "), path.read_text(encoding="utf-8")))
    return docs

def load_text_files(folder):
    texts = []
    for path in sorted(folder.glob("**/*")):
        if path.suffix.lower() in {".txt", ".md"} and path.is_file():
            texts.append(path.read_text(encoding="utf-8"))
    return texts

def build_index(folder):
    chunks = []
    for text in load_text_files(folder):
        chunks.extend(chunk_text(text))
    if not chunks:
        return [], np.empty((0, 0))
    vectors = build_vectors(chunks)
    return chunks, vectors

def print_help():
    print("Commands: /quit, /reload, /help")

def run_chat(chunks, vectors, topics, documents):
    print("RAG Chatbot - ask about your documents")
    print_help()
    history = []
    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query == "/quit":
            break
        if query == "/help":
            print_help()
            continue
        if query == "/reload":
            chunks, vectors = build_index(DATA_DIR)
            topics = list_topics(DATA_DIR)
            documents = load_documents(DATA_DIR)
            history = []
            print(f"Reloaded {len(chunks)} chunks")
            continue
        reply = answer(query, chunks, vectors, topics, history, documents)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": reply})
        history = history[-12:]
        print(f"\nAssistant: {reply}\n")

def main():
    DATA_DIR.mkdir(exist_ok=True)
    topics = list_topics(DATA_DIR)
    documents = load_documents(DATA_DIR)
    chunks, vectors = build_index(DATA_DIR)
    if chunks:
        print(f"Indexed {len(chunks)} chunks from {DATA_DIR}/")
        warmup()
    else:
        print(f"No documents in {DATA_DIR}/. Add files, then /reload")
    run_chat(chunks, vectors, topics, documents)

if __name__ == "__main__":
    main()
