import os

import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
TOP_K = 3
MIN_SCORE = 0.45
LLM_OPTS = {"temperature": 0}
INTENTS = ("greeting", "farewell", "thanks", "meta", "chitchat", "creative", "opinion", "clarify", "factual")

def missing_msg(topics):
    names = ", ".join(topics) if topics else "loaded topics"
    return f"I don't have that in the loaded documents. Ask about {names}."

def bot_facts(topics):
    names = ", ".join(topics) if topics else "none"
    return f"""Bot facts (use only these, never invent or expand acronyms):
- RAG means Retrieval-Augmented Generation
- Loads .txt and .md files from data/
- Chunks text (500 chars, 50 overlap), embeds with {EMBED_MODEL}, retrieves top {TOP_K} by cosine similarity
- Answers with {CHAT_MODEL} using retrieved context only; outside topics are refused
- Loaded topics: {names}
- /help: show commands
- /reload: re-read data/ and rebuild the search index
- /quit: exit"""

def ollama_host():
    host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    if host.startswith("0.0.0.0"):
        return "http://127.0.0.1:11434"
    return host if host.startswith("http") else f"http://{host}"

client = ollama.Client(host=ollama_host())

def chunk_text(text, size=500, overlap=50):
    if not text.strip():
        return []
    step = max(size - overlap, 1)
    return [text[i:i + size] for i in range(0, len(text), step)]

def embed(text, model=EMBED_MODEL):
    return client.embed(model=model, input=text)["embeddings"][0]

def cosine_scores(query_vec, matrix):
    q = np.asarray(query_vec, dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    return m @ q / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-9)

def retrieve(query, chunks, vectors, k=TOP_K):
    scores = cosine_scores(embed(query), vectors)
    idx = np.argsort(scores)[::-1][:min(k, len(scores))]
    return [chunks[i] for i in idx], float(scores[idx[0]]) if len(idx) else 0.0

def intent_prompt(query, topics=None):
    topic_line = f"Loaded topics: {', '.join(topics)}.\n" if topics else ""
    return f"""You classify messages for a RAG chatbot that answers only from loaded documents.

{topic_line}Output exactly one label, nothing else:
greeting | farewell | thanks | meta | chitchat | creative | opinion | clarify | factual

meta = about the bot itself: capabilities, commands, how it works, what it knows, what topics are loaded
factual = world-knowledge questions documents may answer: what is X, explain Y, tell me about Z
greeting = hi, hello, what's up, good morning (casual openers, not real questions)
farewell = bye, goodbye, see you
thanks = thank you, thanks, appreciate it
chitchat = lol, nice, cool, short reactions
creative = write, compose, generate new stories, poems, code
opinion = what do you think, which is better, subjective picks
clarify = follow-up on a prior answer: explain that again, what do you mean

Disambiguation:
what is up -> greeting
what do you know -> meta
what you can do -> meta
how does this work -> meta
what is the Feynman technique -> factual
what is gradient descent -> factual
can you explain how this works -> meta
can you please explain how this work -> meta

Examples:
hi -> greeting
bye -> farewell
thanks -> thanks
what do you know -> meta
what you can do -> meta
Can you please explain how this works? -> meta
What is the Feynman technique? -> factual
explain support vector machine -> factual
write me a poem -> creative
what do you think about jazz -> opinion
explain that again -> clarify
lol nice -> chitchat

Query: {query}
Label:"""

def parse_intent(raw):
    label = raw.strip().lower().split()[0].strip(".:,;\"'")
    return label if label in INTENTS else "factual"

def classify_intent(query, topics=None, model=CHAT_MODEL):
    opts = {**LLM_OPTS, "num_predict": 6}
    raw = client.chat(model=model, messages=[{"role": "user", "content": intent_prompt(query, topics)}], options=opts)["message"]["content"]
    return parse_intent(raw)

def build_prompt(query, context_chunks, topics=None):
    context = "\n\n".join(context_chunks)
    missing = missing_msg(topics)
    return f"You must answer only using the context. Never use outside knowledge. If the context does not contain the answer, reply exactly: {missing}\n\nContext:\n{context}\n\nQuestion: {query}"

def chat(query, context_chunks, topics=None, model=CHAT_MODEL):
    msg = build_prompt(query, context_chunks, topics)
    return client.chat(model=model, messages=[{"role": "user", "content": msg}], options=LLM_OPTS)["message"]["content"]

def meta_chat(query, topics=None, model=CHAT_MODEL):
    msg = f"Answer only using these facts. Never invent details or rename commands.\n\n{bot_facts(topics)}\n\nQuestion: {query}"
    return client.chat(model=model, messages=[{"role": "user", "content": msg}], options=LLM_OPTS)["message"]["content"]

def direct_chat(query, intent, topics=None, model=CHAT_MODEL):
    if intent == "meta":
        return meta_chat(query, topics, model)
    guides = {
        "greeting": "Greet warmly. Invite questions about the loaded topics.",
        "farewell": "Say goodbye briefly.",
        "thanks": "Acknowledge thanks briefly.",
        "chitchat": "Reply casually and briefly.",
        "creative": "Decline creative tasks. Suggest factual questions about loaded topics.",
        "opinion": "Say you answer from loaded documents, not personal opinions.",
        "clarify": "Ask the user to rephrase or name the topic they want clarified.",
    }
    names = ", ".join(topics) if topics else "loaded topics"
    sys = f"RAG chatbot with documents on: {names}. {guides[intent]} Be brief. Never invent facts."
    return client.chat(model=model, messages=[{"role": "system", "content": sys}, {"role": "user", "content": query}], options=LLM_OPTS)["message"]["content"]

def answer(query, chunks, vectors, topics=None):
    if not chunks:
        return "No documents loaded. Add .txt/.md files to data/ and run /reload."
    intent = classify_intent(query, topics)
    missing = missing_msg(topics)
    if intent in ("creative", "opinion"):
        return direct_chat(query, intent, topics)
    if intent == "factual":
        hits, score = retrieve(query, chunks, vectors)
        return chat(query, hits, topics) if score >= MIN_SCORE else missing
    if intent == "clarify":
        hits, score = retrieve(query, chunks, vectors)
        return chat(query, hits, topics) if score >= MIN_SCORE else direct_chat(query, intent, topics)
    return direct_chat(query, intent, topics)
