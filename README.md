# 🧠 LocalMind Agent

An autonomous, fully-local RAG research assistant.
Powered by **Ollama** (llama3.2), **ChromaDB**, **HuggingFace embeddings**, and a
**LangGraph ReAct** agent loop.

## Features

- 📄 Drop PDFs / TXT / Markdown into `data/raw_docs/`
- 🔍 Persistent local vector store (ChromaDB)
- 🤖 Agentic reasoning with tool use (Thought → Action → Observation)
- 🧾 Grounded answers with source citations — no hallucinated facts
- 🖥️ Streamlit UI **and** Rich CLI

## Setup

### 1. Install Ollama and pull the model

```bash
ollama run llama3.2
```

### 2. Create a Python environment

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

Copy `.env` and adjust if needed (defaults work out of the box).

### 4. Add documents

Place your PDFs / TXT / MD files in `data/raw_docs/`.

### 5. Index

```bash
python main.py index --reset
```

## Usage

### CLI — one-off question

```bash
python main.py ask "Summarize section 2 of the design doc"
python main.py ask "What does the paper say about latency?" --mode rag
```

### CLI — interactive REPL

```bash
python main.py shell
```

### Web UI

```bash
streamlit run app.py
```

## Project layout

```
localmind-agent/
├── data/raw_docs/       # your source documents
├── data/vector_db/      # ChromaDB persistence
├── src/                 # core library code
├── prompts/             # system + RAG prompt templates
├── app.py               # Streamlit UI
└── main.py              # CLI
```

## How it works

1. **Loader** ingests every supported file and splits it into overlapping chunks.
2. **Embeddings** converts each chunk using a local sentence-transformer.
3. **Vector store** persists chunks in ChromaDB.
4. **Agent** (LangGraph) receives a question, decides whether to call
   `vector_store_search`, observes the result, and continues until it can
   produce a final, cited answer.
5. **RAG mode** bypasses the agent and directly synthesizes an answer from the
   top-K retrieved chunks using `prompts/rag_prompt.txt`.

## Troubleshooting

| Problem                         | Fix                                              |
| ------------------------------- | ------------------------------------------------ |
| `Connection refused` on Ollama  | Run `ollama serve` in another terminal.          |
| Empty vector store              | Run `python main.py index` after adding files.   |
| Slow first index                | Embedding model is downloading from HuggingFace. |
| `langchain_huggingface` missing | `pip install langchain-huggingface`              |
