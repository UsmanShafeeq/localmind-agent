# LocalMind Agent

> A privacy-focused local document research assistant powered by **RAG, Local LLMs, Vector Search, and Agentic AI**.

LocalMind Agent is a local document research assistant built around **Retrieval-Augmented Generation (RAG)**. It allows users to upload and query **PDF, TXT, Markdown, and `.markdown` documents** using a fully local AI pipeline.

The system uses **Ollama** for local LLM inference, **Hugging Face Sentence Transformers** for embeddings, **ChromaDB** for persistent vector storage, and **LangGraph** for agentic workflows.

---

## Features

* Local LLM inference using **Ollama**
* Retrieval-Augmented Generation (RAG)
* LangGraph-based agent workflow
* PDF document support
* TXT document support
* Markdown and `.markdown` support
* Persistent ChromaDB vector database
* Local Hugging Face embeddings
* `sentence-transformers/all-MiniLM-L6-v2`
* Direct RAG mode
* Agentic AI mode
* Streamlit web interface
* Interactive command-line interface
* Document upload and indexing
* Configurable chunking and retrieval
* Environment-based configuration
* No external LLM API required for the core workflow

---

## Architecture

```text
                    ┌─────────────────────┐
                    │      Documents      │
                    │ PDF / TXT / MD      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Document Loader   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Text Chunking    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Local Embeddings    │
                    │ all-MiniLM-L6-v2    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      ChromaDB       │
                    │   Vector Database   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    User Question    │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │    RAG Mode     │        │  Agentic Mode   │
        │ Direct Retrieval│        │    LangGraph    │
        └────────┬────────┘        └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Ollama        │
                    │      Local LLM      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Final Answer     │
                    └─────────────────────┘
```

---

## Technology Stack

| Component        | Technology            |
| ---------------- | --------------------- |
| Language         | Python                |
| LLM Runtime      | Ollama                |
| Default LLM      | `llama3.2:1b`         |
| Embeddings       | Sentence Transformers |
| Embedding Model  | `all-MiniLM-L6-v2`    |
| Vector Database  | ChromaDB              |
| Agent Framework  | LangGraph             |
| Web Interface    | Streamlit             |
| Configuration    | `.env`                |
| Document Formats | PDF, TXT, Markdown    |

---

## Requirements

Before installing LocalMind Agent, make sure you have:

* Python **3.11 or newer**
* Git
* Ollama
* At least one Ollama model
* Internet connection for the initial embedding model download

### Windows

Windows users may need the **Microsoft Visual C++ 2015–2022 Redistributable (x64)**, especially if PyTorch produces DLL-related errors.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/UsmanShafeeq/localmind-agent.git
cd localmind-agent
```

## 2. Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

# Ollama Setup

LocalMind Agent uses Ollama to run the language model locally.

Start Ollama:

```bash
ollama serve
```

Download the default model:

```bash
ollama pull llama3.2:1b
```

Verify the installed models:

```bash
ollama list
```

You should see:

```text
llama3.2:1b
```

You can use another Ollama model by changing the `MODEL_NAME` variable in the `.env` file.

---

# Configuration

Create a `.env` file in the project root:

```dotenv
MODEL_NAME=llama3.2:1b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

DB_PATH=./data/vector_db
RAW_DOCS_PATH=./data/raw_docs

COLLECTION_NAME=localmind_docs

CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=4

OLLAMA_BASE_URL=http://localhost:11434
```

## Configuration Parameters

| Variable          | Description                             |
| ----------------- | --------------------------------------- |
| `MODEL_NAME`      | Ollama model used for answer generation |
| `EMBEDDING_MODEL` | Hugging Face embedding model            |
| `DB_PATH`         | ChromaDB storage location               |
| `RAW_DOCS_PATH`   | Source document directory               |
| `COLLECTION_NAME` | ChromaDB collection name                |
| `CHUNK_SIZE`      | Document chunk size                     |
| `CHUNK_OVERLAP`   | Overlap between chunks                  |
| `TOP_K`           | Number of retrieved chunks              |
| `OLLAMA_BASE_URL` | Ollama server URL                       |

> **Important:** Do not commit `.env` to GitHub if it contains private credentials or tokens.

Add the following to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

---

# Quick Start

## 1. Add Documents

Place your documents inside:

```text
data/raw_docs/
```

Supported formats:

```text
.pdf
.txt
.md
.markdown
```

Example:

```text
data/
└── raw_docs/
    ├── university_policy.pdf
    ├── research_notes.txt
    ├── project_document.md
    └── security_policy.markdown
```

---

## 2. Build the Vector Index

Run:

```bash
python main.py index --reset
```

This process will:

1. Load documents
2. Extract text
3. Split documents into chunks
4. Generate embeddings
5. Store embeddings in ChromaDB
6. Store document metadata

---

## 3. Ask Questions

Run:

```bash
python main.py ask "What is this document about?"
```

Example:

```bash
python main.py ask "What are the main admission requirements?"
```

---

# RAG Mode

LocalMind provides a direct RAG mode that retrieves relevant document chunks and sends them to the local LLM.

```bash
python main.py ask "Summarize the main risks" --mode rag
```

### RAG Workflow

```text
User Question
      │
      ▼
Generate Query Embedding
      │
      ▼
ChromaDB Similarity Search
      │
      ▼
Retrieve Top-K Chunks
      │
      ▼
Build RAG Prompt
      │
      ▼
Ollama Local LLM
      │
      ▼
Generate Answer
```

---

# Agentic Mode

The default agent mode uses **LangGraph** to create a tool-using workflow.

```text
User Question
      │
      ▼
LangGraph Agent
      │
      ▼
Document Search Tool
      │
      ▼
ChromaDB
      │
      ▼
Relevant Context
      │
      ▼
Local LLM
      │
      ▼
Final Answer
```

This architecture provides a foundation for adding additional tools and more advanced agentic workflows.

---

# Interactive CLI

Start the interactive command-line shell:

```bash
python main.py shell
```

You can then interact with LocalMind through the terminal.

---

# Streamlit Web Interface

LocalMind includes a Streamlit-based web interface.

Start the application:

```bash
streamlit run app.py
```

The interface provides:

* Document upload
* Document indexing
* Interactive chat
* Document question answering
* Local LLM responses
* RAG-based retrieval

### Typical Workflow

```text
Open Streamlit
      │
      ▼
Upload Documents
      │
      ▼
Add & Index
      │
      ▼
Create ChromaDB Index
      │
      ▼
Ask Questions
      │
      ▼
Retrieve Relevant Documents
      │
      ▼
Generate Local LLM Response
```

---

# How It Works

## 1. Document Loading

The `src/loader.py` module loads supported documents.

Supported formats:

* PDF
* TXT
* Markdown
* `.markdown`

---

## 2. Text Chunking

Large documents are divided into smaller overlapping chunks.

Default configuration:

```dotenv
CHUNK_SIZE=800
CHUNK_OVERLAP=120
```

Chunk overlap helps preserve context between adjacent sections.

---

## 3. Embedding Generation

The `src/embeddings.py` module generates local vector embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Text is converted into numerical vectors that can be compared using semantic similarity.

---

## 4. Vector Storage

The `src/vectorstore.py` module manages ChromaDB.

The persistent vector database is stored at:

```text
data/vector_db/
```

This allows the index to be reused between application runs.

---

## 5. Retrieval

When the user asks a question, LocalMind searches ChromaDB for the most relevant document chunks.

The number of retrieved chunks is controlled by:

```dotenv
TOP_K=4
```

---

## 6. Answer Generation

Retrieved document context is passed to the local Ollama model.

```text
Question
   +
Retrieved Context
   +
RAG Prompt
   │
   ▼
Ollama
   │
   ▼
Final Answer
```

---

# Project Structure

```text
localmind-agent/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── prompts/
│   ├── agent_prompt.txt
│   └── rag_prompt.txt
│
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── embeddings.py
│   ├── loader.py
│   ├── tools.py
│   └── vectorstore.py
│
└── data/
    ├── raw_docs/
    └── vector_db/
```

---

# Main Components

| File                 | Responsibility                           |
| -------------------- | ---------------------------------------- |
| `app.py`             | Streamlit web interface                  |
| `main.py`            | CLI entry point                          |
| `src/agent.py`       | LangGraph workflow and answer generation |
| `src/config.py`      | Environment-based configuration          |
| `src/embeddings.py`  | Local embedding initialization           |
| `src/loader.py`      | Document loading and chunking            |
| `src/tools.py`       | Agent tools                              |
| `src/vectorstore.py` | ChromaDB operations                      |
| `prompts/`           | Agent and RAG prompt templates           |
| `data/raw_docs/`     | Source documents                         |
| `data/vector_db/`    | Persistent vector database               |

---

# Privacy

LocalMind is designed around local document processing.

The core workflow is:

```text
Documents
    │
    ▼
Local Embeddings
    │
    ▼
Local ChromaDB
    │
    ▼
Local Ollama LLM
    │
    ▼
Answer
```

No external LLM API is required for the core RAG workflow.

However, internet access may still be required for:

* Installing Python packages
* Downloading the Hugging Face embedding model
* Updating dependencies

Always review your configuration before processing sensitive or confidential documents.

---

# Use Cases

LocalMind Agent can be used for:

* Academic research
* University policy analysis
* Research paper exploration
* Technical documentation
* Cybersecurity documentation
* Internal knowledge bases
* Document question answering
* Personal knowledge management
* Local enterprise knowledge assistants
* AI-powered document research

---

# Troubleshooting

## Vector Store Is Empty

Make sure at least one supported document exists:

```text
data/raw_docs/
```

Then rebuild the index:

```bash
python main.py index --reset
```

---

## Ollama Connection Refused

Make sure Ollama is running:

```bash
ollama serve
```

Verify the configured URL:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Model Not Found

Check installed models:

```bash
ollama list
```

Pull the configured model:

```bash
ollama pull llama3.2:1b
```

---

## Hugging Face Download Warning

The embedding model is downloaded from Hugging Face during the initial setup.

Normal unauthenticated access is sufficient for typical use, although rate limits may apply.

If required, configure:

```dotenv
HF_TOKEN=your_token
```

Never commit your token to GitHub.

---

## PyTorch DLL Error on Windows

If you see an error similar to:

```text
DLL load failed
```

install or update the:

**Microsoft Visual C++ 2015–2022 Redistributable for x64**

After installation:

1. Close the terminal
2. Reopen PowerShell
3. Activate the virtual environment
4. Retry the command

---

# Development

Run the project from the repository root.

### Check CLI Options

```bash
python main.py --help
```

### Rebuild Index

```bash
python main.py index --reset
```

### Ask a Question

```bash
python main.py ask "Explain the main concepts in this document"
```

### Run RAG Mode

```bash
python main.py ask "Summarize the document" --mode rag
```

### Start Interactive Shell

```bash
python main.py shell
```

### Start Web Interface

```bash
streamlit run app.py
```

---

# Roadmap

* [ ] Multi-document collections
* [ ] Source citations in answers
* [ ] Hybrid keyword + vector search
* [ ] Reranking models
* [ ] Conversation memory
* [ ] Document metadata filtering
* [ ] Advanced LangGraph workflows
* [ ] Multi-agent architecture
* [ ] Web search integration
* [ ] Automatic document summarization
* [ ] Research report generation
* [ ] User authentication
* [ ] Role-based document access
* [ ] RAG evaluation and benchmarking
* [ ] Agent observability
* [ ] Docker deployment
* [ ] Production deployment

---

# Contributing

Contributions, bug reports, feature requests, and improvements are welcome.

A typical workflow:

```text
Fork Repository
      │
      ▼
Create Feature Branch
      │
      ▼
Make Changes
      │
      ▼
Test Changes
      │
      ▼
Commit
      │
      ▼
Push
      │
      ▼
Create Pull Request
```

Please test your changes before submitting a pull request.

---

# License

No open-source license has been specified for this project yet.

Until a license is added, the repository should not be assumed to grant permission to reuse, modify, or redistribute the code.

---

# Author

**Usman Shafeeq**

Lecturer
Department of Data Science and Artificial Intelligence
KFUEIT, Rahim Yar Khan

---

## Project Focus

**Local LLM + RAG + Vector Database + Agentic AI**

LocalMind Agent provides a practical foundation for building **privacy-aware, locally hosted document intelligence and agentic AI systems**.
