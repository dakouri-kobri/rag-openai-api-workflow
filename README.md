# RAG Workflow with OpenAI Vector Store

A production-style Retrieval-Augmented Generation (RAG) workflow using OpenAI vector stores, semantic search, metadata filtering, and grounded response generation.

## Project overview

This project demonstrates how to build a clean, reproducible AI/ML workflow around external knowledge retrieval.

The pipeline:

1. Loads local knowledge-base documents
2. Uploads them to an OpenAI vector store
3. Applies chunking and file attributes
4. Performs semantic search
5. Formats retrieved chunks as model context
6. Generates grounded answers with source references

## Why this project matters

Large language models are more useful when they can retrieve relevant information from external sources. This project shows how to build that workflow in a professional Python project structure.

## Features

- OpenAI vector-store ingestion
- Batch file attachment
- Static chunking configuration
- Semantic search
- Query rewriting
- Ranking options and score thresholds
- Metadata filtering
- Grounded answer generation
- CLI interface
- Jupyter notebook walkthrough
- Unit tests
- uv-based dependency management

## Project structure

```text
.
├── data/
│   └── sample/
├── notebooks/
│   └── 01_rag_workflow_openai_vector_stores.ipynb
├── src/
│   └── rag_openai_api_workflow/
│       ├── __init__.py
│       ├── client.py
│       ├── cli.py
│       ├── config.py
│       ├── documents.py
│       ├── generation.py
│       ├── ingest.py
│       ├── retrieval.py
│       ├── storage.py
│       └── vector_store.py
├── tests/
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_documents.py
│   ├── test_generation.py
│   ├── test_ingest.py
│   ├── test_retrieval.py
│   ├── test_storage.py
│   └── test_vector_store.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-openai-api-workflow
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Copy the environment template:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
copy .env.example .env
```

Then add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.4-mini
OPENAI_VECTOR_STORE_ID=
OPENAI_VECTOR_STORE_NAME=rag-portfolio-docs
```

`OPENAI_VECTOR_STORE_ID` can be left empty at first. The project creates a vector store during ingestion and saves the ID locally in `artifacts/vector_store.json`.

## Usage

### Ingest documents

```bash
uv run rag-openai ingest --path data/sample
```

### Search documents

```bash
uv run rag-openai search "What is project-driven learning?"
```

### Ask a grounded question

```bash
uv run rag-openai ask "In 50 words, what is the learning format?"
```

## Notebook demo

The notebook version of the workflow is available at:

```text
notebooks/01_rag_workflow_openai_vector_stores.ipynb
```

It walks through:

1. Loading configuration
2. Creating or reusing a vector store
3. Uploading sample documents
4. Running semantic search
5. Formatting retrieved context
6. Generating a grounded answer

Before committing notebook changes, clear outputs:

```bash
uv run jupyter nbconvert --clear-output --inplace notebooks/01_rag_workflow_openai_vector_stores.ipynb
```

## Example output

```text
The learning format is stage-based and project-driven. Learners build projects step by step, combining theory, coding tasks, tests, and implementation practice. This helps connect abstract concepts to working software. [1]
```

## Tests

```bash
uv run pytest
```

## Linting

```bash
uv run ruff check .
```

## Notes on secrets

The `.env` file is intentionally ignored by Git. Never commit API keys, private vector-store IDs, credentials, or private documents.

Local runtime artifacts are stored in:

```text
artifacts/
```

This folder is also ignored by Git.

## Future improvements

- Add a Streamlit or FastAPI interface
- Add retrieval-quality evaluation metrics
- Add support for multiple document collections
- Add an experiment log for retrieval settings and score thresholds

## Author

Dakouri Kobri

Data Science & AI/ML and Health Science Enthusiast

## License

MIT License
