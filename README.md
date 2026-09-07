# DevRAG

A production-oriented Retrieval-Augmented Generation (RAG) platform designed specifically for technical and developer documentation. DevRAG aims to provide end-to-end capabilities for ingesting, indexing, retrieving, and generating grounded answers from technical documentation using modern LLM and vector search technologies.

## Overview

DevRAG is currently in **early development / architecture phase**. The project is being scaffolded with a clean, modular architecture that separates domain logic, application workflows, infrastructure integrations, and API layers. No features have been implemented yet.

## Planned Capabilities

- Documentation ingestion and processing (Markdown, reStructuredText, HTML, etc.)
- Document parsing, chunking, and embedding generation
- Vector storage and retrieval with hybrid search
- Reranking and context construction
- LLM-based grounded answer generation with source/citation tracking
- Retrieval and answer evaluation pipelines
- REST APIs with FastAPI
- Background processing, caching, and observability
- Docker-based deployment and CI/CD

## Architecture

DevRAG follows a layered architecture:

- **`app/api/`** -- HTTP/API layer (FastAPI routes, schemas, versioning)
- **`app/core/`** -- Application-wide configuration and foundational concerns
- **`app/domain/`** -- Core business/domain concepts independent of external frameworks
- **`app/application/`** -- Application/use-case workflows that orchestrate domain and infrastructure
- **`app/infrastructure/`** -- External technology implementations (databases, LLM providers, etc.)
- **`app/services/`** -- Higher-level reusable services

## Project Structure

```text
devrag/
├── app/
│   ├── api/v1/
│   ├── core/
│   ├── domain/
│   ├── application/
│   ├── infrastructure/
│   └── services/
├── tests/
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
├── scripts/
├── docs/
├── .github/workflows/
├── pyproject.toml
└── README.md
```

## Development Status

| Status | Description |
|--------|-------------|
| **Phase** | Architecture & Scaffolding |
| **Implementation** | None |
| **Dependencies** | None installed |
| **Python version** | 3.13 |

This project is in its earliest stage. The repository contains only the directory structure and project metadata. All features listed under "Planned Capabilities" are to be implemented in future development phases.
