# Legal Agent (Agno + Ollama + FastAPI)

Backend API focused on structure-aware legal extraction for downstream workflows.

## What it does

- Extracts parties, shareholders, clauses, exhibits, relationships, financial terms, cross-references, and graph metadata into structured JSON.
- Supports single-document text and uploaded-file extraction.
- Uses Docling for parsing and a hybrid semantic extraction pipeline on local Ollama models.

## Quick start

1. Install dependencies:

```bash
pip install -e ./libs/agno
pip install -e ./legal_agent
```

2. Configure environment:

```bash
cp legal_agent/.env.example legal_agent/.env
```

3. Run the API:

```bash
cd legal_agent
python -m app.main
```

4. Open docs:
- http://127.0.0.1:8000/docs

## Endpoints

- `GET /health`
- `POST /extract` (single text document)
- `POST /extract-upload` (single uploaded file)

Legacy aliases still point to the same pipeline:
- `POST /v2/extract`
- `POST /v2/extract-upload`

## Model defaults

- Model: `qwen2.5:7b`

Override with env vars in `.env`.

## Notes

- This service is extraction-focused, not legal advice.
- Long documents are processed with Docling-based section-aware chunking.
- The pipeline uses heuristics on every chunk and selectively applies the LLM to the most valuable sections.
- `.txt`, `.md`, `.pdf`, and `.docx` uploads are supported through the current parser stack.

## Extraction Pipeline

- Docling-based document parsing with section-aware chunking and table preservation
- Document classifier with playbook routing
- Playbook-guided extraction for supported contract families
- Hybrid extraction with heuristics on all chunks plus selective LLM passes
- Deterministic merge plus optional financial, exhibit, taxonomy, and relationship refinement
- Lightweight cross-reference and entity graph generation
- Optional LangSmith tracing through:
  - `LEGAL_AGENT_LANGSMITH_ENABLED`
  - `LEGAL_AGENT_LANGSMITH_API_KEY`
  - `LEGAL_AGENT_LANGSMITH_API_URL`
  - `LEGAL_AGENT_LANGSMITH_PROJECT`

## Playbooks

Playbooks live in [playbooks](C:\projects\legal_agent\playbooks) and define:
- expected fields
- clause taxonomy
- risk rules
- cross-reference patterns
- priority terms for chunk routing

Current playbooks include:
- Shareholders Agreement
- NDA
- Lease Agreement
- Employment Agreement
- Service Agreement
- Loan Agreement
- Policy
- Default fallback
