# Legal Agent (Agno + Ollama + FastAPI)

Backend API focused on accurate legal `entities` and `clauses` extraction for downstream actions.

## What it does

- Extracts entities and clauses into structured JSON.
- Supports:
  - Single-document extraction
  - Mixed bundle extraction (multiple documents in one request)
- Uses local Ollama models through Agno typed input/output patterns.

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
- `POST /v1/extract` (single text document)
- `POST /v1/extract-upload` (single uploaded file)
- `POST /v1/extract-bundle` (multiple text documents)
- `POST /v1/extract-bundle-upload` (multiple uploaded files)

The upload parser is fixed to `pypdf` for stable local performance and no external OCR downloads.

## Model defaults

- Model: `qwen3:8b-q4_K_M`

Override with env vars in `.env`.

## Notes

- This service is extraction-focused, not legal advice.
- For very large documents, input is truncated to `LEGAL_AGENT_MAX_TEXT_CHARS`.
- Long documents are processed in chunked extraction mode and merged into one final output across entities, clauses, shareholdings, financial terms, and relationships.
- `.txt`, `.md`, and `.json` uploads are supported directly.
- `.pdf` and `.docx` are supported only if optional parsers are installed:
  - PDF: `pypdf`
  - DOCX: `python-docx`
"# agnoLegalAgent" 
