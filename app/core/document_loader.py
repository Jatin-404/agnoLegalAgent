from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import re

from agno.knowledge.document import Document
from agno.knowledge.chunking.document import DocumentChunking
from agno.knowledge.chunking.recursive import RecursiveChunking
from agno.knowledge.reader.docling_reader import DoclingReader
from fastapi import UploadFile

SUPPORTED_DOCLING_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".xlsx",
    ".csv",
    ".html",
    ".htm",
    ".xml",
    ".md",
    ".markdown",
    ".txt",
}


def _parse_heading_line(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped:
        return None

    bold_match = re.match(r"^\*\*(?P<title>.+?)\*\*$", stripped)
    if bold_match:
        stripped = bold_match.group("title").strip()

    match = re.match(r"^(?P<hashes>#{1,6})\s+(?P<title>.+)$", stripped)
    if match:
        return len(match.group("hashes")), match.group("title").strip()

    numbered = re.match(r"^(?P<number>\d+(?:\.\d+)*[.)]?)\s+(?P<title>.+)$", stripped)
    if numbered:
        return numbered.group("number").count(".") + 1, stripped

    if re.match(r"^[A-Z][A-Z '&/:-]{3,}$", stripped):
        return 1, stripped.title()

    return None


def extract_section_hierarchy(markdown: str) -> list[dict[str, object]]:
    hierarchy: list[dict[str, object]] = []
    stack: list[dict[str, object]] = []

    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        heading = _parse_heading_line(stripped)
        if not heading:
            continue
        level, title = heading

        node = {"level": level, "title": title, "children": []}
        while stack and int(stack[-1]["level"]) >= level:
            stack.pop()

        if stack:
            stack[-1]["children"].append(node)
        else:
            hierarchy.append(node)
        stack.append(node)

    return hierarchy


def extract_table_previews(markdown: str, max_tables: int = 8) -> list[dict[str, object]]:
    previews: list[dict[str, object]] = []
    current_block: list[str] = []

    def flush() -> None:
        nonlocal current_block
        if len(current_block) < 2:
            current_block = []
            return
        header = current_block[0]
        rows = [line for line in current_block[2:] if "|" in line]
        if "|" not in header or not rows:
            current_block = []
            return
        previews.append(
            {
                "header": header,
                "sample_rows": rows[:3],
                "row_count_estimate": len(rows),
            }
        )
        current_block = []

    for line in markdown.splitlines():
        stripped = line.strip()
        if "|" in stripped:
            current_block.append(stripped)
            continue
        flush()
        if len(previews) >= max_tables:
            break

    if len(previews) < max_tables:
        flush()

    return previews[:max_tables]


def _build_docling_metadata(doc: Document, suffix: str) -> dict[str, object]:
    content = doc.content or ""
    return {
        **dict(doc.meta_data or {}),
        "source_type": suffix.lstrip("."),
        "parser": "docling",
        "content_format": "markdown",
        "sections": extract_section_hierarchy(content),
        "table_count": len(extract_table_previews(content, max_tables=100)),
        "table_previews": extract_table_previews(content),
    }


def _split_markdown_sections(document: Document) -> list[Document]:
    lines = document.content.splitlines()
    if not lines:
        return [document]

    sections: list[Document] = []
    current_title = "Preamble"
    current_lines: list[str] = []
    section_index = 0
    base_meta = dict(document.meta_data or {})

    for line in lines:
        heading = _parse_heading_line(line)
        if heading:
            if current_lines:
                section_index += 1
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append(
                        Document(
                            name=document.name,
                            content=content,
                            meta_data={
                                **base_meta,
                                "section_index": section_index,
                                "section_title": current_title,
                            },
                        )
                    )
            current_title = heading[1]
            current_lines = [line]
            continue

        current_lines.append(line)

    if current_lines:
        section_index += 1
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(
                Document(
                    name=document.name,
                    content=content,
                    meta_data={
                        **base_meta,
                        "section_index": section_index,
                        "section_title": current_title,
                    },
                )
            )

    return sections or [document]


def _infer_chunk_type(text: str) -> str:
    stripped = text.strip()
    if "<table" in stripped.lower() or ("\n|" in stripped and "|" in stripped):
        return "table"
    if stripped.startswith("#") or stripped.startswith("**"):
        return "section"
    return "text"


def chunk_legal_document(
    document: Document,
    chunk_size: int,
    max_chunks: int,
) -> tuple[list[Document], bool]:
    section_docs = _split_markdown_sections(document)
    semantic_chunker = DocumentChunking(chunk_size=chunk_size, overlap=0)
    recursive_chunker = RecursiveChunking(chunk_size=chunk_size, overlap=min(400, max(0, chunk_size // 10)))

    chunked_documents: list[Document] = []
    chunk_counter = 0

    current_group_text = ""
    current_group_titles: list[str] = []
    current_group_indices: list[int] = []

    def flush_group() -> None:
        nonlocal current_group_text, current_group_titles, current_group_indices, chunk_counter
        text = current_group_text.strip()
        if not text:
            current_group_text = ""
            current_group_titles = []
            current_group_indices = []
            return

        chunk_counter += 1
        chunked_documents.append(
            Document(
                name=document.name,
                content=text,
                meta_data={
                    **dict(document.meta_data or {}),
                    "section_title": current_group_titles[0] if len(current_group_titles) == 1 else current_group_titles[0],
                    "section_titles": current_group_titles[:],
                    "section_index": current_group_indices[0] if current_group_indices else None,
                    "section_index_range": current_group_indices[:],
                    "chunk_id": chunk_counter,
                    "chunk_type": _infer_chunk_type(text),
                    "source_file": document.name,
                },
            )
        )
        current_group_text = ""
        current_group_titles = []
        current_group_indices = []

    for section_doc in section_docs:
        section_text = section_doc.content.strip()
        if not section_text:
            continue

        section_title = str((section_doc.meta_data or {}).get("section_title") or "Section")
        section_index = (section_doc.meta_data or {}).get("section_index")
        candidate_group = f"{current_group_text}\n\n{section_text}".strip() if current_group_text else section_text

        if len(section_text) > chunk_size:
            flush_group()
            has_large_table = section_text.count("|") > 20 or "<table" in section_text.lower()
            chunker = semantic_chunker if has_large_table or section_text.count("\n\n") >= 2 else recursive_chunker
            section_chunks = chunker.chunk(section_doc)

            for chunk in section_chunks:
                chunk_counter += 1
                base_meta = dict(chunk.meta_data or {})
                section_titles = base_meta.get("section_titles") or [section_title]
                chunk.meta_data = {
                    **base_meta,
                    "section_title": base_meta.get("section_title") or section_title,
                    "section_titles": section_titles,
                    "chunk_id": chunk_counter,
                    "chunk_type": _infer_chunk_type(chunk.content),
                    "source_file": document.name,
                }
                chunked_documents.append(chunk)
            continue

        if current_group_text and len(candidate_group) > chunk_size:
            flush_group()

        current_group_text = f"{current_group_text}\n\n{section_text}".strip() if current_group_text else section_text
        if section_title not in current_group_titles:
            current_group_titles.append(section_title)
        if section_index is not None:
            current_group_indices.append(int(section_index))

    flush_group()

    truncated = len(chunked_documents) > max_chunks
    return chunked_documents[:max_chunks], truncated


def load_and_chunk_legal_document(
    file_path: str | Path,
    chunk_size: int,
    max_chunks: int,
) -> tuple[Document, list[Document], bool]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    reader = DoclingReader(output_format="markdown")
    reader.chunk = False
    documents = reader.read(path, name=path.name)
    if not documents:
        raise ValueError(f"Docling could not extract readable content from {path.name}")
    document = documents[0]
    document.meta_data = _build_docling_metadata(document, suffix)
    chunks, truncated = chunk_legal_document(document, chunk_size=chunk_size, max_chunks=max_chunks)
    return document, chunks, truncated


async def read_upload_document(file: UploadFile) -> Document:
    filename = file.filename or "uploaded_document"
    suffix = Path(filename).suffix.lower()
    data = await file.read()

    if suffix in {".txt", ".md"}:
        content = data.decode("utf-8", errors="replace")
        return Document(
            name=Path(filename).stem,
            content=content,
            meta_data={
                "source_type": suffix.lstrip("."),
                "parser": "text",
                "content_format": "text",
                "sections": extract_section_hierarchy(content),
                "table_count": len(extract_table_previews(content, max_tables=100)),
                "table_previews": extract_table_previews(content),
            },
        )

    if suffix == ".json":
        parsed = json.loads(data.decode("utf-8", errors="replace"))
        content = json.dumps(parsed, indent=2)
        return Document(
            name=Path(filename).stem,
            content=content,
            meta_data={
                "source_type": "json",
                "parser": "text",
                "content_format": "json",
                "sections": extract_section_hierarchy(content),
                "table_count": 0,
                "table_previews": [],
            },
        )

    if suffix in SUPPORTED_DOCLING_EXTENSIONS:
        reader = DoclingReader(output_format="markdown")
        reader.chunk = False
        stream = BytesIO(data)
        stream.name = filename
        documents = await reader.async_read(stream, name=filename)
        if not documents:
            raise ValueError(f"Docling could not extract readable content from {filename}")
        doc = documents[0]
        doc.meta_data = _build_docling_metadata(doc, suffix)
        return doc

    raise ValueError(f"Unsupported file extension: {suffix or '(none)'}")


async def read_upload_text(file: UploadFile) -> str:
    document = await read_upload_document(file)
    return document.content
