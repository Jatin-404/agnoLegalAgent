from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.core.document_loader import read_upload_document
from app.core.schemas import (
    SingleExtractionRequest,
    SingleExtractionResponse,
)
from app.core.service_v2 import run_single_extraction_v2

router = APIRouter()


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "model": settings.v2_model_id,
        "parser": "docling",
    }


@router.post("/extract", response_model=SingleExtractionResponse)
async def extract_single(request: SingleExtractionRequest) -> SingleExtractionResponse:
    try:
        return await run_single_extraction_v2(request, parser_mode="text")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/v2/extract", response_model=SingleExtractionResponse, include_in_schema=False)
async def extract_single_v2_alias(request: SingleExtractionRequest) -> SingleExtractionResponse:
    return await extract_single(request)


@router.post("/extract-upload", response_model=SingleExtractionResponse)
async def extract_upload(
    file: UploadFile = File(...),
    jurisdiction_hint: str | None = Form(default="India"),
) -> SingleExtractionResponse:
    try:
        document = await read_upload_document(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not document.content.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the uploaded file")

    request = SingleExtractionRequest(
        document_name=file.filename or "uploaded_document",
        document_text=document.content,
        jurisdiction_hint=jurisdiction_hint,
        document_metadata=document.meta_data,
    )
    try:
        return await run_single_extraction_v2(request, parser_mode="docling")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/v2/extract-upload", response_model=SingleExtractionResponse, include_in_schema=False)
async def extract_upload_v2(
    file: UploadFile = File(...),
    jurisdiction_hint: str | None = Form(default="India"),
) -> SingleExtractionResponse:
    return await extract_upload(file=file, jurisdiction_hint=jurisdiction_hint)
