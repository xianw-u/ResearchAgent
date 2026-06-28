from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    APIError,
    ConfigCheckResponse,
    HealthResponse,
    LogAnalysisResponse,
    PaperQAResponse,
    ReproductionAuditResponse,
)
from src.api.services import (
    decode_text_file,
    run_config_check,
    run_log_analysis,
    run_paper_qa,
    run_reproduction_audit,
)

API_VERSION = "0.2.0"

app = FastAPI(
    title="ResearchAgent API",
    version=API_VERSION,
    description=(
        "Service API for paper-grounded QA, training-log diagnosis, "
        "MMSeg config inspection, and reproducibility audit reports."
    ),
    contact={"name": "ResearchAgent"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse, tags=["System"])
def root() -> HealthResponse:
    return HealthResponse(version=API_VERSION)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    return HealthResponse(version=API_VERSION)


@app.post(
    "/api/v1/papers/qa",
    response_model=PaperQAResponse,
    responses={400: {"model": APIError}},
    tags=["Paper"],
)
async def paper_qa(
    file: Annotated[UploadFile, File(description="Paper PDF file")],
    question: Annotated[str, Form(description="Question grounded in the uploaded paper")],
    top_k: Annotated[int, Form(ge=1, le=10)] = 3,
    include_profile: Annotated[bool, Form()] = True,
) -> PaperQAResponse:
    _ensure_suffix(file.filename, {".pdf"})
    pdf_bytes = await file.read()
    return _handle(lambda: run_paper_qa(pdf_bytes, file.filename or "paper.pdf", question, top_k, include_profile))


@app.post(
    "/api/v1/logs/analyze",
    response_model=LogAnalysisResponse,
    responses={400: {"model": APIError}},
    tags=["Experiment Logs"],
)
async def analyze_training_log(
    file: Annotated[UploadFile, File(description="Training log, txt, log, or jsonl")],
) -> LogAnalysisResponse:
    _ensure_suffix(file.filename, {".txt", ".log", ".jsonl"})
    text = decode_text_file(await file.read(), file.filename or "training.log")
    _, response = _handle(lambda: run_log_analysis(text, file.filename or "training.log"))
    return response


@app.post(
    "/api/v1/configs/check",
    response_model=ConfigCheckResponse,
    responses={400: {"model": APIError}},
    tags=["MMSeg Config"],
)
async def check_mmseg_config(
    file: Annotated[UploadFile, File(description="MMSegmentation Python config")],
) -> ConfigCheckResponse:
    _ensure_suffix(file.filename, {".py", ".txt"})
    text = decode_text_file(await file.read(), file.filename or "config.py")
    _, response = _handle(lambda: run_config_check(text, file.filename or "config.py"))
    return response


@app.post(
    "/api/v1/reproduction/audit",
    response_model=ReproductionAuditResponse,
    responses={400: {"model": APIError}},
    tags=["Reproducibility"],
)
async def reproduction_audit(
    paper: Annotated[UploadFile | None, File(description="Optional paper PDF")] = None,
    training_log: Annotated[UploadFile | None, File(description="Optional training log")] = None,
    config: Annotated[UploadFile | None, File(description="Optional MMSeg config")] = None,
    title: Annotated[str, Form()] = "ResearchAgent Reproduction Audit Report",
    notes: Annotated[str, Form()] = "",
) -> ReproductionAuditResponse:
    paper_bytes = await paper.read() if paper else None
    log_text = decode_text_file(await training_log.read(), training_log.filename or "training.log") if training_log else None
    config_text = decode_text_file(await config.read(), config.filename or "config.py") if config else None

    return _handle(
        lambda: run_reproduction_audit(
            pdf_bytes=paper_bytes,
            pdf_name=paper.filename if paper else None,
            log_text=log_text,
            log_name=training_log.filename if training_log else None,
            config_text=config_text,
            config_name=config.filename if config else None,
            title=title,
            notes=notes,
        )
    )


def _ensure_suffix(file_name: str | None, allowed_suffixes: set[str]) -> None:
    suffix = ""
    if file_name and "." in file_name:
        suffix = "." + file_name.rsplit(".", 1)[-1].lower()
    if suffix not in allowed_suffixes:
        allowed = ", ".join(sorted(allowed_suffixes))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Expected one of: {allowed}.")


def _handle(callback):
    try:
        return callback()
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ResearchAgent API failed: {exc}") from exc

