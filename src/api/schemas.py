from typing import Any

from pydantic import BaseModel, Field


class APIError(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "ResearchAgent API"
    version: str


class Finding(BaseModel):
    level: str
    message: str


class SourceSnippet(BaseModel):
    page: int
    score: float
    text: str


class PaperQAResponse(BaseModel):
    file_name: str
    question: str
    answer: str
    sources: list[SourceSnippet]
    profile: dict[str, Any] | None = None


class LogAnalysisResponse(BaseModel):
    file_name: str
    row_count: int
    columns: list[str]
    records: list[dict[str, Any]]
    diagnosis: list[str]
    best_miou: dict[str, Any] | None = None


class ConfigCheckResponse(BaseModel):
    file_name: str
    extracted: dict[str, Any]
    findings: list[Finding]


class ReproductionAuditResponse(BaseModel):
    paper_profile: dict[str, Any] | None = None
    config_check: ConfigCheckResponse | None = None
    log_analysis: LogAnalysisResponse | None = None
    config_consistency: list[dict[str, Any]] = Field(default_factory=list)
    reproduction_gap: list[dict[str, Any]] = Field(default_factory=list)
    report_markdown: str

