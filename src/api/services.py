import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config_checker import ConfigCheckResult, check_config
from src.log_analyzer import LogAnalysisResult, analyze_log
from src.paper_profile_extractor import extract_paper_profile
from src.paper_qa import PaperIndex, answer_question, build_paper_index, extract_pdf_text
from src.report_generator import build_report
from src.reproduction_audit import analyze_reproduction_gap, compare_config_to_profile


def run_paper_qa(
    pdf_bytes: bytes,
    file_name: str,
    question: str,
    top_k: int = 3,
    include_profile: bool = True,
) -> dict[str, Any]:
    if not pdf_bytes:
        raise ValueError("Uploaded PDF is empty.")
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    index = _build_index_from_pdf_bytes(pdf_bytes, file_name)
    answer = answer_question(index, question.strip(), top_k=top_k)
    profile = extract_paper_profile(index.chunks) if include_profile else None

    return {
        "file_name": file_name,
        "question": question.strip(),
        "answer": answer["answer"],
        "sources": answer["sources"],
        "profile": profile,
    }


def run_log_analysis(text: str, file_name: str) -> tuple[LogAnalysisResult, dict[str, Any]]:
    result = analyze_log(text, file_name=file_name)
    response = serialize_log_result(result)
    return result, response


def run_config_check(text: str, file_name: str) -> tuple[ConfigCheckResult, dict[str, Any]]:
    result = check_config(text, file_name=file_name)
    response = serialize_config_result(result)
    return result, response


def run_reproduction_audit(
    pdf_bytes: bytes | None,
    pdf_name: str | None,
    log_text: str | None,
    log_name: str | None,
    config_text: str | None,
    config_name: str | None,
    title: str,
    notes: str = "",
) -> dict[str, Any]:
    paper_profile = None
    log_result = None
    config_result = None
    log_response = None
    config_response = None

    if pdf_bytes:
        index = _build_index_from_pdf_bytes(pdf_bytes, pdf_name or "paper.pdf")
        paper_profile = extract_paper_profile(index.chunks)

    if log_text and log_text.strip():
        log_result, log_response = run_log_analysis(log_text, log_name or "training.log")

    if config_text and config_text.strip():
        config_result, config_response = run_config_check(config_text, config_name or "config.py")

    config_consistency = compare_config_to_profile(paper_profile, config_result)
    reproduction_gap = analyze_reproduction_gap(paper_profile, log_result)
    report = build_report(
        title=title,
        paper_profile=paper_profile,
        log_result=log_result,
        config_result=config_result,
        config_consistency=config_consistency,
        reproduction_gap=reproduction_gap,
        notes=notes,
    )

    return {
        "paper_profile": _jsonable(paper_profile),
        "config_check": config_response,
        "log_analysis": log_response,
        "config_consistency": _jsonable(config_consistency),
        "reproduction_gap": _jsonable(reproduction_gap),
        "report_markdown": report,
    }


def serialize_log_result(result: LogAnalysisResult, record_limit: int = 500) -> dict[str, Any]:
    dataframe = result.dataframe.copy()
    records = dataframe.head(record_limit).where(pd.notnull(dataframe), None).to_dict(orient="records")
    return {
        "file_name": result.file_name,
        "row_count": int(len(dataframe)),
        "columns": [str(column) for column in dataframe.columns],
        "records": _jsonable(records),
        "diagnosis": result.diagnosis,
        "best_miou": _jsonable(result.best_miou),
    }


def serialize_config_result(result: ConfigCheckResult) -> dict[str, Any]:
    return {
        "file_name": result.file_name,
        "extracted": _jsonable(result.extracted),
        "findings": _jsonable(result.findings),
    }


def decode_text_file(file_bytes: bytes, file_name: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode {file_name}. Please upload a UTF-8 text file.")


def _build_index_from_pdf_bytes(pdf_bytes: bytes, file_name: str) -> PaperIndex:
    suffix = Path(file_name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(pdf_bytes)
        temp_path = Path(temp_file.name)

    try:
        pages = extract_pdf_text(temp_path)
        return build_paper_index(pages)
    finally:
        temp_path.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, pd.DataFrame):
        return value.where(pd.notnull(value), None).to_dict(orient="records")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value

