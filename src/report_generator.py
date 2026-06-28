from datetime import datetime
from typing import Any


def build_report(
    title: str,
    paper_answer: dict[str, Any] | None = None,
    paper_profile: dict[str, Any] | None = None,
    log_result: Any | None = None,
    config_result: Any | None = None,
    config_consistency: list[dict[str, Any]] | None = None,
    reproduction_gap: list[dict[str, Any]] | None = None,
    notes: str = "",
) -> str:
    report_title = title or "ResearchAgent Reproduction Audit Report"
    lines: list[str] = [
        f"# {report_title}",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "> If the paper does not provide source code or complete experimental settings, ResearchAgent can only perform evidence-based analysis from the paper text, user config, and training logs. It cannot guarantee exact reproduction of the authors' implementation.",
        "",
        "## 1. Paper Reference Profile",
        "",
    ]

    if paper_profile:
        _append_paper_profile(lines, paper_profile)
    else:
        lines.append("No structured paper profile is available.")

    lines.extend(["", "## 2. Missing Reproducibility Details", ""])
    if paper_profile and paper_profile.get("missing_details"):
        lines.extend(f"- {item}" for item in paper_profile["missing_details"])
    else:
        lines.append("- No major missing detail was detected from the extracted profile.")

    lines.extend(["", "## 3. Config Consistency Check", ""])
    if config_result:
        lines.append(f"Config file: `{config_result.file_name}`")
        lines.append("")
        lines.append("### Extracted Fields")
        for key, value in config_result.extracted.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
        lines.append("### Consistency Against Paper Profile")
        _append_audit_items(lines, config_consistency or [])
        lines.append("")
        lines.append("### Original Config Findings")
        for finding in config_result.findings:
            lines.append(f"- **{finding['level']}**: {finding['message']}")
    else:
        lines.append("No config check result is available.")

    lines.extend(["", "## 4. Training Log Diagnosis", ""])
    if log_result:
        lines.append(f"Log file: `{log_result.file_name}`")
        if log_result.best_miou:
            lines.append(f"- Best mIoU: `{log_result.best_miou['mIoU']:.4f}`")
        lines.append("")
        lines.append("### Diagnosis")
        for item in log_result.diagnosis:
            lines.append(f"- {item}")
    else:
        lines.append("No training log analysis result is available.")

    lines.extend(["", "## 5. Reproduction Gap Analysis", ""])
    _append_audit_items(lines, reproduction_gap or [])

    lines.extend(["", "## 6. Suggested Next Actions", ""])
    issues = _collect_issues(log_result, config_result)
    if issues:
        lines.append("Prioritize these issues:")
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- No major issue was automatically identified.")
    lines.extend([
        "- Fill missing reproducibility details from source code, appendix, or author repository if available.",
        "- Validate the best checkpoint on the target validation set.",
        "- Compare config settings against the paper profile before running long experiments.",
        "- Add ablation experiments only after the baseline reproduction gap is understood.",
    ])

    if notes.strip():
        lines.extend(["", "### Additional Notes", "", notes.strip()])

    return "\n".join(lines).strip() + "\n"


def _append_paper_profile(lines: list[str], profile: dict[str, Any]) -> None:
    for key in ("task", "method_name", "datasets", "metrics", "model_backbone", "baselines", "ablation_items"):
        field = profile.get(key, {})
        lines.append(f"- **{key}**: `{field.get('value', 'not specified')}` ({field.get('evidence_level', 'missing')})")
        _append_profile_sources(lines, field)

    lines.extend(["", "### Training Settings", ""])
    for key, field in profile.get("training_settings", {}).items():
        lines.append(f"- **{key}**: `{field.get('value', 'not specified')}` ({field.get('evidence_level', 'missing')})")
        _append_profile_sources(lines, field)

    lines.extend(["", "### Reported Results", ""])
    for item in profile.get("reported_results", []):
        lines.append(
            f"- Dataset: `{item.get('dataset', 'not specified')}`, "
            f"Metric: `{item.get('metric', 'not specified')}`, "
            f"Value: `{item.get('value', 'not specified')}` "
            f"({item.get('evidence_level', 'missing')})"
        )
        _append_profile_sources(lines, item)

    missing = profile.get("missing_details", [])
    if missing:
        lines.extend(["", "### Missing Details From Profile", ""])
        lines.extend(f"- {item}" for item in missing)


def _append_audit_items(lines: list[str], items: list[dict[str, Any]]) -> None:
    if not items:
        lines.append("- No profile-based audit result is available.")
        return
    for item in items:
        lines.append(
            f"- **{item.get('level', 'UNKNOWN')}** `{item.get('item', '')}`: "
            f"{item.get('message', '')} "
            f"(paper: `{item.get('paper', 'not specified')}`, user: `{item.get('user', 'not specified')}`)"
        )


def _append_profile_sources(lines: list[str], field: dict[str, Any]) -> None:
    sources = field.get("evidence_sources", [])
    for source in sources[:2]:
        page = source.get("page", "?")
        text = source.get("text", "")
        lines.append(f"  - Evidence [p.{page}]: {text}")


def _collect_issues(log_result: Any | None, config_result: Any | None) -> list[str]:
    issues: list[str] = []
    if log_result:
        for item in log_result.diagnosis:
            lowered = item.lower()
            if any(keyword in lowered for keyword in ("nan", "inf", "does not clearly", "plateau", "high")):
                issues.append(item)
    if config_result:
        for finding in config_result.findings:
            if finding["level"] in {"ERROR", "WARNING"}:
                issues.append(finding["message"])
    return issues
