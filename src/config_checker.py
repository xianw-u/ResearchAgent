import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ConfigCheckResult:
    file_name: str
    extracted: dict[str, Any]
    findings: list[dict[str, str]]


def check_config(text: str, file_name: str = "config.py") -> ConfigCheckResult:
    extracted = extract_config_fields(text)
    findings = run_checks(text, extracted)
    return ConfigCheckResult(file_name=file_name, extracted=extracted, findings=findings)


def extract_config_fields(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "base_configs": _extract_base_configs(text),
        "dataset_type": _extract_string_assignment(text, "dataset_type"),
        "data_root": _extract_string_assignment(text, "data_root"),
        "num_classes": _extract_int_key(text, "num_classes"),
        "crop_size": _extract_tuple_assignment(text, "crop_size"),
        "img_scale": _extract_key_or_assignment(text, "img_scale"),
        "optimizer": _extract_optimizer(text),
        "lr": _extract_lr(text),
        "weight_decay": _extract_float_key(text, "weight_decay"),
        "batch_size": _extract_batch_size(text),
        "pretrained": _extract_string_key_or_assignment(text, "pretrained"),
        "load_from": _extract_string_assignment(text, "load_from"),
        "scheduler": _detect_scheduler(text),
        "pipeline_steps": _extract_pipeline_steps(text),
    }
    return {key: value for key, value in fields.items() if value not in (None, [], {})}


def run_checks(text: str, fields: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    has_base_configs = bool(fields.get("base_configs"))

    if has_base_configs:
        findings.append(_finding(
            "INFO",
            "This config references _base_ files. Missing fields may be inherited; upload the base configs together for a complete check.",
        ))

    dataset_type = fields.get("dataset_type")
    num_classes = fields.get("num_classes")
    if dataset_type == "SUIMDataset":
        if num_classes != 6:
            findings.append(_finding("WARNING", "SUIMDataset usually has 6 classes; confirm num_classes matches annotations."))
        else:
            findings.append(_finding("INFO", "SUIMDataset class count appears consistent with the built-in rule."))
    elif num_classes is not None:
        findings.append(_finding("INFO", "Dataset class count cannot be confirmed by built-in rules; manually verify num_classes."))
    else:
        findings.append(_finding("WARNING", "num_classes was not detected; verify decode head and auxiliary head class settings."))

    data_root = fields.get("data_root")
    if not data_root:
        findings.append(_finding(_missing_level(has_base_configs), _missing_message("data_root", has_base_configs)))
    else:
        data_root_path = Path(str(data_root))
        resolved_path = data_root_path if data_root_path.is_absolute() else PROJECT_ROOT / data_root_path
        if not resolved_path.exists():
            findings.append(_finding(
                "WARNING",
                (
                    "Detected data_root does not exist locally. "
                    f"config data_root=`{data_root}`; resolved path=`{resolved_path}`. "
                    "If the dataset is stored elsewhere, update data_root or ignore this local-path warning."
                ),
            ))
        else:
            findings.append(_finding("INFO", f"data_root exists locally at `{resolved_path}`."))

    crop_size = fields.get("crop_size")
    img_scale = fields.get("img_scale")
    if crop_size and img_scale and crop_size != img_scale:
        findings.append(_finding("INFO", "Training crop_size and test/img_scale differ; confirm this is intentional."))

    optimizer = fields.get("optimizer")
    if not optimizer:
        findings.append(_finding("INFO" if has_base_configs else "ERROR", _missing_message("optimizer", has_base_configs)))
    else:
        findings.append(_finding("INFO", f"Detected optimizer: {optimizer}."))

    lr = fields.get("lr")
    if lr is None:
        findings.append(_finding(_missing_level(has_base_configs), _missing_message("learning rate", has_base_configs)))
    elif lr > 1e-3:
        findings.append(_finding("WARNING", "Learning rate is larger than 1e-3; confirm it matches model and batch size."))
    else:
        findings.append(_finding("INFO", f"Detected learning rate: {lr}."))

    if "pretrained" not in fields and "load_from" not in fields:
        findings.append(_finding("INFO", "No pretrained/load_from setting detected; confirm whether Transformer backbone needs pretrained weights."))

    pipeline_steps = fields.get("pipeline_steps", {})
    train_steps = set(pipeline_steps.get("train", []))
    test_steps = set(pipeline_steps.get("test", [])) | set(pipeline_steps.get("val", []))
    if train_steps:
        if "LoadAnnotations" not in train_steps:
            findings.append(_finding("WARNING", "train pipeline may miss LoadAnnotations."))
        for required in ("Resize", "Normalize", "Pad"):
            if required not in train_steps:
                findings.append(_finding("INFO", f"train pipeline may miss {required}; verify preprocessing."))
    if "Normalize" in train_steps and test_steps and "Normalize" not in test_steps:
        findings.append(_finding("WARNING", "train pipeline has Normalize but val/test pipeline may not."))

    if not fields.get("scheduler"):
        findings.append(_finding(_missing_level(has_base_configs), _missing_message("scheduler/lr_config/param_scheduler", has_base_configs)))
    else:
        findings.append(_finding("INFO", f"Detected scheduler config: {fields['scheduler']}."))

    return findings


def _finding(level: str, message: str) -> dict[str, str]:
    return {"level": level, "message": message}


def _missing_level(has_base_configs: bool) -> str:
    return "INFO" if has_base_configs else "WARNING"


def _missing_message(field_name: str, has_base_configs: bool) -> str:
    if has_base_configs:
        return f"{field_name} was not detected in the uploaded text; it may be defined in _base_ configs."
    return f"{field_name} was not detected."


def _extract_base_configs(text: str) -> list[str]:
    match = re.search(r"\b_base_\s*=\s*(\[[^\]]+\]|['\"][^'\"]+['\"])", text, re.DOTALL)
    if not match:
        return []
    raw_value = match.group(1)
    try:
        value = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        found = re.findall(r"['\"]([^'\"]+\.py)['\"]", raw_value)
        return found
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _extract_string_assignment(text: str, name: str) -> str | None:
    match = re.search(rf"\b{name}\s*=\s*['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else None


def _extract_string_key_or_assignment(text: str, name: str) -> str | None:
    assignment = _extract_string_assignment(text, name)
    if assignment:
        return assignment
    match = re.search(rf"['\"]{name}['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else None


def _extract_int_key(text: str, name: str) -> int | None:
    match = re.search(rf"\b{name}\s*=\s*(\d+)|['\"]{name}['\"]\s*:\s*(\d+)", text)
    if not match:
        return None
    return int(next(group for group in match.groups() if group is not None))


def _extract_float_key(text: str, name: str) -> float | None:
    match = re.search(rf"\b{name}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)|['\"]{name}['\"]\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if not match:
        return None
    return float(next(group for group in match.groups() if group is not None))


def _extract_tuple_assignment(text: str, name: str) -> tuple[int, ...] | None:
    match = re.search(rf"\b{name}\s*=\s*(\([^)]+\))", text)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None
    if isinstance(value, tuple):
        return tuple(int(item) for item in value if isinstance(item, int))
    return None


def _extract_key_or_assignment(text: str, name: str) -> Any:
    assignment = _extract_tuple_assignment(text, name)
    if assignment:
        return assignment
    match = re.search(rf"['\"]{name}['\"]\s*:\s*(\([^)]+\)|\[[^\]]+\]|[-+]?\d+)", text)
    if not match:
        return None
    try:
        return ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return match.group(1)


def _extract_optimizer(text: str) -> str | None:
    match = re.search(r"optimizer\s*=\s*dict\([^)]*type\s*=\s*['\"]([^'\"]+)['\"]", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"['\"]optimizer['\"]\s*:\s*\{[^}]*['\"]type['\"]\s*:\s*['\"]([^'\"]+)['\"]", text, re.DOTALL)
    return match.group(1) if match else None


def _extract_lr(text: str) -> float | None:
    optimizer_block = re.search(r"optimizer\s*=\s*dict\((.*?)\)", text, re.DOTALL)
    if optimizer_block:
        match = re.search(r"\blr\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", optimizer_block.group(1))
        if match:
            return float(match.group(1))
    return _extract_float_key(text, "lr")


def _extract_batch_size(text: str) -> int | None:
    for name in ("batch_size", "samples_per_gpu"):
        value = _extract_int_key(text, name)
        if value is not None:
            return value
    return None


def _detect_scheduler(text: str) -> str | None:
    if "param_scheduler" in text:
        return "param_scheduler"
    if "lr_config" in text:
        return "lr_config"
    if re.search(r"\bscheduler\b", text):
        return "scheduler"
    return None


def _extract_pipeline_steps(text: str) -> dict[str, list[str]]:
    steps: dict[str, list[str]] = {}
    for pipeline_name in ("train_pipeline", "test_pipeline", "val_pipeline"):
        match = re.search(rf"{pipeline_name}\s*=\s*\[(.*?)\]", text, re.DOTALL)
        if not match:
            continue
        found = re.findall(r"type\s*=\s*['\"]([^'\"]+)['\"]|['\"]type['\"]\s*:\s*['\"]([^'\"]+)['\"]", match.group(1))
        key = pipeline_name.replace("_pipeline", "")
        steps[key] = [left or right for left, right in found]
    return steps
