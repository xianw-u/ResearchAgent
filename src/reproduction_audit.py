import math
import re
from typing import Any


AuditItem = dict[str, Any]


def compare_config_to_profile(profile: dict[str, Any] | None, config_result: Any | None) -> list[AuditItem]:
    if not profile or not config_result:
        return []

    config = getattr(config_result, "extracted", {}) or {}
    items = [
        _compare_num_classes(profile, config),
        _compare_dataset(profile, config),
        _compare_input_size(profile, config),
        _compare_string_setting("optimizer", _training_value(profile, "optimizer"), config.get("optimizer")),
        _compare_numeric_setting("learning_rate", _training_value(profile, "learning_rate"), config.get("lr"), tolerance=0.15),
        _compare_string_setting("scheduler", _training_value(profile, "scheduler"), config.get("scheduler")),
        _compare_numeric_setting("batch_size", _training_value(profile, "batch_size"), config.get("batch_size"), tolerance=0.0),
        _compare_pretrained(profile, config),
    ]
    return [item for item in items if item]


def analyze_reproduction_gap(profile: dict[str, Any] | None, log_result: Any | None) -> list[AuditItem]:
    if not profile or not log_result:
        return []

    items: list[AuditItem] = []
    items.append(_compare_reported_miou(profile, getattr(log_result, "best_miou", None)))
    items.append(_compare_training_length(profile, getattr(log_result, "dataframe", None), getattr(log_result, "best_miou", None)))
    items.append(_analyze_loss_trend(getattr(log_result, "diagnosis", [])))
    items.append(_compare_lr_schedule(profile, getattr(log_result, "dataframe", None)))
    return [item for item in items if item]


def _compare_num_classes(profile: dict[str, Any], config: dict[str, Any]) -> AuditItem:
    datasets = _list_value(profile.get("datasets"))
    config_classes = config.get("num_classes")
    expected = None
    if any(name.upper() == "SUIM" for name in datasets):
        expected = 6
    if expected is None:
        return _unknown("num_classes", "Paper profile does not provide a reliable class-count rule.", "not specified", config_classes)
    if config_classes is None:
        return _unknown("num_classes", "Config does not expose num_classes.", expected, "not specified")
    if int(config_classes) == expected:
        return _match("num_classes", "Config num_classes matches the paper dataset rule.", expected, config_classes)
    return _mismatch("num_classes", "Config num_classes differs from the paper dataset rule.", expected, config_classes)


def _compare_dataset(profile: dict[str, Any], config: dict[str, Any]) -> AuditItem:
    datasets = _list_value(profile.get("datasets"))
    dataset_type = str(config.get("dataset_type", ""))
    data_root = str(config.get("data_root", ""))
    if not datasets:
        return _unknown("dataset_type / data_root", "Paper profile did not identify a dataset.", "not specified", dataset_type or data_root or "not specified")
    if not dataset_type and not data_root:
        return _unknown("dataset_type / data_root", "Config does not expose dataset_type or data_root.", datasets, "not specified")
    target = f"{dataset_type} {data_root}".lower()
    if any(dataset.lower() in target for dataset in datasets):
        return _match("dataset_type / data_root", "Config dataset information appears consistent with the paper dataset.", datasets, target)
    return _unknown("dataset_type / data_root", "Dataset naming differs or is indirect; manual verification is needed.", datasets, target)


def _compare_input_size(profile: dict[str, Any], config: dict[str, Any]) -> AuditItem:
    paper_size = _training_value(profile, "input_size")
    config_size = config.get("crop_size") or config.get("img_scale")
    paper_tuple = _size_tuple(paper_size)
    config_tuple = _size_tuple(config_size)
    if not paper_tuple:
        return _unknown("crop_size / input_size", "Paper profile does not specify input_size.", paper_size, config_size)
    if not config_tuple:
        return _unknown("crop_size / input_size", "Config does not expose crop_size or img_scale.", paper_size, config_size)
    if paper_tuple == config_tuple:
        return _match("crop_size / input_size", "Config input size matches the paper profile.", paper_tuple, config_tuple)
    return _mismatch("crop_size / input_size", "Config input size differs from the paper profile.", paper_tuple, config_tuple)


def _compare_pretrained(profile: dict[str, Any], config: dict[str, Any]) -> AuditItem:
    model_terms = " ".join(_list_value(profile.get("model_backbone"))).lower()
    has_transformer = "transformer" in model_terms or "vit" in model_terms or "swin" in model_terms
    config_weight = config.get("pretrained") or config.get("load_from")
    if not has_transformer:
        return _unknown("pretrained / load_from", "Paper profile does not clearly require pretrained Transformer weights.", "not specified", config_weight or "not specified")
    if config_weight:
        return _match("pretrained / load_from", "Config includes pretrained/load_from for a Transformer-style backbone.", "Transformer-style backbone", config_weight)
    return _unknown("pretrained / load_from", "Transformer-style backbone detected, but config does not expose pretrained/load_from.", "Transformer-style backbone", "not specified")


def _compare_string_setting(name: str, paper_value: Any, config_value: Any) -> AuditItem:
    if _is_missing(paper_value):
        return _unknown(name, f"Paper profile does not specify {name}.", paper_value, config_value)
    if config_value in (None, "", []):
        return _unknown(name, f"Config does not expose {name}.", paper_value, "not specified")
    paper_norm = str(paper_value).lower()
    config_norm = str(config_value).lower()
    if name == "scheduler" and paper_norm == "poly" and config_norm in {"lr_config", "param_scheduler", "scheduler"}:
        return _match(name, "Config exposes a scheduler field; verify that its policy is poly in the expanded config.", paper_value, config_value)
    if paper_norm in config_norm or config_norm in paper_norm:
        return _match(name, f"Config {name} matches the paper profile.", paper_value, config_value)
    return _mismatch(name, f"Config {name} differs from the paper profile.", paper_value, config_value)


def _compare_numeric_setting(name: str, paper_value: Any, config_value: Any, tolerance: float) -> AuditItem:
    paper_number = _number(paper_value)
    config_number = _number(config_value)
    if paper_number is None:
        return _unknown(name, f"Paper profile does not specify {name}.", paper_value, config_value)
    if config_number is None:
        return _unknown(name, f"Config does not expose {name}.", paper_value, "not specified")
    if tolerance == 0.0:
        matched = paper_number == config_number
    else:
        denominator = max(abs(paper_number), 1e-12)
        matched = abs(paper_number - config_number) / denominator <= tolerance
    if matched:
        return _match(name, f"Config {name} is consistent with the paper profile.", paper_value, config_value)
    return _mismatch(name, f"Config {name} differs from the paper profile.", paper_value, config_value)


def _compare_reported_miou(profile: dict[str, Any], best_miou: dict[str, Any] | None) -> AuditItem:
    paper_miou = _reported_metric(profile, "miou")
    user_miou = _number(best_miou.get("mIoU")) if best_miou else None
    if paper_miou is None:
        return _unknown("best mIoU gap", "Paper profile does not provide a reported mIoU.", "not specified", user_miou)
    if user_miou is None:
        return _unknown("best mIoU gap", "Log does not expose best mIoU.", paper_miou, "not specified")
    paper_norm, user_norm = _same_miou_scale(paper_miou, user_miou)
    gap = user_norm - paper_norm
    level = "MATCH" if abs(gap) <= 0.02 else "MISMATCH"
    message = f"User best mIoU differs from paper by {gap:+.4f}."
    return _item(level, "best mIoU gap", message, paper_norm, user_norm)


def _compare_training_length(profile: dict[str, Any], dataframe: Any, best_miou: dict[str, Any] | None) -> AuditItem:
    paper_length = _number(_training_value(profile, "iterations_or_epochs"))
    if paper_length is None:
        return _unknown("training length", "Paper profile does not specify total training length.", "not specified", best_miou)
    user_length = _max_log_position(dataframe)
    if user_length is None:
        return _unknown("training length", "Log does not expose enough iter/epoch records to estimate total training length.", paper_length, "not specified")
    ratio = user_length / paper_length if paper_length else 0
    if 0.5 <= ratio <= 1.2:
        return _match("training length", "Parsed log length is close to the paper's stated total training length. The paper does not claim the best checkpoint occurs at this endpoint.", paper_length, user_length)
    return _mismatch("training length", "Parsed log length is far from the paper's stated total training length; verify whether the run is incomplete or uses a different schedule.", paper_length, user_length)


def _analyze_loss_trend(diagnosis: list[str]) -> AuditItem:
    joined = " ".join(diagnosis).lower()
    if "loss generally decreases" in joined:
        return _match("loss trend", "Loss decreases normally in the parsed log.", "decreasing loss expected", "decreasing loss observed")
    if "loss does not clearly decrease" in joined or "nan" in joined or "inf" in joined:
        return _mismatch("loss trend", "Loss trend suggests a training stability or convergence issue.", "stable decreasing loss expected", diagnosis)
    return _unknown("loss trend", "Log diagnosis does not contain enough loss signal.", "stable decreasing loss expected", diagnosis)


def _compare_lr_schedule(profile: dict[str, Any], dataframe: Any) -> AuditItem:
    paper_scheduler = _training_value(profile, "scheduler")
    if _is_missing(paper_scheduler):
        return _unknown("lr schedule", "Paper profile does not specify scheduler.", "not specified", "not evaluated")
    if dataframe is None or "lr" not in getattr(dataframe, "columns", []):
        return _unknown("lr schedule", "Log does not contain learning-rate records.", paper_scheduler, "not specified")
    lr = dataframe["lr"].dropna()
    if lr.empty:
        return _unknown("lr schedule", "Log learning-rate column is empty.", paper_scheduler, "not specified")
    if lr.nunique() > 1:
        return _match("lr schedule", "Log learning rate changes over time, consistent with a scheduled run.", paper_scheduler, "changing lr observed")
    return _unknown("lr schedule", "Learning rate is constant in the parsed log; schedule consistency cannot be confirmed.", paper_scheduler, "constant lr observed")


def _max_log_position(dataframe: Any) -> float | None:
    if dataframe is None:
        return None
    columns = getattr(dataframe, "columns", [])
    if "iter" in columns:
        values = dataframe["iter"].dropna()
        if not values.empty:
            return float(values.max())
    if "epoch" in columns:
        values = dataframe["epoch"].dropna()
        if not values.empty:
            return float(values.max())
    return None


def _training_value(profile: dict[str, Any], key: str) -> Any:
    return _field_value(profile.get("training_settings", {}).get(key))


def _field_value(field: Any) -> Any:
    if isinstance(field, dict):
        return field.get("value", "not specified")
    return field


def _list_value(field: Any) -> list[str]:
    value = _field_value(field)
    if value in (None, "not specified"):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _reported_metric(profile: dict[str, Any], metric_name: str) -> float | None:
    for result in profile.get("reported_results", []):
        metric = str(result.get("metric", "")).lower()
        if metric_name in metric:
            return _number(result.get("value"))
    return None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    text = str(value).replace("−", "-").replace("×", "x")
    scientific = re.search(r"([-+]?\d*\.?\d+)\s*[xX]\s*10\s*(-?\s*\d+)", text)
    if scientific:
        try:
            base = float(scientific.group(1))
            exponent = int(scientific.group(2).replace(" ", ""))
            return base * (10 ** exponent)
        except ValueError:
            return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


# Overrides the basic parser above with support for paper-style values such as
# "160k iterations" and "6 x 10^-5" / "6 × 10−5".
def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)

    text = str(value).replace("\u2212", "-").replace("\u00d7", "x")
    scientific = re.search(r"([-+]?\d*\.?\d+)\s*[xX]\s*10\s*(-?\s*\d+)", text)
    if scientific:
        try:
            base = float(scientific.group(1))
            exponent = int(scientific.group(2).replace(" ", ""))
            return base * (10 ** exponent)
        except ValueError:
            return None

    k_match = re.search(r"([-+]?\d*\.?\d+)\s*[kK]\b", text)
    if k_match:
        try:
            return float(k_match.group(1)) * 1000
        except ValueError:
            return None

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _same_miou_scale(paper_value: float, user_value: float) -> tuple[float, float]:
    if paper_value > 1 and user_value <= 1:
        return paper_value / 100.0, user_value
    if paper_value <= 1 and user_value > 1:
        return paper_value, user_value / 100.0
    return paper_value, user_value


def _size_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, tuple) and len(value) >= 2:
        return int(value[0]), int(value[1])
    numbers = re.findall(r"\d+", str(value))
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    return None


def _is_missing(value: Any) -> bool:
    return value in (None, "", [], "not specified")


def _match(item: str, message: str, paper: Any, user: Any) -> AuditItem:
    return _item("MATCH", item, message, paper, user)


def _mismatch(item: str, message: str, paper: Any, user: Any) -> AuditItem:
    return _item("MISMATCH", item, message, paper, user)


def _unknown(item: str, message: str, paper: Any, user: Any) -> AuditItem:
    return _item("UNKNOWN", item, message, paper, user)


def _item(level: str, item: str, message: str, paper: Any, user: Any) -> AuditItem:
    return {
        "level": level,
        "item": item,
        "message": message,
        "paper": paper if paper not in (None, "", []) else "not specified",
        "user": user if user not in (None, "", []) else "not specified",
    }
