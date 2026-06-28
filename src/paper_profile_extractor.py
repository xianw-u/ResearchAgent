import re
from dataclasses import asdict, dataclass
from typing import Any

from src.paper_qa import PaperChunk
from src.utils import truncate_text


NOT_SPECIFIED = "not specified"


@dataclass
class EvidenceSource:
    page: int
    text: str


@dataclass
class ProfileField:
    value: Any
    evidence_level: str
    evidence_sources: list[EvidenceSource]


GENERAL_FIELDS = (
    "task",
    "method_name",
    "datasets",
    "metrics",
    "model_backbone",
    "baselines",
    "ablation_items",
)

TRAINING_FIELDS = (
    "optimizer",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "input_size",
    "scheduler",
    "iterations_or_epochs",
)


def extract_paper_profile(chunks: list[PaperChunk]) -> dict[str, Any]:
    profile = {
        "task": _extract_task(chunks),
        "method_name": _extract_method_name(chunks),
        "datasets": _extract_keyword_list(chunks, _dataset_patterns()),
        "metrics": _extract_keyword_list(chunks, _metric_patterns()),
        "model_backbone": _extract_keyword_list(chunks, _model_patterns()),
        "training_settings": _extract_training_settings(chunks),
        "reported_results": _extract_reported_results(chunks),
        "baselines": _extract_keyword_list(chunks, _baseline_patterns()),
        "ablation_items": _extract_ablation_items(chunks),
    }
    profile["missing_details"] = _build_missing_details(profile)
    return _to_plain_dict(profile)


def _field(value: Any, level: str, sources: list[EvidenceSource] | None = None) -> ProfileField:
    return ProfileField(value=value, evidence_level=level, evidence_sources=sources or [])


def _missing_field() -> ProfileField:
    return _field(NOT_SPECIFIED, "missing", [])


def _explicit_field(value: Any, sources: list[EvidenceSource]) -> ProfileField:
    return _field(value, "explicit", sources)


def _extract_task(chunks: list[PaperChunk]) -> ProfileField:
    task_patterns = [
        ("underwater semantic segmentation", r"\bunderwater semantic segmentation\b"),
        ("semantic segmentation", r"\bsemantic segmentation\b"),
        ("image segmentation", r"\bimage segmentation\b"),
        ("object detection", r"\bobject detection\b"),
        ("image classification", r"\bimage classification\b"),
    ]
    for label, pattern in task_patterns:
        sources = _find_sources(chunks, pattern, limit=3)
        if sources:
            return _explicit_field(label, sources)
    return _missing_field()


def _extract_method_name(chunks: list[PaperChunk]) -> ProfileField:
    patterns = [
        r"\b(?:called|named)\s+([A-Z][A-Za-z0-9_-]{2,})\b",
        r"\b(?:we\s+)?propose\s+(?:a|an)?\s*(?:novel\s+)?(?:method|model|network|framework)?\s*(?:called|named)?\s*([A-Z][A-Za-z0-9_-]{2,})\b",
        r"\b([A-Z][A-Za-z0-9_-]{2,})\s+(?:is\s+)?(?:a\s+)?(?:novel\s+)?(?:method|model|network|framework)\b",
    ]
    for pattern in patterns:
        match = _first_regex_match(chunks, pattern)
        if match:
            value, source = match
            return _explicit_field(value, [source])
    return _missing_field()


def _extract_keyword_list(chunks: list[PaperChunk], patterns: dict[str, str]) -> ProfileField:
    found: dict[str, EvidenceSource] = {}
    for label, pattern in patterns.items():
        sources = _find_sources(chunks, pattern, limit=1)
        if sources:
            found[label] = sources[0]
    if not found:
        return _missing_field()
    return _explicit_field(sorted(found), list(found.values())[:6])


def _extract_training_settings(chunks: list[PaperChunk]) -> dict[str, ProfileField]:
    extractors = {
        "optimizer": lambda: _extract_first_value(chunks, r"\b(?:optimizer|optimized by|optimizer is)\s*(?:=|:|is|with)?\s*([A-Za-z]+W?)\b"),
        "learning_rate": lambda: _extract_numeric_value(chunks, r"\b(?:base\s+)?(?:learning rate|lr)\s*(?:=|:|is|of|was set to)?\s*"),
        "weight_decay": lambda: _extract_numeric_value(chunks, r"\bweight decay\s*(?:=|:|is|of|was set to)?\s*"),
        "batch_size": lambda: _extract_first_value(chunks, r"\bbatch size\s*(?:=|:|is|of|was set to)?\s*(\d+)\b"),
        "input_size": lambda: _extract_first_value(chunks, r"\b(?:input size|crop size|image size|cropped to)\s*(?:=|:|is|of|was set to)?\s*(\d+\s*[xX×,]\s*\d+)\b"),
        "scheduler": lambda: _extract_first_value(chunks, r"\b(poly|cosine|step|multi-step|linear warmup|warmup)\s+(?:learning rate\s+)?(?:scheduler|schedule|policy)\b"),
        "iterations_or_epochs": lambda: _extract_first_value(chunks, r"\b(\d+\s*(?:K|k)?\s*(?:iterations|iters|epochs))\b"),
    }
    settings: dict[str, ProfileField] = {}
    for key, extractor in extractors.items():
        result = extractor()
        settings[key] = _explicit_field(result[0], [result[1]]) if result else _missing_field()
    return settings


def _extract_reported_results(chunks: list[PaperChunk]) -> list[dict[str, Any]]:
    result_pattern = re.compile(
        r"\b(?P<dataset>[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)?)?\s*"
        r"(?P<metric>mIoU|IoU|mAcc|aAcc|accuracy|F1)\s*"
        r"(?:of|=|:|is|reaches|achieves)?\s*"
        r"(?P<value>\d+(?:\.\d+)?\s*%?)",
        flags=re.IGNORECASE,
    )
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for chunk in chunks:
        for sentence in _sentences(chunk.text):
            for match in result_pattern.finditer(sentence):
                metric = match.group("metric")
                value = match.group("value")
                dataset = (match.group("dataset") or NOT_SPECIFIED).strip()
                key = (dataset.lower(), metric.lower(), value)
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "dataset": dataset,
                    "metric": metric,
                    "value": value,
                    "evidence_level": "explicit",
                    "evidence_sources": [asdict(_source(chunk.page, sentence))],
                })
                if len(results) >= 12:
                    return results
    return [{
        "dataset": NOT_SPECIFIED,
        "metric": NOT_SPECIFIED,
        "value": NOT_SPECIFIED,
        "evidence_level": "missing",
        "evidence_sources": [],
    }]


def _extract_ablation_items(chunks: list[PaperChunk]) -> ProfileField:
    sources = _find_sources(chunks, r"\bablation\b|\bwithout\b|\bw/o\b|\bcomponent\b|\bmodule\b", limit=6)
    if not sources:
        return _missing_field()
    values = sorted({truncate_text(source.text, 180) for source in sources})
    return _explicit_field(values, sources)


def _build_missing_details(profile: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in GENERAL_FIELDS:
        field = profile[key]
        if isinstance(field, ProfileField) and field.evidence_level == "missing":
            missing.append(key)
    for key in TRAINING_FIELDS:
        field = profile["training_settings"][key]
        if field.evidence_level == "missing":
            missing.append(f"training_settings.{key}")
    if profile["reported_results"][0]["evidence_level"] == "missing":
        missing.append("reported_results")
    return missing


def _find_sources(chunks: list[PaperChunk], pattern: str, limit: int = 3) -> list[EvidenceSource]:
    regex = re.compile(pattern, flags=re.IGNORECASE)
    sources: list[EvidenceSource] = []
    for chunk in chunks:
        for sentence in _sentences(chunk.text):
            if regex.search(sentence):
                sources.append(_source(chunk.page, sentence))
                break
        if len(sources) >= limit:
            break
    return sources


def _extract_first_value(chunks: list[PaperChunk], pattern: str) -> tuple[str, EvidenceSource] | None:
    match = _first_regex_match(chunks, pattern)
    return match if match else None


def _extract_numeric_value(chunks: list[PaperChunk], prefix_pattern: str) -> tuple[str, EvidenceSource] | None:
    number_pattern = r"([0-9]+(?:\.[0-9]+)?\s*(?:[eE]\s*[-−]?\s*\d+|[xX×]\s*10\s*[−-]?\s*\d+)?)"
    match = _first_regex_match(chunks, prefix_pattern + number_pattern)
    if not match:
        return None
    value, source = match
    return _normalize_numeric_expression(value), source


def _first_regex_match(chunks: list[PaperChunk], pattern: str) -> tuple[str, EvidenceSource] | None:
    regex = re.compile(pattern, flags=re.IGNORECASE)
    for chunk in chunks:
        for sentence in _sentences(chunk.text):
            match = regex.search(sentence)
            if match:
                return match.group(1).strip(), _source(chunk.page, sentence)
    return None


def _normalize_numeric_expression(value: str) -> str:
    compact = re.sub(r"\s+", "", value).replace("−", "-").replace("×", "x").lower()
    scientific = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)x10(-?\d+)", compact)
    if scientific:
        base = float(scientific.group(1))
        exponent = int(scientific.group(2))
        return f"{base * (10 ** exponent):.12g}"
    compact = compact.replace("e-", "e-").replace("e+", "e+")
    return compact


def _source(page: int, text: str) -> EvidenceSource:
    return EvidenceSource(page=page, text=truncate_text(text, 420))


def _sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if len(part.strip()) > 6]


def _to_plain_dict(value: Any) -> Any:
    if isinstance(value, ProfileField):
        return {
            "value": value.value,
            "evidence_level": value.evidence_level,
            "evidence_sources": [asdict(source) for source in value.evidence_sources],
        }
    if isinstance(value, dict):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    return value


def _dataset_patterns() -> dict[str, str]:
    names = ["SUIM", "Cityscapes", "ADE20K", "Pascal VOC", "COCO", "CamVid", "SUN RGB-D", "NYUDv2", "ImageNet"]
    return {name: rf"\b{re.escape(name)}\b" for name in names}


def _metric_patterns() -> dict[str, str]:
    names = ["mIoU", "IoU", "mAcc", "aAcc", "Pixel Accuracy", "Accuracy", "F1"]
    return {name: rf"\b{re.escape(name)}\b" for name in names}


def _model_patterns() -> dict[str, str]:
    names = ["Transformer", "CNN", "ResNet", "Swin Transformer", "ViT", "SegFormer", "DeepLab", "U-Net", "HRNet", "MobileNet"]
    return {name: rf"\b{re.escape(name)}\b" for name in names}


def _baseline_patterns() -> dict[str, str]:
    names = ["FCN", "PSPNet", "DeepLabV3", "DeepLabV3+", "U-Net", "SegNet", "OCRNet", "HRNet", "SegFormer", "Mask2Former"]
    return {name: rf"\b{re.escape(name)}\b" for name in names}
