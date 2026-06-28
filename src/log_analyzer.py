import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd


METRIC_KEYS = ("loss", "lr", "mIoU", "mAcc", "aAcc")


@dataclass
class LogAnalysisResult:
    file_name: str
    dataframe: pd.DataFrame
    diagnosis: list[str]
    best_miou: dict[str, Any] | None


def analyze_log(text: str, file_name: str = "training.log") -> LogAnalysisResult:
    records = parse_log_records(text)
    if not records:
        raise ValueError("No supported training metrics were found in the uploaded log.")

    dataframe = pd.DataFrame(records)
    dataframe = _normalize_dataframe(dataframe)
    best_miou = find_best_miou(dataframe)
    diagnosis = diagnose_training(dataframe, best_miou)

    return LogAnalysisResult(
        file_name=file_name,
        dataframe=dataframe,
        diagnosis=diagnosis,
        best_miou=best_miou,
    )


def parse_log_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parsed = _parse_json_line(line)
        if parsed is None:
            parsed = _parse_text_line(line)

        if parsed and any(key in parsed for key in METRIC_KEYS):
            records.append(parsed)
    return records


def _parse_json_line(line: str) -> dict[str, Any] | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    record: dict[str, Any] = {}
    for key in ("iter", "iteration", "step", "epoch", *METRIC_KEYS):
        if key in data:
            target_key = "iter" if key in {"iteration", "step"} else key
            record[target_key] = _safe_number(data[key])
    return {key: value for key, value in record.items() if value is not None}


def _parse_text_line(line: str) -> dict[str, Any]:
    record: dict[str, Any] = {}

    iter_match = re.search(r"\b(?:iter|iteration)\s*[:=/ ]\s*(\d+)", line, re.IGNORECASE)
    epoch_match = re.search(r"\bepoch\s*[:=/ ]\s*(\d+)", line, re.IGNORECASE)
    if iter_match:
        record["iter"] = int(iter_match.group(1))
    if epoch_match:
        record["epoch"] = int(epoch_match.group(1))

    for key in METRIC_KEYS:
        pattern = rf"\b{re.escape(key)}\b\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        match = re.search(pattern, line)
        if match:
            record[key] = _safe_number(match.group(1))

    return record


def _safe_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        if number.is_integer():
            return int(number)
        return number
    return None


def _normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    if "iter" not in dataframe.columns:
        dataframe["iter"] = range(1, len(dataframe) + 1)

    for column in dataframe.columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe = dataframe.sort_values(by=["epoch", "iter"] if "epoch" in dataframe.columns else ["iter"])
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def find_best_miou(dataframe: pd.DataFrame) -> dict[str, Any] | None:
    if "mIoU" not in dataframe.columns or dataframe["mIoU"].dropna().empty:
        return None
    idx = dataframe["mIoU"].idxmax()
    row = dataframe.loc[idx].dropna().to_dict()
    return row


def diagnose_training(dataframe: pd.DataFrame, best_miou: dict[str, Any] | None) -> list[str]:
    diagnosis: list[str] = []

    if "loss" in dataframe.columns:
        loss = dataframe["loss"].dropna()
        if loss.empty:
            diagnosis.append("No valid loss values were found.")
        elif loss.apply(lambda value: not math.isfinite(value)).any():
            diagnosis.append("Detected NaN or inf loss; check learning rate, data quality, and gradient stability.")
        elif len(loss) >= 2 and loss.iloc[-1] < loss.iloc[0]:
            diagnosis.append("Loss generally decreases, suggesting the training process is basically normal.")
        else:
            diagnosis.append("Loss does not clearly decrease; consider checking the optimizer, learning rate, and data pipeline.")

    if "mIoU" in dataframe.columns:
        miou = dataframe["mIoU"].dropna()
        if len(miou) >= 4:
            recent_gain = miou.iloc[-1] - miou.iloc[max(0, len(miou) - 4)]
            if abs(recent_gain) < 0.002:
                diagnosis.append("mIoU has changed little recently; the model may be entering a plateau.")
            elif recent_gain > 0:
                diagnosis.append("mIoU is still improving in recent records.")
        if best_miou:
            location = []
            if "epoch" in best_miou:
                location.append(f"epoch {int(best_miou['epoch'])}")
            if "iter" in best_miou:
                location.append(f"iter {int(best_miou['iter'])}")
            where = ", ".join(location) if location else "the parsed log"
            diagnosis.append(f"Best mIoU is {best_miou['mIoU']:.4f}, observed at {where}.")

    if "lr" in dataframe.columns:
        lr = dataframe["lr"].dropna()
        if not lr.empty and (lr <= 0).any():
            diagnosis.append("Detected non-positive learning rate; verify scheduler output.")
        elif not lr.empty and lr.max() > 1e-2:
            diagnosis.append("Learning rate is relatively high; confirm it matches model and batch size.")

    if not diagnosis:
        diagnosis.append("Parsed metrics successfully, but there is not enough signal for a detailed diagnosis.")
    return diagnosis


def plot_metrics(dataframe: pd.DataFrame) -> dict[str, plt.Figure]:
    figures: dict[str, plt.Figure] = {}
    x_col = "iter" if "iter" in dataframe.columns else dataframe.index

    for metric in ("loss", "mIoU", "lr"):
        if metric not in dataframe.columns or dataframe[metric].dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(dataframe[x_col], dataframe[metric], marker="o", linewidth=1.6)
        ax.set_title(metric)
        ax.set_xlabel("iter")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        figures[metric] = fig
    return figures
