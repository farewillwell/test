#!/usr/bin/env python3
"""Aggregate qselect debug JSONL by sampling configuration."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
from typing import Any, Iterable

import numpy as np


CORE_METRICS = (
    "cand_pair_l2_mean",
    "cand_pair_l2_per_dim_mean",
    "exec_pair_l2_mean",
    "exec_pair_l2_per_dim_mean",
    "best_vs_first_l2",
    "best_vs_first_l2_per_dim",
    "best_vs_first_exec_l2",
    "best_vs_first_exec_l2_per_dim",
    "cand_std_trans",
    "cand_std_rot",
    "cand_std_grip",
    "cand_saturation_frac",
    "cand_would_clip_frac",
    "cand_saturated_candidate_frac",
    "cand_would_clip_candidate_frac",
    "exec_cand_saturation_frac",
    "exec_cand_would_clip_frac",
    "exec_cand_saturated_candidate_frac",
    "exec_cand_would_clip_candidate_frac",
    "selected_saturation_frac",
    "selected_would_clip_frac",
    "exec_selected_saturation_frac",
    "exec_selected_would_clip_frac",
)
Q_METRICS = ("q_std", "q_gap", "q_top1_top2_gap")
LAYER_METRICS = (
    "pair_l2_mean",
    "pair_l2_per_dim_mean",
    "exec_pair_l2_mean",
    "exec_pair_l2_per_dim_mean",
    "std_all",
    "exec_std_all",
)
BEST_THRESHOLDS = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)


def load_jsonl(path: str | pathlib.Path) -> list[dict[str, Any]]:
    path = pathlib.Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            if isinstance(record.get("stats"), dict):
                records.append(record)
    if not records:
        raise ValueError(f"No candidate diagnostic records found in {path}")
    return records


def _finite_metric_values(
    records: Iterable[dict[str, Any]], metric: str, *, section: str = "stats"
) -> np.ndarray:
    values: list[float] = []
    for record in records:
        values_section = record.get(section, {})
        if metric not in values_section:
            continue
        value = float(values_section[metric])
        if not math.isfinite(value):
            raise ValueError(f"Metric {metric!r} contains non-finite value {value!r}")
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def _threshold_fractions(records: list[dict[str, Any]], metric: str) -> dict[str, float]:
    values = _finite_metric_values(records, metric)
    if values.size == 0:
        return {}
    return {
        f"{threshold:.0e}": float(np.mean(values <= threshold))
        for threshold in BEST_THRESHOLDS
    }


def _group_summary(
    records: list[dict[str, Any]],
    *,
    sample_mode: str,
    noise_scale: float | None,
    noise_strategy: str,
) -> dict[str, Any]:
    include_q = sample_mode == "qselect" or (
        sample_mode == "overall" and any(bool(record.get("q_metrics_available")) for record in records)
    )
    metrics: dict[str, dict[str, float]] = {}
    for metric in CORE_METRICS:
        values = _finite_metric_values(records, metric)
        if values.size:
            metrics[metric] = _distribution(values)
    for prefix, section in (
        ("noise", "noise_stats"),
        ("raw_action", "raw_action_stats"),
    ):
        for metric in LAYER_METRICS:
            values = _finite_metric_values(records, metric, section=section)
            if values.size:
                metrics[f"{prefix}_{metric}"] = _distribution(values)
    if include_q:
        q_records = [record for record in records if bool(record.get("q_metrics_available"))]
        for metric in Q_METRICS:
            values = _finite_metric_values(q_records, metric)
            if values.size:
                metrics[metric] = _distribution(values)

    histogram: collections.Counter[str] = collections.Counter()
    for record in records:
        stats = record["stats"]
        histogram[str(int(stats["best_idx"]))] += 1
    sorted_histogram = dict(sorted(histogram.items(), key=lambda item: int(item[0])))

    return {
        "sample_mode": sample_mode,
        "noise_scale": noise_scale,
        "noise_strategy": noise_strategy,
        "num_requests": len(records),
        "q_metrics_available": include_q,
        "metrics": metrics,
        "best_idx_histogram": sorted_histogram,
        "fraction_best_idx_is_0": float(histogram.get("0", 0) / len(records)),
        "best_vs_first_threshold_fractions": _threshold_fractions(records, "best_vs_first_l2"),
        "best_vs_first_exec_threshold_fractions": _threshold_fractions(
            records, "best_vs_first_exec_l2"
        ),
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("records is empty")
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        mode = str(record.get("sample_mode", "unknown"))
        scale = float(record.get("noise_scale", 0.0))
        strategy = str(record.get("noise_strategy", "unknown"))
        grouped[(mode, scale, strategy)].append(record)

    groups = [
        _group_summary(items, sample_mode=mode, noise_scale=scale, noise_strategy=strategy)
        for (mode, scale, strategy), items in sorted(grouped.items())
    ]
    return {
        "num_requests": len(records),
        "overall": _group_summary(
            records,
            sample_mode="overall",
            noise_scale=None,
            noise_strategy="mixed",
        ),
        "groups": groups,
    }


def diagnose_group(group: dict[str, Any]) -> str:
    mode = str(group["sample_mode"])
    prefix = (
        f"[{mode} scale={group['noise_scale']} strategy={group['noise_strategy']} "
        f"requests={group['num_requests']}]"
    )
    lines = [prefix]
    if mode == "random":
        lines.extend(
            [
                "If cand_pair_l2 is small: proposal expansion did not produce meaningful action diversity.",
                "If best_idx is mostly 0 or best_vs_first_l2 is tiny: random selection rarely changes executed actions.",
                "If full-horizon diversity is large but exec-prefix diversity is small: diversity appears only in unexecuted tail actions.",
                "If saturation/would-clip fractions are high: proposal scale may be pushing actions onto or beyond environment bounds.",
            ]
        )
    elif mode == "qselect":
        lines.extend(
            [
                "If cand_pair_l2 is small and q_std is small: proposal expansion did not produce meaningful candidate diversity.",
                "If cand_pair_l2 is large but q_std/q_gap is small: candidates are diverse, but critic cannot distinguish them.",
                "If cand_pair_l2 and q_gap are large but success does not improve: critic may be miscalibrated on off-manifold candidates.",
                "If best_idx is mostly 0 or best_vs_first_l2 is tiny: Q-select does not actually change executed actions.",
                "If full-horizon diversity is large but exec-prefix diversity is small: diversity appears only in unexecuted tail actions.",
                "If saturation/would-clip fractions are high: selected proposals may be dominated by environment-boundary actions.",
            ]
        )
    else:
        lines.append("No candidate-level interpretation is defined for this sample mode.")
    return "\n".join(lines)


def write_summary_json(summary: dict[str, Any], path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv_row(group: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_mode": group["sample_mode"],
        "noise_scale": group["noise_scale"],
        "noise_strategy": group["noise_strategy"],
        "num_requests": group["num_requests"],
        "q_metrics_available": group["q_metrics_available"],
        "fraction_best_idx_is_0": group["fraction_best_idx_is_0"],
        "best_idx_histogram": json.dumps(group["best_idx_histogram"], sort_keys=True),
        "best_vs_first_threshold_fractions": json.dumps(
            group["best_vs_first_threshold_fractions"], sort_keys=True
        ),
        "best_vs_first_exec_threshold_fractions": json.dumps(
            group["best_vs_first_exec_threshold_fractions"], sort_keys=True
        ),
    }
    for metric, distribution in group["metrics"].items():
        for statistic, value in distribution.items():
            row[f"{metric}_{statistic}"] = value
    return row


def write_summary_csv(summary: dict[str, Any], path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_csv_row(group) for group in summary["groups"]]
    fields = sorted({key for row in rows for key in row})
    leading = [
        "sample_mode",
        "noise_scale",
        "noise_strategy",
        "num_requests",
        "q_metrics_available",
        "fraction_best_idx_is_0",
        "best_idx_histogram",
        "best_vs_first_threshold_fractions",
        "best_vs_first_exec_threshold_fractions",
    ]
    fields = leading + [field for field in fields if field not in leading]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.diag_jsonl)
    summary = build_summary(records)
    write_summary_json(summary, args.out_json)
    write_summary_csv(summary, args.out_csv)
    print(f"num_requests: {summary['num_requests']}")
    for group in summary["groups"]:
        print(diagnose_group(group))


if __name__ == "__main__":
    main()
