#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from qselect_debug_server import (
    append_diag_jsonl,
    build_diagnostic_record,
    compute_candidate_stats,
    extract_request_metadata,
    maybe_save_candidate_record,
    strip_request_metadata,
)


def summary_api():
    from summarize_qselect_diag import (
        build_summary,
        diagnose_group,
        write_summary_csv,
        write_summary_json,
    )

    return build_summary, diagnose_group, write_summary_csv, write_summary_json


class CandidateStatsTest(unittest.TestCase):
    def test_two_candidates_full_and_prefix(self) -> None:
        candidates = np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[3.0, 4.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        stats = compute_candidate_stats(
            candidates,
            np.array([1.0, 3.0], dtype=np.float32),
            1,
            exec_horizon=1,
        )

        self.assertEqual(stats["num_candidates"], 2)
        self.assertEqual(stats["horizon"], 2)
        self.assertEqual(stats["action_dim"], 2)
        self.assertAlmostEqual(stats["cand_pair_l2_mean"], 5.0)
        self.assertAlmostEqual(stats["cand_pair_l2_per_dim_mean"], 2.5)
        self.assertAlmostEqual(stats["exec_pair_l2_mean"], 5.0)
        self.assertAlmostEqual(stats["exec_pair_l2_per_dim_mean"], 5.0 / np.sqrt(2.0))
        self.assertAlmostEqual(stats["best_vs_first_l2"], 5.0)
        self.assertAlmostEqual(stats["best_vs_first_exec_l2"], 5.0)
        self.assertAlmostEqual(stats["q_mean"], 2.0)
        self.assertAlmostEqual(stats["q_std"], 1.0)
        self.assertAlmostEqual(stats["q_gap"], 1.0)
        self.assertAlmostEqual(stats["q_range"], 2.0)
        self.assertAlmostEqual(stats["q_top1_top2_gap"], 2.0)
        self.assertAlmostEqual(stats["cand_std_trans"], 0.875)
        self.assertEqual(stats["cand_std_rot"], 0.0)
        self.assertEqual(stats["cand_std_grip"], 0.0)

    def test_saturation_and_would_clip(self) -> None:
        candidates = np.array(
            [
                [[0.99, 1.01]],
                [[0.0, -1.2]],
            ],
            dtype=np.float32,
        )
        stats = compute_candidate_stats(candidates, np.zeros(2, dtype=np.float32), 0)

        self.assertAlmostEqual(stats["cand_saturation_frac"], 0.75)
        self.assertAlmostEqual(stats["cand_would_clip_frac"], 0.5)
        self.assertAlmostEqual(stats["cand_saturated_candidate_frac"], 1.0)
        self.assertAlmostEqual(stats["cand_would_clip_candidate_frac"], 1.0)
        self.assertAlmostEqual(stats["selected_saturation_frac"], 1.0)
        self.assertAlmostEqual(stats["selected_would_clip_frac"], 0.5)
        self.assertAlmostEqual(stats["exec_cand_saturation_frac"], 0.75)
        self.assertAlmostEqual(stats["exec_selected_would_clip_frac"], 0.5)

    def test_single_candidate_has_zero_pair_and_top_gap(self) -> None:
        stats = compute_candidate_stats(
            np.zeros((1, 2, 7), dtype=np.float32),
            np.array([2.5], dtype=np.float32),
            0,
        )

        self.assertEqual(stats["cand_pair_l2_mean"], 0.0)
        self.assertEqual(stats["cand_pair_l2_std"], 0.0)
        self.assertEqual(stats["q_top1_top2_gap"], 0.0)
        self.assertEqual(stats["cand_std_grip"], 0.0)

    def test_invalid_exec_horizon_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exec_horizon"):
            compute_candidate_stats(
                np.zeros((2, 2, 7), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                0,
                exec_horizon=3,
            )

    def test_metadata_aliases_and_numpy_scalars(self) -> None:
        metadata = extract_request_metadata(
            {
                "metadata": {"trial_id": np.int64(7), "step": np.int32(20)},
                "task_id": np.int64(3),
            }
        )
        self.assertEqual(metadata, {"episode_id": 7, "env_step": 20, "task_id": 3})

    def test_missing_metadata_is_none(self) -> None:
        self.assertEqual(
            extract_request_metadata({"prompt": "pick up the bowl"}),
            {"episode_id": None, "env_step": None, "task_id": None},
        )

    def test_strip_request_metadata_keeps_model_inputs(self) -> None:
        obs = {
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "move bowl",
            "episode_id": 8,
            "env_step": 10,
            "task_id": 2,
            "metadata": {"trial_id": 8},
        }
        model_obs = strip_request_metadata(obs)
        self.assertEqual(set(model_obs), {"observation/state", "prompt"})
        self.assertIn("episode_id", obs)


class RecordingTest(unittest.TestCase):
    def test_build_random_record_has_layer_stats_and_metadata(self) -> None:
        args = SimpleNamespace(
            sample_mode="random",
            noise_scale=1.5,
            noise_strategy="base",
            num_action_samples=2,
            policy_dir="/models/pi05",
            critic_path="",
            diag_exec_horizon=1,
        )
        candidates = np.array(
            [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]],
            dtype=np.float32,
        )
        record = build_diagnostic_record(
            args=args,
            request_count=4,
            obs={"episode_id": 9, "env_step": 30, "task_id": 2, "prompt": "move bowl"},
            candidates=candidates,
            scores=np.zeros(2, dtype=np.float32),
            best_idx=1,
            noise=candidates + 1.0,
            raw_actions=candidates + 2.0,
            model_ms=12.0,
            select_ms=0.2,
        )

        self.assertEqual(record["request"], 4)
        self.assertEqual(record["episode_id"], 9)
        self.assertEqual(record["env_step"], 30)
        self.assertEqual(record["task_id"], 2)
        self.assertFalse(record["q_metrics_available"])
        self.assertIn("cand_pair_l2_per_dim_mean", record["stats"])
        self.assertIn("pair_l2_per_dim_mean", record["noise_stats"])
        self.assertIn("exec_pair_l2_mean", record["raw_action_stats"])

    def test_help_does_not_load_model(self) -> None:
        env = dict(os.environ)
        env["action_horizon"] = "10"
        result = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).with_name("qselect_debug_server.py")), "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--diag-jsonl", result.stdout)
        self.assertNotIn("QSelector", result.stdout + result.stderr)

    def test_jsonl_append_and_periodic_npz(self) -> None:
        record = {
            "request": 2,
            "sample_mode": "random",
            "q_metrics_available": False,
            "stats": {"cand_pair_l2_mean": 1.0},
        }
        candidates = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        scores = np.zeros(2, dtype=np.float32)
        selected = candidates[1]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            jsonl = tmp_path / "diag.jsonl"
            append_diag_jsonl(jsonl, record)
            loaded = json.loads(jsonl.read_text(encoding="utf-8").strip())
            self.assertEqual(loaded["request"], 2)

            no_path = maybe_save_candidate_record(
                request_count=1,
                every=2,
                record_dir=tmp_path / "records",
                candidates=candidates,
                scores=scores,
                best_idx=1,
                selected_action=selected,
                stats=record["stats"],
                noise=candidates + 1,
                raw_actions=candidates + 2,
            )
            self.assertIsNone(no_path)

            saved_path = maybe_save_candidate_record(
                request_count=2,
                every=2,
                record_dir=tmp_path / "records",
                candidates=candidates,
                scores=scores,
                best_idx=1,
                selected_action=selected,
                stats=record["stats"],
                noise=candidates + 1,
                raw_actions=candidates + 2,
            )
            self.assertEqual(saved_path.name, "request_000002.npz")
            with np.load(saved_path) as data:
                self.assertEqual(
                    set(data.files),
                    {
                        "candidates",
                        "scores",
                        "best_idx",
                        "selected_action",
                        "stats_json",
                        "noise",
                        "raw_actions",
                    },
                )
                self.assertEqual(int(data["best_idx"]), 1)


def make_stats(best_idx: int, best_distance: float, q_std: float) -> dict[str, float | int]:
    return {
        "best_idx": best_idx,
        "cand_pair_l2_mean": 2.0,
        "cand_pair_l2_per_dim_mean": 0.2,
        "exec_pair_l2_mean": 1.0,
        "exec_pair_l2_per_dim_mean": 0.1,
        "best_vs_first_l2": best_distance,
        "best_vs_first_exec_l2": best_distance / 2.0,
        "q_std": q_std,
        "q_gap": q_std * 2.0,
        "q_top1_top2_gap": q_std / 2.0,
        "cand_std_trans": 0.2,
        "cand_std_rot": 0.1,
        "cand_std_grip": 0.3,
        "cand_saturation_frac": 0.05,
        "cand_would_clip_frac": 0.01,
        "exec_cand_saturation_frac": 0.04,
        "exec_cand_would_clip_frac": 0.005,
    }


def make_layer_stats(multiplier: float) -> dict[str, float | int]:
    return {
        "pair_l2_mean": 3.0 * multiplier,
        "pair_l2_per_dim_mean": 0.3 * multiplier,
        "exec_pair_l2_mean": 2.0 * multiplier,
        "exec_pair_l2_per_dim_mean": 0.2 * multiplier,
        "std_all": 0.1 * multiplier,
        "exec_std_all": 0.05 * multiplier,
    }


class SummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "sample_mode": "random",
                "noise_scale": 1.0,
                "noise_strategy": "base",
                "q_metrics_available": False,
                "stats": make_stats(0, 0.0, 0.0),
                "noise_stats": make_layer_stats(1.0),
                "raw_action_stats": make_layer_stats(2.0),
            },
            {
                "sample_mode": "random",
                "noise_scale": 1.0,
                "noise_strategy": "base",
                "q_metrics_available": False,
                "stats": make_stats(2, 0.02, 0.0),
                "noise_stats": make_layer_stats(1.0),
                "raw_action_stats": make_layer_stats(2.0),
            },
            {
                "sample_mode": "qselect",
                "noise_scale": 1.5,
                "noise_strategy": "hubu",
                "q_metrics_available": True,
                "stats": make_stats(1, 0.3, 0.4),
                "noise_stats": make_layer_stats(1.5),
                "raw_action_stats": make_layer_stats(2.5),
            },
        ]

    def test_summary_groups_modes_and_threshold_ratios(self) -> None:
        build_summary, _, _, _ = summary_api()
        summary = build_summary(self.records)

        self.assertEqual(summary["num_requests"], 3)
        self.assertEqual(len(summary["groups"]), 2)
        random_group = next(g for g in summary["groups"] if g["sample_mode"] == "random")
        qselect_group = next(g for g in summary["groups"] if g["sample_mode"] == "qselect")

        self.assertEqual(random_group["num_requests"], 2)
        self.assertEqual(random_group["best_idx_histogram"], {"0": 1, "2": 1})
        self.assertAlmostEqual(random_group["fraction_best_idx_is_0"], 0.5)
        self.assertAlmostEqual(
            random_group["best_vs_first_threshold_fractions"]["1e-03"],
            0.5,
        )
        self.assertFalse(random_group["q_metrics_available"])
        self.assertNotIn("q_std", random_group["metrics"])
        self.assertAlmostEqual(random_group["metrics"]["noise_pair_l2_mean"]["mean"], 3.0)
        self.assertAlmostEqual(
            random_group["metrics"]["raw_action_pair_l2_per_dim_mean"]["mean"],
            0.6,
        )
        self.assertTrue(qselect_group["q_metrics_available"])
        self.assertIn("q_std", qselect_group["metrics"])
        self.assertAlmostEqual(qselect_group["metrics"]["q_std"]["mean"], 0.4)

    def test_random_diagnosis_does_not_interpret_q(self) -> None:
        build_summary, diagnose_group, _, _ = summary_api()
        random_group = next(
            group for group in build_summary(self.records)["groups"]
            if group["sample_mode"] == "random"
        )
        text = diagnose_group(random_group).lower()
        self.assertNotIn("critic", text)
        self.assertNotIn("q_std", text)
        self.assertNotIn("q_gap", text)

    def test_json_and_csv_outputs_have_one_row_per_group(self) -> None:
        build_summary, _, write_summary_csv, write_summary_json = summary_api()
        summary = build_summary(self.records)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            json_path = tmp_path / "summary.json"
            csv_path = tmp_path / "summary.csv"
            write_summary_json(summary, json_path)
            write_summary_csv(summary, csv_path)

            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["num_requests"], 3)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["sample_mode"] for row in rows}, {"random", "qselect"})
            random_row = next(row for row in rows if row["sample_mode"] == "random")
            self.assertEqual(random_row.get("q_std_mean", ""), "")


if __name__ == "__main__":
    unittest.main()
