# Q-select Layered Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated pi05 Q-select debug server and offline summarizer that identify which layer of proposal expansion fails without modifying any existing production file.

**Architecture:** Copy the current websocket server into `ours/debug`, then add pure NumPy metrics plus request-level JSONL/NPZ recording around its existing candidate path. Keep aggregation in a standalone standard-library/NumPy CLI, and keep all launch/docs/tests inside the new directory.

**Tech Stack:** Python 3.11, NumPy, JAX/OpenPI runtime already installed on sciem, standard-library `unittest`, Bash.

---

### Task 1: Establish isolated files and RED tests

**Files:**
- Create: `ours/debug/test_qselect_diag.py`
- Create: `ours/debug/qselect_debug_server.py` (after the RED run)
- Create: `ours/debug/summarize_qselect_diag.py` (after the RED run)
- Copy: `ours/debug/diverse_vis_ref.py` (after the RED run)

- [ ] **Step 1: Write failing metric and metadata tests**

Create tests importing the wished-for APIs:

```python
from qselect_debug_server import compute_candidate_stats, extract_request_metadata

def test_two_candidates_full_and_prefix(self):
    candidates = np.array([[[0., 0.], [0., 0.]], [[3., 4.], [0., 0.]]])
    stats = compute_candidate_stats(candidates, np.array([1., 3.]), 1, exec_horizon=1)
    self.assertAlmostEqual(stats["cand_pair_l2_mean"], 5.0)
    self.assertAlmostEqual(stats["cand_pair_l2_per_dim_mean"], 2.5)
    self.assertAlmostEqual(stats["exec_pair_l2_mean"], 5.0)
    self.assertAlmostEqual(stats["exec_pair_l2_per_dim_mean"], 5.0 / np.sqrt(2.0))

def test_saturation_and_would_clip(self):
    candidates = np.array([[[0.99, 1.01]], [[0.0, -1.2]]])
    stats = compute_candidate_stats(candidates, np.zeros(2), 0)
    self.assertAlmostEqual(stats["cand_saturation_frac"], 3 / 4)
    self.assertAlmostEqual(stats["cand_would_clip_frac"], 2 / 4)

def test_metadata_aliases(self):
    meta = extract_request_metadata({"metadata": {"trial_id": 7, "step": 20}, "task_id": 3})
    self.assertEqual(meta, {"episode_id": 7, "env_step": 20, "task_id": 3})
```

- [ ] **Step 2: Write failing grouped-summary tests**

Create temporary JSONL containing one random group and one qselect group. Assert that `build_summary(records)` produces separate keys for both tuples, threshold fractions exist, and the random group has `q_metrics_available == False` with no q metric aggregates.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
cd /data/huangdi/heliqun/pi0/ours/debug
/data/huangdi/heliqun/pi0/openpi/pi_env/bin/python -m unittest -v test_qselect_diag.py
```

Expected: import failure because `qselect_debug_server.py` and `summarize_qselect_diag.py` do not yet exist.

- [ ] **Step 4: Copy only into the new directory**

Copy `ours/Qselect/server.py` to `ours/debug/qselect_debug_server.py` and `ours/Qselect/diverse_vis.py` to `ours/debug/diverse_vis_ref.py`. Do not edit the sources.

### Task 2: Implement pure candidate and metadata diagnostics (GREEN)

**Files:**
- Modify: `ours/debug/qselect_debug_server.py`
- Test: `ours/debug/test_qselect_diag.py`

- [ ] **Step 1: Make debug-server imports isolated and help-safe**

Set `QSELECT_DIR = THIS_DIR.parent / "Qselect"`, insert it only for compatibility imports, and move `from selector import QSelector` into the qselect-only initializer so `--help` never constructs or imports the critic.

- [ ] **Step 2: Implement validated pure metrics**

Implement public function `compute_candidate_stats(candidates, scores, best_idx, exec_horizon=None)`. It validates `[N,H,A]`, `[N]`, finite values, index bounds, and prefix bounds before computing upper-triangle pair distances. Per-dim distance is exactly `distance / sqrt(number_of_flattened_scalars)`. Population std uses `ddof=0`; saturation uses `abs(x) >= 0.99`; would-clip uses `abs(x) > 1.0`. The returned keys include every requested original key plus `cand_pair_l2_per_dim_*`, `exec_pair_l2_per_dim_*`, `best_vs_first_l2_per_dim`, `best_vs_first_exec_l2_per_dim`, and full/prefix candidate/selected saturation and would-clip metrics.

- [ ] **Step 3: Implement metadata extraction**

Use deterministic alias priority:

```python
episode_id: episode_id, trial_id, rollout_id, episode
env_step: env_step, step, step_id, timestep
task_id: task_id
```

Search top-level first, then `metadata`; normalize NumPy scalars to JSON scalars and return `None` when absent.

- [ ] **Step 4: Run metric tests and verify GREEN**

Run the Task 1 unittest command. Expected: metric and metadata cases pass; summary cases still fail because summarizer is absent.

### Task 3: Implement grouped offline summarizer (GREEN)

**Files:**
- Create: `ours/debug/summarize_qselect_diag.py`
- Test: `ours/debug/test_qselect_diag.py`

- [ ] **Step 1: Implement record loading and aggregation**

`load_jsonl(path)` reports malformed line numbers and rejects empty candidate-record sets. `build_summary(records)` creates `overall` plus groups keyed by JSON-safe labels built from `(sample_mode, noise_scale, noise_strategy)`.

For each numeric metric compute a dictionary with exactly five finite float fields named `mean`, `std`, `p10`, `p50`, and `p90`, using NumPy population standard deviation and percentiles.

Compute best-index histogram, index-zero fraction, and fractions `<= 1e-6, 1e-4, 1e-3, 1e-2, 1e-1` for full and prefix best-vs-first. Omit q aggregates from random groups.

- [ ] **Step 2: Implement JSON/CSV/stdout outputs**

JSON stores nested overall/groups. CSV writes one row per group with flattened metric columns and JSON-encoded histograms/threshold ratios. Stdout prints proposal/selection/saturation conclusions for random, and adds critic conclusions only for qselect.

- [ ] **Step 3: Run all unit tests and verify GREEN**

Run the Task 1 unittest command. Expected: all tests pass.

### Task 4: Integrate request-level recording into the copied server

**Files:**
- Modify: `ours/debug/qselect_debug_server.py`
- Test: `ours/debug/test_qselect_diag.py`

- [ ] **Step 1: Add RED integration tests**

Test a lightweight recorder helper with a temporary directory: append one JSONL record, load it with `json.loads`, verify `allow_nan=False` compatibility, save request 2 when interval is 2, and verify NPZ contains `candidates`, `scores`, `best_idx`, `selected_action`, `stats_json`, `noise`, and `raw_actions`.

- [ ] **Step 2: Run integration tests and verify RED**

Expected: missing helper/behavior assertion failure.

- [ ] **Step 3: Add CLI/dataclass fields**

Add `--diag-jsonl`, `--diag-save-candidates-every`, `--diag-record-dir`, `--diag-exec-horizon`, `--log-every`, and expose the copied server's required `--num-steps`, `--score-horizon`, `--default-prompt`, and `--record` fields without changing defaults.

- [ ] **Step 4: Capture layer data and write diagnostics**

Return actual noise from `_sample_batched_raw_actions`; convert raw actions once for raw-layer metrics. After selection, build the record with ISO-8601 UTC time, mode/noise/policy/critic metadata, optional request metadata, `q_metrics_available`, env `stats`, `noise_stats`, and `raw_action_stats`. Append JSONL with flush and save interval NPZ.

- [ ] **Step 5: Replace periodic log with one `[diag]` line**

Include request/mode/scale/strategy/best, full/prefix pair L2 and per-dim values, best-vs-first, q values, trans/rot/grip std, saturation/would-clip, model_ms, and select_ms. Prefix exactly `[diag]`; append `q_available=0` in random mode so zero q values cannot be mistaken for critic output.

- [ ] **Step 6: Run all unit tests and verify GREEN**

Expected: all tests pass with no model/checkpoint loaded.

### Task 5: Add launcher and operator documentation

**Files:**
- Create: `ours/debug/run_debug_server.sh`
- Create: `ours/debug/README.md`

- [ ] **Step 1: Write launcher**

Place exactly the eight user-editable variables first. Below them hardcode pi0/OpenPI paths, pi05 config, Python, port, seed, diag paths, validation, mkdir, and foreground execution redirected to `${OUT_DIR}/policy_server.log`. Require critic only for qselect.

- [ ] **Step 2: Write README**

Document simple; random at 1.0 and 1.5/2.0; qselect at 1.0/1.5/2.0; what each diagnoses; two-terminal LIBERO main/eval usage; JSONL summary usage; grep usage; boundary-proxy semantics; optional metadata request fields; and the fact that no diverse directory existed but `diverse_vis.py` was copied.

- [ ] **Step 3: Run shell syntax and help checks**

Run:

```bash
bash -n /data/huangdi/heliqun/pi0/ours/debug/run_debug_server.sh
/data/huangdi/heliqun/pi0/openpi/pi_env/bin/python /data/huangdi/heliqun/pi0/ours/debug/qselect_debug_server.py --help
```

Expected: exit 0; no checkpoint load or CUDA initialization message.

### Task 6: Final verification and production-file audit

**Files:**
- Verify only: `ours/debug/*`

- [ ] **Step 1: Run fresh full verification**

```bash
cd /data/huangdi/heliqun/pi0
/data/huangdi/heliqun/pi0/openpi/pi_env/bin/python -m unittest -v ours/debug/test_qselect_diag.py
/data/huangdi/heliqun/pi0/openpi/pi_env/bin/python -m py_compile ours/debug/qselect_debug_server.py ours/debug/summarize_qselect_diag.py
bash -n ours/debug/run_debug_server.sh
```

- [ ] **Step 2: Exercise summarizer end-to-end**

Generate a tiny temporary JSONL through the test fixture, run both `--out-json` and `--out-csv`, parse both outputs, and verify random stdout contains no critic interpretation.

- [ ] **Step 3: Audit isolation**

List mtimes/checksums of the four protected production files and verify all newly written implementation files are under `ours/debug`. Do not rely on repository status because this checkout is blocked by Git dubious-ownership protection.

- [ ] **Step 4: Decide GPU smoke test safely**

The launcher intentionally has empty MODEL_PATH/CRITIC_PATH, so a default invocation must fail fast before GPU use. Report that validation result; do not start a long-running server without concrete checkpoints supplied by the user.
