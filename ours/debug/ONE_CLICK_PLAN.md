# One-click Q-select Debug Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `run_debug_server.sh` into a visible one-command server/eval/summary workflow with episode metadata and automatic server cleanup.

**Architecture:** Copy the production collector into `ours/debug` and modify only the copy to attach rollout identifiers to websocket requests. Rework the existing launcher using the proven `vis.sh` background-process/readiness/trap pattern, then invoke the debug eval and summary sequentially.

**Tech Stack:** Bash, Python 3.11, LIBERO environment, OpenPI websocket client, standard-library unittest.

---

### Task 1: Debug eval metadata

**Files:**
- Create: `ours/debug/qselect_debug_eval.py`
- Create: `ours/debug/test_qselect_debug_eval.py`

- [ ] Write a test that imports `build_policy_element`, passes `episode_id=7`, `env_step=20`, and `task_id=3`, then asserts the returned element contains these identifiers plus the original image/state/prompt fields.
- [ ] Run `/data/huangdi/heliqun/pi0/openpi/examples/libero/libero_env/bin/python -m unittest -v test_qselect_debug_eval.py`; expect import failure because `qselect_debug_eval.py` does not exist.
- [ ] Copy `ours/Qselect/collect.py` to `ours/debug/qselect_debug_eval.py` without editing the source.
- [ ] Extend `build_policy_element` with keyword-only identifiers and pass `trial_id`, `env_action_step`, and `task_id` from `run_one_episode` at every replan.
- [ ] Re-run the test; expect one passing test.

### Task 2: One-click launcher

**Files:**
- Modify: `ours/debug/run_debug_server.sh`

- [ ] Add top-level `TASK_SUITE`, `TASK_ID`, `NUM_TRIALS`, and `INITIAL_STATE_OFFSET` controls.
- [ ] Before model loading, use a one-shot socket connect to reject an already occupied port 8000.
- [ ] Start the server in the background, install an EXIT/INT/TERM cleanup trap, and poll readiness for up to 900 seconds. Every 10 seconds print elapsed time and the latest server log line; fail with the log tail if the process exits.
- [ ] Run `qselect_debug_eval.py` with LIBERO Python, `HF_LEROBOT_HOME=${OUT_DIR}`, no episode saving flags, metrics at `${OUT_DIR}/eval_metrics.jsonl`, and output mirrored through `tee` to `${OUT_DIR}/eval.log`.
- [ ] Stop the server after eval. For random/qselect, run `summarize_qselect_diag.py` and create `summary.json`, `summary.csv`, and `summary.stdout`; for simple, skip candidate summary explicitly.
- [ ] Run `bash -n`. While the old server occupies port 8000, run the launcher and verify it fails in preflight without starting a second model process.

### Task 3: Documentation and verification

**Files:**
- Modify: `ours/debug/README.md`
- Modify: `ours/debug/DESIGN.md`

- [ ] Replace the two-terminal instructions with one-command usage and document all outputs, progress messages, cleanup behavior, and the requirement to stop an old server first.
- [ ] Run both unittest files, py_compile all three debug Python entrypoints, shell syntax, launcher occupied-port preflight, and production-file SHA-256 audit.
