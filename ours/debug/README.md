# pi05 Q-select 分层诊断工具

本目录是一套独立的诊断实现。它不修改 `ours/iter.py`、`ours/Qselect/server.py`、`ours/Qselect/collect.py`、`ours/Qselect/selector.py` 或其他生产流程。

## 文件

- `qselect_debug_server.py`：debug websocket server；记录 noise、raw model action、env-space action、critic 与选择层。
- `qselect_debug_eval.py`：从生产 collector 复制的 debug eval；自动发送 episode/task/step metadata 并记录 success metrics。
- `summarize_qselect_diag.py`：按 `(sample_mode, noise_scale, noise_strategy)` 分组汇总 JSONL。
- `run_debug_server.sh`：参考 `ours/Qselect/vis.sh`，一键完成 server、ready wait、eval、summary 和 cleanup。
- `test_qselect_diag.py`：纯统计、记录、metadata 与 summary 的单元测试。
- `diverse_vis_ref.py`：`ours/Qselect/diverse_vis.py` 的只读副本。

仓库中没有找到名称包含 `diverse` 的目录；找到的是 `ours/Qselect/diverse_vis.py` 文件，因此复制为 `diverse_vis_ref.py` 供参考。

## 一键运行

只编辑 `run_debug_server.sh` 开头的变量。除 model/noise 配置外，可以设置 `TASK_SUITE`、`TASK_ID`、`NUM_TRIALS` 和 `INITIAL_STATE_OFFSET`。pi05 policy checkpoint 必须包含 `params/`；qselect 模式还必须填写存在的 critic checkpoint。

如果此前手动启动的 server 仍在运行，先在旧 terminal 按 `Ctrl+C`。脚本会在加载模型前检查端口 8000；端口被占用时立即退出，不会再加载第二份模型。

```bash
cd /data/huangdi/heliqun/pi0
bash ours/debug/run_debug_server.sh
```

脚本复用了 `run.sh` / `ours/Qselect/vis.sh` 的关键运行约定，包括 `action_horizon=10`、OpenPI Python、JAX/CUDA cache、ready polling、trap cleanup 和 `LD_LIBRARY_PATH` 处理。固定使用 `pi05_libero_awbc`，不兼容 pi0。模型加载期间每 10 秒打印最新 server log；eval 日志实时显示。eval 或 summary 结束/失败时都会自动停止 server。

输出位置：

```text
${OUT_DIR}/policy_server.log
${OUT_DIR}/diag.jsonl
${OUT_DIR}/records/request_*.npz
${OUT_DIR}/eval.log
${OUT_DIR}/eval_metrics.jsonl
${OUT_DIR}/summary.json
${OUT_DIR}/summary.csv
${OUT_DIR}/summary.stdout
```

快速查看周期诊断：

```bash
grep "\[diag\]" "${OUT_DIR}/policy_server.log"
```

`simple` 没有 candidates，不会伪造 candidate JSONL；它只打印 baseline infer 日志。`random` 的 scores 是全零占位，记录中 `q_metrics_available=false`，summary 不汇总或解释这些 q 值。

## 四组诊断

每组修改脚本顶部变量后运行同一个脚本；它会自动使用相同 task/seed/initial-state 范围完成 LIBERO eval。

1. `SAMPLE_MODE="simple"`
   - 建立 baseline action execution 与 success rate。
   - 不产生 candidate diagnostics。
2. `SAMPLE_MODE="random"`, `NOISE_SCALE=1.0`
   - 检查默认 proposal sampling 是否真实改变 rollout distribution。
   - 只解释 noise/raw/env action、selection 与 saturation，不解释 q。
3. `SAMPLE_MODE="random"`, `NOISE_SCALE=1.5` 或 `2.0`
   - 检查增大 noise 后的多样性是否从 noise 层传到 raw action、env action 和 executed prefix。
   - 同时检查是否只是把 action 推到 `[-1,1]` 边界。
4. `SAMPLE_MODE="qselect"`, `NOISE_SCALE=1.0`、`1.5`、`2.0`
   - 检查 critic 的 `q_std/q_gap/top-gap` 是否随 candidate diversity 增大。
   - 检查 best index 与 best-vs-first 是否真正改变 executed actions，再与 success rate 对齐。

不要把不同配置追加到同一个 `diag.jsonl`，除非有意让 summary 在同一文件中分组比较。最稳妥的做法是每组使用不同 `OUT_DIR`。

## Eval 与 success 对齐

一键脚本调用 `qselect_debug_eval.py`，不再调用生产 collector。debug eval 在每次重新规划请求中发送：

- `episode_id=trial_id`
- `env_step=当前已执行 action 数`
- `task_id=当前 LIBERO task`

server 会在送入模型 transform 前移除这些 debug-only 字段，但将它们写入 `diag.jsonl`。`eval_metrics.jsonl` 使用相同的 `task_id/trial_id` 并记录 success，因此可以用 `(task_id, episode_id)` 对齐。

## 汇总 JSONL

```bash
cd /data/huangdi/heliqun/pi0
/data/huangdi/heliqun/pi0/openpi/pi_env/bin/python \
  ours/debug/summarize_qselect_diag.py \
  --diag-jsonl /data/aoss/heliqun/qselect_debug/diag.jsonl \
  --out-json /data/aoss/heliqun/qselect_debug/summary.json \
  --out-csv /data/aoss/heliqun/qselect_debug/summary.csv
```

JSON 包含 overall 和分组结果；CSV 每个 `(sample_mode, noise_scale, noise_strategy)` 一行。每个指标输出 mean/std/p10/p50/p90，并包含：

- best-index histogram 与 `fraction_best_idx_is_0`；
- `best_vs_first_l2` 和 executed-prefix 版本小于等于 `1e-6/1e-4/1e-3/1e-2/1e-1` 的比例；
- full horizon / executed prefix 的原始 L2 与 per-dim L2；
- noise layer、raw model action layer 与 env-space candidate layer 的 full/prefix L2、per-dim L2 和 std；
- translation/rotation/gripper std；
- candidate/selected action saturation 与 would-clip 比例。

per-dim L2 定义为 `L2 / sqrt(H*A)`，用于消除 horizon/action dimension 对数值规模的机械影响。

## 失效模式解释

- noise diversity 小：noise strategy 自身没有拉开 proposal。
- noise 大但 raw action diversity 小：生成模型对 noise 不敏感。
- raw action 大但 env-space diversity 小：output transform 压缩了差异。
- env candidate diversity 大但 q std/gap 小：critic 无法区分候选（只对 qselect 解释）。
- q gap 大、动作也改变但 success 不提升：critic 可能对 off-manifold candidate 失准。
- best index 多为 0 或 best-vs-first 阈值比例很高：选择没有实质改变动作。
- full horizon 大但 executed prefix 小：差异主要出现在未执行的 tail。
- saturation/would-clip 高：noise scale 可能主要制造边界动作，而不是有效控制多样性。

`saturation` 使用 `abs(action) >= 0.99`；`would_clip` 使用 `abs(action) > 1.0`。它们是相对于 LIBERO 常用 `[-1,1]` action boundary 的代理统计，诊断 server 不会实际裁剪或修改动作。

## 基础检查

```bash
cd /data/huangdi/heliqun/pi0
export action_horizon=10
(cd ours/debug && /data/huangdi/heliqun/pi0/openpi/pi_env/bin/python -m unittest -v test_qselect_diag.py)
(cd ours/debug && PYTHONPATH="/data/huangdi/heliqun/pi0/openpi/third_party/libero:${PYTHONPATH:-}" \
  /data/huangdi/heliqun/pi0/openpi/examples/libero/libero_env/bin/python -m unittest -v test_qselect_debug_eval.py)
/data/huangdi/heliqun/pi0/openpi/pi_env/bin/python -m py_compile \
  ours/debug/qselect_debug_server.py \
  ours/debug/summarize_qselect_diag.py \
  ours/debug/qselect_debug_eval.py
bash -n ours/debug/run_debug_server.sh
/data/huangdi/heliqun/pi0/openpi/pi_env/bin/python ours/debug/qselect_debug_server.py --help
```
