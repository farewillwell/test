# Q-select 分层诊断工具设计

## 目标与边界

在 `/data/huangdi/heliqun/pi0/ours/debug/` 新建一套独立工具，定位 proposal noise 在以下链路中的失效层：

`noise diversity -> raw generated action diversity -> env-space candidate diversity -> critic score diversity -> selected action difference -> rollout success`

现有 `ours/iter.py`、`ours/Qselect/server.py`、`ours/Qselect/collect.py`、`ours/Qselect/selector.py` 及其他生产文件仅作为只读参考，不做任何修改。运行时固定使用 `pi05_libero_awbc`，不实现 pi0 兼容。

## 文件与职责

- `qselect_debug_server.py`：从生产 server 复制出的独立服务；采集 noise、raw action、env action、critic 和选择结果，写 JSONL/NPZ，并打印 `[diag]` 日志。
- `summarize_qselect_diag.py`：离线读取 JSONL，输出 JSON、CSV、best-index histogram、分位数和诊断结论模板。
- `run_debug_server.sh`：固定 OpenPI/pi05 路径和参数，仅允许用户编辑需求中列出的顶部变量。
- `README.md`：四组实验、LIBERO client 配合方式、指标解释和已复制的旧参考说明。
- `diverse_vis_ref.py`：只读复制 `ours/Qselect/diverse_vis.py`；仓库中未找到名称含 `diverse` 的目录，仅找到此文件。
- `test_qselect_diag.py`：使用标准库 `unittest` 覆盖统计口径、边界输入和汇总输出。

## 数据流与统计口径

debug server 保留原 websocket 协议。`simple` 直接委托原 policy infer，不伪造候选统计。`random` 和 `qselect` 生成候选时保留实际 noise 与 output transform 前 raw actions，最终 env-space candidates 进入 `compute_candidate_stats`。

候选 pair L2 将每个 `[H, A]` 展平，仅统计上三角的不同候选对；`N < 2` 时 pair 指标为 `0.0`。除原始 L2 外，full horizon 与 executed prefix 都记录 per-dim L2，定义为 `L2 / sqrt(H * A)`，消除 horizon/action dim 对量级的机械影响。`cand_std_*` 是先沿 candidate 维计算逐坐标 population std，再对指定 horizon/action slice 求均值。translation 使用 `:3`，rotation 使用 `3:6`，gripper 在 action dim 至少为 7 时使用 `6:7`，不存在的 slice 返回 `0.0`。执行前缀由 `diag_exec_horizon` 决定，`0` 使用完整 horizon，越界时报错。

`scores` 必须为 `[N]`；random 模式使用全零占位，并在顶层明确 `sample_mode=random` 与 `q_metrics_available=false`。top1-top2 gap 在少于两个 score 时为 `0.0`。汇总器不会针对 random 的占位 q 值计算或输出 critic 结论。所有输出转为 Python `int`/`float`，JSON 使用 `allow_nan=False`，非有限输入直接报错，避免产生不可移植 JSON。

env-space candidates 与 selected action 同时记录 full/prefix 的边界统计：绝对值最大值、`abs(x) >= 0.99` 的 saturation fraction、`abs(x) > 1.0` 的 would-clip fraction，以及至少一个维度命中上述条件的 candidate fraction。诊断代码不修改或裁剪 action；“clipping”是按 LIBERO 常用 `[-1, 1]` 边界计算的代理指标，并在 README 中明确这一点。

`best_vs_first_l2` 和 executed-prefix 版本除均值/分位数外，汇总器还报告小于等于 `1e-6、1e-4、1e-3、1e-2、1e-1` 各阈值的 request 比例，用于区分“索引不同但动作近似相同”和“选择真正改变动作”。

每个 candidate request 立即追加并 flush 一条 JSONL。顶层 `stats` 保留需求指定的 env-space 字段并加入 per-dim 与边界字段；阈值比例由汇总器跨 request 计算。额外的 `noise_stats` 与 `raw_action_stats` 记录前两层多样性。NPZ 按 `request_count % K == 0` 保存，至少包含 candidates、scores、best_idx、selected_action、stats_json，并额外保存 noise/raw actions。

服务从请求顶层及可选 `metadata` 字典中尽力提取 `episode_id`、`env_step`、`task_id`（同时接受 `trial_id/rollout_id`、`step/step_id/timestep` 等常见别名），并记录 prompt。当前生产版 `ours/Qselect/collect.py` 和 `openpi/examples/libero/main.py` 不发送这些字段，因此不做不可靠推断，缺失时写 `null`；README 给出在“复制到 debug 的 client/eval”中附加这三个标量字段的示例，不修改现有 client。

## 启动与错误处理

启动脚本固定：

- `PI0_ROOT=/data/huangdi/heliqun/pi0`
- `OPENPI_ROOT=/data/huangdi/heliqun/pi0/openpi`
- `POLICY_CONFIG=pi05_libero_awbc`
- OpenPI 环境 Python 与端口 8000

脚本在启动前检查 MODEL_PATH、checkpoint 的 `params/`、qselect 的 CRITIC_PATH、Python 可执行文件，并创建 `${OUT_DIR}` 与 records。它参考 `ours/Qselect/vis.sh` 完成一键闭环：后台启动 debug server、轮询端口 ready、运行 debug eval、生成 success metrics、运行 JSONL summary，最后通过 trap 清理 server。模型加载期间每 10 秒打印最新 server log，eval 日志通过 `tee` 同时显示并保存，避免长时间静默。`QSelector` 及 critic 相关依赖延迟到 qselect policy 初始化时导入；argparse 的 `--help` 在构造 policy/model 前退出，因此不会加载 checkpoint 或占用 GPU。

新增 `qselect_debug_eval.py`，从 `ours/Qselect/collect.py` 复制后只修改 debug 副本。每次重新规划时，请求附带 `episode_id=trial_id`、`env_step`、`task_id`；eval 输出 `${OUT_DIR}/eval_metrics.jsonl`，使 server diagnostic JSONL 能按 `(task_id, episode_id)` 与 success 对齐。默认不保存 rollout dataset episode 文件，只创建轻量 manifest；task suite/id/trials/initial offset 作为启动脚本顶部的 eval 变量。

## 汇总与诊断

汇总器拒绝空文件和损坏 JSONL，并按 `(sample_mode, noise_scale, noise_strategy)` 分组；同时保留 overall。每组对需求指标及新增 per-dim/saturation 指标计算 mean/std/p10/p50/p90，输出 best-index histogram、index-0 比例和 best-vs-first 阈值比例。JSON 使用嵌套结构；CSV 每个 group 一行，histogram/threshold ratios 以 JSON 字符串保存。stdout 按组打印诊断，random 组只解释 proposal/action/selection/saturation，不解释 q metrics；qselect 组才解释 critic 区分度。成功率仍由 episode metadata 对齐后的 LIBERO eval 指标对照，不声称 JSONL 自身测得 success uplift。

## 验证

测试先失败再实现。完成后运行：

1. `unittest`：统计、per-dim L2、执行前缀、阈值比例、saturation/would-clip、random 不解释 q、单候选、metadata 提取、debug eval request metadata、分组汇总和坏输入。
2. `python -m py_compile`：debug server、summary 与 debug eval。
3. `bash -n`：一键启动脚本语法。
4. 使用 `--help` 做无 GPU import/CLI smoke test。
5. occupied-port preflight：旧 server 占用 8000 时，一键脚本必须在模型加载前退出。
6. 仅在 MODEL_PATH/CRITIC_PATH 已配置、端口空闲且无长期占用风险时运行完整 server/eval smoke；否则明确记录原因。
