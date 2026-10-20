# 最终门禁汇总

日期：2026-10-20
范围：第三种子牌组交叉、Linux 候选包真实运行、200 局 checkpoint 门禁、下一轮微调方案。
本任务未启动长时间训练。

## 1. 第三种子牌组交叉

牌组：`official_dragapult` vs `agent/deck.csv`（未修改当前牌组）。
协议：`eval_h2h.py --opponent selfmirror --mode cross`，seed `20261023`，先 10 局冒烟再 100 局，座位严格 50/50。

| 方向 | 胜率 | Wilson 95% CI | 座位 0 | 座位 1 | 平局/异常/非法 | 平均长度 | 耗时 |
|---|---:|---|---:|---:|---|---:|---:|
| official_dragapult（候选座位） | 56.0% (56/100) | [46.2%, 65.3%] | 78% | 34% | 0/0/0 | 87.0 applies | 69.7 s |
| agent/deck.csv（候选座位） | 41.0% (41/100) | [31.9%, 50.8%] | 64% | 18% | 0/0/0 | 90.2 applies | 74.3 s |

三种子并列：A 方向 55.0%（21，60 局）/ 57.0%（22）/ 56.0%（23）；22+23 合并 56.5% [49.6%, 63.2%]。B 方向只有 22/23：44.0% / 41.0%，合并 42.5% [35.9%, 49.4%]。
座位效应大于牌组效应；分座位后 dragapult 在两个座位都优于当前牌组（79% vs 69%，34% vs 16%）。
详见 `runs/evaluation/deck_search/third_seed_report.md`。

## 2. Linux 推理是否实际通过

**未无条件通过。**

- 通过：WSL2 Ubuntu 24.04 + Python 3.12.3 + kaggle-environments 1.33.0 + torch 2.14.1+cpu；官方 `libcg.so` 加载正常，无缺失动态库；环境先加载时 `import main` 成功，牌组请求返回 60 张，`battle_start` 接受（`errorPlayer=-1`）。
- 通过：真实引擎双座位对局完成（如 seat 0 67 步 / 27 次决策、seat 1 21 步 / 4 次决策；文件路径加载 78 步），状态 DONE，动作合法，0 非法。
- 通过：真实 observation 上 `would_ko` Search API 正例，`attackId=1044`，`flags={13:[1.0,1.0,1.0]}`，写回成功；Search begin/step/release/end = 17/18/17/17，无异常。
- **失败/阻塞**：干净解释器直接 `import main`（在任何 `kaggle_environments` 导入之前）会在 `sdk_cg/sim.py:58` 的第二次 `GameInitialize()` 触发 SIGABRT：`buffer full. capacity:7`，退出码 134。环境先加载的顺序可规避，但不能把 Linux 兼容性写成无条件通过。
- 未验证：真实 Kaggle 线上 runner 的导入顺序；若线上先导入 agent，会命中该崩溃。详见 `runs/evaluation/linux_runtime_report.md`。

## 3. 200 局模型对比

协议：`scripts/h2h_ckpt.py`，候选 `ckpt_999424` vs 基线 `ckpt_500736`，200 局；牌组池 50/50（当前牌组 + official_dragapult）；`PKMN_TURN_CAP=200`，`--live 1`，Python 编码回退；座位 flip 107/93。

| 运行 | 结果 | 平局 | 异常 | 非法 |
|---|---:|---:|---:|---:|
| 主运行 | **117/200 = 58.5%**，CI [51.6%, 65.1%] | 0 | 0 | 未计数 |
| instrumented 复跑 | 118/200 = 59.0%，CI [52.1%, 65.6%] | 0 | 0 | **0 clamp / 0 forfeit** |

- `errs=0`；`h2h_ckpt.py` 的 `errs` 计入 `draws`，本批两者均为 0。
- 与前序 30 局 66.7% 的差异来自小样本/不同池；与 `eval_h2h` 的 44.5%（旧牌组）/69.0%（dragapult）不是同一协议。混合池的线性预期 `(0.445+0.690)/2=56.75%` 与实测 58.5% 接近，但这是推断，不是直接测量。
- 原生编码不可用（`native_encode_available=False`）；本结果只适用于 Python 编码口径。
详见 `runs/evaluation/final_gate_report.md`。

## 4. 下一轮微调建议预算

- 当前 1M 训练的真实配置：`num_envs=16, num_steps=64` → batch 1024，976 次迭代；`global_step=999424` 是环境步，不是梯度步。
- 实测 E16 吞吐约 86.6 环境步/s：
  - 0.5M ≈ 1.6 h（首阶段建议）
  - 2M ≈ 6.4 h（第二阶段上限）
  - 5M ≈ 16 h（可选第三阶段）
  - 10M ≈ 32 h；20M ≈ 64 h（当前证据不支持直接启动）
- 初始化：`--init-from runs/train-20m-e16-1m/ckpt_999424.pt`（纯权重，fresh optimizer）。`latest.pt` 才是含优化器的完整恢复点；两者不要混用。
- 牌组分布建议：60% `TRAIN_TOP15`（30 副，每副 0.02）+ 25% `official_dragapult` + 15% 当前 `agent/deck.csv`。任何单牌组不超过 25%。
- **接口前置**：当前 `--decks` 只接受预定义名称，`top15` 不包含这两副目标牌组，`official` 不含当前牌组；要执行该分布，必须先给 `resolve_deck_pool()` 增加 `json:<path>` 或新命名池，并做 1k 步 smoke。未完成前不得宣称该分布已生效。
- 门禁：Stage 1 结束用 400 局 `h2h_ckpt.py` 对 `ckpt_999424`，要求合并 ≥55% 且 Wilson 下界 >50%；同时看分牌组、旧基线、随机合法、自镜像和 held-out TOP50。失败即停止，保留原 checkpoint。
- 训练接口**没有随机策略对手分支**；随机合法策略只用于离线 `eval_h2h.py`，不能宣称已加入 PPO。
详见 `runs/evaluation/next_training_plan.md`。

## 5. 尚未验证的事项

1. Linux 提交包在真实 Kaggle runner 的 agent 导入顺序下是否稳定；当前独立 `import main` 会 SIGABRT。
2. Linux 提交 agent 自身在测试对局中未触发可 KO 攻击选项；`would_ko=1.0` 正例来自脚本代理，不是提交模型。
3. 原生编码（`native_encode=True`）路径不可用，所有结果均为 Python 编码回退。
4. 三牌组混合池的接口扩展、权重文件和 1k 步 smoke 尚未实现/运行。
5. 5e-5 LR、teacher KL 0.005、held-out TOP50 的具体牌组尚未做 A/B 或选定。
6. 10–20M 步微调的实际效果和耗时未验证；本文只给出外推。
7. 未验证 Kaggle 线上引擎与本地官方 `libcg.so` 的行为差异；Linux 测试只覆盖 WSL CPU。

## 6. 新增/修改文件与执行命令

### 新增文件（本任务）

```text
runs/evaluation/deck_search/third_seed_report.md
runs/evaluation/deck_search/third_seed/smoke_drag_vs_control_20261023_10.json
runs/evaluation/deck_search/third_seed/smoke_control_vs_drag_20261023_10.json
runs/evaluation/deck_search/third_seed/cross_drag_vs_control_20261023_100.json
runs/evaluation/deck_search/third_seed/cross_control_vs_drag_20261023_100.json
runs/evaluation/linux_runtime_report.md
runs/evaluation/final_gate_report.md
runs/evaluation/next_training_plan.md
runs/evaluation/final_summary.md
runs/evaluation/final_gate/pool_control_plus_dragapult.json
runs/evaluation/final_gate/pool_control_plus_dragapult_w.npy
runs/evaluation/final_gate/h2h_cand_vs_base_200_r0of1.json
runs/evaluation/final_gate/h2h_cand_vs_base_200_instr_r0of1.json
runs/evaluation/final_gate/illegal_counts_200.json
runs/evaluation/final_gate/run_h2h_instrumented.py
runs/evaluation/final_gate/smoke/smoke_cand_vs_base_10_r0of1.json
```

WSL 独立目录（不在仓库内，未修改 Windows 原始包）：

```text
~/pkmn-linux-verify-20261020/
  pylibs/ pkg/ linux_verify.py linux_verify_result.json
  linux_verify_mirror.py linux_verify_mirror_result.json
  probe_would_ko.py probe_would_ko_result.json
  probe_would_ko_multi.py probe_would_ko_multi_result.json
  test_runner_order.py debug_*.py
```

### 未修改

```text
agent/deck.csv
runs/evaluation/deck_search/final/*.json
runs/train-20m-e16-1m/ckpt_999424.pt
runs/train-20m-e16-1m/ckpt_500736.pt
runs/train-20m-e16-1m/latest.pt
Windows 原始 submission_rl_transformer2_dragapult_nodist/ 与 .tar.gz
```

仓库中 `rl/gates.py`、`rl/train_selfplay.py`、`sdk_cg/sim.py`、`tests/test_gates.py` 等未提交改动是任务开始前已存在的，不是本任务产生；本任务未改这些文件。

### 关键命令

```bash
# 第三种子：10 局冒烟 + 100 局，双向
PYTHONPATH=".;C:/Users/Swcmb/AppData/Local/Temp/opencode/bench-shim" PKMN_TURN_CAP=200 \
python runs/evaluation/eval_h2h.py --opponent selfmirror --mode cross --games 100 \
  --seed 20261023 \
  --pool-json runs/evaluation/deck_search/pools/official_dragapult.json \
  --net-pool-json runs/evaluation/deck_search/pools/control_deckcsv.json \
  --out runs/evaluation/deck_search/third_seed/cross_drag_vs_control_20261023_100.json

# 反向：交换 --pool-json / --net-pool-json

# 200 局主门禁
PYTHONPATH=".;C:/Users/Swcmb/AppData/Local/Temp/opencode/bench-shim" PKMN_TURN_CAP=200 \
python scripts/h2h_ckpt.py runs/train-20m-e16-1m/ckpt_999424.pt \
  runs/train-20m-e16-1m/ckpt_500736.pt 200 \
  --pool-json runs/evaluation/final_gate/pool_control_plus_dragapult.json \
  --pool-weights runs/evaluation/final_gate/pool_control_plus_dragapult_w.npy \
  --sample 2 --seed 777 --live 1 --progress-every 50 \
  --out runs/evaluation/final_gate --tag h2h_cand_vs_base_200

# 200 局非法动作计数复跑
PYTHONPATH=".;C:/Users/Swcmb/AppData/Local/Temp/opencode/bench-shim" PKMN_TURN_CAP=200 \
python runs/evaluation/final_gate/run_h2h_instrumented.py

# WSL Linux 验证
wsl.exe -d Ubuntu-24.04 -e bash -lc "cd ~/pkmn-linux-verify-20261020 && \
  PYTHONPATH=./pylibs:./pkg python3 linux_verify.py"
wsl.exe -d Ubuntu-24.04 -e bash -lc "cd ~/pkmn-linux-verify-20261020 && \
  PYTHONPATH=./pylibs:./pkg python3 probe_would_ko_multi.py"
wsl.exe -d Ubuntu-24.04 -e bash -lc "cd ~/pkmn-linux-verify-20261020 && \
  PYTHONPATH=./pylibs:./pkg python3 test_runner_order.py"
```
