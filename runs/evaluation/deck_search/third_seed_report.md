# 第三种子牌组交叉验证报告

日期：2026-10-20
状态：已完成实测；`agent/deck.csv` 未修改。

## 1. 结论

- 第三种子 `20261023` 的 100 局双向结果与种子 `20261022` 高度一致：
  - `official_dragapult` 作为候选座位对 `agent/deck.csv`：56/100 = 56.0%，Wilson 95% CI [46.2%, 65.3%]。
  - `agent/deck.csv` 作为候选座位对 `official_dragapult`：41/100 = 41.0%，Wilson 95% CI [31.9%, 50.8%]。
- 两个方向、两种子（22/23）合并后：`official_dragapult` 视角 113/200 = 56.5%，CI [49.6%, 63.2%]；`agent/deck.csv` 视角 85/200 = 42.5%，CI [35.9%, 49.4%]。
- 座位效应远大于牌组效应：同 checkpoint 自对弈下，座位 0 的胜率在两组牌组上都接近 69%–79%，座位 1 仅 16%–34%。因此必须看分座位结果；只看总胜率会把座位偏差误读成牌组优势。
- 在分座位口径下，`official_dragapult` 在两个座位都优于当前牌组：座位 0 为 79% 对 69%，座位 1 为 34% 对 16%（种子 22+23 合并，各 100 局）。这支持“候选牌组更强”，但这一结论只对当前 `ckpt_999424` 的贪心策略成立。
- 第三种子 100 局：0 平局、0 异常、0 非法动作；双向共 200 局同样为 0/0/0。

## 2. 实验设计

- 评估脚本：`runs/evaluation/eval_h2h.py`，`--opponent selfmirror --mode cross`。
  - 两个座位由同一个 checkpoint（`runs/train-20m-e16-1m/ckpt_999424.pt`，global step 999424）驱动，因此测的是**牌组差异**，不是模型差异。
  - `--pool-json` 决定候选座位牌组；`--net-pool-json` 决定对手座位牌组。两个池各只有 1 副 60 张牌组，实际等价于固定牌组对局。
  - 座位严格交替（`gi % 2`），10 局为 5/5，100 局为 50/50。
- 固定牌组：
  - `runs/evaluation/deck_search/pools/official_dragapult.json`：`official_dragapult`，60 张，内容与 `runs/evaluation/deck_search/dragapult_deck.csv` 一致。
  - `runs/evaluation/deck_search/pools/control_deckcsv.json`：`control_deckcsv`，60 张，内容与 `agent/deck.csv` 逐行一致。
- 运行环境：Windows，`D:\ProgramFiles\deeplearning\python.exe`，torch 2.14.0+cu130，CUDA 可用，`PKMN_TURN_CAP=200`。
- 引擎：仓库 `sdk_cg/` 缺 `cg.dll`，运行通过 `PYTHONPATH` 指向已有的临时 shim
  `C:/Users/Swcmb/AppData/Local/Temp/opencode/bench-shim`；`eval_h2h.py` 启动时把平台
  `kaggle_environments.envs.cabt.cg.sim` 注入为 `sdk_cg.sim`，复用同一 `cg.dll` 句柄。
- 命令（方向 A，`official_dragapult` 为候选座位）：

```text
PYTHONPATH=".;C:/Users/Swcmb/AppData/Local/Temp/opencode/bench-shim" PKMN_TURN_CAP=200 \
python runs/evaluation/eval_h2h.py --opponent selfmirror --mode cross --games 10 \
  --seed 20261023 \
  --pool-json runs/evaluation/deck_search/pools/official_dragapult.json \
  --net-pool-json runs/evaluation/deck_search/pools/control_deckcsv.json \
  --out runs/evaluation/deck_search/third_seed/smoke_drag_vs_control_20261023_10.json
```

方向 B 将 `--pool-json` 与 `--net-pool-json` 互换。正式评估把 `--games 10` 改为 `--games 100`，输出文件名中的 `smoke_..._10` 改为 `cross_..._100`。

## 3. 第三种子结果

### 3.1 冒烟（10 局）

| 方向 | 候选座位牌组 | 对手座位牌组 | 胜/负 | 胜率 | 座位 0 | 座位 1 | 平局 | 异常 | 非法 |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| A | official_dragapult | agent/deck.csv | 8/2 | 80.0% | 4/5 (80%) | 4/5 (80%) | 0 | 0 | 0 |
| B | agent/deck.csv | official_dragapult | 4/6 | 40.0% | 3/5 (60%) | 1/5 (20%) | 0 | 0 | 0 |

冒烟仅用于确认流程可运行，不用于效果结论；10 局区间过宽。

### 3.2 正式（100 局）

| 方向 | 候选座位牌组 | 对手座位牌组 | 胜/负 | 胜率 | Wilson 95% CI | 座位 0 | 座位 1 |
|---|---|---|---:|---:|---|---:|---:|
| A | official_dragapult | agent/deck.csv | 56/44 | 56.0% | [46.2%, 65.3%] | 39/50 (78%) | 17/50 (34%) |
| B | agent/deck.csv | official_dragapult | 41/59 | 41.0% | [31.9%, 50.8%] | 32/50 (64%) | 9/50 (18%) |

| 方向 | 平局 | 异常 | 非法动作 | 平均 `_apply` 次数 | 中位 `_apply` | P95 `_apply` | 平均回合 | 耗时 | 吞吐 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 0 | 0 | 87.0 | 97.0 | 99.1 | 74.5 | 69.7 s | 1.44 局/s |
| B | 0 | 0 | 0 | 90.2 | 97.0 | 99.0 | 76.1 | 74.3 s | 1.35 局/s |

- 胜率分母为已决出对局数（`decided`）；本批 100/100 全部决出，因此与 100 局总胜率相同。
- 非法动作统计同时包含 clamp 与 forfeit，`rate_per_decision = 0`。
- 两方向合计 200 局：0 平局、0 异常、0 非法动作。

## 4. 与之前两种子的并列比较

### 4.1 总胜率

| 种子 | 局数 | 方向 A：official_dragapult 视角 | 方向 B：agent/deck.csv 视角 |
|---|---:|---|---|
| 20261021 | 60 | 55.0% (33/60)，CI [42.5%, 66.9%] | 未运行 |
| 20261022 | 100 | 57.0% (57/100)，CI [47.2%, 66.3%] | 44.0% (44/100)，CI [34.7%, 53.8%] |
| 20261023 | 100 | 56.0% (56/100)，CI [46.2%, 65.3%] | 41.0% (41/100)，CI [31.9%, 50.8%] |
| 22+23 合并 | 200 | 56.5% (113/200)，CI [49.6%, 63.2%] | 42.5% (85/200)，CI [35.9%, 49.4%] |
| 21+22+23 合并（仅 A） | 260 | 56.2% (146/260)，CI [50.1%, 62.1%] | — |

三种子的 A 方向点估计落在 55.0%–57.0%，区间高度重叠；第三种子没有出现与种子 22 相反的信号。方向 B 只有种子 22/23 两次完整 100 局，点估计 44.0% 与 41.0%，合并后 42.5%。

### 4.2 分座位

| 数据 | 座位 0 | 座位 1 |
|---|---|---|
| 种子 21，A | 21/30 (70.0%) | 12/30 (40.0%) |
| 种子 22，A | 40/50 (80.0%) | 17/50 (34.0%) |
| 种子 23，A | 39/50 (78.0%) | 17/50 (34.0%) |
| 22+23 合并，A | 79/100 (79.0%)，CI [70.0%, 85.8%] | 34/100 (34.0%)，CI [25.5%, 43.7%] |
| 种子 22，B | 37/50 (74.0%) | 7/50 (14.0%) |
| 种子 23，B | 32/50 (64.0%) | 9/50 (18.0%) |
| 22+23 合并，B | 69/100 (69.0%)，CI [59.4%, 77.2%] | 16/100 (16.0%)，CI [10.1%, 24.4%] |

观察：

1. **座位效应稳定且很大**。同一 checkpoint、同一牌组池，座位 0 的胜率约 69%–79%，座位 1 约 16%–34%。这不是牌组差异，是当前 JSON 引擎/环境下的先后手或座位实现偏差。
2. **牌组效应在分座位后仍一致**。`official_dragapult` 在座位 0 比当前牌组高约 10 个百分点（79% 对 69%），在座位 1 高约 18 个百分点（34% 对 16%）。两个座位方向一致，说明候选牌组不是只靠座位运气。
3. **总胜率接近互补**。种子 22 两方向 57% + 44% = 101；种子 23 为 56% + 41% = 97。差异来自固定种子下换牌组会改变洗牌/引擎随机流，两次运行不是逐局严格镜像。
4. **第三种子没有改变结论方向**：A 方向合并 56.5% 的 CI 下界 49.6%，跨过 50%；仅凭 200 局还不能把“候选牌组严格优于当前牌组”写成高置信结论，但三种子点估计一致、分座位方向一致，证据强于单个 100 局结果。

## 5. 解读与限制

- 本实验固定同一个 checkpoint 自对弈，只回答“在这套策略下哪副牌组更强”，不能回答“换牌组后模型是否仍保持同等决策质量”。
- 两副牌组的差异不只是卡表：`official_dragapult` 与当前牌组属于不同 archetype，牌组强度、策略适配度、引擎内 matchup 都可能贡献差异。本实验没有拆分这些因素。
- 总胜率受座位偏置强烈影响。后续报告和门禁应以**分座位胜率**为主，总胜率只作为汇总展示。
- 方向 A 的种子 21 只有 60 局，且没有对应的方向 B；三种子合并的 260 局只覆盖 A 方向，不能当作双向平衡样本。
- `eval_h2h.py` 的输出 JSON 记录了 `pool_json`，未记录 `net_pool_json`；本报告的对手牌组由执行命令确定。若后续要机器复核，建议在结果旁保存完整命令行（本报告第 2 节已给出）。
- 未验证：不同 checkpoint、不同贪心/采样策略、Kaggle 线上引擎与本地 JSON 引擎之间的座位偏置是否相同。

## 6. 产物

新增文件：

```text
runs/evaluation/deck_search/third_seed/smoke_drag_vs_control_20261023_10.json
runs/evaluation/deck_search/third_seed/smoke_control_vs_drag_20261023_10.json
runs/evaluation/deck_search/third_seed/cross_drag_vs_control_20261023_100.json
runs/evaluation/deck_search/third_seed/cross_control_vs_drag_20261023_100.json
runs/evaluation/deck_search/third_seed_report.md
```

未修改：

```text
agent/deck.csv
runs/evaluation/deck_search/final/*.json
runs/train-20m-e16-1m/ckpt_999424.pt
runs/train-20m-e16-1m/ckpt_500736.pt
```
