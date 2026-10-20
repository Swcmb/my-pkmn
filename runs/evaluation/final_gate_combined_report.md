# Pokémon TCG 最终门禁与微调方案（合并报告）

日期：2026-10-20

本文件按顺序合并以下五份报告，原文内容未改动：

1. [第三种子牌组交叉验证报告](#第三种子牌组交叉验证报告)
2. [Linux 候选提交包真实运行验证报告](#linux-候选提交包真实运行验证报告)
3. [200 局 checkpoint 对比门禁报告](#200-局-checkpoint-对比门禁报告)
4. [下一轮短程微调方案（暂不启动训练）](#下一轮短程微调方案暂不启动训练)
5. [最终门禁汇总](#最终门禁汇总)


---

<!-- source: runs/evaluation/deck_search/third_seed_report.md -->

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


---

<!-- source: runs/evaluation/linux_runtime_report.md -->

# Linux 候选提交包真实运行验证报告

日期：2026-10-20
候选包：`submission_rl_transformer2_dragapult_nodist.tar.gz`
SHA-256：`09b2929c5f9cf77759a087b09ad202bf05bdf32b36d9ac00e89ca2505b1d9d28`
结论：**未无条件通过**。在“先加载 `kaggle_environments` / CABT 环境，再加载 agent”的顺序下，真实引擎双座位对局可以完成；在干净解释器中直接 `import main` 会在 `sdk_cg/sim.py` 的第二次 `GameInitialize()` 处触发原生崩溃（SIGABRT）。因此不能把 Linux 兼容性写成“Linux 运行通过”。

## 1. 测试环境

| 项 | 值 |
|---|---|
| 发行版 | Ubuntu 24.04.4 LTS |
| 内核 | Linux 6.6.87.2-microsoft-standard-WSL2 x86_64 |
| Python | 3.12.3（系统解释器） |
| PyTorch | 2.14.1+cpu |
| CUDA | 不可用（WSL CPU 推理） |
| kaggle-environments | 1.33.0 |
| glibc | 2.39 |
| 隔离依赖目录 | `~/pkmn-linux-verify-20261020/pylibs`（`pip --target`，未改系统 Python） |
| 解包目录 | `~/pkmn-linux-verify-20261020/pkg` |
| 原始包 | 只读复制；Windows 原始包、模型、checkpoint 未修改 |

`kaggle-environments` 通过官方 PyPI 安装；Tsinghua 镜像未返回该包，按规则回退到上游。安装内容包含官方 CABT 引擎：

```text
/home/swcmb/pkmn-linux-verify-20261020/pylibs/kaggle_environments/envs/cabt/cg/libcg.so
```

## 2. 包内容与引擎加载路径

压缩包共 19 个文件，`sdk_cg/` 只有 5 个 `.py`，**不包含** `cg.dll` / `libcg.so`，符合“运行时解析引擎依赖”的候选方案。`pkg/` 内没有原生库。

`kaggle_environments.envs.cabt.cg.sim` 的实际加载路径：

```text
libcg._name = /home/swcmb/pkmn-linux-verify-20261020/pylibs/kaggle_environments/envs/cabt/cg/libcg.so
```

动态依赖检查：

```text
file: ELF 64-bit LSB shared object, x86-64, dynamically linked, BuildID=...
ldd:
  linux-vdso.so.1
  libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6
  libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1
  libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
  libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6
  /lib64/ld-linux-x86-64.so.2
```

没有缺失依赖。`libcg.so` 需要 glibc 2.39 环境（WSL 提供），未发现 `GLIBC_... not found`。

## 3. import 顺序：一个真实的阻塞

### 3.1 干净解释器直接 `import main`：失败

命令（工作目录 `pkg/`，`PYTHONPATH=../pylibs`）：

```text
python3 -c "import sdk_cg.sim"
```

输出：

```text
buffer full. capacity:7
terminate called after throwing an instance of 'std::runtime_error'
  what():  buffer full. capacity:7
Fatal Python error: Aborted
...
  File "/home/swcmb/pkmn-linux-verify-20261020/pkg/sdk_cg/sim.py", line 58 in <module>
```

退出码 134（SIGABRT）。`sdk_cg/sim.py` 第 58 行是 `lib.GameInitialize()`。

原因（已从代码与调用顺序确认）：

1. `_load_lib()` 先检查 `sys.modules["kaggle_environments.envs.cabt.cg.sim"]`。
2. 干净进程中该模块不存在，随后 `_engine_candidates()` 导入 `kaggle_environments.envs.cabt.cg`；导入父包会加载 CABT 环境，环境再加载 `cg.sim` 并调用一次 `GameInitialize()`。
3. `sdk_cg/sim.py` 随后又用 `ctypes.cdll.LoadLibrary` 加载同一 `libcg.so`，因为 `_loaded_here=True`，再次调用 `GameInitialize()`。
4. 官方引擎禁止二次初始化，抛出 `buffer full. capacity:7` 并 abort。

这不是静态链接问题，是运行期引擎初始化顺序问题。

### 3.2 先加载 `kaggle_environments`，再加载 agent：通过

以下两种顺序都能 `import main` 并返回 60 张牌组：

```text
python3 -c "import kaggle_environments; import main; print(len(main.DECK))"
python3 -c "import kaggle_environments.envs.cabt.cg.sim as s; import main; print(len(main.DECK))"
```

输出 `60`。此时 `sdk_cg.sim._loaded_here == False`，说明它复用了官方 `cg.sim` 的句柄，没有二次初始化。

### 3.3 按 Kaggle runner 的文件路径加载：通过

用 `env.run(["/.../pkg/main.py", opponent])` 让 `kaggle_environments` 先创建环境、再按文件路径导入 agent，真实对局完成：

```text
ok=true, steps=78, status=["DONE","DONE"], reward=[1,-1]
```

这与“环境先加载、agent 后加载”的实际 runner 顺序一致，但不能消除 3.1 的独立 `import main` 崩溃。候选包若被任何先导入 agent 的 harness 加载，会直接 abort。

## 4. 牌组请求与合法性

- `main.DECK` 长度 60，内容与 `runs/evaluation/deck_search/dragapult_deck.csv` 完全一致。
- `agent({"select": None})` 返回 60 个整数，与 `main.DECK` 相同。
- 直接调用官方 `battle_start(DECK, DECK)`：`errorPlayer = -1`、`errorType = 0`，引擎接受该 60 张牌组。
- 首决策调用序号为 2（第一次调用是牌组请求），说明 `select is None` 与后续选择分支都被真实执行。

## 5. 双座位真实对局

环境：`kaggle_environments.make("cabt", configuration={"bo": 1})`，对手为合法脚本代理或同牌组代理；每个 agent 在同一进程中只驱动一个座位。

| 测试 | 座位 | 步数 | 最终状态 | 奖励（本 agent） | 决策数 | 合法动作 |
|---|---:|---:|---|---:|---:|---:|
| `linux_verify.py` 第 1 局 | 0 | 67 | DONE/DONE | -1 | 27 | 27/27 |
| `linux_verify.py` 第 2 局 | 1 | 21 | DONE/DONE | +1 | 4 | 4/4 |
| `test_runner_order.py`（文件路径加载） | 0 | 78 | DONE/DONE | +1 | — | — |
| `probe_would_ko.py`（脚本代理） | 0/1 | 42 | DONE/DONE | +1/-1 | — | — |

- 所有测试对局均以 `DONE/DONE` 结束，没有 `ERROR` / `INVALID` / `TIMEOUT`。
- 动作合法性检查：动作是整数列表、无重复、长度不超过 `maxCount`、索引落在 `option` 范围内；上述 agent 对局中 `action_bad = 0`。
- 对局长度差异较大（21–78 步），部分对局在模型尚未展开攻击前就结束；这不影响“真实引擎可运行”的判断，但说明该 CPU 测试样本不足以评估策略强度。

## 6. `would_ko` Search API 与特征写回

### 6.1 提交 agent 自身的调用

在 `linux_verify.py` 的两局中，`_WK_ON=True`，`annotate_would_ko` 被调用 31 次；共产生 12 个攻击选项 flag，全部 `ko=0.0`，写回 12/12 成功，Search 调用 `begin=12, step=12, release=12, end=12`，无异常。提交 agent 在测试对局中没有遇到可 KO 的攻击选项，因此没有正例。

### 6.2 真实引擎上的正例

为验证 `would_ko=True` 的 Search 路径，使用**脚本化合法代理**（不是提交模型）在同一官方引擎、同一 CABT 环境中驱动对局；在真实 main select 中捕获到攻击选项 `attackId=1044`，调用 `would_ko_flags` 并写回：

```text
seat=1, turn=18, n_attack=1, attack_ids=[1044]
flags = {"13": [1.0, 1.0, 1.0]}   # ko_rate, expected_prizes, win_rate
writeback_ok = true
search = begin 17 / begin_ok 17 / step 18 / step_ok 18 / release 17 / end 17
begin_err = [] ; step_err = []
```

这证明官方 `libcg.so` 上 `SearchBegin` / `SearchStep` / `SearchRelease` / `SearchEnd` 可以正常调用，`would_ko=1.0` 的结果能写回 option 的 `would_ko` 字段。该正例来自脚本代理，提交 agent 自身尚未在测试中触发同类正例。

## 7. 结论与阻塞

已确认：

1. WSL2 Ubuntu 24.04 可以安装并加载官方 `kaggle-environments 1.33.0` 与 `libcg.so`。
2. 官方引擎接受候选牌组（60 张，`errorPlayer=-1`）。
3. 在环境先加载的顺序下，提交包可以导入、请求牌组、完成双座位真实对局，返回合法动作。
4. `would_ko` Search API 与特征写回在真实 observation 上工作，正例 `ko=1.0`。

未通过 / 剩余阻塞：

1. **干净解释器直接 `import main` 会 SIGABRT**（`buffer full. capacity:7`，`sdk_cg/sim.py:58`）。这是运行期初始化顺序缺陷，不能把 Linux 兼容性写成无条件通过。
2. 提交 agent 自身在本次 CPU 测试中未出现可 KO 的攻击选项，`would_ko` 正例由脚本代理覆盖；提交 agent 的 `would_ko=True` 行为仍未在真实对局中直接观察到。
3. 未验证真实 Kaggle 线上 runner 的 agent 导入顺序；如果线上先导入 agent 再创建环境，会命中第 1 条崩溃。
4. 未验证 GPU / CUDA 推理路径；本报告只覆盖 WSL CPU。

## 8. 产物与命令

新增（均在 WSL `~/pkmn-linux-verify-20261020/`，不在 Windows 原始包内）：

```text
pylibs/                              # 隔离依赖
pkg/                                 # 解包后的候选包
linux_verify.py
linux_verify_result.json
linux_verify_mirror.py
linux_verify_mirror_result.json
probe_would_ko.py
probe_would_ko_result.json
probe_would_ko_multi.py
probe_would_ko_multi_result.json
debug_game.py / debug_short.py / debug_attacks.py / debug_options.py
test_runner_order.py
```

主要命令：

```bash
python3 -m pip install --target ./pylibs -i https://pypi.org/simple kaggle-environments
python3 -m pip install --target ./pylibs --index-url https://download.pytorch.org/whl/cpu torch
tar -xzf submission_rl_transformer2_dragapult_nodist.tar.gz -C pkg
PYTHONPATH=./pylibs:./pkg python3 linux_verify.py
PYTHONPATH=./pylibs:./pkg python3 probe_would_ko_multi.py
PYTHONPATH=./pylibs:./pkg python3 test_runner_order.py
```

未修改：

```text
Windows 原始 submission_rl_transformer2_dragapult_nodist/
Windows 原始 submission_rl_transformer2_dragapult_nodist.tar.gz
runs/train-20m-e16-1m/ckpt_999424.pt
runs/train-20m-e16-1m/ckpt_500736.pt
```


---

<!-- source: runs/evaluation/final_gate_report.md -->

# 200 局 checkpoint 对比门禁报告

日期：2026-10-20
候选：`runs/train-20m-e16-1m/ckpt_999424.pt`（global step 999424）
基线：`runs/train-20m-e16-1m/ckpt_500736.pt`（global step 500736）
结论：**门禁通过，但优势中等**。候选 117/200 = **58.5%**，Wilson 95% CI **[51.6%, 65.1%]**；0 平局、0 异常、0 非法动作。CI 下界高于 50%，支持候选优于基线；点估计低于此前 100 局 `eval_h2h.py` 口径的 69.0%，原因是牌组池与座位/协议不同，不能直接混用。

## 1. 复核项

| 项 | 值 | 证据 |
|---|---|---|
| 候选模型 | `runs/train-20m-e16-1m/ckpt_999424.pt` | 脚本参数；JSON `a`/`step_a` |
| 候选 global step | 999424 | `torch.load(...).global_step` |
| 基线模型 | `runs/train-20m-e16-1m/ckpt_500736.pt` | JSON `b`/`step_b` |
| 基线 global step | 500736 | `torch.load(...).global_step` |
| 候选 SHA-256 | `b8a3be1d247eac45af0bbc38a207336d6f702c82fd0b19e5f787cc7addf08a1e` | `sha256sum` |
| 基线 SHA-256 | `b02b0724ebbb90916e9d495cbc4ab73b30f6bb761cea993366798942b5477a3d` | `sha256sum` |
| 两边 `net_config` | 完全相同：transformer2, d_model=544, ff=256, nhead=8, nlayers=5, value_atoms=51, would_ko=true | checkpoint `net_config` |
| 牌组池 | `runs/evaluation/final_gate/pool_control_plus_dragapult.json` | 新建，未覆盖旧文件 |
| 池内容 | 2 副 60 张：`control_deckcsv`（= `agent/deck.csv`）、`official_dragapult` | 池 JSON `names`/`counts` |
| 池权重 | `[0.5, 0.5]` | `pool_control_plus_dragapult_w.npy` |
| 采样 | `--sample 2`，两副牌都被选中；每副 100 局 | `_field_candidates(2,200)` = [0,1,0,1,…] |
| 座位分配 | `field_winrate_fast` 的随机 flip，seed 固定为 `4321`；200 局中座位 0 = 107、座位 1 = 93 | `np.random.default_rng(4321+1_000_003).integers(0,2,200)` |
| 运行参数 | `PKMN_TURN_CAP=200`，`--live 1`，`--rank 0 --world 1`，`--seed 777`（池采样） | 命令行/JSON |
| 设备 | CUDA，RTX 5060 Laptop，torch 2.14.0+cu130 | 运行环境 |
| 编码路径 | **Python 编码回退**（`native_encode_available=False`）；原生 `GetBinaryObs`/`AdvanceLog` 不在本机引擎导出 | `rl.gates._native_encode_available()` |

## 2. 主结果

命令：

```text
PYTHONPATH=".;C:/Users/Swcmb/AppData/Local/Temp/opencode/bench-shim" \
PKMN_TURN_CAP=200 \
python scripts/h2h_ckpt.py \
  runs/train-20m-e16-1m/ckpt_999424.pt \
  runs/train-20m-e16-1m/ckpt_500736.pt 200 \
  --pool-json runs/evaluation/final_gate/pool_control_plus_dragapult.json \
  --pool-weights runs/evaluation/final_gate/pool_control_plus_dragapult_w.npy \
  --sample 2 --seed 777 --live 1 --progress-every 50 \
  --out runs/evaluation/final_gate --tag h2h_cand_vs_base_200
```

| 指标 | 值 |
|---|---:|
| 对局数 | 200 |
| 候选胜 | 117 |
| 候选负 | 83 |
| 胜率（已决出） | 0.5850 |
| Wilson 95% CI | [0.5157, 0.6511] |
| 平局 / 截断 | 0 |
| 异常（spawn/step） | 0 |
| 耗时/吞吐 | 约 0.6 局/s（进度日志） |

`h2h_ckpt.py` 的 `errs` 同时计入 `draws`；本批 `errs=0, draws=0`。

## 3. 非法动作核验

`scripts/h2h_ckpt.py` → `field_winrate_fast` 不输出非法动作计数。为核实，另用
`runs/evaluation/final_gate/run_h2h_instrumented.py` 包一层
`TwoSidedSelfPlayEnv.step`，参数、池、种子与主运行完全一致，重跑 200 局：

| 指标 | 值 |
|---|---:|
| 完成局数 | 200 |
| step 调用数 | 18741 |
| illegal clamp | 0 |
| illegal forfeit | 0 |
| 异常 | 0 |
| 候选胜 | 118/200 = 59.0%（CI [52.1%, 65.6%]） |

- 两次运行相差 1 局（117 vs 118），来自 GPU/引擎的非严格确定性；均在主结果 CI 内。
- 动作 mask 在 `rl/policy.py:635` 由 `logits_value` 应用（`masked_fill(action_mask < 0.5, -1e9)`），策略不会直接选到被 mask 的动作；环境侧仍保留 clamp/forfeit 兜底。本批未触发兜底。
- 结论：**200 局内 0 异常、0 非法动作成立**（以 instrumented 重跑为直接证据）。

## 4. 与前序结果的关系

| 来源 | 协议 | 结果 | 可比性 |
|---|---|---|---|
| 本报告 | `h2h_ckpt.py`，双牌组池 50/50，200 局，随机座位 flip 107/93 | 58.5% (117/200) | 本轮主结果 |
| `runs/evaluation/h2h_ckpt/h2h_999424_vs_500736_r0of1.json` | 同一脚本，30 局，`sample=30` 的旧池，旧 runner 包装 | 66.7% (20/30)，CI [48.8%, 80.8%] | 小样本、不同池；仅方向参考 |
| `runs/evaluation/final_report.md` §0 | `eval_h2h.py` 对 `ckpt_500736`，两种子各 100 局，旧牌组 | 44.5% [37.8%, 51.4%] | 不同协议/牌组 |
| 同上 | `eval_h2h.py` 对 `ckpt_500736`，两种子各 100 局，`official_dragapult` | 69.0% [62.3%, 75.0%] | 不同协议/牌组 |

- 本轮池是 50/50 混合；若用前序两副牌的分牌组估计做线性预期：
  `(0.445 + 0.690) / 2 = 0.5675`，与实测 58.5% 接近。这个 56.75% 是**推断**，不是本轮直接测量；本轮 `h2h_ckpt.py` 不输出分牌组胜负。
- 因此“58.5% 低于此前 69.0%”不是模型退化，主要来自牌组池构成与协议差异。不能把 69.0% 与 58.5% 当作同一口径的前后对比。
- 旧 30 局 66.7% 的 CI 很宽，且池/座位不同，不能用来否定本轮 58.5%。

## 5. 编码口径

- 本轮 `native_encode_available=False`，所有 200 局走 **Python 编码回退**。
- 原生编码路径需要 `encode_native` + 引擎 `GetBinaryObs`/`AdvanceLog`；当前 bench-shim 的 `cg.dll` 不导出这两符号，强行开启会在第一局抛
  `AttributeError: function 'GetBinaryObs' not found`（前序报告已记录）。
- 本轮结果只对“Python 编码 + 当前引擎”有效；不能与任何 `native_encode=True` 的历史数字直接合并或比较。前序 `eval_h2h.py` 也使用 `native_encode=False`，但牌组池/座位协议不同，仍不能直接混算。

## 6. 门禁判定

| 检查 | 结果 |
|---|---|
| 200 局完成 | 是 |
| 0 异常 | 是（两次运行均为 0） |
| 0 非法动作 | 是（instrumented 运行 clamp=0, forfeit=0） |
| 候选胜率 > 50% 且 CI 下界 > 50% | 是（58.5%，CI 下界 51.6%） |
| 与旧基线相比有稳定改善 | 方向成立；幅度中等，且受池构成影响 |
| 可直接扩大训练预算 | 否；58.5% 只说明相对旧基线的边际优势，尚不足以支持 10–20M 步的大预算 |

**建议**：保留 `ckpt_999424.pt` 作为当前候选；微调先做小规模、可中止的试跑，并以本报告的分牌组/分座位补充评估作为门禁，而不是直接按 10–20M 步启动。

## 7. 产物

新增：

```text
runs/evaluation/final_gate/pool_control_plus_dragapult.json
runs/evaluation/final_gate/pool_control_plus_dragapult_w.npy
runs/evaluation/final_gate/h2h_cand_vs_base_200_r0of1.json
runs/evaluation/final_gate/h2h_cand_vs_base_200_instr_r0of1.json
runs/evaluation/final_gate/illegal_counts_200.json
runs/evaluation/final_gate/run_h2h_instrumented.py
runs/evaluation/final_gate_report.md
```

未修改：

```text
scripts/h2h_ckpt.py
rl/gates.py（本轮未新增改动；仓库已有未提交的 _field_candidates / native_encode 回退改动）
runs/train-20m-e16-1m/ckpt_999424.pt
runs/train-20m-e16-1m/ckpt_500736.pt
agent/deck.csv
```


---

<!-- source: runs/evaluation/next_training_plan.md -->

# 下一轮短程微调方案（暂不启动训练）

日期：2026-10-20
初始化：`runs/train-20m-e16-1m/ckpt_999424.pt`（global step 999424，纯权重）
目标：在不扩大风险的前提下，验证“继续微调能否稳定超过当前候选”，而不是直接烧 10–20M 步。

## 1. 训练接口核查结果

### 1.1 计步方式

- `--total-timesteps` 是**全局环境步预算**（`rl/train_selfplay.py:130`）。
- 每次迭代采集 `num_envs × num_steps` 个环境步；`global_step += batch_size * world`（`:468-469`, `:1931`）。
- 默认 `num_envs=64, num_steps=256` → 每迭代 16,384 环境步；当前 1M 训练实际使用 `num_envs=16, num_steps=64` → 每迭代 1,024 环境步（`latest.pt` 的 `args` 可核实）。
- `ckpt_999424` 的 999,424 = 976 × 1,024，正好 976 次迭代；因此“10–20M 步”是约 9,766–19,531 次迭代的**环境步**，不是 10–20M 次梯度更新。

### 1.2 梯度更新频率

- 每次迭代收集一个 rollout（`batch_size` 条转移），再做一次 PPO 更新。
- `batch_size = num_envs × num_steps`；`minibatch_size = batch_size / num_minibatches`（`:468-469`）。
- 默认 `num_minibatches=16`、`grad_accum=1`、`update_epochs=1`（`:161-172`）→ **每个 rollout 16 次优化器 step**，1 个 epoch。
- 当前 1M 配置（batch 1024）下：976 次迭代 = 15,616 次优化器 step。
- 若扩到 10M 步（batch 1024）：约 9,766 次迭代 = 156,256 次优化器 step；20M 约 312,512 次。

### 1.3 checkpoint 与恢复

| 文件 | 内容 | 用途 |
|---|---|---|
| `ckpt_{global_step}.pt` | net、args、net_config、global_step（无优化器/teacher） | 评估/导出/冷启动 |
| `latest.pt` | net、optimizer、teacher、global_step、sched_anchor、ft_best（两网模式另含 net_b/opt_b/t_b/cg_state） | 唯一完整恢复点 |

- `--init-from <ckpt>`：严格加载权重，**优化器/LR schedule 全新**，teacher 从该权重复制（`:671-676`）。
- `--resume <latest.pt>`：加载 net+optimizer+teacher+global_step，LR schedule 从恢复点继续（`:729-752`）。
- `ckpt_999424.pt` **不含 `opt`**；虽然 `--resume` 遇到缺失 `opt` 会标 `opt=FRESH`，但语义上它是“净权重初始化”。本轮方案使用 `--init-from ckpt_999424.pt`，避免把旧优化器状态与新的牌组分布混在一起。
- checkpoint 保存：`latest.pt` 每次保存；`ckpt_{global_step}.pt` 在 `global_step % save_every` 触发（`:2123-2140`），默认 `save_every=500000`。
- 实测加载开销（warm cache，CPU）：`ckpt_999424.pt` 0.019–0.028 s；`latest.pt` 0.053–0.072 s。相对小时级训练可忽略；冷盘未测。

### 1.4 日志定义

- stdout 每次迭代（`--log-every 1`）：`step / it / sps / avg / tch_wr / pg / v_ce / ent / clip / kl`（`:2089-2121`）。
- `matchup_r{rank}.jsonl`：每次迭代写 `{"step", "cells"}`，按牌组对记录对局计数（`:1942-1949`）。
- `value_diag.jsonl`：`--value-diag` 默认开启，每 `--diag-every 20` 次迭代汇总价值头分桶指标（`:438-443`）。
- 训练日志的 `tch_wr` 是训练内 teacher 对局指标，**不能替代独立对战门禁**；本方案的门禁全部用 `h2h_ckpt.py` / `eval_h2h.py`。

### 1.5 牌组接口

- `resolve_deck_pool()` 只接受预定义名称（`rl/train_selfplay.py:55-90`）：
  - `top15` → `TRAIN_TOP15`，实际 **30 副**（不是 15 副）；
  - `sample` → `agent/deck.csv`（当前提交牌组）；
  - `official` → `rl/decks.DECKS` 的 4 副官方牌组，含 `dragapult`，**不含**当前牌组；
  - `all` → 4 副官方 + sample，**不含** TOP15。
- 已核实：当前 `agent/deck.csv` 与 `official_dragapult` 都**不在** `TRAIN_TOP15` / `TRAIN_TOP50` 中；`TRAIN_TOP15` 与 `TRAIN_TOP50` 也不含这两副目标牌组。
- 因此“TOP15 + 当前牌组 + official_dragapult”的混合分布**当前接口不能直接表达**。执行前必须先做一个小的接口扩展：
  - 在 `resolve_deck_pool()` 中支持 `json:<path>`（与 `--finetune-deck` 的 `_resolve_deck_set` 类似），读取 `{"decks": [[60 ints], ...]}`；
  - 或用新的命名池常量实现同样效果；
  - 扩展后先跑 1 次 ≤1k 步的 smoke，确认 pool 长度、权重长度、日志中的 `[decks]` 行与实际一致，再进入正式阶段。
- `--deck-weights-file` 已经是现成接口（`:643-652`）：`np.load` 后按 pool 顺序一一对应；权重会归一化。

### 1.6 随机策略对手

- `rl/train_selfplay.py` **没有随机合法策略对手分支**。训练侧只有：对称自博弈、frozen `--opponent-ckpt`（finetune）、teacher/旧策略锚点；`random` 只出现在 seed、座位和 minibatch shuffle。
- 因此不能宣称“随机策略已加入 PPO 训练”。本方案的随机合法策略只出现在**离线评估矩阵**（`eval_h2h.py --opponent random`）。
- 若将来要把随机对手加入训练，需要单独实现 + 单测 + 小规模 A/B；本轮不做。

### 1.7 编码路径

- 本机引擎不导出 `GetBinaryObs` / `AdvanceLog`，`native_encode_available=False`；`--native-encode` 会失败。
- 训练与门禁必须使用默认的 **Python 编码回退**；不要开启 `--native-encode`。所有结果只与 Python 编码口径比较。

## 2. “10–20M 步”的实际预算与耗时

实测来源：`runs/bench-20m/report.md`（E16，`num_envs=16, num_steps=64`，JSON/Python 编码，compile 关闭）。

| 预算（环境步） | 迭代数（batch 1024） | 优化器 step（×16） | E16 端到端 86.6 步/s | 说明 |
|---:|---:|---:|---:|---|
| 0.5M | ~488 | ~7,808 | ~1.6 h | 建议的第一阶段试跑 |
| 1M | ~976 | ~15,616 | ~3.2 h | 与当前 1M 训练同量级 |
| 2M | ~1,953 | ~31,248 | ~6.4 h | 第二阶段上限 |
| 5M | ~4,883 | ~78,128 | ~16.0 h | 可选第三阶段 |
| 10M | ~9,766 | ~156,256 | ~32.1 h | 不建议直接启动 |
| 20M | ~19,531 | ~312,500 | ~64.1 h | 当前证据不支持 |

- 当前 1M 训练的时间戳（500736→999424 约 1h42m）与 E16 实测吞吐一致；1M 全量约 3.2–3.4 h。
- E32 在 benchmark 中 CUDA OOM，不可用；E8 约 81.5 步/s，略慢。推荐 E16。
- 上述是**连续运行**估算；加门禁、保存、恢复、环境启动，实际多 10–20%。
- `latest.pt` 243 MB、`ckpt` 81 MB；按 250k/500k 保存，I/O 开销远小于训练本身。恢复的主要成本是重新加载 243 MB 优化器状态和启动环境，实测 warm load <0.1 s，冷盘需另测。

## 3. 牌组分布与采样权重

### 3.1 目标分布

| 组 | 牌组数 | 权重 | 单副权重 |
|---|---:|---:|---:|
| `TRAIN_TOP15` | 30 | 0.60 | 0.02 / 副 |
| `official_dragapult` | 1 | 0.25 | 0.25 |
| 当前提交牌组（`agent/deck.csv`） | 1 | 0.15 | 0.15 |
| 合计 | 32 | 1.00 | — |

### 3.2 设计理由

- 60% 保留 generalist 覆盖，避免把模型拉回单一 archetype；TOP15 的 30 副按均匀权重，不人为放大某一副。
- 25% 给候选提交牌组 `official_dragapult`，使模型有机会专门改善它；上限 25%，避免单牌组过拟合。
- 15% 保留当前 `agent/deck.csv`，作为“不遗忘对照牌组”和回退保险；它的权重低于候选牌组，但不为零。
- 不把任何单副牌组放到 >25%；如果后续要提高候选牌组权重，必须先通过分牌组门禁，且每次最多 +5pp。

### 3.3 防过拟合措施

1. 分牌组评估：dragapult、当前牌组、TOP15 中随机抽 5 副、以及 3 副**未参与训练**的 TOP50 牌组（held-out）分别跑独立对局。
2. 牌组交叉：dragapult vs 当前牌组双向各 100 局，确认候选牌组优势没有被训练分布改写。
3. 训练日志的 `matchup_r0.jsonl` 只用于观察采样覆盖，不作为效果结论；效果结论只看离线 h2h。
4. 若 dragapult 提升但当前牌组/held-out 下降 >3pp，立即停止并保留 `ckpt_999424.pt`。

### 3.4 执行前的接口产物（尚未创建/未测试）

```text
runs/finetune-20261020/pool_top15_deploy.json
runs/finetune-20261020/pool_top15_deploy_w.npy
```

- `pool_top15_deploy.json`：`{"decks": [30 副 TOP15..., dragapult, control], "counts": [60...], "names": [...]}`。
- `pool_top15_deploy_w.npy`：`[0.02] * 30 + [0.25, 0.15]`，`np.float64`。
- 由于 `resolve_deck_pool()` 目前不接受 `json:`，上述文件只有在完成 1.5 节的接口扩展后才能被训练读取。**在扩展和 smoke 通过前，不得宣称该分布已生效。**

## 4. 分阶段微调方案

### Stage 0：冻结基线（已完成）

- 当前候选：`ckpt_999424.pt`。
- 200 局混合池：117/200 = 58.5%，CI [51.6%, 65.1%]；0 异常/0 非法。
- 三种子牌组交叉：dragapult 视角 56.5%（22+23 合并），control 视角 42.5%。
- 这些数字是本方案的门禁基线。
- 混合池的 random/self-mirror 基线尚未单独测；Stage 1 前先补测同池同种子的 100 局 random 与 self-mirror，作为回归阈值。不要用 dragapult/control 单牌组的历史 random 数字替代混合池基线。

### Stage 1：0.5M 环境步试跑（建议首先执行）

> 下列命令中的 `--decks json:` 需要先完成 1.5 节的接口扩展；当前代码会报 `unknown --decks`。未完成扩展前，只能用 `--decks top15`/`official`/`sample` 的单一预定义池，不能执行本方案的三牌组分布。

```text
--init-from runs/train-20m-e16-1m/ckpt_999424.pt
--decks json:runs/finetune-20261020/pool_top15_deploy.json
--deck-weights-file runs/finetune-20261020/pool_top15_deploy_w.npy
--num-envs 16 --num-steps 64 --num-minibatches 16 --update-epochs 1
--total-timesteps 500000
--lr 5e-5 --warmup-steps 50000 --decay-steps 500000 --lr-min-ratio 0.1
--teacher-kl-coef 0.005 --teacher-value-coef 0.005
--teacher-refresh 5000000
--save-every 250000 --log-every 1 --value-diag
--out runs/finetune-20261020/stage1
```

- 预算：约 1.6 h（E16），约 488 次迭代、7,808 次优化器 step。
- `--init-from` 让 global_step 从 0 开始，日志中的 step 是**本轮相对步数**；不要与 999,424 混读。
- `--teacher-kl-coef 0.005` / `--teacher-value-coef 0.005` 是建议的稳定项；teacher 从 `ckpt_999424` 复制，`--teacher-refresh 5M` 大于本轮预算，不会中途换 teacher。
- 训练结束后，用 `ckpt_{step}.pt`（net-only）做评估；用 `latest.pt` 只在需要继续训练时恢复。

**Stage 1 中途检查（250k 步）**：

- 0 异常、0 非法动作；KL 没有连续 10 次 >0.05；entropy 没有快速塌缩。
- 与 `ckpt_999424` 的 200 局 h2h 快速门禁：胜率 <50% 或 0 非法不成立 → 立即停止，保留原 checkpoint。

**Stage 1 结束门禁（500k 步）**：

- 主门禁：`h2h_ckpt.py new vs ckpt_999424`，双牌组池，2 个种子 × 200 局 = 400 局；合并胜率 **≥55%** 且 Wilson 95% 下界 **>50%**。
- 安全：0 异常、0 非法动作、平局 ≤2%。
- 牌组保持：dragapult vs control 双向各 100 局，dragapult 视角 ≥54%，control 视角 ≤46%，且相对 Stage 0 不下降 >2pp。
- 保留项：random 100 局不低于 Stage 0 同池同种子 -3pp；self-mirror 100 局在 45–55% 且 0 异常；held-out TOP50 3 副不低于 Stage 0 -3pp。
- 若主门禁失败：停止，不进入 Stage 2；保留 `ckpt_999424.pt` 和 Stage 1 的中间 ckpt 作为证据。

### Stage 2：扩展到累计 2M 环境步（仅在 Stage 1 通过后）

- 从 Stage 1 的最佳 ckpt 继续：优先用 `--resume stage1/latest.pt`（保留优化器），或重新 `--init-from` Stage 1 最佳 net（更干净，但优化器重置）。
- 预算语义不同：`--resume` 时 `--total-timesteps` 是**累计**预算（Stage 2 设为 2,000,000，global_step 从 Stage 1 继续）；`--init-from` 时是**本轮新增**预算（设为 1,500,000，global_step 从 0 开始）。不要把两者混用。
- 追加 1.5M 步（累计 2M），约 4.8 h；每 500k 步跑一次 200 局快速门禁。
- 门禁阈值同 Stage 1；若两次连续 500k 窗口的 h2h 点估计没有上升（或低于 53%），停止。
- 累计 2M 仍无明显改善 → 停止，不进入 Stage 3。

### Stage 3：可选累计 5M（不建议现在启动）

- 仅当 Stage 2 的 2M 结果相对 0.5M/1M 呈单调改善，且分牌组/held-out 没有回归时，追加 3M（累计 5M，约 16 h）。
- 10M（约 32 h）和 20M（约 64 h）**不建议在当前证据下启动**；需要先把 5M 的门禁做扎实，再讨论。

## 5. 评估矩阵（每阶段结束）

| 项 | 命令/协议 | 用途 | 阈值 |
|---|---|---|---|
| 主改善门禁 | `h2h_ckpt.py new ckpt_999424 200` ×2 seeds，双牌组池 | 新模型是否超过当前候选 | 合并 ≥55%，CI 下界 >50% |
| 旧基线 | `eval_h2h.py --opponent baseline --mode mirror --games 100`，双牌组池 | 是否仍胜 `ckpt_500736` | ≥55%（Stage 0 为 58.5%） |
| 自镜像 | `eval_h2h.py --opponent selfmirror --mode mirror --games 100` | 管线/座位 sanity | 45–55%，0 异常 |
| 随机合法 | `eval_h2h.py --opponent random --mode mirror --games 100` | 回归下限（离线，非训练对手） | 不低于 Stage 0 -3pp |
| 牌组交叉 A | `eval_h2h.py --mode cross`，dragapult 候选、control 对手，100 局 | 候选牌组是否仍占优 | dragapult 视角 ≥54% |
| 牌组交叉 B | 反向，control 候选、dragapult 对手，100 局 | 对称检查 | control 视角 ≤46% |
| held-out | 3 副未参与训练的 TOP50 牌组，各 100 局 vs baseline | 防遗忘 | 不低于 Stage 0 -3pp |

- 所有门禁必须使用**独立对战**，不得用训练内 `tch_wr` 代替。
- `h2h_ckpt.py` 的座位 flip 使用字面量 `seed=4321`，`--seed` 主要改变池采样顺序；两个种子不是独立的座位随机化。若需要座位平衡的补充证据，用 `eval_h2h.py` 的严格 50/50 协议，但 `eval_h2h.py` 只能对固定 baseline，不能替代 ckpt-vs-ckpt 主门禁。
- `h2h_ckpt.py` 不输出非法动作计数；需要时用 `runs/evaluation/final_gate/run_h2h_instrumented.py` 的同类包装器补计。
- 编码口径固定为 Python 回退；不得把任何 `native_encode=True` 历史数字混入。

## 6. 中止与保留策略

立即停止的条件（任一成立）：

1. 任意门禁出现异常 >0 或非法动作 >0。
2. KL 连续 10 次 >0.05，或 entropy 相对 Stage 0 下降 >30%。
3. 250k/500k 快速门禁 <50%，或主门禁 400 局 <55%。
4. dragapult 提升但 control/held-out 下降 >3pp。
5. 实际吞吐低于 70 步/s，导致 0.5M 阶段超过 2.5 h（检查环境/显存后再决定）。
6. 训练日志出现 NaN/Inf 或 checkpoint 保存失败。

保留策略：

- 永远保留 `runs/train-20m-e16-1m/ckpt_999424.pt` 和 `latest.pt`，不覆盖。
- Stage 1/2/3 各自独立输出目录；失败的 ckpt 不删除，作为证据。
- 只有主门禁通过的 ckpt 才进入下一阶段；否则回退到 `ckpt_999424.pt`。

## 7. 未实现 / 未验证

1. **三牌组池接口未实现**：`resolve_deck_pool()` 目前不接受 `json:`；本方案第 3.4 节的文件和权重只是设计，尚未创建/测试。
2. **随机策略对手未加入 PPO 训练**：训练接口无该分支；随机合法策略只用于离线 `eval_h2h.py`。
3. **5e-5 / teacher KL 0.005 是建议值**：未在本模型上做 LR/系数的 A/B；首阶段必须以 KL、entropy 和门禁结果决定是否调整。
4. **held-out TOP50 牌组未选定**：执行前需要从 `TRAIN_TOP50` 中排除 TOP15 后固定 3 副，并记录名称/ID。
5. **Linux 提交包导入顺序阻塞未修复**：`import main` 先于 `kaggle_environments` 会 SIGABRT；微调不解决该问题。部署前必须修复或明确 runner 顺序，并重新跑 `linux_runtime_report.md` 的验证。
6. **训练吞吐只测到 E16/E8/E32**；若改变 `num_envs`/`num_steps`/compile 设置，耗时需重新实测。

## 8. 结论

- “10–20M 步”在当前硬件上是 **32–64 小时**的连续训练，不是一次短试跑；当前 58.5% 的边际优势不足以支持直接投入该预算。
- 建议先做 **0.5M 步（约 1.6 h）** 的 Stage 1，用 400 局独立 h2h 和分牌组门禁决定是否继续；只有 Stage 1 通过，才扩展到 2M（约 6.4 h）；5M 作为可选中止点；10–20M 暂不启动。
- 执行前先完成三牌组池的接口扩展与 1k 步 smoke；不要把“计划中的分布”写成“已生效的训练配置”。


---

<!-- source: runs/evaluation/final_summary.md -->

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

