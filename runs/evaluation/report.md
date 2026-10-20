# PPO 自博弈模型 step=999424 评估报告

评估日期：2026-10-19（本地时钟）。所有数字为本地实测产物，来源文件在每节标注。未验证项统一列在"未覆盖项与信息不足声明"一节，正文中不出现未实测的胜负推断。

## 1. 结论速览

正式对战（每组 100 局，固定种子 20261019，先后手严格 50/50 均衡）：

| 对手 | 局数 | 胜率 | 95% CI（Wilson） | 胜/负 | 平局 | 异常 | 非法动作 |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| 基线 `ckpt_500736` | 100 | 0.6200（62.00%） | [0.5221, 0.7090] | 62/38 | 0 | 0 | 0 |
| 随机合法策略 | 100 | 0.4300（43.00%） | [0.3373, 0.5278] | 43/57 | 0 | 0 | 0 |
| 同架构未训练初始化 | 100 | 0.2200（22.00%） | [0.1500, 0.3107] | 22/78 | 0 | 0 | 0 |

显著性判读：

- 对基线 `ckpt_500736`：CI 下界 0.5221 > 0.5，候选显著占优。
- 对随机合法策略：CI [0.3373, 0.5278] 跨 0.5，结果不显著；点估计 0.4300 落在 0.5 之下。
- 对同架构未训练初始化：CI 上界 0.3107 < 0.5，候选显著占优。

补充事实：

- 300 局正式对战全部无异常退出、无平局、无非法动作（`illegal.total = 0`，clamp/forfeit 四路计数全 0）。
- 座位效应显著：候选在与基线的对局中 seat0 胜率 0.84（42/50），seat1 胜率 0.40（20/50）。总胜率相对 0.5 的偏差不能当作座位均衡证据，座位均衡应按 `seat_balance.games_seat0/games_seat1` 各为 50 核验。
- 对照组 30 局交叉验证（见第 5 节）给出非传递性证据：未训练初始化胜基线 0.90（27/30），而候选胜基线 0.62（100 局）。
- `value_diag` 的 `sign_acc`/`mse_out` 与 PPO rollout buffer 同源，属于在训 on-policy 数据，按局不独立（见第 6 节）。
- 提交包结构性校验全部通过；`would_ko=True` 的推理路径本地无法验证（见第 7、8 节）。

## 2. 评估对象与候选产物

| 项 | 值 | 来源 |
| :--- | :--- | :--- |
| 候选模型 | `runs/train-20m-e16-1m/ckpt_999424.pt`（step=999424） | `eval_baseline_100.json` 顶层 `cand_ckpt`/`cand_step` |
| 基线对手模型 | `runs/train-20m-e16-1m/ckpt_500736.pt`（step=500736） | `eval_baseline_100.json` `results[0].meta` |
| 牌池（正式评估） | `runs/evaluation/pool_top15.json`（30 副） | 三份 `eval_*_100.json` 顶层 `pool_json` |
| 牌池（备用） | `runs/evaluation/pool_top50.json`（74 副） | 牌池构建脚本产物 |
| 提交包（tar） | `submission_rl_transformer2.tar.gz`（75.2 MB） | `export.log` L2 |
| 提交包（解包目录） | `submission_rl_transformer2/` | `verify_submission.log` |

网络配置（`eval_untrained_100.json` `results[0].meta.net_config` 全文）：

```json
{"arch": "transformer2", "d_model": 544, "nhead": 8, "nlayers": 5, "ff": 256,
 "static": false, "would_ko": true, "split_heads": false,
 "value_categorical": false, "value_atoms": 51, "value_vmax": 1.0}
```

两个牌池均已通过引擎合法性校验（`ALL DECKS ENGINE-OK`），每副 60 张。

## 3. 评估协议

- **回合上限**：`PKMN_TURN_CAP=200`。`eval_h2h.py` L31 起对 `PKMN_TURN_CAP` 的 pre-parse 位于 `import rl` 之前（`eval_h2h.py` L31-L32 注释），训练与评估共读同一份环境上限，候选侧不存在环境截断。
- **编码**：`native_encode=False` + `would_ko=True`（`eval_h2h.py` L146-L147），与候选 `ckpt_999424` 的 `net_config`（§2）一致。
- **对战形式**：mode=mirror，`gi % 2` 严格均衡先后手（`eval_h2h.py` L141-L148，L142 行内注释 `# 严格均衡先后手`）；每对手 100 局；`live_n=1` 串行；每局种子 `int(seed + gi)`。
- **牌组**：每对手一个牌池，stride 7 轮转（`eval_h2h.py` L128-L130）。
- **CI**：Wilson score 95%（z=1.959963984540054，`eval_h2h.py` L77-L78）。

### 3.1 对手定义

| 组 | 定义 | 备注 |
|---|---|---|
| `baseline` | `ckpt_500736.pt` 同配置同码镜像对局 | §1 现有基线 |
| `random` | 随机合法策略（legal-only），标签 `random_legal` | 不消耗候选算力路径之外的模型 |
| `untrained` | 与候选同 `net_config` 的随机初始化网络，`init_seed=20261019`，输入张量形状派生自 `ckpt_999424` 权重 | 同架构零能力下界 |

## 4. 正式对战结果（每对手 100 局，seed=20261019）

| 对手（label） | 胜率 | 95% CI | 分先后手胜率 (seat0 / seat1，各 50 局) | 平局 | 异常 | 非法动作（候选侧） | 平均对局长度（applies） | 平均回合数（turn） | 耗时 |
|---|---|---|---|---|---|---|---|---|---|
| baseline `ckpt_500736` | 0.62 | [0.5221, 0.7090] | 0.84 (42/50) / 0.40 (20/50) | 0 | 0 | 0 | 106.92（median 98, p95 159.2, max 707） | 79.14（p95 95, max 96） | 444.46 s（0.23 g/s） |
| random `random_legal` | 0.43 | [0.3373, 0.5278] | 0.42 (21/50) / 0.44 (22/50) | 0 | 0 | 0 | 108.54（median 112, p95 201.6, max 387） | 36.79（p95 74.5, max 126） | 143.34 s（0.70 g/s） |
| untrained `untrained_same_config` | 0.22 | [0.1500, 0.3107] | 0.20 (10/50) / 0.24 (12/50) | 0 | 0 | 0 | 116.66（median 113, p95 203.75, max 313） | 77.19（p95 96, max 96） | 438.82 s（0.23 g/s） |

来源：`eval_baseline_100.json`、`eval_random_100.json`、`eval_untrained_100.json` 的 `results[0]`。显著性判读同 §1（baseline 显著占优；random 不显著且点估计低于 0.5；untrained 候选显著占优）。

- 平局：三组 `draw_breakdown` 的 `turn_cap/max_steps/true_draw` 均全 0。
- 非法动作：三组候选侧 `clamp=forfeit=0`，`rate_per_decision=0.0`（含 `clamp_candidate`、`forfeit_candidate` 全 0）。
- 对局长度口径说明：`applies` 为逐局实际动作次数，max 值（707 / 387 / 313）可远超平均，且与 `turn`（回合数，max 96–126）不同量纲，两列不可混算。
- 座位效应：仅 baseline 组 seat0/seat1 胜率差 0.44（0.84 vs 0.40）；random、untrained 两组座位差 0.02–0.04。先手优势主要体现在对基线的镜像局中。

## 5. 对照局与交叉对局（30 局，seed=20261019，PKMN_TURN_CAP=200）

来源：`ctl_base_random_30.json`、`ctl_base_selfmirror_30.json`、`ctl_base_untrained_30.json`、`ctl_selfmirror_30.json`。

### 5.1 基线自镜像（`ckpt_500736` vs 自身）

`ctl_base_selfmirror_30.json`：候选侧（基线）胜率 0.4667（14/30），CI [0.3023, 0.6386]；seat0 胜率 0.8667（13/15）、seat1 0.0667（1/15）——基线对局座位效应极强。平均 applies 97.87，平均 turn 94.5（max 95）。平局 0，非法 0。

### 5.2 基线 vs 随机

`ctl_base_random_30.json`：基线胜率 0.5667（17/30），CI [0.3920, 0.7262]。随机对手 100 局组胜率 0.43（§4），30 局对照 0.5667，方向一致。

### 5.3 基线 vs 未训练初始化

`ctl_base_untrained_30.json`：基线胜率 0.1000（3/30），CI [0.0346, 0.2562]，即未训练初始化胜基线 0.90（27/30）。对照 §4 候选 100 局胜基线 0.62（CI [0.5221, 0.7090]）——未训练胜基线（0.90）与候选胜基线（0.62）呈现非传递性；候选反而输给未训练（§4 候选视角 0.22）。30 局小样本下 CI 宽，仅作方向证据。

### 5.4 ckpt 直对（候选 999424 vs 基线 500736，30 局，seed=777）

来源：`h2h_ckpt/h2h_999424_vs_500736_r0of1.json`，对拍 runner 为临时脚本 `run_h2h_ckpt.py`（位于 `C:\Users\Swcmb\AppData\Local\Temp\opencode\`，未入库；调用仓库自带 `scripts/h2h_ckpt.py`，L77-L79 座位传字面量 `seed=4321, ..., mirror=True`，不保证 15/15 均衡）。

| 模式 | 结果 | 证据 |
| :--- | :--- | :--- |
| raw（`native_encode=True`，训练侧口径） | 失败，0 局完成 | 临时目录 `h2h_raw.log` 尾行逐字 `AttributeError: function 'GetBinaryObs' not found` |
| patched（`native_encode=False` + `_Pinned` 覆写 `_sample_decks`，与 §3 评测口径一致） | 候选胜 20/30（wr=0.6667），draws=0 | `h2h_ckpt/h2h_999424_vs_500736_r0of1.json`：`wins=20, decided=30, draws=0, seed=777, rank=0, world=1, turn_cap=200` |

- 该 JSON 无 `winrate`/`winrate_ci95`/`errs` 键，0.6667（20/30）为 `wins/games` 直接计算，无 CI。
- 与 §4 候选 100 局口径（0.62，CI [0.5221, 0.7090]，seed=20261019）同向，但口径不同（30 局小样本 + 座位字面量不均衡 + 单世界 rank=0），本条仅作方向证据，§1 结论不以此为准。

## 6. `value_diag` 核验

数据来源：`runs/train-20m-e16-1m/value_diag.jsonl`（34 条记录，step=122880 起、step=983040 止，无 999424 行）。分桶统计中间产物：`runs/evaluation/value_diag_buckets.json`（L3 note 逐字：`k=距终局步数; short=1-4, mid=5-16, long=17-64; n-weighted approx for sign_acc (draw weight not separable)`）。

### 6.1 样本独立性核验：`sign_acc`/`mse_out` 使用的不是独立样本

- `rl/value_diag.py` docstring（L1-L3）逐字：`FREE at training time: reuses the PPO rollout buffer, no extra games.`——诊断与 PPO 训练共享同一份 rollout buffer，属于在训 on-policy 数据，无留出集。
- 同一局（episode）在窗口内的所有 step 逐行计入，且共享同一终局标签：`rl/value_diag.py` L63 `b[0] += 1` 对全部行计数，L72 `if s < 0 or done[s, e] > 0.5:  # crossed into the previous episode -> stop` 界定 episode 边界。因此按局不独立，`sign_acc` 是 n-weighted 近似（bucket note 已声明）。
- 终局标签本身是真实的：`rl/rollout.py` L97-L98 docstring 逐字 `terms = {(step, env): terminal-rewards-dict} for every REAL (non-truncated) terminal in the window`；L302-L305 `if d and "terminal" in info:` → `terms[(step, e)] = info["terminal"]`，即截断回合不产生 `terms` 记录，被诊断跳过。

### 6.2 标签与统计口径

- `sign_acc` 排除平局：`rl/value_diag.py` L64 逐字 `if r != 0.0:                       # draws carry no sign information`，其分母只计非平局行。
- `mse_out` 分母含平局：`rl/value_diag.py` L63 `b[0] += 1` 计全部行，`b[3] += (v - r) ** 2` 对全部行累加。两个指标分母不同，不可直接对照。
- 分桶变量 `k` 是距终局步数（1-4 / 5-16 / 17-64），不是全局对局长度；§6 的"长局/短局"按此口径统计。

### 6.3 关键数值（`value_diag_buckets.json` `rows`）

| step | short n / sa / mse | mid n / sa / mse | long n / sa / mse |
| ---: | :--- | :--- | :--- |
| 716800（观察到的低点） | 686 / 0.7653 / 0.5832 | 1790 / 0.7670 / 0.5846 | 3177 / 0.7668 / 0.6389 |
| 962560 | 663 / 0.8959 / 0.3157 | 1742 / 0.9053 / 0.3019 | 3117 / 0.8765 / 0.3755 |
| 983040（末次记录） | 329 / 0.8815 / 0.3995 | 854 / 0.9040 / 0.3744 | 1514 / 0.9359 / 0.3628 |

- 末次记录 step=983040 的 `long.mse_out` 为 0.3627851986721401（全精度值），四舍五入 0.3628。
- 训练末期 `sign_acc` 稳定在 0.88-0.94 区间；`mean_absV`（末次 0.5117）表明输出尺度未塌缩到 0。
- 无 step=999424 的记录：`value_diag.jsonl` 止于 983040，本核验不能覆盖候选 checkpoint 本身对应的训练点，只能给出其前一个记录点（983040）的状态。

## 7. 提交包校验

导出命令与产物：`python scripts/export_rl_submission.py --backend transformer2 --ckpt runs\train-20m-e16-1m\ckpt_999424.pt`。`export.log` 逐字三行：

```text
backend=transformer2  trained to step 999424
wrote D:\pkmn-kaggle\submission_rl_transformer2.tar.gz (75.2 MB)
top-level contents: ['EN_Card_Data.csv', 'attack_data.py', 'buff_data.py', 'card_features.py', 'deck.csv', 'decks.py', 'effect_data.py', 'enc_constants.py', 'encoding.py', 'main.py', 'model.pt', 'policy.py', 'sdk_cg', 'search_agent.py']
```

### 7.1 包结构（`verify_submission.log`，解包实况）

- tar `member_count=19`（18 个文件 + 1 个目录条目 `sdk_cg/`）、`has_pycache=false`、`has_dot_git=false`；解包目录 `C:\Users\Swcmb\AppData\Local\Temp\subsmoke_zh72avpa`。解包后仓库内解包目录 `submission_rl_transformer2/` 实测 31 文件 = tar 的 18 文件 + 13 个运行 `main.py` 产生的 `.pyc`（顶层 10 个 + `sdk_cg/` 3 个），属解包目录污染，非 tar 内容。
- `deck.csv` 实况（tar 内）：`deck_len=60`、`deck_unique=11`、`deck_counts` 全文：

```json
{"3": 33, "721": 2, "722": 4, "723": 4, "1092": 1, "1121": 2, "1145": 2,
 "1163": 2, "1219": 4, "1227": 4, "1262": 2}
```

本地独立复核解包目录 `submission_rl_transformer2/deck.csv`：60 行、文件 191 字节、唯一牌组 11 个，与 log 一致。

### 7.2 接口与导入（`clean_import_test.json`，权威导入证据）

- `import_main_ok=true`：干净环境（`sdk_cg_present_at_start=[]`）下 `import main` 成功。
- `main_wk_on=true`、`main_has_sa2=true`：`main.py` 启动时 `would_ko=True` 生效，且 `search_agent` 模块可载入。
- `modules_sdk_cg_after_import_main=[]`：`main` 导入路径不触碰平台 `sdk_cg`。
- `explicit_import_api_ok=false`：显式 `from sdk_cg import api` 抛 `FileNotFoundError: Could not find module 'D:\pkmn-kaggle\submission_rl_transformer2\sdk_cg\cg.dll'`，traceback 落在 `sdk_cg/api.py` L6 → `sdk_cg/sim.py` L29。
- 判定：被验证的是"平台是否预注入 `sdk_cg`"。哨兵文件 `platform_agent\sdk_cg\__init__.py` 建立后，`find_spec` 仍指向解包目录内的 `sdk_cg`（`find_spec_locations` 仅 `submission_rl_transformer2\sdk_cg`），说明本机无任何 `kaggle_environments` 可观测性（`find_spec=None` 于平台包），平台预注入行为本地不可判定，见 §8/§10。
- 导入证据以 `clean_import_test.json` 为权威，取代被 shim 预注入污染的 `verify_submission.log` 中的 import 段（其 L3 逐字：`[shim] sdk_cg.sim -> 平台引擎别名已注入（补 7 声明）`，该日志生成于 shim 已注入的进程内，不能证明干净环境行为）。

### 7.3 `would_ko=True` 推理路径的本地验证缺口

- 导出内容由 `wk = bool(ck.net_config.would_ko)` 门控（`main.py`），`main_wk_on=true` 只证明开关状态，不证明该推理路径可跑通。
- 必需符号证据（两条独立路径）：
  - `pe_exports.py` 对 bench-shim dll（实体 `C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim\kaggle_environments\envs\cabt\cg\cg.dll`，1659392 B）输出 `COUNT 14`，清单逐字：`AgentStart, AllAttack, AllCard, BattleFinish, BattleStart, BattleStartReverse, GameInitialize, GetBattleData, SearchBegin, SearchEnd, SearchRelease, SearchStep, Select, VisualizeData`——无 `WouldKO`/`GetSerialObs`/`GetBinaryObs`。
  - `check_dll_exports.log` 逐字：`WouldKO False` / `GetSerialObs False` / `GetBinaryObs False` / `AgentStart True` / `SearchBegin True` / `PKMN_WK_NATIVE=None` / `wk_native_enabled()=False`。
- 结论：本机可用引擎缺 `would_ko=True` 所需的 3 个符号，该推理路径本地无法验证；§10 声明。

## 8. 已知缺陷与风险

1. **`would_ko=True` 推理路径本地未验证**（§7.3）。提交包 `main.py` 在 `would_ko=True` 下需调用 `WouldKO` 符号（经 `annotate_would_ko`），而本机两个可用引擎（bench-shim `cabt/cg.dll` 与缺失的本仓 `sdk_cg/cg.dll`）均不导出该符号。`sim.py` 对 3 个搜索/枚举符号（`WouldKO`/`GetSerialObs`/`GetBinaryObs`）无 hasattr 守卫，符号缺失时导入即失败。本地能做的全部动作合法性验证（§4 三组 100 局 `illegal.total=0`）发生在 `rl` 侧 harness，不在提交包 `main.py` 的 `would_ko` 路径上。
2. **patched 对拍口径偏移**：§5.4 patched 模式用 `native_encode=False` + `_Pinned` 覆写 `_sample_decks` 运行。非 native 的 `_encode` 会跳过 `annotate_would_ko`，`OPT_WK=11` 列 `[11:14]` 零填充，与训练输入分布不同。§5.4 的 0.6667 在该偏移口径下成立，不可外推到 `native_encode=True` 的 raw 口径（raw 口径 0 局完成，仅有失败面证据）。
3. **座位效应**：§4 候选 vs 基线 seat0 胜率 0.84 vs seat1 0.40；§5.1 基线自镜像 seat0 13/15 vs seat1 1/15。镜像局先手优势显著。总胜率 0.62 是 50/50 座位混合值，座位敏感的对局（如平台固定座位）下结论可能漂移。
4. **非传递性战绩**（§5.3）：未训练初始化胜基线 0.90（27/30）、候选胜基线 0.62（100 局）、候选输未训练 0.22（100 局）。基线 `ckpt_500736` 的行为被后训练模型系统性克制；0.62 的"胜基线"不能线性外推为对所有对手的优势。
5. **`value_diag` 与候选 checkpoint 不对齐**（§6）：末次记录 step=983040 < 999424，诊断止于候选训练点之前 16384 步；且诊断样本为同 buffer on-policy 数据，`sign_acc` 是 n-weighted 近似。
6. **`eval_h2h.py` 的 `label` 字段硬编码**（L110/L113 按 `kind` 写死），`ctl_base_selfmirror_30.json` 的 `meta.step=500736` 而其 label 写 `selfmirror_ckpt_999424`——JSON 内 label 不可信，本报告全部口径以 `results[0].meta` 为准。
7. **引擎硬约束**：同进程绝不二次 `LoadLibrary`（CardTable 跨句柄共享，双加载触发 assert/状态损坏），故 `live_n=1` 串行、吞吐 0.22-0.93 g/s。复现 300 局正式 + 120 局对照共约 1421 s（§4 三组合计 1026.6 s，§5 四组 `elapsed_s` 合计 394.6 s：117.29/106.29/32.42/138.62 s），加 §5.4 patched 30 局约 25 分钟以内。

## 9. 复现命令

正式 100 局（本机在 `D:\pkmn-kaggle` 下；`bench-shim` 路径为本机临时目录）：

```bat
cd /d D:\pkmn-kaggle
set PYTHONPATH=.;C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim
set PKMN_TURN_CAP=200
py runs\evaluation\eval_h2h.py --opponent all --games 100 --seed 20261019 --out runs\evaluation\eval_baseline_100.json
```

实际执行的三组命令：同上，`--opponent` 依次为 `baseline`/`random`/`untrained`，`--out` 依次为 `eval_baseline_100.json`/`eval_random_100.json`/`eval_untrained_100.json`（候选均为 `--ckpt runs\train-20m-e16-1m\ckpt_999424.pt`）。对照 30 局：`--games 30 --seed 20261019`，`--out ctl_*.json`，其中 `--ckpt` 分别为候选 999424（`--opponent selfmirror`）与基线 500736（`--opponent` 依次为 `selfmirror`/`random`/`untrained`）。`value_diag` 分桶：临时脚本 `value_diag_stats.py`（`C:\Users\Swcmb\AppData\Local\Temp\opencode\`）读 `runs/train-20m-e16-1m/value_diag.jsonl` 写 `runs/evaluation/value_diag_buckets.json`。ckpt 对拍（§5.4）：`cmd /c "set H2H_MODE=patched && set PYTHONPATH=.&& python C:\Users\Swcmb\AppData\Local\Temp\opencode\run_h2h_ckpt.py > h2h_patched.log 2>&1"`。导出：`python scripts/export_rl_submission.py --backend transformer2 --ckpt runs\train-20m-e16-1m\ckpt_999424.pt`（`export.log` 见 §7）。

临时目录（`C:\Users\Swcmb\AppData\Local\Temp\opencode\`）内引用物：`export.log`、`verify_submission.log`、`h2h_raw.log`、`h2h_patched.log`、`check_dll_exports.log`、`clean_import_test.json`、`run_h2h_ckpt.py`、`ctl_dump.txt`、`value_diag_stats.py`。这些文件仓库内无副本，引用时以上述绝对路径为准。

## 10. 未覆盖项与信息不足声明

1. **提交包未宣称可运行于真实平台沙箱**。比赛已于 2026-08-16 结束，无法获得真实沙箱验证。本地证据（§7）只覆盖：干净导入 `main.py` 成功（`import_main_ok=true`）、tar 结构完整（19 成员、无 `__pycache__`/`.git`）、`deck.csv` 60 行 11 种。`would_ko=True` 推理路径与平台 `sdk_cg` 预注入行为均属未覆盖项。
2. **初始模型缺失**。任务书"初始模型"在仓库中无对应 checkpoint（`ckpt_20480.pt` 不存在），故 §3 用同 `net_config` 随机初始化（`init_seed=20261019`）替代，口径为"同架构零能力下界"而非训练起点 0 步模型。
3. **`value_diag` 无 999424 记录**（§6.3），候选 checkpoint 自身的价值头状态未直接观测。
4. **动作合法性验证范围**：仅在 `rl` 侧 harness（`eval_h2h.py` + bench-shim 引擎）内验证（三组 100 局 `illegal.total=0`）。提交包 `main.py` 自身动作生成路径（含 `would_ko` 注入）的动作合法性未在本地验证。
5. **平台 `sdk_cg` 预注入不可判定**：本机 `find_spec(None)`、bench-shim 无 `sdk_cg` 模块名匹配，哨兵测试（§7.2）证明解包目录内的 `sdk_cg/` 遮蔽平台包。平台若预注入同名模块，导入路径将优先命中平台副本；若无，则提交包自带的 `sdk_cg/` 因缺 `cg.dll` 导入失败。两种情形的行为均未观测。
6. **§5.4 raw 模式失败面**（`GetBinaryObs` 缺失）是 bench-shim 引擎的固有限制，不代表候选模型缺陷；但同样意味着 `native_encode=True` 口径下的对局胜率在本环境不可得。
7. **对照 30 局样本量小**：CI 宽（如 §5.3 0.10 的 CI [0.0346, 0.2562]），仅作方向证据；正式结论以 §4 100 局为准。

---

所有正文数字来源：`eval_*.json`/`ctl_*.json`/`value_diag_buckets.json`/`h2h_ckpt/h2h_999424_vs_500736_r0of1.json`（仓库 `runs/evaluation/`）与 `clean_import_test.json`/`verify_submission.log`/`export.log`/`h2h_raw.log`/`h2h_patched.log`/`check_dll_exports.log`（`C:\Users\Swcmb\AppData\Local\Temp\opencode\`，§9 列明）。报告完稿。
