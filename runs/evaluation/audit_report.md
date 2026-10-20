# PPO 自博弈模型异常战绩与提交推理链路审计报告

审计日期：2026-10-19（本地时钟）。全部数字来自本机实测；未实测的推断单独标记为「未验证」。
本报告只使用现有 checkpoint、现有评估代码与本机官方引擎二进制，没有启动训练，没有覆盖
`runs/train-20m-e16-1m/ckpt_*.pt` 与 `submission_rl_transformer2.tar.gz`。

## 0. 结论摘要

1. **评估实现未发现能解释异常战绩的缺陷。** 座位映射、动作掩码、终局归属、随机策略与
   untrained 定义均按代码语义正确实现；仪器自检通过：候选自镜像 0.56（CI 含 0.5）、旧基线
   自镜像 0.500、非法动作与异常在全部正式对局中为 0。批量前向与提交包单样本前向在 241 个
   决策上 argmax 完全一致（0 分歧）。
2. **「候选 62% 胜旧基线」可信且可复现**：第二个种子复测 0.58，两组合并 120/200 = 0.600，
   Wilson 95% CI [0.531, 0.665]。这是真实进步，不是座位或牌组偏差。
3. **「候选 43% 负随机合法」可复现，但旧基线同样只有 42%**（第二个种子 0.45；合并
   88/200 = 0.440，CI [0.373, 0.509]）。两个独立训练的 checkpoint 在镜像同牌组协议下都略低于
   0.5，说明这是这批训练策略对该引擎/该协议的共性，而非候选单独损坏；具体机制尚未确证（见 §4.2）。
   另外，把评估池换成只含提交牌组 `deck.csv` 后，候选对随机只有 **0.20**（60 局，CI [0.118, 0.318]），
   说明提交牌组本身是独立风险项（§4.5）。
4. **「候选 22% 负同架构随机初始化」不可复现，不能作为能力下界**。`untrained` 对手是单次
   随机初始化抽样：同一候选在 init_seed 20261019/20261021/20261020 下分别得到 0.22 / 0.48 /
   0.70；旧基线在 20261019/20261021/20261022 下得到 0.10 / 0.433 / 0.433。此外同一 seed 重跑
   结果也不同（0.10 → 0.20），因为官方引擎内部 RNG 不可设置种子。22% 是单样本，不是稳定性质。
5. **提交包 `would_ko=True` 路径的真实依赖不是 `WouldKO`/`GetSerialObs`/`GetBinaryObs`**，而是
   官方 search API（`AgentStart`、`SearchBegin`、`SearchStep`、`SearchEnd`、`SearchRelease`）。
   用官方引擎二进制补齐后，提交包 `main.py` 的完整推理路径本地跑通：牌组请求返回 60 张、
   连续 86 局 0 异常、0 非法返回、`would_ko` 标注真实写入。**但现有 tar 不含任何引擎二进制**：
   按原样解包后 `from sdk_cg import api` 即报 `FileNotFoundError`，第一个需要 would_ko 标注的
   决策就会崩。这是当前提交包的首要阻塞，修复方式见 §5.6。

## 1. 审计对象与证据文件

| 项 | 值 |
|---|---|
| 候选 | `runs/train-20m-e16-1m/ckpt_999424.pt`，`global_step=999424` |
| 对照 | `runs/train-20m-e16-1m/ckpt_500736.pt`，`global_step=500736` |
| 网络配置（两者相同） | `transformer2, d_model=544, nhead=8, nlayers=5, ff=256, static=False, would_ko=True, split_heads=False, value_categorical=False` |
| 提交包 | `submission_rl_transformer2.tar.gz`（19 成员，75.2 MB，无 `.dll/.so/.dylib`） |
| 提交包内模型 | `model.pt`，`global_step=999424`，`net_config` 与候选一致 |
| 评估驱动 | `runs/evaluation/eval_h2h.py` |
| 环境 | `D:\ProgramFiles\deeplearning\python.exe`，torch 2.14.0+cu130，CUDA 可用 |
| 引擎 | 官方 `kaggle_environments` wheel 的 CABT 二进制（`cg.dll` 1,659,392 B；来源见 §5.2） |
| 证据目录 | `runs/evaluation/audit_artifacts/`（新增日志与汇总表）、`runs/evaluation/*.json` |
| 版本一致性 | `eval_h2h.py` 的 mtime（今日 12:21）早于全部 `eval_*_100.json`（12:25-12:39），即已有 100 局数据产生后评估代码未被改动；本轮新增 JSON 均由同一版本产生 |

## 2. 第一阶段：评估实现审计

### 2.1 每局座位与牌组的对应关系

- `eval_h2h.py` L141-L148：`cs = gi % 2` 严格均衡候选座位；`d0, d1 = (od, cd) if cs else (cd, od)`，
  即候选牌组始终落在 `cs` 座位。`_Counted._sample_decks`（L64-L66）固定返回 `(decks[0], decks[1])`，
  不再随机抽座位。
- 每步行动路由：L167-L176 按 `b["seat"] == b["cs"]` 把活动局面分到候选网络组或对手组；
  `b["seat"]` 来自环境 `reset/step` 返回的 `yourIndex`（`rl/env_selfplay.py` L192、L238），
  与被激活座位一致。
- 结论：候选/对手与 seat0/seat1 的对应正确；100 局中两个座位各 50 局（表中 `n0/n1` 均为 50）。

### 2.2 终局胜负归属与平局

- 终局取 `info["terminal"]`（`rl/env_selfplay.py` L216-L221，来自 `_terminal_rewards` L288-L301：
  胜者 +1、负者 -1、平局 0），`eval_h2h.py` L228-L244 用 `t[cs] > t[1-cs]` 判胜负。
- 截断（turn-cap / max_steps）单列，不计胜负（L229-L237）；非法提交造成的 forfeit 由
  `env._forfeit` 计数（L222-L227）。
- 全部正式 100 局与对照 30 局中 `draws=0`、`errs=0`、`illegal.total=0`；仅 seedB 的
  candidate-vs-untrained 出现 2 个 `max_steps` 截断平局（`draw_breakdown.max_steps=2`，见
  `ctl2_cand_vs_untrained_100_seedB.json`）。

### 2.3 `random_legal` 是否真从合法动作中随机

- `eval_h2h.py` L180-L185：`mask = obs["action_mask"] > 0.5`，`leg = flatnonzero(mask)`，
  `rng.choice(leg)`；无合法动作时取 0（但由 mask 构造保证不会发生）。
- 掩码一致性由构造保证：环境的合法掩码与编码器输出使用**同一个函数** `build_mask`
  （`rl/encoding.py` L125-L147；编码入口 L1414 把 `build_mask(sel, picked)` 写入 `action_mask`），
  环境 `step` 也用同一函数校验（`rl/env_selfplay.py` L202-L207）。因此随机策略与网络看到的
  动作索引、掩码、玩家状态完全同源。
- 政策网络侧另有内部掩码：`rl/policy.py` L635 `logits.masked_fill(action_mask < 0.5, -1e9)`，
  贪心 argmax 不可能选非法动作；实测 clamp/forfeit 全 0。

### 2.4 `untrained_same_config` 的定义

- `eval_h2h.py` L100-L107：先加载候选 ckpt 取 `net_config`，`torch.manual_seed(seed)` 后
  `build_token_net(ct, nc)`，得到同结构、同观测/动作空间、默认随机初始化的网络；推理同样走
  `logits_value(...).argmax(-1)`。定义本身正确。
- **但它是「一次随机初始化抽样」而不是固定对手**：init 只由 `seed` 决定。实测同一候选在不同
  init_seed 下对它的胜率为 0.22 / 0.30 / 0.48 / 0.70（§3.4）。因此该组不能当作模型的
  「对抗能力下界」，只能当作单次抽样。

### 2.5 镜像、牌组轮换、种子

- `mode=mirror`：两座位同一副牌（L144 `od = cd`），牌力逐局抵消，胜率反映纯策略差。
- 牌组轮换：`cand_seq[i] = pool[(i*7) % len(pool)]`（L128-L130），30 副池、stride 7 与 30 互质，
  100 局覆盖全部牌组各 3–4 次。
- 种子：`--seed` 决定牌序（此处固定覆盖）、随机策略动作、untrained 初始化；`env.reset(seed)` 只
  重置 Python 侧 RNG（`rl/env_selfplay.py` L162-L164）。
- **官方引擎内部 RNG 不可设置种子**：同一 seed、同一牌组、同一进程内重跑，逐局序列不同
  （`audit_artifacts/sm_rr.log` 与 `sm_agent.log` 的 `"reproducible": false`；评测框架层面
  同一 seed 30 局 base-vs-untrained 两次为 0.10 / 0.20）。这意味着「固定种子」只固定了协议与
  策略侧随机，不能保证逐局可复现；胜率结论必须以 CI 和多种子复测为准。

### 2.6 JSON 元数据与硬编码 label

- 真实 ckpt 路径与步数已核对：所有 `eval_*.json`/`ctl*.json` 顶层 `cand_ckpt/cand_step` 与
  meta 中的对手路径/步数均指向实际文件（`audit_artifacts/audit_meta.log` 逐行记录）。
- `eval_h2h.py` L110/L113 的 `label` 按 `kind` 硬编码，**不可信**：`ctl_base_selfmirror_30.json`
  的 label 写 `selfmirror_ckpt_999424`，而 `meta.ckpt`/`meta.step` 是 500736。本报告所有口径
  一律取 `meta` 与顶层 `cand_*`，不取 label。

### 2.7 仓库既有评估路径的缺陷（与本次结论的边界）

- `rl/gates.py` L235 调用 `_field_candidates(...)`，全仓库无定义（grep 仅命中该调用与任务书）。
  因此 `scripts/h2h_ckpt.py` 走 `field_winrate_fast` 必然 `NameError`，除非 monkeypatch。
- `scripts/h2h_ckpt.py` L77-L79 通过 `field_winrate_fast` 使用 `_gate_env`（`rl/gates.py` L104），
  该环境硬编码 `native_encode=True`，本机官方引擎缺 `GetBinaryObs`，无法直接运行。
  这两点解释了为什么本次审计改用 `eval_h2h.py`（JSON 路径）而不是仓库自带 runner。
- 结论：**「仓库自带 runner 不可用」是已确认缺陷，但它不是 62%/43%/22% 的原因**——本次所有
  数字来自独立驱动 `eval_h2h.py`，且该驱动的统计逻辑已自检（§2.1-L2.4、§3.5）。

### 2.8 仪器一致性实测

- **前向路径对拍**：在真实对局中对同一批 obs 同时计算「提交包单样本 `_select`」与
  「harness 批量 + `bucketed_opt_len_np`」两个 argmax，241 个决策 **0 分歧**
  （`audit_artifacts/parity.json`，`mismatch_rate=0.0`，opt_len 桶全为 32）。
- **自镜像对称性**：旧基线自镜像 100 局总胜率 0.500（seat0 0.84 / seat1 0.16，即该模型的
  先手优势极强，但总量精确回到 0.5）；候选自镜像 0.56（CI [0.462, 0.653]）。候选并不存在
  「作为候选一侧被系统性压低」的迹象。
- 综合：没有发现候选/对手之间的环境或数据路径不对称。

## 3. 第二阶段：最小化交叉对局实测

### 3.1 协议

- 驱动：`runs/evaluation/eval_h2h.py`（复用 `TwoSidedSelfPlayEnv` + `rl/policy.py` + `rl/encoding.py`）。
- 固定 `PKMN_TURN_CAP=200`；`mode=mirror`（两座位同牌组）；`live=1`（官方 JSON 引擎是进程级单例，
  必须串行）；每局 `seed+gi`；座位 `gi % 2` 严格 50/50。
- 小样本先行：三组新对局各 10 局冒烟（基线 vs 随机 0.6、基线 vs 未训练 0.0、候选自镜像 0.6；
  均 0 异常 / 0 非法），随后全部跑 100 局。
- 覆盖任务要求的 6 组：候选×3（基线/随机/未训练）、旧基线×2（随机/未训练）、候选自镜像；
  另加旧基线自镜像作为仪器对称性对照。第二种子（20261020）复测候选×3。

### 3.2 100 局矩阵（所有数字来自对应 JSON 的 `results[0]`）

| 视角 ckpt | 对手 | 局 | seed | 胜/负/平(截断) | 异常 | 非法(候选) | 胜率 | Wilson 95% CI | seat0 胜率 (n) | seat1 胜率 (n) | 平均 apply / turn | 耗时 (g/s) | 文件 |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| 999424 | 基线 500736 | 100 | 19 | 62/38/0 | 0 | 0 | 0.6200 | [0.5221, 0.7090] | 0.84 (50) | 0.40 (50) | 106.9 / 79.1 | 444.5 s (0.22) | eval_baseline_100.json |
| 999424 | random_legal | 100 | 19 | 43/57/0 | 0 | 0 | 0.4300 | [0.3373, 0.5278] | 0.42 (50) | 0.44 (50) | 108.5 / 36.8 | 143.3 s (0.70) | eval_random_100.json |
| 999424 | untrained (init 19) | 100 | 19 | 22/78/0 | 0 | 0 | 0.2200 | [0.1500, 0.3107] | 0.20 (50) | 0.24 (50) | 116.7 / 77.2 | 438.8 s (0.23) | eval_untrained_100.json |
| 500736 | random_legal | 100 | 19 | 42/58/0 | 0 | 0 | 0.4200 | [0.3280, 0.5179] | 0.42 (50) | 0.42 (50) | 98.9 / 37.7 | 28.9 s (3.46) | ctl2_base_vs_random_100.json |
| 500736 | untrained (init 19) | 100 | 19 | 10/90/0 | 0 | 0 | 0.1000 | [0.0552, 0.1744] | 0.18 (50) | 0.02 (50) | 120.3 / 91.7 | 98.3 s (1.02) | ctl2_base_vs_untrained_100.json |
| 999424 | 自身镜像 | 100 | 19 | 56/44/0 | 0 | 0 | 0.5600 | [0.4623, 0.6533] | 0.66 (50) | 0.46 (50) | 93.2 / 64.1 | 73.1 s (1.37) | ctl2_cand_selfmirror_100.json |
| 500736 | 自身镜像 | 100 | 19 | 50/50/0 | 0 | 0 | 0.5000 | [0.4038, 0.5962] | 0.84 (50) | 0.16 (50) | 98.0 / 94.7 | 74.5 s (1.34) | ctl2_base_selfmirror_100.json |
| 999424 | 基线 500736 | 100 | 20 | 58/42/0 | 0 | 0 | 0.5800 | [0.4821, 0.6720] | 0.88 (50) | 0.28 (50) | 99.0 / 79.5 | 79.7 s (1.26) | ctl2_cand_vs_baseline_100_seedB.json |
| 999424 | random_legal | 100 | 20 | 45/55/0 | 0 | 0 | 0.4500 | [0.3561, 0.5476] | 0.46 (50) | 0.44 (50) | 94.8 / 33.6 | 33.0 s (3.03) | ctl2_cand_vs_random_100_seedB.json |
| 999424 | untrained (init 20) | 100 | 20 | 69/29/2 | 0 | 0 | 0.7041 | [0.6074, 0.7854] | 0.82 (50) | 0.56 (50) | 220.8 / 72.2 | 202.2 s (0.49) | ctl2_cand_vs_untrained_100_seedB.json |

合并（同一对局的两个种子，仅作为点估计汇总）：候选 vs 基线 **120/200 = 0.600**，CI [0.531, 0.665]；
候选 vs 随机 **88/200 = 0.440**，CI [0.373, 0.509]。

### 3.3 分座位与座位效应

- 候选 vs 基线：seat0 0.84/0.88，seat1 0.40/0.28 —— 两次一致，候选在 seat1 明显吃亏。
- 候选 vs 随机：0.42/0.44 与 0.46/0.44 —— 无座位差。
- 自镜像：候选 seat0 0.66 / seat1 0.46；旧基线 seat0 0.84 / seat1 0.16。
  **结论：该引擎存在显著的先手（seat0）优势，且旧基线的策略更依赖先手；候选的相对进步部分来自
  它把 seat1 胜率从 0.16 提高到 0.46。** 总胜率因 50/50 座位混合而不受此偏差影响。

### 3.4 方差、复现性与多初始化扫描

`untrained` 是「按 init seed 抽一个随机网络」，同一候选/基线的结果随抽样大幅波动：

| 视角 | init_seed | 局数 | 胜率 | 文件 |
|---|---:|---:|---:|---|
| 候选 999424 | 20261019 | 100 | 0.2200 | eval_untrained_100.json |
| 候选 999424 | 20261019 | 30 | 0.3000 | wk_untrained_cand_s20261019.json |
| 候选 999424 | 20261021 | 30 | 0.4828（1 max_steps 平局） | wk_untrained_cand_s20261021.json |
| 候选 999424 | 20261020 | 100 | 0.7041（2 max_steps 平局） | ctl2_cand_vs_untrained_100_seedB.json |
| 基线 500736 | 20261019 | 30 / 100 / 30（重跑） | 0.1000 / 0.1000 / 0.2000 | ctl_base_untrained_30 / ctl2_base_vs_untrained_100 / wk_untrained_base_s20261019 |
| 基线 500736 | 20261021 | 30 | 0.4333 | wk_untrained_base_s20261021.json |
| 基线 500736 | 20261022 | 30 | 0.4333 | wk_untrained_base_s20261022.json |

两件事同时成立：
1. **init seed 效应**：同一候选在 0.22–0.70 之间波动；同一基线在 0.10–0.433 之间波动。
2. **运行间不可复现**：基线 vs init19 的 30 局两次分别 0.10 / 0.20；random-vs-random 对照两次
   11/9 与 6/14（`audit_artifacts/sm_rr.log`，`reproducible=false`）。官方引擎内部洗牌/掷币 RNG
   无法设置种子，同 seed 只固定了协议与策略随机。

因此单次 0.22 不能作为「候选被随机初始化模型碾压」的证据。

### 3.5 非法动作、异常与平局

- 100 局矩阵共 1000 局：`illegal.total = 0`（clamp/forfeit 四路计数全 0）、`errs = 0`；
  平局仅 2 个（seedB untrained，`max_steps` 截断，非 turn-cap）。
- 非法动作率按决策计为 0.0（`illegal.rate_per_decision`）；随机策略与两个网络的动作都未触发
  环境 clamp 或提交异常。

## 4. 异常战绩的可信解释

### 4.1 候选 vs 旧基线 62%——真实、可复现

两个种子 0.62 / 0.58，合并 0.600 [0.531, 0.665]，CI 下界显著高于 0.5；另有一条独立证据：
仓库 runner 的 30 局直对（patched 口径）为 20/30 = 0.667（`h2h_ckpt/h2h_999424_vs_500736_r0of1.json`）。
分座位看，候选把 seat1 胜率从基线的 0.16 提升到 0.46，说明进步不是靠先手优势放大。

### 4.2 候选 vs 随机合法 43%——可复现，但属于两个 checkpoint 的共性

- 候选 0.43 / 0.45；**旧基线 0.42**（100 局，另一进程、另一视角）。两个独立训练点的结果几乎相同，
  且候选与基线的合并点估计都在 0.44 左右。
- 牌组因素已单独测量：把评估池换成只含 `deck.csv` 的 60 局，候选对随机为 **0.20**（12/60，
  Wilson CI [0.118, 0.318]，seat0 0.267 / seat1 0.133，0 异常 / 0 非法）。独立驱动（提交包
  `main.py` + 官方引擎，镜像 deck.csv，40×2 局）为 8/32 与 6/34，与该 0.20 一致（合计 16/86 ≈ 0.186）。
  也就是说，43% 是「池牌组 + 镜像」下的数字，换成提交牌组会进一步跌到 0.20。
- 已排除的解释：非法动作（0）、掩码错位（同一 `build_mask`）、奖励/终局归属（自镜像对称）、
  前向路径差异（0/241 分歧）、座位偏差（座位均衡且随机组无座位差）。
- **仍未确证的解释**（按可能性排序，均未单独做消融）：
  1. 自博弈专化：策略在「结构化、可预测」的对手分布上被优化，均匀随机产生的状态分布是 OOD
     （随机对手平均对局 33–38 turn，显著短于对基线的 79 turn，见 §3.2 的 turn 列）。
  2. 镜像协议放大了这一点：两座位同牌组时牌力优势被抵消，随机策略只需「不犯错地交换资源」即可
     拿到接近 50% 的期望；已有证据是随机组座位胜率完全平坦（0.42/0.44、0.46/0.44）。
  3. 训练数据（top50 池 74 副，`matchup_r0.jsonl` header `pool_size=74`）中没有任何随机策略样本，
     价值/策略头对「无结构对手」没有校准信号。
  4. 牌组结构：提交牌组只有 10 张宝可梦，换成该牌组后对随机胜率从 0.43–0.45 跌到 0.20（§4.5）。
- 结论：**这不是「评估代码把候选算错了」**，而是这批 checkpoint 的稳定行为特征；但它确实说明
  当前的胜率不能代表对任意弱策略的碾压能力。

### 4.3 候选 vs 同架构随机初始化 22%——不可复现，不能作为能力下界

如 §3.4：0.22 / 0.30 / 0.48 / 0.70 四个点来自同一候选与四个不同的随机网络；旧基线 0.10 / 0.20 /
0.433 / 0.433。单次 22% 的 95% CI [0.150, 0.311] 只描述「对这一个随机网络」的不确定性，不描述
「对随机初始化模型总体」的不确定性。任务书要求警惕的「非传递性」在这组数据里并不能成立为结论：
它建立在一次抽样之上，换一个 init seed 就消失。

### 4.4 非传递性判断

现有最强的一组非传递证据是「同一 init19 随机网络胜基线 90%、胜候选 78%」。由于该随机网络只被
抽样一次、且引擎 RNG 不可复现，这一现象应记为「一次抽样下的风格克制」，**不足以支持代码有错**；
也不足以支持「候选比基线更差」——候选对基线的直接对局是 0.60 [0.531, 0.665]，方向相反。

### 4.5 提交牌组 `deck.csv` 的独立风险（新增发现）

- `agent/deck.csv`（与 tar 内一致，sha256 前缀 `1156379af39e`）共 60 张、11 种：**33 张基础水能量、
  10 张宝可梦**（2 Kyogre、4 Snover、4 Mega Abomasnow ex）、17 张训练家/工具/竞技场。
- 该 60 张牌**不在**训练池 `TRAIN_TOP15`(30) / `TRAIN_TOP50`(74) / `pool_top15` / `pool_top50` 中
  （精确多重集与顺序均不匹配）；与池中最接近的牌组重叠 49/60 张。
- 独立驱动中用 deck.csv 镜像对随机策略：6 局 2 胜 + 40×2 局 8 胜/6 胜（合计 16/86 ≈ 0.186），
  全部终局原因中 `no_active` 占绝对多数（38/40、40/40）——牌组只有 10 张宝可梦，铺场被清后
  很容易因「无出战宝可梦」直接判负。
- **牌组消融（已测）**：`runs/evaluation/ctl2_cand_vs_random_deckcsv_60.json`（60 局，仅含
  `deck.csv` 的镜像池，随机对手）：胜率 **0.20**，Wilson CI [0.118, 0.318]，seat0 0.267 / seat1 0.133，
  0 平局 / 0 异常 / 0 非法；平均 turn **16.0**（池牌组同对局为 36.8）。对局明显更短，与「10 张宝可梦
  被清场后因 `no_active` 判负」的结构性解释一致。
- 结论：提交牌组是「候选对随机偏弱」的**主要放大器**：池牌组 0.43–0.45 → 提交牌组 0.20。
  该差异不能归因于座位或牌力偏差（两座位同牌组、牌力抵消）；它混合了「牌组结构」与「策略对
  该牌组的熟悉度」两个因素，本次未分离。
- **未测**：同一消融对基线/未训练对手的效果（原批次中这两组未完成）。

## 5. 第三阶段：`would_ko=True` 推理链路与提交包

### 5.1 `would_ko=True` 的真实执行路径（逐调用）

1. `submission_rl_transformer2/main.py` L28：`_WK_ON = bool(_CK["net_config"].get("would_ko"))`；
   L29-L30：`if _WK_ON: import search_agent as _SA2`。
2. `main.py` L56-L57：每个**决策 obs**调用一次 `_SA2.annotate_would_ko(obs, DECK, _ENC)`。
3. `search_agent.py` L394-L403：`annotate_would_ko` → `would_ko_flags` → `write_would_ko`，
   把结果写入 `obs["select"]["option"][i]` 的 `would_ko / would_ko_prizes / would_ko_win`。
4. `would_ko_flags`（L318-L390）在 **L335** 执行 `from sdk_cg import api`，随后 L348
   `api.search_begin(api.to_observation_class(obs), **_determinize(...))`、L351 `api.search_step(...)`、
   L367 `api.search_release(...)`、L371 `api.search_end()`（`annotate_would_ko` 在 L394，经
   `would_ko_flags` 调用同一路径）。
5. `sdk_cg/api.py` L517-L639：这些函数直接绑定 `lib.AgentStart / lib.SearchBegin / lib.SearchStep /
   lib.SearchEnd / lib.SearchRelease`（`sdk_cg/sim.py` L36-L71 声明 argtypes）。
6. 编码器 `rl/encoding.py` L325-L327（提交包同构）读取这三个字段填入 `opt_attr[:, 11:14]`
   （`rl/enc_constants.py` L43-L44：`OPT_WK=11`、`OPT_PICKED=14`）。

**依赖结论（与旧报告 §7.3/§8.1 相反，已用代码与符号表核实）：**

| 符号 | main.py 的 would_ko 路径是否需要 | 实际使用者 |
|---|---|---|
| `AgentStart` / `SearchBegin` / `SearchStep` / `SearchEnd` / `SearchRelease` | **需要**（官方引擎导出，本机实测 True） | `sdk_cg/api.py` → `search_agent` |
| `AllCard` / `AllAttack` | 否（仅 `api.all_card_data/all_attack`） | 卡表工具 |
| `WouldKO` | **不需要** | 仅 `rl/wk_native.py`（`PKMN_WK_NATIVE=1` 的训练加速路径） |
| `GetSerialObs` | **不需要** | 仅 `rl/wk_native.py` |
| `GetBinaryObs` / `AdvanceLog` | **不需要** | 仅 `rl/env.py`、`rl/native_enc.py`、`native_encode/`（`native_encode=True` 训练路径） |

即：`WouldKO/GetSerialObs/GetBinaryObs` 来自自研 `libcg_custom.so` / `libcg_wk.so`（`native_encode/README.md`
L83-L85、`rl/wk_native.py` L4-L14），是**训练侧**的私有导出；提交包推理路径完全不依赖它们。

### 5.2 引擎/SDK 资源盘点

- 仓库 `sdk_cg/` 只有 5 个 `.py`，**没有任何二进制**；`README.md` L39-L40 明确写
  “the engine binaries themselves are proprietary and not included -- get them from the competition page”。
- 本机存在官方 CABT 引擎二进制：`C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim\kaggle_environments\envs\cabt\cg\`
  下的 `cg.dll`（1,659,392 B）、`libcg.so`（1,458,504 B）、`libcg-arm64.so`、`libcg.dylib`。
  来源（Prompt/实际 steps-s 基准测试报告 L39）：从 PyPI `kaggle_environments` wheel 抽取
  `envs/cabt/cg/*` 原字节，外层 `__init__.py` 为本地 0 字节骨架。**这是官方引擎二进制，不是伪造导出。**
- 实测导出：`GameInitialize, BattleStart, BattleFinish, GetBattleData, Select, VisualizeData,
  AgentStart, SearchBegin, SearchStep, SearchEnd, SearchRelease, AllCard, AllAttack` 均存在；
  `WouldKO / GetSerialObs / GetBinaryObs / AdvanceLog` 不存在（与 §5.1 一致，且不影响 main.py）。

### 5.3 提交包「按原样」验证（不添加任何文件）

用 `tarfile` 解包 `submission_rl_transformer2.tar.gz` 到干净临时目录：

- 19 个成员：`EN_Card_Data.csv, attack_data.py, buff_data.py, card_features.py, deck.csv, decks.py,
  effect_data.py, enc_constants.py, encoding.py, main.py, model.pt, policy.py, sdk_cg/（5 个 py）, search_agent.py`；
  **`.dll/.so/.dylib` 列表为空**。
- `import main`：**成功**；`_WK_ON = True`。
- 牌组请求分支（`agent({"select": None})`）：**成功**，返回 60 个 int（与 `deck.csv` 一致）。
- `from sdk_cg import api`：**失败** — `FileNotFoundError: Could not find module
  '...\subclean\sdk_cg\cg.dll'`。
- 由于 `main.py` 的决策路径在第一个决策就调用 `annotate_would_ko`，**按原样提交时，平台若不在
  agent 目录内提供引擎二进制，第一个决策即抛异常**（`audit_artifacts/tarcheck.log`）。

### 5.4 补齐官方二进制后的端到端验证（临时副本，不修改原包）

方法：把 tar 内容解到临时目录，仅把官方 `cg.dll` 放入 `sdk_cg/`，然后用 `sdk_cg.game` 直接驱动
真实对局；提交包 `main.py` 作为 seat0/seat1（每局交替）的 agent，另一座位用**同一 `build_mask`**
的随机合法策略。开局显式调用一次 `agent({"select": None})` 模拟平台的牌组请求并重置 tracker。

| 指标 | 6 局冒烟 | 40 局（run1） | 40 局（run2，同 seed） |
|---|---:|---:|---:|
| 异常 | 0 | 0 | 0 |
| 非法返回（索引越界/非 int） | 0 | 0 | 0 |
| 胜/负/平 | 2/4/0 | 8/32/0 | 6/34/0 |
| 含 would_ko 标注的决策 / 写入单元格 | 116 / 54 | 635 / 229 | 519 / 181 |
| 平均对局长度 | 14–186 apply | 14–136 apply | 15–136 apply |
| 终局原因 | — | `no_active` 38/40、`0_prize` 2 | `no_active` 40/40 |

- 结论：**提交包入口、`agent(obs)` 返回格式（`list[int]` 选项索引）、60 张牌组文件、
  `would_ko=True` 的完整推理路径（含 search API 调用与 3 列特征写入）均在本地真实引擎上跑通。**
- 40 局两跑使用同一 seed 但结果不同（8/32 vs 6/34），再次印证 §2.5 的引擎 RNG 不可复现。
- 前向一致性另见 §2.8：提交包单样本与 harness 批量 argmax 241 决策 0 分歧。

### 5.5 关闭 `would_ko` 的方案评估（不推荐直接关闭）

- 维度：`opt_attr` 宽度不变（`OPT_STRUCT=15`），关闭只把第 11–13 列恒置 0；观测维度与动作空间
  与 `would_ko=False` 的模型完全兼容。
- 输入分布：在被测样本中，`opt_attr[:, 11:14]` 有非零值的决策为 **4/241（1.66%）**；带攻击选项
  （annotate 会写入键）的决策为 229/635（36%）。也就是说，关闭后受影响的是“含攻击选项的决策”
  中的一小部分，但确实是训练时存在、推理时缺失的特征。
- 结论：关闭 `would_ko` 在形状上安全，在分布上**不等价**；在没有 `would_ko=False` 训练出的
  checkpoint 做对照之前，不能宣称“模型兼容”。本次不建议修改导出，维持 `would_ko=True`。

### 5.6 剩余阻塞（提交包层面）

1. **引擎二进制缺失（阻塞级，已实证）**：tar 内无 `cg.dll/libcg.so`；干净环境 `from sdk_cg import api`
   直接失败，首个决策崩。修复：在打包前把官方 Linux 二进制复制进 `sdk_cg/`（本机已有
   `libcg.so` 1,458,504 B），重新导出 tar；导出脚本 `scripts/export_rl_submission.py` L202-L203 的
   `copytree` 会原样带上。
2. **平台注入行为不可判定（未验证）**：若 Kaggle 运行时把引擎二进制放进 agent 目录，则当前 tar 可跑；
   但比赛已于 2026-08-16 结束，无法取得真实沙箱验证。`main.py` 的 `sys.path.insert(0, _HERE)` 会
   让包内 `sdk_cg` 优先于任何平台副本，因此“平台另外提供 sdk_cg”并不能绕过缺二进制的问题，
   除非平台直接写入 agent 目录。
3. **Linux 二进制未实跑（未验证）**：本机端到端用的是 Windows `cg.dll`；`libcg.so` 仅做了存在性
   与来源核对，未在 Linux 下运行。

### 5.7 已确认问题清单（证据与代码位置）

| # | 问题 | 严重度 | 证据 | 代码位置 |
|---|---|---|---|---|
| 1 | 提交 tar 不含引擎二进制，干净环境首个 would_ko 决策即 `FileNotFoundError` | 阻塞 | `audit_artifacts/tarcheck.log`（19 成员，二进制列表为空） | `main.py` L29-L30；`search_agent.py` L335；`sdk_cg/sim.py` L29 |
| 2 | `untrained` 是单次随机初始化抽样，22% 不可复现（0.22–0.70） | 评估方法 | §3.4；`wk_untrained_*.json` | `eval_h2h.py` L100-L107 |
| 3 | 官方引擎内部 RNG 不可设置种子，同 seed 重跑结果不同 | 复现性 | `sm_rr.log`（`reproducible=false`）；0.10 vs 0.20 | `rl/env_selfplay.py` L162-L164（只 seed Python RNG） |
| 4 | 提交牌组只有 10 张宝可梦/33 张能量，不在训练池；对随机仅 0.20 | 高 | `audit_artifacts/deck_probe_out.txt`；`ctl2_cand_vs_random_deckcsv_60.json` | `agent/deck.csv` |
| 5 | `label` 硬编码，与真实 ckpt 不符 | 数据质量 | `ctl_base_selfmirror_30.json`（label 999424 / meta 500736） | `eval_h2h.py` L110/L113 |
| 6 | `_field_candidates` 未定义，仓库 runner 不可用 | 中 | grep 仅 L235 调用；`scripts/h2h_ckpt.py` 依赖它 | `rl/gates.py` L235 |
| 7 | 旧报告把 `would_ko` 依赖说成 `WouldKO` 导出 | 文档错误 | §5.1 符号表 + 实测导出清单 | `runs/evaluation/report.md` §7.3/§8.1 |

## 6. 决策：是否值得继续训练、优先修复什么

**值得继续训练，但先修提交包与评估协议；当前 checkpoint 可以进入下一阶段验证。**

支持证据：
- 候选对旧基线 0.62 / 0.58（合并 0.600 [0.531, 0.665]），且各自自镜像中的 seat1 胜率由基线的
  0.16 提升到候选的 0.46；这是稳定、可复现的进步。
- 候选自镜像 0.56（CI 含 0.5），旧基线 0.50；仪器无偏向。
- 全部正式对局 0 非法动作 / 0 异常；仅 2 个 `max_steps` 截断平局（seedB untrained 组）。

优先修复顺序：
1. **提交包引擎二进制**（阻塞级）：把官方 `libcg.so` 放进 `sdk_cg/` 后重打包；验收标准 =
   干净目录解包 → `import main` → 首个决策 `annotate_would_ko` 不抛异常。
2. **评估协议**：`untrained` 改为至少 5 个 init seed 的分布报告（中位数 + 区间），禁止用单点
   0.22 作为“能力下界”；所有对局报告注明引擎 RNG 不可复现，并以多种子 + CI 呈现。
3. **牌组**：`deck.csv` 已测得对随机仅 0.20（池牌组 0.43–0.45）；应把它加入训练/评估分布，
   并补测它对基线/未训练对手的胜率。当前牌组只有 10 张宝可梦、33 张能量且不在训练池中
   （最接近 49/60），是独立于策略强度的风险项。
4. **仓库 runner**：修复 `rl/gates.py` L235 的 `_field_candidates`，否则 `scripts/h2h_ckpt.py`
   与 `field_winrate_fast` 仍不可直接使用。
5. **不建议**直接关闭 `would_ko` 或以 `native_encode=False` 的结果冒充训练口径；两者都已给出
   明确的分布差异实测（§5.5）。

## 7. 未验证项声明

1. 真实 Kaggle 平台是否在 agent 目录注入引擎二进制：**未验证**（比赛已结束，无法取得沙箱）。
2. 候选 43% vs 随机合法策略的机制（自博弈专化 / OOD / 镜像协议）：**未确证**，只排除了实现层面
   的 bug 与座位偏差；未做对手分布消融。
3. 牌组消融只完成了「对随机对手」一组（0.20 vs 池牌组 0.43–0.45）；**对基线/未训练对手的牌组
   消融未测**。
4. `untrained` 总体的胜率分布：**样本不足**（每个 init seed 仅 30–100 局；init seed 本身只抽了
   4–7 个）。
5. Linux `libcg.so` 在 Kaggle 运行时的可用性/glibc 兼容性：**未验证**（仅 Windows `cg.dll` 实跑）。
6. 旧报告 `report.md` 关于 `would_ko` 依赖 `WouldKO` 的表述：**已证伪**（§5.1），以本报告为准。

## 8. 复现命令

环境（Windows，`cmd`）：

```bat
cd /d D:\pkmn-kaggle
set PYTHONPATH=.;C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim
set PKMN_TURN_CAP=200
D:\ProgramFiles\deeplearning\python.exe runs\evaluation\eval_h2h.py ^
  --ckpt runs\train-20m-e16-1m\ckpt_999424.pt --opponent baseline --games 100 --seed 20261020 ^
  --out runs\evaluation\ctl2_cand_vs_baseline_100_seedB.json
```

- 提交牌组消融（已完成 60 局，仅随机对手）：加
  `--pool-json runs\evaluation\pool_deckcsv_only.json --games 60 --opponent random ^
  --out runs\evaluation\ctl2_cand_vs_random_deckcsv_60.json`。
- 其余 100 局组：把 `--ckpt/--opponent/--seed/--out` 换成 §3.2 表中的组合
  （`--opponent` ∈ `baseline|random|untrained|selfmirror`；基线视角用
  `--ckpt runs\train-20m-e16-1m\ckpt_500736.pt`）。
- 10 局冒烟：`--games 10 --out runs\evaluation\tmp_smoke_*.json`（已产出于本轮）。
- 多 init seed 扫描：`--opponent untrained --games 30 --seed <init_seed>`。
- 提交包干净验证：`python C:\...\opencode\tarcheck.py`（解包原 tar，验证 import/牌组/缺二进制失败点）。
- 提交包端到端验证：`python C:\...\opencode\subsmoke2b.py agent 40 20261019 replay`（临时副本 + 官方
  `cg.dll`；脚本位于本机临时目录，仓库内副本见 `runs/evaluation/audit_artifacts/` 的日志与 JSON）。
- 前向对拍：`python C:\...\opencode\parity2.py 12`。
- 汇总表：`python C:\...\opencode\table.py`（输出 `runs/evaluation/audit_artifacts/table.md`）。

证据文件索引：`runs/evaluation/audit_artifacts/` 下 `table.md`（全部 JSON 汇总）、`ctl2.log`
（7 组 100 局原始输出）、`untrained_sweep.log`（多 init seed）、`parity.json`（前向对拍）、
`subsmoke2_agent.json` / `subsmoke2_rr.json`（独立驱动与随机对照）、`tarcheck.log`（原样提交包）、
`deck_probe_out.txt`（牌组构成）、`audit_subs.log`（源码差异与符号表）。
