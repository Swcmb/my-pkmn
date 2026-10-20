# 牌组验证与交叉评估报告（新卡牌库）

日期：2026-10-20　目录：`runs/evaluation/card_database_migration/`
执行脚本：`runs/card_database_migration/20261020/validate_decks.py`、`deck_cross_eval.py`
原始证据：`runs/card_database_migration/20261020/deck_engine_checks.json`、`deck_cross_eval.json`

## 1. 口径一致性（只变牌组）

| 项 | 值 |
|:---|:---|
| checkpoint | `runs/train-20m-e16-1m/ckpt_999424.pt`　SHA `b8a3be1d247eac45af0bbc38a207336d6f702c82fd0b19e5f787cc7addf08a1e`　step 999424 |
| net_config | `transformer2, d_model=544, nhead=8, nlayers=5, ff=256, static=false, would_ko=true, value_atoms=51` |
| 卡牌库 | `EN_Card_Data.csv`（迁移后）　SHA `05aa569e3003a838b8503a9d928795e5aea7407ab19dd9f3d9c4b1293b6bb4b4`　vocab 1268 / 1267 张卡 / feat_dim 59 |
| 引擎 | 官方 `kaggle_environments==1.33.0` CABT（`D:\pkmn-kaggle\.engine-shim`，Windows `cg.dll`），卡牌 1431 / 招式 1755 |
| 编码路径 | Python `TokenEncoder` + 引擎 would_ko 标注（`native_encode=False`） |
| 环境参数 | `PKMN_TURN_CAP=200`、`live=1`（JSON 引擎 battle_ptr 为进程级单例）、argmax 贪心 |
| 版本 | Python 3.12.14 / PyTorch 2.14.0+cu130 / device cuda |
| 协议来源 | 复用 `runs/evaluation/eval_h2h.py::run_one`，未重写对战逻辑 |
| 种子与座位 | 2 个独立种子（20261020 / 20261021），每 cell 40 局，座位 `cs = gi % 2` 逐局严格交替，双向（谁当候选方各跑一次） |

历史评估结果一律未被复用为本库结论；`agent/deck.csv`、`runs/evaluation/deck_search/*`、池 JSON 与权重文件均保留为历史资料。

## 2. 候选牌组与来源

| 候选 | 来源 | SHA-256 |
|:---|:---|:---|
| `dragapult_newlib` | `runs/evaluation/deck_search/dragapult_deck.csv` | `30c8c7365c75f38fd6e7e1d8543c42ce7055ed6fd1c6e9eb244e44484b78e724` |
| `current_agent_deck` | `agent/deck.csv`（当前提交牌组） | `1156379af39e71bc83eecb50d1e04c5cc480501d293621fcc89c2d355d99be78` |
| `fixture_iono` | `rl/decks.py:_DECKS_SPEC['iono']`（补充合法候选） | 代码内置 |

旧 Dragapult 牌组在新库下的逐 ID 核对结果：**60 张全部命中**，无 ID 缺失、无进化链告警。
新库 ID↔卡名 1:1 保持不变（1267/1267），因此旧牌组的每个 ID 仍指向同一张卡，
不需要按名称重新映射；同时已按新库重新核对 HP/能量/招式/ACE SPEC 标记。

## 3. 合法性验证

### 3.1 Python 层（新库 + 引擎元数据）

5 副候选（含 `mega_lucario`、`mega_abomasnow` 两个额外 fixture）全部通过：
恰好 60 张、ID 均存在于新库与引擎、同名 ≤4（基本能量除外）、ACE SPEC ≤1、
至少 1 张基础宝可梦（Dragapult 候选含 9 张基础宝可梦）、进化前置名称均在牌组内。
`python_rejects = []`。

### 3.2 真实 CABT 引擎层

对 5 副候选做镜像 + 全部交叉配对开局，共 **15 种配对，全部被引擎接受**
（`accepted: true, errorPlayer: -1, errorType: 0`），`engine_rejects = []`。
引擎开局校验不依赖 Python 侧检查结果。

## 4. 双向交叉评估结果（同 checkpoint，仅牌组变量）

每 cell 40 局；`dr/er/il` = 平局 / 异常 / 非法动作；`len` = 平均决策提交数。

| cell | seed | 胜率 | Wilson 95% CI | 分座位胜率 s0/s1 | len | dr/er/il |
|:---|:---|:---|:---|:---|:---|:---|
| dragapult vs current | 20261020 | 0.450 | [0.307, 0.602] | 0.65 / 0.25 | 84.8 | 0/0/0 |
| dragapult vs iono | 20261020 | 0.500 | [0.352, 0.648] | 0.55 / 0.45 | 96.2 | 0/0/0 |
| current vs dragapult | 20261020 | 0.500 | [0.352, 0.648] | 0.70 / 0.30 | 86.4 | 0/0/0 |
| current vs iono | 20261020 | 0.600 | [0.446, 0.737] | 0.70 / 0.50 | 103.0 | 0/0/0 |
| iono vs dragapult | 20261020 | 0.500 | [0.352, 0.648] | 0.60 / 0.40 | 97.8 | 0/0/0 |
| iono vs current | 20261020 | 0.350 | [0.221, 0.505] | 0.45 / 0.25 | 94.2 | 0/0/0 |
| **dragapult vs random** | 20261020 | 0.475 | [0.329, 0.625] | 0.30 / 0.65 | 104.8 | 0/0/0 |
| **current vs random** | 20261020 | 0.175 | [0.087, 0.319] | 0.15 / 0.20 | 60.9 | 0/0/0 |
| **iono vs random** | 20261020 | 0.275 | [0.161, 0.428] | 0.20 / 0.35 | 124.0 | 0/0/0 |
| dragapult vs current | 20261021 | 0.500 | [0.352, 0.648] | 0.60 / 0.40 | 84.3 | 0/0/0 |
| dragapult vs iono | 20261021 | 0.525 | [0.375, 0.671] | 0.60 / 0.45 | 92.0 | 0/0/0 |
| current vs dragapult | 20261021 | 0.475 | [0.329, 0.625] | 0.60 / 0.35 | 84.7 | 0/0/0 |
| current vs iono | 20261021 | 0.475 | [0.329, 0.625] | 0.50 / 0.45 | 104.8 | 0/0/0 |
| iono vs dragapult | 20261021 | 0.550 | [0.398, 0.693] | 0.50 / 0.60 | 96.8 | 0/0/0 |
| iono vs current | 20261021 | 0.575 | [0.422, 0.715] | 0.55 / 0.60 | 94.8 | 0/0/0 |
| **dragapult vs random** | 20261021 | 0.525 | [0.375, 0.671] | 0.60 / 0.45 | 101.0 | 0/0/0 |
| **current vs random** | 20261021 | 0.150 | [0.071, 0.291] | 0.00 / 0.30 | 53.5 | 0/0/0 |
| **iono vs random** | 20261021 | 0.225 | [0.123, 0.375] | 0.15 / 0.30 | 118.6 | 0/0/0 |

合并两种子（wins/decided）：

| 配对 | 合并胜率 |
|:---|:---|
| dragapult vs current | 38/80 = 0.475 |
| current vs dragapult | 39/80 = 0.487 |
| dragapult vs iono | 41/80 = 0.512 |
| iono vs dragapult | 42/80 = 0.525 |
| current vs iono | 43/80 = 0.537 |
| iono vs current | 37/80 = 0.463 |
| **dragapult vs random** | **40/80 = 0.500** |
| **iono vs random** | **20/80 = 0.250** |
| **current vs random** | **13/80 = 0.163** |

全部 18 个 cell：**0 平局、0 异常、0 非法动作**，两座位局数严格相等（各 20）。

## 5. 结论与 `best_deck.csv`

1. 同 checkpoint 对称对抗（牌组 vs 牌组）三对六向胜率都在 0.46–0.54，即三副牌组在
   “同强策略”下强度接近，胜负在噪声区间内。
2. 真正拉开差距的是 vs 随机合法基线：`dragapult 0.500 > iono 0.250 > current 0.163`。
   Kaggle 实盘的对手质量分布宽，弱对手占比高，因此该维度直接决定期望得分。
3. 选择 `dragapult_newlib` 作为提交牌组：`runs/evaluation/card_database_migration/best_deck.csv`
   （60 行，SHA `30c8c7365c75f38fd6e7e1d8543c42ce7055ed6fd1c6e9eb244e44484b78e724`，
   与已验证候选逐字节一致；引擎开局已确认接受）。
4. 迁移回归对照：历史库上 `current_agent_deck` vs random 为 0.20
   （`runs/evaluation/ctl2_cand_vs_random_deckcsv_60.json`，60 局），
   迁移库同协议复现 0.163（80 局，CI 内一致），说明牌组结论没有因卡牌库替换而漂移。

## 6. 已知边界

- 引擎 1.33.0 实际卡牌池为 1431 张，本库覆盖 ID 1–1267；ID 1268–1431 的 164 张新卡
  （140 宝可梦、8 支援者、6 物品、5 特殊能量、3 竞技场、2 道具）在编码器里被夹到 UNK 行。
  本次三副候选牌组的 ID 均 ≤1267，不受影响；若对手使用这些新卡，本策略会把它们当作 UNK 处理。
- 新 CSV 与本机引擎在 ID 15（Team Rocket's Energy 能量枚举）与 ID 979（Koraidon ex 招式表）
  两处存在版本差；本次以用户提供的 `cards.csv` 为权威来源，两处差异已归档在
  `runs/card_database_migration/20261020/engine_vs_csv_comparison.json`。
- 牌组搜索本身（重新训练/新卡池扩展）不在本次任务范围，未启动任何训练。
