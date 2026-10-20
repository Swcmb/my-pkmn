# 评估门禁重验报告（迁移后的新卡牌库）

日期：2026-10-20　脚本：`runs/card_database_migration/20261020/stage5_gates.py`
原始记录：`runs/card_database_migration/20261020/stage5_gates.json`
协议：复用 `runs/evaluation/eval_h2h.py::run_one`（未重写对战逻辑），`scripts/h2h_ckpt.py` 与
`rl/gates.py` 的调度口径保持一致；`rl/gates.py` 的 `_field_candidates` 缺失缺陷已由工作区版本补齐，
`tests/test_gates.py` 覆盖该修复（48 项测试全通过）。

## 1. 统一口径（所有 cell 相同）

| 项 | 值 |
|:---|:---|
| 卡牌库 | `EN_Card_Data.csv`（迁移后）SHA `05aa569e3003a838b8503a9d928795e5aea7407ab19dd9f3d9c4b1293b6bb4b4`，vocab 1268 |
| 引擎 | 官方 `kaggle_environments==1.33.0` CABT（`D:\pkmn-kaggle\.engine-shim\…\cg.dll`，卡牌 1431 / 招式 1755） |
| 编码路径 | Python `TokenEncoder` + 引擎 `would_ko` 标注（`native_encode=False`，与训练一致） |
| 环境 | `PKMN_TURN_CAP=200`、`live=1`（JSON 引擎 battle_ptr 为进程级单例）、贪心 argmax |
| 版本 | Python 3.12.14 / PyTorch 2.14.0+cu130 / device cuda / Windows 10.0.26100 |
| 座位 | `cs = gi % 2` 逐局严格交替（每 cell 两座位局数相等） |
| checkpoint | 候选 `ckpt_999424.pt` SHA `b8a3be1d247e…` step 999424；基线 `ckpt_500736.pt` SHA `b02b0724ebbb…` step 500736；两者 `net_config` 相同 |
| 牌组 | `dragapult_newlib` SHA `30c8c7365c75…`；`current_agent_deck`（`agent/deck.csv`）SHA `1156379af39e…`；held-out `pool_top50.json` SHA `f97624da164b…` |

每个 checkpoint 路径/SHA/step、牌组文件与 SHA、新库 SHA、种子、版本、编码路径、座位分配、
平局/异常/非法动作计数均逐条写入 `stage5_gates.json`。

## 2. 结果矩阵（每 cell 60 局）

| cell | 牌组 | seed | 胜率 | Wilson 95% CI | 分座位 s0/s1 | 平/异/非 | 平均决策数 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| cand_vs_random | dragapult_newlib | 20261020 | 0.567 | [0.441, 0.684] | 0.40 / 0.73 | 0/0/0 | 107 |
| cand_vs_random | dragapult_newlib | 20261021 | 0.650 | [0.524, 0.758] | 0.60 / 0.70 | 0/0/0 | 110 |
| cand_selfmirror | dragapult_newlib | 20261020 | 0.567 | [0.441, 0.684] | 0.60 / 0.53 | 0/0/0 | 78 |
| cand_selfmirror | dragapult_newlib | 20261021 | 0.450 | [0.331, 0.575] | 0.57 / 0.33 | 0/0/0 | 88 |
| cand_vs_untrained | dragapult_newlib | 20261020 | 0.833 | [0.720, 0.907] | 0.90 / 0.77 | 0/0/0 | 118 |
| base_vs_random | dragapult_newlib | 20261020 | 0.600 | [0.474, 0.714] | 0.60 / 0.60 | 0/0/0 | 114 |
| cand_vs_random | current_agent_deck | 20261020 | 0.267 | [0.171, 0.390] | 0.20 / 0.33 | 0/0/0 | 48 |
| cand_vs_random | current_agent_deck | 20261021 | 0.200 | [0.118, 0.318] | 0.13 / 0.27 | 0/0/0 | 52 |
| cand_selfmirror | current_agent_deck | 20261020 | 0.450 | [0.331, 0.575] | 0.67 / 0.23 | 0/0/0 | 92 |
| cand_selfmirror | current_agent_deck | 20261021 | 0.517 | [0.393, 0.638] | 0.53 / 0.50 | 0/0/0 | 89 |
| cand_vs_untrained | current_agent_deck | 20261020 | 0.283 | [0.185, 0.408] | 0.17 / 0.40 | 0/0/0 | 58 |
| base_vs_random | current_agent_deck | 20261020 | 0.000 | [0.000, 0.060] | 0.00 / 0.00 | 0/0/0 | 39 |
| cand_vs_baseline_heldout | pool_top50（交叉） | 20261020 | 0.517 | [0.393, 0.638] | 0.83 / 0.20 | 0/0/0 | 109 |
| cand_vs_baseline_heldout | pool_top50（交叉） | 20261021 | 0.500 | [0.377, 0.623] | 0.73 / 0.27 | 0/0/0 | 101 |

**合计 840 局：0 平局、0 异常、0 非法动作**（`illegal.clamp/forfeit` 全 0，
`err_samples` 全空）。

## 3. 门禁判定

1. **模型有效性**：候选 `ckpt_999424` + `dragapult_newlib` 对同配置随机初始化网络 0.833
   （CI 下界 0.720，远超 0.5）；自镜像 0.567/0.450（合并 61/120 = 0.508，落在镜像应有的 ~0.5）。
   两项同时成立说明：策略确实学到了东西，且评估口径没有系统性座位/牌组泄漏。
2. **牌组效应**：同一 checkpoint 换牌组，vs random 从 0.233（`agent/deck.csv` 合并 28/120）
   提到 0.608（`dragapult_newlib` 合并 73/120）。`best_deck.csv` 取 Dragapult。
3. **候选 vs 旧基线**：held-out `pool_top50` 交叉 0.517/0.500（合并 0.508，CI 跨 0.5）——
   两个 checkpoint 在新库下强度相当，候选没有显著优势；分座位 0.83/0.20 显示
   先手座位效应主导胜负，与历史报告记录的座位效应方向一致。
4. **基线对照**：`base_vs_random` 在 Dragapult 上 0.600、在当前牌组上 0.000，
   说明牌组对基线同样关键，且当前牌组是主要短板。
5. **新旧库结果分开**：
   - 新库（本报告）：同协议、同 harness、同 turn_cap。
   - 旧库（历史，未混入本报告计算）：`runs/evaluation/ctl2_cand_vs_random_deckcsv_60.json`
     给出 `agent/deck.csv` vs random = 0.20（60 局，turn_cap 200，同一 `run_one` 协议）。
     新库同 cell 为 0.233（120 局）与 0.267/0.200 分种子 → 同置信区间内一致，
     迁移未造成胜率漂移。历史文档中 Dragapult 的 0.690/0.445 属另一协议（不同池与调度），
     不与本表混算。

## 4. 结论

在新卡牌库上，统一数据与运行时设置的重验没有出现异常或非法动作，候选 checkpoint 判定为
**可用且与迁移库兼容**；牌组门禁选定 `runs/evaluation/card_database_migration/best_deck.csv`。
本报告与 `deck_validation_report.md`、`linux_runtime_report.md`、
`runs/card_database_migration/20261020/ckpt_compat_report.json` 共同构成最终导出前的证据链。
