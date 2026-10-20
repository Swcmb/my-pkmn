# 卡牌库迁移 · 全部报告合订

生成日期：2026-10-20　项目：`D:\pkmn-kaggle`
最终提交包：`submission_rl_transformer2_db_migrated.tar.gz`
SHA-256 `8d4ec3ccdebd563c259efa12bfe1061493dcaff7c1f0154fcfff6f128d58ae71`（75,126,979 B）

本文件按执行顺序合订本次迁移的六份报告，正文逐段保留原文（仅标题下沉一级，代码块与日志原文不动）。
来源文件仍然存在且未被修改：

| 部分 | 来源文件 |
|:---|:---|
| 第 1 部分 · 阶段 1 · 备份 / 校验 / schema 与 ID 对比（含方案 A 落地结果） | `runs/card_database_migration/20261020/stage1_migration_report.md` |
| 第 2 部分 · 牌组验证与双向交叉评估 | `runs/evaluation/card_database_migration/deck_validation_report.md` |
| 第 3 部分 · Linux（WSL2）真实运行时验证（含最终包复验） | `runs/evaluation/card_database_migration/linux_runtime_report.md` |
| 第 4 部分 · 评估门禁重验（迁移后新库） | `runs/evaluation/card_database_migration/final_gate_report.md` |
| 第 5 部分 · 最终导出前门禁核对表 | `runs/evaluation/card_database_migration/pre_export_checklist.md` |
| 第 6 部分 · 最终综合报告 | `runs/evaluation/card_database_migration/final_report.md` |
## 目录

1. 第 1 部分 · 阶段 1 · 备份 / 校验 / schema 与 ID 对比（含方案 A 落地结果）
2. 第 2 部分 · 牌组验证与双向交叉评估
3. 第 3 部分 · Linux（WSL2）真实运行时验证（含最终包复验）
4. 第 4 部分 · 评估门禁重验（迁移后新库）
5. 第 5 部分 · 最终导出前门禁核对表
6. 第 6 部分 · 最终综合报告
7. 第 0 部分 · 待用户决定的两项与微调状态（已前置）

---

# 待用户决定的两项与微调状态

## 待决定 1  · `best_deck.csv` 是否加入 `rl/decks_train.py` 的 `TRAIN_TOP50` 池

`--finetune-deck` 只接受池内**牌组名**（`rl/train_selfplay.py:312`），
`--gate-decks-file` 接受的是池下标 JSON 列表（同文件 L294），两者都无法直接指向牌组文件。
实测 `TRAIN_TOP50` 共 74 张池内牌组，`best_deck.csv` 的 60 张序列**无同序列匹配**，
最接近的是 `k06_dragapult_ex_fezandipiti_ex_latias_ex_me`（与提交牌组不同）。
选项：(a) 把 `best_deck.csv` 作为新命名条目入池（一处数据改动，需批准）；
(b) 用池内 `k06_dragapult…` 训练并接受差异；(c) 不入池，微调只做通用自博弈。
本次未擅自修改池文件。

## 0.5M 微调状态

配置、预计耗时（约 1.6 h 纯训练 / 规划 2 h 机器时间，按实测 88.095 env-step/s，
来源 `runs/bench-20m/report.md` E16 推荐配置）与六项验收阈值见合订本第 6 部分 §9。
**训练未启动**，等待用户明确决定。

## 本次会话用量

约 56 轮、3376 秒。


---

# 第 1 部分 · 阶段 1 · 备份 / 校验 / schema 与 ID 对比（含方案 A 落地结果）

## 卡牌库迁移 · 阶段 1 报告（备份 / 校验 / schema 与 ID 对比）

日期：2026-10-20　执行目录：`runs/card_database_migration/20261020/`

### 1. 权威卡牌库定位

| 类别 | 路径 | 角色 |
|:---|:---|:---|
| 原始卡牌数据（权威库） | `D:\pkmn-kaggle\EN_Card_Data.csv` | `rl/card_features.py` `_build()` 唯一 CSV 入口；`rl/decks.py`、`scripts/deck_search.py`、`scripts/deck_rank_blend.py`、`scripts/scout_subs.py`、`scripts/export_rl_submission.py --csv` 均读取它；提交包内副本由导出脚本 `shutil.copy(args.csv, …/EN_Card_Data.csv)` 生成 |
| 派生数据 | `rl/card_features.py` 运行时缓存的 59 维特征矩阵（`lru_cache`，不落盘）；`rl/buff_data.py`（由 `all_attack()` + `EN_Card_Data.csv` 生成的静态表，随代码提交）；`rl/effect_data.py`（招式/特性 multi-hot） | 需在库变更后按依赖关系单独处理 |
| 引擎内部卡牌库 | `kaggle_environments/envs/cabt/cg/{cg.dll,libcg.so,…}` 内嵌，经 `sdk_cg.api.all_card_data()` / `all_attack()` 暴露 | 不在仓库内，是运行时真值来源；观测中的 cardId 由它定义 |
| 提交牌组 | `agent/deck.csv` | 60 行 cardId，**不是**卡牌库 |

### 2. 备份与新库落地校验（SHA-256 全部一致）

| 文件 | SHA-256 |
|:---|:---|
| 原库 `EN_Card_Data.csv` | `aca07ac210be6fa23297174e4855eb900a54e267fa73b31e1397bf84bed89dc5` |
| 备份 `runs/card_database_migration/20261020/original_cards.csv` | `aca07ac2…bed89dc5`（与原库逐字节一致 ✅） |
| 新源 `E:\Files\Downloads\cards.csv` | `a198d909eae4d04d7fdac07cd14bf5a8d870015e3356d1c286071ff76264a23b` |
| 副本 `…/downloaded_cards.csv` | `a198d909…64a23b`（与源一致 ✅） |

### 3. 新库自身质量校验（`compare_databases.py`）

- 编码：UTF-8 **带 BOM**（原库无 BOM）；读取需 `encoding="utf-8-sig"`。
- 行数 1268（含表头）→ 真实卡牌 **1267** 张；`cardId` 全部为整数、无空值、**无重复**、不可解析行 0。
- 列 21 个：`cardId,name,cardType,cardTypeName,pokemonType,energyType,evolutionType,hp,retreatCost,weakness,resistance,basic,stage1,stage2,ex,megaEx,tera,aceSpec,evolvesFrom,attacks,skills`。
- 布局：一卡一行，招式打包为 `attackId:名称(伤害)[能量索引,…]`，特性打包在 `skills`。
- 该 schema 与 `sdk_cg/api.py` 的 `CardData` dataclass 字段一一对应 → 判定为**引擎格式导出**。

### 4. 新旧 ID 映射（`id_mapping_comparison.json`）

| 指标 | 结果 |
|:---|:---|
| 旧库唯一卡 ID | 1267（范围 1–1267） |
| 新库唯一卡 ID | 1267（范围 1–1267） |
| ID 集合是否相同 | **完全相同**（新增 0 / 删除 0） |
| 同 ID 卡名是否一致 | **1267/1267 一致**（无重编号、无名称漂移） |

结论：ID 是稳定语义键，未发生重新分配。

### 5. 字段级差异（`schema_comparison.json`、`field_diff_details`、`classify_field_diffs.py`）

按“表示差异”和“真实语义差异”分类：

| 字段 | 数量 | 性质 | 说明 |
|:---|:---|:---|:---|
| HP `n/a` → `0` | 206 | 表示 | 训练家/能量卡在旧库写 `n/a`，新库写 `0`；`_parse_int` 两者都得 0，特征不变 |
| Retreat `n/a` → `0` | 246 | 表示 | 同上 |
| 能量类型 空/缺 → `COLORLESS` | 191 | 表示 | 旧库训练家卡 `Type` 留空，新库显式给枚举 |
| 招式伤害 `n/a`/`20×` → `0` | 333 | 主要为表示 | 旧库把变动伤害写成 `20×`（`_parse_int`→20），新库给基值 0；**这一类会真实改变 `max_dmg` 特征** |
| 招式条数不一致 | 250 | 混合 | 旧库逐招式的行展开方式与新库打包列表在部分卡上不一致（含 `Tera` 这类 0 伤招式是否计入） |
| 能量类型实质变化 | 39 | 真实 | 多为特殊能量的多属性表达（旧 `{A}{A}`/`{A}` vs 新 `RAINBOW`/`COLORLESS` 枚举） |
| `evolvesFrom` 变化 | 5 | 真实 | 进化前名称差异，需逐卡确认 |

### 6. 三方对比：新库 vs 本机引擎 1.33.0（`engine_vs_csv_comparison.json`）

本机引擎（PyPI `kaggle_environments==1.33.0` 的 CABT 二进制，见 `runs/evaluation/final_gate_combined_report.md` L170）实测：

| 指标 | 值 |
|:---|:---|
| 引擎卡牌总数 | **1431**（ID 1–1431 连续） |
| 引擎招式总数 | 1755 |
| 新 CSV ∩ 引擎 | 1267，其中 **1265 张逐字段完全一致** |
| 新 CSV ∩ 引擎 差异 | 仅 2 处：`id 15 Team Rocket's Energy`（CSV `COLORLESS` vs 引擎 `TEAM_ROCKET`）、`id 979 Koraidon ex`（CSV 招式 `1408:Tera(50)/1409:Orichalcum Fang(200)` vs 引擎 `1408:Orichalcum Fang(50)/1409:Impact Blow(200)`） |
| 仅引擎有（ID 1268–1431） | **164 张**：宝可梦 140、支援者 8、物品 6、特殊能量 5、竞技场 3、道具 2（例：`Nitro {R} Energy`、`Beedrill ex`、`Chesnaught`、`Prism Tower`、`Fossil Quarry`） |
| 仅 CSV 有 | 0 |

判读：新 CSV ≈ 引擎库在 ID 1–1267 上的导出，但**版本略旧于 1.33.0**（少了后 164 张新卡，且 2 处字段落后）。因此“替换本地卡牌库”存在两个不同目标，必须区分：
- 让 Python 侧特征表跟上**新 CSV**（1267 卡）；
- 让 Python 侧特征表跟上**引擎真值**（1431 卡）。

### 7. 关键阻塞与结论

1. **直接覆盖会使权威库失效（已实证）**：新 CSV 缺少读取契约要求的列，用 `downloaded_cards.csv` 直接覆盖 `EN_Card_Data.csv` 后：
   - `rl/card_features.CardTable(NEW)` → `KeyError: 'Card ID'`
   - `rl/decks.py` 风格的 `csv.DictReader(...)['Card ID']` → `KeyError: 'Card ID'`
   - 结果：训练入口、Agent 编码器、导出脚本全部崩溃。按任务 §4.5「校验失败则保留原数据库」，**原库当前保持不变**（哈希仍为 `aca07ac2…`）。
2. **checkpoint 兼容性（初判，待阶段 2 回归验证）**：`ckpt_999424.pt`（step 999424，sha `b8a3be1d247eac45af0bbc38a207336d6f702c82fd0b19e5f787cc7addf08a1e`）与 `ckpt_500736.pt`（step 500736，sha `b02b0724ebbb90916e9d495cbc4ab73b30f6bb761cea993366798942b5477a3d`）`net_config` 相同（`transformer2, d_model=544, nhead=8, nlayers=5, ff=256, static=false, would_ko=true, value_atoms=51`），`card_emb.weight = (1269, 544)` = `vocab_size(1268)+1`，其中 0=PAD、1268=UNK。由于 ID↔卡名 1:1 未变，`card_emb` 行语义保持；`would_ko` 三列来自引擎观测而非 CSV，不受库替换影响。风险点集中在 `static=false` 路径之外的 CSV 派生输入（`max_dmg` 变动伤害、`tera` 标记、`effect_data`/`buff_data` 生成表）。
3. **引擎二进制当前缺失（阻塞级，新出现）**：本次会话开始时 `C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim\…\cg.dll` 可用（阶段 2 引擎导出即由它完成），随后该临时目录被清理，当前 `Temp/opencode` 已不存在；WSL2 侧 `python3` 无 `kaggle_environments`，但历史验证目录 `/home/swcmb/pkmn-linux-verify-20261020/pylibs/…/libcg.so` 仍在。§6.2「必须用真实 CABT 引擎开局校验」、§6.3 交叉评估、§7 WSL2 运行验证、以及导出命令都依赖引擎，需要先恢复（在 conda 环境 `pip install --no-deps kaggle-environments==1.33.0`，或重建 shim）。
4. 旧产物全部保留：未改动 `agent/deck.csv`、`runs/evaluation/**`、三个历史提交目录与 `submission_rl_transformer2*.tar.gz`、未触碰任何 checkpoint。

### 8. 需要你决定的分叉（阻塞后续阶段）

“替换本地卡牌库”在证据下有三种可执行语义，代价与风险不同，见对话中的选项。产出文件：
`original_cards.csv`、`downloaded_cards.csv`、`schema_comparison.json`、`id_mapping_comparison.json`、`engine_all_card_data.json`、`engine_all_attack.json`、`engine_vs_csv_comparison.json`、`compare_databases.py`、`compare_three_way.py`、`classify_field_diffs.py`。

### 9. 执行结果（当日更新：方案 A 已落地，分叉已关闭）

用户选定 **A 保格式换内容** + **conda 环境 / 永久 shim 恢复引擎**。实际执行：

1. 适配器 `build_migrated_library.py` 用新库数据重生成符合旧列契约的库
   （`EN_Card_Data.migrated.csv`，1766 行 / 1267 卡）。规则：以新库为权威重建
   Stage/Rule/Previous stage/HP/Type/Weakness/Resistance/Retreat/Move Name/Cost/Damage；
   新 schema 无法表达的 `Expansion`、`Collection No.`、`Category`（Ancient/Future/Tera(...)/Trainer's 标签）
   按稳定 ID 从前库继承；`RAINBOW`/`TEAM_ROCKET` 沿用前库能量写法，避免既有 one-hot 语义被单枚举压平。
2. 覆盖前校验：`CardTable` 读取通过，`vocab_size=1268`、`feat_dim=59` 与原库一致；
   ID→卡名 1267/1267 保持；`agent/deck.csv` 与 `dragapult_deck.csv` 各 60 张、库内缺失 0。
3. 同卷临时文件 + `os.replace` 原子替换 `EN_Card_Data.csv`；替换后 SHA
   `05aa569e3003a838b8503a9d928795e5aea7407ab19dd9f3d9c4b1293b6bb4b4`，
   原库 SHA `aca07ac2…bed89dc5` 仅存在于备份文件。
4. 特征矩阵差异定量（501/1268 张卡有变化）：只落在 dim12 `type_C`（195 张）、
   dim22 `type_A`（2 张）、dim45 `n_attacks`（75 张）、dim46 `max_dmg`（250 张）。
   dim 38–41（SPECIAL_TAGS，含 `_tera_flag` 依赖的 Tera 维）零变化，
   而该维是本 checkpoint（`static=false`）唯一真正进入前向的 CSV 派生路径。
5. 引擎恢复：`Temp/opencode/bench-shim` 已被系统清理；改为仓库内永久 shim
   `D:\pkmn-kaggle\.engine-shim`（官方 `kaggle_environments==1.33.0` 副本，包 `__init__` 链降级为桩以避开
   `jsonschema` 重依赖），并写入 `.gitignore`。引擎实测 1431 卡 / 1755 招式。
6. 上述 §7.1 的「原库当前保持不变」为本阶段中间态，已由本节取代；
   后续结论见 `ckpt_compat_report.json` 与
   `runs/evaluation/card_database_migration/{deck_validation_report.md, linux_runtime_report.md, final_gate_report.md, pre_export_checklist.md}`。

---

# 第 2 部分 · 牌组验证与双向交叉评估

## 牌组验证与交叉评估报告（新卡牌库）

日期：2026-10-20　目录：`runs/evaluation/card_database_migration/`
执行脚本：`runs/card_database_migration/20261020/validate_decks.py`、`deck_cross_eval.py`
原始证据：`runs/card_database_migration/20261020/deck_engine_checks.json`、`deck_cross_eval.json`

### 1. 口径一致性（只变牌组）

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

### 2. 候选牌组与来源

| 候选 | 来源 | SHA-256 |
|:---|:---|:---|
| `dragapult_newlib` | `runs/evaluation/deck_search/dragapult_deck.csv` | `30c8c7365c75f38fd6e7e1d8543c42ce7055ed6fd1c6e9eb244e44484b78e724` |
| `current_agent_deck` | `agent/deck.csv`（当前提交牌组） | `1156379af39e71bc83eecb50d1e04c5cc480501d293621fcc89c2d355d99be78` |
| `fixture_iono` | `rl/decks.py:_DECKS_SPEC['iono']`（补充合法候选） | 代码内置 |

旧 Dragapult 牌组在新库下的逐 ID 核对结果：**60 张全部命中**，无 ID 缺失、无进化链告警。
新库 ID↔卡名 1:1 保持不变（1267/1267），因此旧牌组的每个 ID 仍指向同一张卡，
不需要按名称重新映射；同时已按新库重新核对 HP/能量/招式/ACE SPEC 标记。

### 3. 合法性验证

#### 3.1 Python 层（新库 + 引擎元数据）

5 副候选（含 `mega_lucario`、`mega_abomasnow` 两个额外 fixture）全部通过：
恰好 60 张、ID 均存在于新库与引擎、同名 ≤4（基本能量除外）、ACE SPEC ≤1、
至少 1 张基础宝可梦（Dragapult 候选含 9 张基础宝可梦）、进化前置名称均在牌组内。
`python_rejects = []`。

#### 3.2 真实 CABT 引擎层

对 5 副候选做镜像 + 全部交叉配对开局，共 **15 种配对，全部被引擎接受**
（`accepted: true, errorPlayer: -1, errorType: 0`），`engine_rejects = []`。
引擎开局校验不依赖 Python 侧检查结果。

### 4. 双向交叉评估结果（同 checkpoint，仅牌组变量）

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

### 5. 结论与 `best_deck.csv`

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

### 6. 已知边界

- 引擎 1.33.0 实际卡牌池为 1431 张，本库覆盖 ID 1–1267；ID 1268–1431 的 164 张新卡
  （140 宝可梦、8 支援者、6 物品、5 特殊能量、3 竞技场、2 道具）在编码器里被夹到 UNK 行。
  本次三副候选牌组的 ID 均 ≤1267，不受影响；若对手使用这些新卡，本策略会把它们当作 UNK 处理。
- 新 CSV 与本机引擎在 ID 15（Team Rocket's Energy 能量枚举）与 ID 979（Koraidon ex 招式表）
  两处存在版本差；本次以用户提供的 `cards.csv` 为权威来源，两处差异已归档在
  `runs/card_database_migration/20261020/engine_vs_csv_comparison.json`。
- 牌组搜索本身（重新训练/新卡池扩展）不在本次任务范围，未启动任何训练。

---

# 第 3 部分 · Linux（WSL2）真实运行时验证（含最终包复验）

## Linux（WSL2）真实运行时验证报告 · 卡牌库迁移

日期：2026-10-20　目录：`runs/evaluation/card_database_migration/`
证据文件：`runs/card_database_migration/20261020/{wsl_stage4.log, wsl_stage4_rerun.log, wsl_stage4_results.json, wsl_stage4c.json, init_repro_results_windows.json, win_init_repro.py, wsl_stage4_run.py, wsl_stage4_cases.py, wsl_stage4b_harness.py, wsl_stage4c_instrumented.py}`

### 1. 运行环境（实测）

| 项 | 值 |
|:---|:---|
| 发行版 | Ubuntu-24.04（WSL2），内核 `Linux 6.6.87.2-microsoft-standard-WSL2 x86_64` |
| Python | 3.12.3（系统解释器） |
| PyTorch | 2.14.1+cpu（`~/pkmn-linux-verify-20261020/pylibs`） |
| glibc | 2.39（Ubuntu GLIBC 2.39-0ubuntu8.9） |
| 引擎 | 官方 `kaggle-environments==1.33.0`，`…/envs/cabt/cg/libcg.so`（1,458,504 B） |
| 暂存目录 | `~/pkmn-migration-verify-20261020/pkg`（干净解包） |
| 只读性 | 未改动 Windows 侧原提交包、checkpoint 与历史评估产物 |

### 2. 暂存包构造（迁移叠加）

以现有候选包 `submission_rl_transformer2_dragapult_nodist.tar.gz`
（SHA `09b2929c5f9cf77759a087b09ad202bf05bdf32b36d9ac00e89ca2505b1d9d28`）为基线解包到空目录，
再叠加本次迁移产物：

| 覆盖项 | 来源 | SHA-256 |
|:---|:---|:---|
| `EN_Card_Data.csv` | 迁移后的仓库权威库 | `05aa569e3003a838b8503a9d928795e5aea7407ab19dd9f3d9c4b1293b6bb4b4` |
| `sdk_cg/*.py` | 仓库当前版本（含本次引擎修复） | `sim.py` 见 `wsl_stage4_results.json.staging.staged` |
| `deck.csv` | `runs/evaluation/card_database_migration/best_deck.csv` | `30c8c7365c75f38fd6e7e1d8543c42ce7055ed6fd1c6e9eb244e44484b78e724` |

包内不含 `cg.dll` / `libcg.so` 等专有二进制：引擎依赖在运行时由
`sdk_cg/sim.py` 解析（包内二进制 → 官方 `kaggle_environments` 安装），符合“不嵌入未确认可分发二进制”的要求。

### 3. 缺陷定位：单向句柄复用保护

历史结论（`runs/evaluation/linux_runtime_report.md`，同日前一次会话）是：干净解释器
`import main` 在“第二次 `GameInitialize()`”处 SIGABRT。本次在 WSL2 复现并定位到更精确的机制：

- 官方 `kaggle_environments/envs/cabt/cg/sim.py` 在 **import 期** 就执行
  `LoadLibrary(libcg.so)` + `lib.GameInitialize()`。
- 同一路径的 dll/so 会被进程复用为 **同一个 HMODULE/dlopen 句柄**。
- 修复前的 `sdk_cg/sim.py::_load_lib()` 只在“官方模块已在 `sys.modules`”时复用句柄；
  若本模块先被导入（`import main` 的自然顺序），它会自己加载并初始化，
  随后官方模块再导入时 **在同一句柄上二次初始化** → 引擎内部固定容量缓冲区溢出：

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  buffer full. capacity:7
```

修复前实测（`runs/card_database_migration/20261020/wsl_stage4.log`，每例独立进程）：

| 用例 | 退出码 | 现象 |
|:---|:---|:---|
| `import_clean`（干净解释器 `import main`） | **-6（SIGABRT）** | `buffer full. capacity:7` |
| `env_first`（官方环境先加载再导入 agent） | 0 | 通过 |
| `by_path`（按文件路径加载 agent） | **-6** | 同上 |
| `deck_request`（60 张牌组 + 引擎开局） | **-6** | 同上 |

#### 修复（最小范围）

`sdk_cg/sim.py::_load_lib()` 改为**优先导入官方 sim 模块**，让官方模块完成它那唯一一次
`GameInitialize`，本模块复用其 `lib` 句柄；仅当官方包不可导入时才自行加载并初始化一次。
保留 `SearchBegin/SearchStep/SearchEnd/SearchRelease` 等全部真实声明与生命周期，
不伪造符号、不关闭 `would_ko`、失败时原始错误照常抛出。

### 4. 修复后验证（WSL2，官方 libcg.so）

六个用例各自独立进程，全部通过（`wsl_stage4_rerun.log` + `wsl_stage4_results.json`）：

| 用例 | 退出码 | 关键结果 |
|:---|:---|:---|
| 干净 `import main` | 0 | 模块可导入，`deck_len=60` |
| 官方环境先加载再导入 agent | 0 | `make("cabt")` 成功；同一引擎句柄 |
| 按文件路径加载 agent | 0 | `spec_from_file_location` 加载 `pkg/main.py`，`agent` 可调用 |
| 60 张牌组请求 + 引擎开局 | 0 | `deck_len=60`、`unknown_ids=[]`、`engine_accepted=true`、`errorType=0`、基础宝可梦 9 张 |
| 双座位真实对局 | 0 | 见下表（官方 harness 口径） |
| `would_ko=True` 真实 Search API | 0 | `SearchBegin/SearchStep/SearchEnd/SearchRelease` 齐备 |

#### 4.1 官方 harness 真实对局（插桩，`wsl_stage4c.json`）

用 `kaggle_environments.make("cabt")` + `env.run([agent, random_legal])` 跑真实对局，
候选 agent 分别坐 0/1 号位各 3 局：

| 座位 | 局 | 步数 | reward（agent 视角） | 异常 |
|:---|:---|:---|:---|:---|
| 0 | 0 | 322 | 胜 | 无 |
| 0 | 1 | 337 | 胜 | 无 |
| 0 | 2 | 297 | 负 | 无 |
| 1 | 0 | 279 | 负 | 无 |
| 1 | 1 | 176 | 胜 | 无 |
| 1 | 2 | 274 | 胜 | 无 |

- 6 局全部正常完赛并产生胜负（`rewards` 为 `[1,-1]`/`[-1,1]`），0 异常、0 非法动作。
- 引擎加载与初始化：`loads=["libcg.so"]`（1 次），`game_initialize_calls=1`（1 次）。

#### 4.2 `would_ko` 真实正例

同一批 harness 对局内插桩 `search_agent.would_ko_flags`（引擎 Search API 路径，非构造数据）：

```text
calls=387  非空攻击选项集=53  正例 KO=21  正例终局胜=3
samples: {"ko":1.0,"prizes":0.0,"win":1.0} / {"ko":1.0,"prizes":1.0,"win":0.0} ×2
```

来源标注：正例取自真实引擎对局中候选 agent 自身产生的攻击 observation（`main.agent`
内部调用 `annotate_would_ko`），未使用任何伪造符号或静默降级。

#### 4.3 Windows 侧同机制对照（`init_repro_results_windows.json`）

WSL 故障后在 Windows 用官方 `cg.dll` 复现同一句柄语义（子进程：先导变体 `sdk_cg.sim`，
再导官方 `kaggle_environments.envs.cabt.cg.sim`）：

| 变体 | 退出码 | 结果 |
|:---|:---|:---|
| 修复前 `_load_lib` | 1 | 官方模块 `sim.py:34 lib.GameInitialize()` 抛 C++ 异常 `OSError 0xe06d7363`（与 Linux 的 `buffer full. capacity:7` 同源） |
| 修复后 `_load_lib` | 0 | `loads=["cg.dll"]`、`game_initialize_calls=1`、`same_handle=true`、开局 `errorType=0` |

### 5. Windows 回归

- `PYTHONPATH=.engine-shim pytest tests/ -q` → **48 passed**（含 gates、selfplay、encoding、decks 等）。
- 阶段 2 兼容性回归（`ckpt_compat_report.json`）：迁移库下两个 checkpoint 严格加载通过、
  745 个真实观测 logits/value 与备份库逐位一致、采样动作 0 非法、`would_ko` 136/136 调用无错误。

### 6. 剩余边界

1. WSL2 发行版在本次会话末段无法启动：挂载 `F:\DevOffload\WSL\Ubuntu-24.04\ext4.vhdx` 报
   `Wsl/Service/CreateInstance/MountDisk/HCS/E_ACCESSDENIED`（外部环境故障，未做任何修复尝试）。
   本报告所需的全部 Linux 证据均在该故障之前采集并落盘；仅“Linux 侧 old/new 机制对照”这一
   非必需项尚未补做（Windows 侧等价对照已完成）。
2. 本机 glibc 2.39 与 Kaggle 官方运行镜像的 glibc 一致性：本机验证通过，Kaggle 镜像实测待提交后确认。
3. 最终提交包尚未导出（按任务要求由用户手动执行导出命令），因此本报告验证的是**与将要导出的包同构的暂存包**
   （同一 `main.py` 模板、同一迁移库、同一 `sdk_cg` 修复）。

### 7. 结论

在真实官方 CABT 运行时（WSL2 + `libcg.so` 1.33.0）下，迁移库暂存包的两处核心问题均已闭环：
引擎初始化在本进程内唯一（干净 `import main`、官方环境先加载、按路径加载三种顺序全部 exit 0），
60 张牌组被引擎接受，双座位各 3 局真实对局无异常无非法动作，`would_ko` 走真实 Search API
并捕获到 21 个正例。Linux 运行时验证达到本阶段要求。

### 8. 最终提交包的 Linux 复验（导出后补做，2026-10-20）

§6.1 记录的 WSL2 不可用为外部变更：会话后段该发行版默认用户改为 `ubuntu`
（`/etc/wsl.conf` `default=ubuntu`），原 `/home/swcmb`（含历史 pylibs 验证目录）在该实例中已不存在。
在 `~/pkmn-venv` 重建 Linux 验证环境：`get-pip.py` → Tsinghua 索引未返回 pip（回退上游 PyPI 成功）、
`numpy 2.5.4`、`torch 2.14.1+cpu`（`download.pytorch.org/whl/cpu`）、`kaggle_environments 1.33.0`
（含官方 `libcg.so`）。

对**最终包本体**（`submission_rl_transformer2_db_migrated.tar.gz`，
SHA `8d4ec3ccdebd563c259efa12bfe1061493dcaff7c1f0154fcfff6f128d58ae71`）干净解包后复跑全部用例
（`final_pkg_wsl_results.json`）：

| 用例 | 退出码 | 关键结果 |
|:---|:---|:---|
| 干净 `import main` | 0 | `deck_len=60`，`engine_lib=…/kaggle_environments/envs/cabt/cg/libcg.so` |
| 官方环境先加载再导入 agent | 0 | `make("cabt")` 成功，句柄复用 |
| 按文件路径加载 agent | 0 | `has_agent=true` |
| 60 张牌组 + 引擎开局 | 0 | `unknown_ids=[]`、`engine_accepted=true`、`errorType=0`、基础宝可梦 9 张 |
| 双座位真实对局（官方 harness） | 0 | 见下 |
| `would_ko=True` 真实 Search API | 0 | 见下 |

官方 harness 对局（`make("cabt") + env.run([agent, random_legal])`，双座位各 3 局，
`final_pkg_harness.json`）：

```text
seat0: 胜 胜 负 (steps 238/215/252)   seat1: 胜负负 (steps 159/278/249)
errors: []   loads=["libcg.so"]   game_initialize_calls=1
would_ko: calls=346  非空攻击选项集=36  正例 KO=18  正例终局胜=9
samples: {"ko":1.0,"prizes":1.0,"win":0.0} / {"ko":1.0,"prizes":3.0,"win":0.0} ×2
```

对照说明：同一批对局若改用手写最小循环（不经官方 harness），会在引擎给出
`minCount=1、options=0` 的退化 select 时抛 `IndexError`；已用
`final_pkg_engine_error_trace.json` 定位为驱动在该退化状态的提交语义差异，
官方 harness 与训练 env 路径均为 0 异常，包本身无非法动作输出。

结论：**最终包在 WSL2 官方运行时下的真实 CABT 对局通过**，第 6.1 项遗留（Linux 侧 old/new 机制对照）
已由 Windows 侧对照 + 最终包 Linux 全用例覆盖。

---

# 第 4 部分 · 评估门禁重验（迁移后新库）

## 评估门禁重验报告（迁移后的新卡牌库）

日期：2026-10-20　脚本：`runs/card_database_migration/20261020/stage5_gates.py`
原始记录：`runs/card_database_migration/20261020/stage5_gates.json`
协议：复用 `runs/evaluation/eval_h2h.py::run_one`（未重写对战逻辑），`scripts/h2h_ckpt.py` 与
`rl/gates.py` 的调度口径保持一致；`rl/gates.py` 的 `_field_candidates` 缺失缺陷已由工作区版本补齐，
`tests/test_gates.py` 覆盖该修复（48 项测试全通过）。

### 1. 统一口径（所有 cell 相同）

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

### 2. 结果矩阵（每 cell 60 局）

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

### 3. 门禁判定

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

### 4. 结论

在新卡牌库上，统一数据与运行时设置的重验没有出现异常或非法动作，候选 checkpoint 判定为
**可用且与迁移库兼容**；牌组门禁选定 `runs/evaluation/card_database_migration/best_deck.csv`。
本报告与 `deck_validation_report.md`、`linux_runtime_report.md`、
`runs/card_database_migration/20261020/ckpt_compat_report.json` 共同构成最终导出前的证据链。

---

# 第 5 部分 · 最终导出前门禁核对表

## 最终导出前门禁核对表（等待用户手动执行导出）

日期：2026-10-20　状态：**9 项前置条件全部通过**，导出命令交由用户执行。

| # | 条件 | 结果 | 证据 |
|:--|:---|:---|:---|
| 1 | 新卡牌库替换成功，哈希与记录数已验证 | ✅ 权威库 `EN_Card_Data.csv` SHA `05aa569e3003a838b8503a9d928795e5aea7407ab19dd9f3d9c4b1293b6bb4b4`，1766 数据行 / 1267 张唯一卡 | `runs/card_database_migration/20261020/{original_cards.csv, downloaded_cards.csv, stage1_migration_report.md}`；原库备份 SHA `aca07ac2…bed89dc5` |
| 2 | 所有必要字段可正常解析 | ✅ `CardTable` vocab 1268 / feat_dim 59；`pytest tests/ -q` → **48 passed** | `build_migrated_library.py` 校验输出、`runs/card_database_migration/20261020/verify_run.log` |
| 3 | checkpoint 与新库输入语义兼容 | ✅ 两个 ckpt 均 `load_state_dict(strict=True)` 成功；745 个真实观测的 logits/value 与备份库**逐位一致**（maxdiff 0.0）；唯一进入前向的 CSV 派生路径 `_tera_flag`（维 40）逐位不变 | `runs/card_database_migration/20261020/ckpt_compat_report.json` |
| 4 | `best_deck.csv` 已生成且恰好 60 张 | ✅ `runs/evaluation/card_database_migration/best_deck.csv`，60 行，SHA `30c8c7365c75f38fd6e7e1d8543c42ce7055ed6fd1c6e9eb244e44484b78e724` | `deck_validation_report.md` §2/§5 |
| 5 | `best_deck.csv` 通过真实引擎验证 | ✅ Python 层 0 拒绝；真实 CABT 15 种开局配对全接受（`errorType=0`）；WSL2 官方 harness 下 60 张开局 `unknown_ids=[]` | `deck_engine_checks.json`、`wsl_stage4_results.json` |
| 6 | Windows 与 WSL2 运行时测试达标 | ✅ Windows：48 测试通过、真实对局与 would_ko 136/136 无错误；WSL2：6 用例全 exit 0、双座位各 3 局真实对局完赛、`GameInitialize` 恰 1 次、would_ko 正例 21 | `ckpt_compat_report.json`、`linux_runtime_report.md`、`wsl_stage4c.json` |
| 7 | 新的独立评估无异常与非法动作 | ✅ 840 局：0 平局 / 0 异常 / 0 非法动作；候选 vs untrained 0.833，自镜像 0.508 | `stage5_gates.json`、`final_gate_report.md` |
| 8 | `export_rl_submission.py` 参数经源码与 `--help` 核实 | ✅ 支持 `--ckpt --backend{transformer2,mcts2} --deck --csv --out --builddir --n-sims --n-det`；`--csv` 默认即仓库权威库（迁移后）；导出前强制校验 `deck.csv` 恰好 60 张 | `scripts/export_rl_submission.py` L158-213 + `--help` 实测 |
| 9 | 不嵌入未确认可分发的专有引擎二进制 | ✅ 打包清单只含 `sdk_cg/*.py`（无 `cg.dll/libcg.so`）；运行时由 `sdk_cg/sim.py` 解析官方包 | `export_rl_submission.py` `_bundle_engine_sim()`、`package contents` 审计 |

### 与任务书原文命令的**唯一差异**

任务书里的 `$env:PYTHONPATH` 指向 `C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim`。
该临时目录已在本会话中被系统清理（`Temp/opencode` 不复存在），因此按用户选定的方案，
改用仓库内永久 shim：`D:\pkmn-kaggle\.engine-shim`（官方 `kaggle_environments==1.33.0` 包副本，
包 `__init__` 链已降级为桩以避开 `jsonschema` 等重依赖；已写入 `.gitignore`，不入库）。
其余参数与任务书一致。

### 待用户执行的命令

```powershell
cd D:\pkmn-kaggle

$env:PYTHONPATH = "D:\pkmn-kaggle\.engine-shim"
$env:PYTHONUNBUFFERED = "1"

D:\ProgramFiles\deeplearning\python.exe scripts\export_rl_submission.py `
  --ckpt runs\train-20m-e16-1m\ckpt_999424.pt `
  --backend transformer2 `
  --deck runs\evaluation\card_database_migration\best_deck.csv `
  --out submission_rl_transformer2_db_migrated.tar.gz `
  --builddir submission_rl_transformer2_db_migrated
```

执行前复核（已在本文档 §1 完成）：`best_deck.csv` 存在 ✅、checkpoint 兼容性已验证 ✅、
参数受支持 ✅、打包库为迁移后卡牌数据 ✅、输出路径 `submission_rl_transformer2_db_migrated.tar.gz`
与 `submission_rl_transformer2_db_migrated/` 均**不存在**，不会覆盖任何已有产物 ✅。

### 已知边界（不影响门禁，但需记录）

1. 引擎 1.33.0 实际卡池 1431 张，本库覆盖 ID 1–1267；ID 1268–1431 的 164 张新卡在编码器中被夹到 UNK 行。
   选定牌组全部 ID ≤1267，不受影响。
2. `cards.csv` 与本机引擎在 ID 15、ID 979 两处存在版本差，以用户提供的 `cards.csv` 为权威来源。
3. WSL2 发行版在会话末段无法启动（挂载 `ext4.vhdx` 报 `E_ACCESSDENIED`，外部环境故障）。
   Linux 侧全部必需证据在故障前已采集落盘。

---

# 第 6 部分 · 最终综合报告

## 卡牌库迁移最终报告

日期：2026-10-20　项目：`D:\pkmn-kaggle`
提交包：`submission_rl_transformer2_db_migrated.tar.gz`
SHA-256 `8d4ec3ccdebd563c259efa12bfe1061493dcaff7c1f0154fcfff6f128d58ae71`　大小 75,126,979 B

导出命令由用户本人执行（AI 未运行、未间接运行）；本报告汇总其后所有验证。

### 1. 原数据库备份

| 项 | 值 |
|:---|:---|
| 原权威库 | `D:\pkmn-kaggle\EN_Card_Data.csv`（赛事导出，一招式一行，17 列） |
| 原库 SHA-256 | `aca07ac210be6fa23297174e4855eb900a54e267fa73b31e1397bf84bed89dc5` |
| 备份位置 | `runs/card_database_migration/20261020/original_cards.csv` |
| 备份哈希 | 与原库逐字节一致 ✅ |
| 引擎内部库 | 独立于 CSV：`sdk_cg.api.all_card_data()` 由官方 CABT 二进制提供（1431 卡 / 1755 招式） |
| 派生数据 | `rl/card_features.py` 运行时特征矩阵（缓存不落盘）、`rl/buff_data.py`、`rl/effect_data.py`（按依赖单独评估，未盲目覆盖） |

### 2. 新数据库

| 项 | 值 |
|:---|:---|
| 新源 | `E:\Files\Downloads\cards.csv`，SHA `a198d909eae4d04d7fdac07cd14bf5a8d870015e3356d1c286071ff76264a23b` |
| 迁移副本 | `runs/card_database_migration/20261020/downloaded_cards.csv`（哈希一致） |
| 实际记录数 | 1267 张唯一卡（`cardId` 1–1267 连续；无空值、无重复、无不可解析行） |
| 覆盖后权威库 | `EN_Card_Data.csv` SHA `05aa569e3003a838b8503a9d928795e5aea7407ab19dd9f3d9c4b1293b6bb4b4`，1766 数据行 / 1267 卡 |
| 覆盖方式 | 同卷临时文件 + `os.replace` 原子替换；覆盖前完成备份、格式校验与 schema 检查 |

新 CSV 为**引擎格式**（`cardId,name,cardType,…,attacks,skills` 21 列、一卡一行），与项目读取契约
（`Card ID / Card Name / Move Name / …`）不兼容：直接覆盖后 `CardTable` 与 `decks.py` 均 `KeyError: 'Card ID'`。
按用户选定的方案 A，用新库数据重生成符合旧列契约的库（`build_migrated_library.py`），
新库无法表达的 `Expansion / Collection No. / Category 版本标签` 按稳定 ID 从前库继承，
`RAINBOW / TEAM_ROCKET` 沿用前库能量写法以免 one-hot 语义被单枚举压平。

### 3. 新旧 schema 与 ID 语义差异

- ID 集合：`ids_equal_as_sets = true`，新增 0、删除 0；同 ID 卡名 **1267/1267 一致** → ID 是稳定语义键，未重新分配。
- 与本机引擎（1.33.0，1431 卡）：共享 1267 个 ID 中 1265 张逐字段完全一致，仅 2 处版本差
  （id 15 `Team Rocket's Energy` 能量枚举；id 979 `Koraidon ex` 招式表）。
  引擎另有 164 张 ID 1268–1431 的新卡不在任何一份 CSV 中（140 宝可梦 / 8 支援者 / 6 物品 / 5 特殊能量 / 3 竞技场 / 2 道具）。
- 字段差异分类（旧→新）：表示差异为主 —— HP `n/a→0` 206、Retreat `n/a→0` 246、能量列缺失→`COLORLESS` 191、
  变动伤害 `20×/n/a→基值` 333；真实语义差异 —— 能量类型 39、进化前名称 5、招式条数 250（含 `Tera` 等 0 伤招式是否计入）。
- 特征矩阵实际位移：501/1268 张卡的向量有变，只落在 dim12 `type_C`(195)、dim22 `type_A`(2)、
  dim45 `n_attacks`(75)、dim46 `max_dmg`(250)；SPECIAL_TAGS 维 38–41 **零变化**。

### 4. 旧 checkpoint 兼容性结论

**判定：兼容，可继续使用**（依据是输入级与输出级证据，不是“能加载”）。

| 检查 | 候选 `ckpt_999424.pt` | 基线 `ckpt_500736.pt` |
|:---|:---|:---|
| SHA-256 | `b8a3be1d247eac45af0bbc38a207336d6f702c82fd0b19e5f787cc7addf08a1e`（step 999424） | `b02b0724ebbb90916e9d495cbc4ab73b30f6bb761cea993366798942b5477a3d`（step 500736） |
| `net_config` | `transformer2, d_model=544, nhead=8, nlayers=5, ff=256, static=false, would_ko=true, value_atoms=51` | 同 |
| `card_emb.weight` | `(1269, 544)` = `vocab_size(1268)+1`，0=PAD、1268=UNK | 同 |
| 严格加载（备份库 vs 迁移库） | 均 `missing/unexpected = 空` | 均通过 |
| 745 个真实观测前向对比 | logits 与 value **逐位一致**（max abs diff 0.0） | 同 |
| 采样动作合法性 | 0 非法 | 0 非法 |
| `would_ko` 真实 Search API | 136/136 调用无错误 | — |

关键点：`static=false` 使 59 维 CSV 特征矩阵不进入该网络前向
（`build_token_net` 仅在 `static=true` 时传 `card_feat`），唯一残留的 CSV→输入路径是
`_tera_flag`（读 dim 40，来自 `Category`），而该列按稳定 ID 继承、逐位未变。
`would_ko` 三列来自引擎观测，与 CSV 无关。
风险留档：若将来启用 `static=true` 或重训，`max_dmg`/`n_attacks` 维已因迁移改变，需按新库重算基线。

### 5. 牌组验证与独立评估

`runs/evaluation/card_database_migration/deck_validation_report.md` 摘要：

- 合法性：Python 层 5 副候选 0 拒绝（60 张、同名 ≤4、ACE SPEC ≤1、≥1 基础宝可梦、进化链完整）；
  **真实 CABT 引擎 15 种配对开局全接受**（`errorType=0`）。
- 双向交叉（同 checkpoint，仅变牌组；2 种子 × 40 局 / cell，座位 `cs = gi % 2` 严格交替）：
  牌组对牌组全部落在 0.46–0.54；vs 随机合法基线拉开差距 ——
  Dragapult `40/80 = 0.500`、iono `20/80 = 0.250`、当前 `agent/deck.csv` `13/80 = 0.163`。
- 全部 18 cell：0 平局、0 异常、0 非法动作。
- 选定 `runs/evaluation/card_database_migration/best_deck.csv`（60 张，SHA `30c8c7365c75…b78e724`）。

`final_gate_report.md` 摘要（新库统一口径，840 局）：
候选+Dragapult vs random 合并 `73/120 = 0.608`；vs 同配置未训练网络 `0.833 [0.720, 0.907]`；
自镜像 `61/120 = 0.508`；候选 vs 旧基线在 held-out `pool_top50` 上 `61/120 = 0.508`（CI 跨 0.5，强度相当，座位效应主导）；
基线+当前牌组 vs random `0.000`。总计 **0 平局 / 0 异常 / 0 非法动作**。
新旧库胜率分开报告：当前牌组 vs random 旧库同协议 0.20（60 局）→ 新库 0.233（120 局），一致。

### 6. Windows 与 WSL2 真实运行验证

**引擎初始化缺陷已修复**：官方 `…/cg/sim.py` 在 import 期即 `LoadLibrary + GameInitialize`，
同一 so/dll 路径共用一个句柄；修复前的保护是单向的（只认“官方模块已导入”），
干净 `import main` 会由本模块先初始化、官方模块再在同一句柄上二次初始化 →
原生 `std::runtime_error: buffer full. capacity:7`（Linux SIGABRT，Windows 表现为 C++ 异常）。
修复：`sdk_cg/sim.py::_load_lib()` 优先导入官方 sim 模块并复用其句柄，仅在官方包不可用时自行加载并初始化一次；
保留 Search API 全部真实声明与生命周期，不伪造符号、不关闭 `would_ko`、失败照常抛错。

- Windows：`pytest tests/ -q` → **48 passed**；修复前后对照
  （`init_repro_results_windows.json`）old → 官方模块第 34 行 `GameInitialize()` 抛 `OSError 0xe06d7363`，
  new → `loads=["cg.dll"]`、`game_initialize_calls=1`、`same_handle=true`、开局 `errorType=0`。
- WSL2（Ubuntu-24.04，Python 3.12.3，torch 2.14.1+cpu，glibc 2.39，官方 `libcg.so`）：
  修复前 `import_clean / by_path / deck_request` exit **-6**；修复后 6 用例全部 exit 0
  （`wsl_stage4_rerun.log`、`wsl_stage4_results.json`）。
- 最终包 Linux 复验：6 用例 exit 0，`deck_request` 显示 `deck_len=60`、`unknown_ids=[]`、
  `engine_accepted=true`、`errorType=0`、基础宝可梦 9 张（`final_pkg_wsl_results.json`）。
- 最终包官方 harness 对局（`make("cabt") + env.run`，双座位各 3 局）：
  6 局全部 error=null、rewards  decisive（agent 3 胜 3 负）、`loads=["libcg.so"]`、`game_initialize_calls=1`；
  `would_ko` 插桩 `calls=346 / 非空 36 / KO 正例 18 / 终局胜正例 9`，
  样本 `{"ko":1.0,"prizes":3.0,"win":0.0}`（`final_pkg_harness.json`）。

### 7. 提交包内容与一致性

| 检查 | 结果 |
|:---|:---|
| tar 可读、19 成员、扁平结构 | ✅ |
| `model.pt` ↔ `ckpt_999424.pt` | ✅ 88 个张量逐位相等；`global_step=999424`；`net_config` 一致；键精简为 `net/net_config/global_step` |
| `deck.csv` ↔ `best_deck.csv` | ✅ 逐字节一致，60 行 |
| `EN_Card_Data.csv` ↔ 迁移后权威库 | ✅ 逐字节一致（`05aa569e…`） |
| 专有引擎二进制 | ✅ 未打包（`*.dll/*.so/*.dylib` 列表为空，`sdk_cg/` 仅 5 个 `.py`） |
| 打包代码 ↔ 仓库源码 | ✅ 13 个模块 byte-identical；唯一差异为 `encoding.py` 中 `from ._encfast import`（位于 `try/except`，包内无 `_encfast`，落回纯 Python 编码，与 `native_encode=False` 口径一致；属既有打包细节，非本次迁移引入） |
| 运行验证 | ✅ 干净解包 `import main` exit 0；首个决策 `[1]`、连续决策 `[[0],[4],[5],[5],[6]]`；官方 harness 6 局正常完赛 |
| 旧产物 | ✅ 三个历史 tar、`agent/deck.csv`、所有 checkpoint 与 `runs/evaluation/**` 未被覆盖或删除 |

### 8. 未解决项与下一步

1. **引擎卡池比两份 CSV 更宽**：本机/官方 1.33.0 有 1431 张卡，库只覆盖 1–1267；ID 1268–1431 被夹到 UNK 行。
   选定牌组不受影响，但对手若使用这些新卡，策略会把它们当 UNK 处理。
   要消除该盲区需按引擎真值扩库（vocab 1268→1432、`card_emb` 增行），属需重训路径，未在本次范围内启动。
2. `cards.csv` 与引擎在 id 15 / id 979 两处版本差，以用户提供的 `cards.csv` 为权威（已归档差异证据）。
3. WSL2 发行版在会话中发生外部变更：默认用户变为 `ubuntu`、原 `/home/swcmb`（含旧 pylibs）消失、
   一度无法挂载 `ext4.vhdx`（`E_ACCESSDENIED`）。本次在 `~/pkmn-venv` 重建了 Linux 验证环境
   （get-pip → 上游 PyPI，因 Tsinghua 索引未返回包；torch 走 `download.pytorch.org/whl/cpu`）。
   旧 pylibs 内的历史证据文件仍在 Windows 侧报告里引用，路径已失效但不影响结论可复现性
   （新环境已复跑同类用例）。
4. Kaggle 线上镜像的 glibc/引擎版本一致性：本机 glibc 2.39 验证通过，线上实测需提交后确认。
5. 微调（0.5M 环境步）未启动，按任务要求等待用户明确决定。

### 9. 0.5M 环境步微调配置建议（未执行，等待用户决定）

前提已满足：数据库、模型兼容性、牌组合法性、部署运行、独立评估全部通过。
以下为**初步配置**，非执行记录。

```powershell
cd D:\pkmn-kaggle
$env:PYTHONPATH = "D:\pkmn-kaggle\.engine-shim"
$env:PYTHONUNBUFFERED = "1"
$env:PKMN_TURN_CAP = "200"

D:\ProgramFiles\deeplearning\python.exe rl\train_selfplay.py `
  --arch transformer2 --d-model 544 --nhead 8 --nlayers 5 `
  --init-from runs\train-20m-e16-1m\ckpt_999424.pt `
  --total-timesteps 500000 --num-envs 16 --update-epochs 1 --num-steps 64 `
  --lr 2e-5 --lr-schedule cosine --anneal-lr --gamma 0.997 --gae-lambda 0.95 `
  --num-minibatches 16 --clip-coef 0.2 --ent-coef 0.002 --vf-coef 0.5 --max-grad-norm 0.5 `
  --would-ko --decks top50 `
  --gate-pairs mirror --promote-winrate 0.6
```

理由与依据：

1. **配置延续性**：`ckpt_999424` 的采集参数是 `num_envs=16 / update_epochs=1 / num_steps=64 /
   lr=1e-4 cosine / gamma=0.997`（checkpoint `args` 实测），`--num-envs 16` 也是本机吞吐基准
   `runs/bench-20m/report.md` 的推荐配置（稳定段 88.095 env-step/s，显存峰值 6163/8151 MiB；
   E32 在第 31/40 次迭代 CUDA OOM，不可用）。
2. **步长与 lr**：0.5M 只做适配性微调，`lr=2e-5`（约 1/5 原值）+ cosine 退火，避免破坏已收敛的
   `card_emb` 行语义；未加 `--lr-restart`。
3. **预计耗时**：按实测 88 env-step/s 折算 500000/88 ≈ 5682 s ≈ **1.6 h 纯训练**；
   含启动、检查点与门禁对局，按 **2 h 机器时间**规划（该吞吐含 `would_ko` 引擎模拟）。
4. **牌组维度的已知缺口**：`--finetune-deck` 只接受 `rl/decks_train.py` 池内**牌组名**，
   而 `best_deck.csv` 的 60 张序列在 74 张池内牌组中**无同名同序列匹配**
   （最接近的是 `k06_dragapult_ex_fezandipiti_ex_latias_ex_me`，与本候选不同）。
   `--gate-decks-file` 接受的是池下标 JSON 列表，也不能直接指向牌组文件。
   因此要让微调真正“锁定并提交牌组一致”，需先二选一：
   （a）把 `best_deck.csv` 作为新命名条目加入 `TRAIN_TOP50` 池（一处数据改动，需你批准）；
   （b）直接用池内 `k06_dragapult…` 训练，并接受其与提交牌组的差异。
   本项未擅自修改池文件。
5. **卡池盲区优先**：引擎 1.33.0 有 1431 张卡而库只覆盖 1267 张，微调期间 ID 1268–1431 仍会被夹到 UNK。
   若目标是提升实盘对局质量，先扩库（并处理 `card_emb` 从 1269 行扩到 1432 行的权重迁移）比先微调更有价值；
   否则微调会继续学习一张“看不见的牌”。

验收标准（微调后必须逐项实测，不接受“训练完成”本身）：

| 门禁 | 阈值 |
|:---|:---|
| 自镜像（同牌组） | 胜率落在 45–55%，0 异常、0 非法动作 |
| vs random（`best_deck.csv`，mirror，≥120 局） | 不低于微调前 0.608 − 3pp |
| vs 未训练同配置网络 | ≥ 0.75 |
| vs 微调前 ckpt（held-out `pool_top50` cross） | ≥ 0.53（对应 `--cross-gate-delta 0.03` 的晋级裕度） |
| 非法动作率 / spawn 异常 | 0 / 0，且退出码 0 |
| 运行时回归 | `pytest tests/ -q` 全通过；WSL2 最终包用例全 exit 0 |

### 10. 需要用户决定的两项

1. 微调前是否先扩库到引擎真值 1431 卡（涉及 `card_emb` 权重迁移与重训代价）。
2. `best_deck.csv` 是否加入 `rl/decks_train.py` 的 `TRAIN_TOP50` 池，以便微调锁定提交牌组。
