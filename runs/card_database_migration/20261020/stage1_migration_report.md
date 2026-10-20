# 卡牌库迁移 · 阶段 1 报告（备份 / 校验 / schema 与 ID 对比）

日期：2026-10-20　执行目录：`runs/card_database_migration/20261020/`

## 1. 权威卡牌库定位

| 类别 | 路径 | 角色 |
|:---|:---|:---|
| 原始卡牌数据（权威库） | `D:\pkmn-kaggle\EN_Card_Data.csv` | `rl/card_features.py` `_build()` 唯一 CSV 入口；`rl/decks.py`、`scripts/deck_search.py`、`scripts/deck_rank_blend.py`、`scripts/scout_subs.py`、`scripts/export_rl_submission.py --csv` 均读取它；提交包内副本由导出脚本 `shutil.copy(args.csv, …/EN_Card_Data.csv)` 生成 |
| 派生数据 | `rl/card_features.py` 运行时缓存的 59 维特征矩阵（`lru_cache`，不落盘）；`rl/buff_data.py`（由 `all_attack()` + `EN_Card_Data.csv` 生成的静态表，随代码提交）；`rl/effect_data.py`（招式/特性 multi-hot） | 需在库变更后按依赖关系单独处理 |
| 引擎内部卡牌库 | `kaggle_environments/envs/cabt/cg/{cg.dll,libcg.so,…}` 内嵌，经 `sdk_cg.api.all_card_data()` / `all_attack()` 暴露 | 不在仓库内，是运行时真值来源；观测中的 cardId 由它定义 |
| 提交牌组 | `agent/deck.csv` | 60 行 cardId，**不是**卡牌库 |

## 2. 备份与新库落地校验（SHA-256 全部一致）

| 文件 | SHA-256 |
|:---|:---|
| 原库 `EN_Card_Data.csv` | `aca07ac210be6fa23297174e4855eb900a54e267fa73b31e1397bf84bed89dc5` |
| 备份 `runs/card_database_migration/20261020/original_cards.csv` | `aca07ac2…bed89dc5`（与原库逐字节一致 ✅） |
| 新源 `E:\Files\Downloads\cards.csv` | `a198d909eae4d04d7fdac07cd14bf5a8d870015e3356d1c286071ff76264a23b` |
| 副本 `…/downloaded_cards.csv` | `a198d909…64a23b`（与源一致 ✅） |

## 3. 新库自身质量校验（`compare_databases.py`）

- 编码：UTF-8 **带 BOM**（原库无 BOM）；读取需 `encoding="utf-8-sig"`。
- 行数 1268（含表头）→ 真实卡牌 **1267** 张；`cardId` 全部为整数、无空值、**无重复**、不可解析行 0。
- 列 21 个：`cardId,name,cardType,cardTypeName,pokemonType,energyType,evolutionType,hp,retreatCost,weakness,resistance,basic,stage1,stage2,ex,megaEx,tera,aceSpec,evolvesFrom,attacks,skills`。
- 布局：一卡一行，招式打包为 `attackId:名称(伤害)[能量索引,…]`，特性打包在 `skills`。
- 该 schema 与 `sdk_cg/api.py` 的 `CardData` dataclass 字段一一对应 → 判定为**引擎格式导出**。

## 4. 新旧 ID 映射（`id_mapping_comparison.json`）

| 指标 | 结果 |
|:---|:---|
| 旧库唯一卡 ID | 1267（范围 1–1267） |
| 新库唯一卡 ID | 1267（范围 1–1267） |
| ID 集合是否相同 | **完全相同**（新增 0 / 删除 0） |
| 同 ID 卡名是否一致 | **1267/1267 一致**（无重编号、无名称漂移） |

结论：ID 是稳定语义键，未发生重新分配。

## 5. 字段级差异（`schema_comparison.json`、`field_diff_details`、`classify_field_diffs.py`）

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

## 6. 三方对比：新库 vs 本机引擎 1.33.0（`engine_vs_csv_comparison.json`）

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

## 7. 关键阻塞与结论

1. **直接覆盖会使权威库失效（已实证）**：新 CSV 缺少读取契约要求的列，用 `downloaded_cards.csv` 直接覆盖 `EN_Card_Data.csv` 后：
   - `rl/card_features.CardTable(NEW)` → `KeyError: 'Card ID'`
   - `rl/decks.py` 风格的 `csv.DictReader(...)['Card ID']` → `KeyError: 'Card ID'`
   - 结果：训练入口、Agent 编码器、导出脚本全部崩溃。按任务 §4.5「校验失败则保留原数据库」，**原库当前保持不变**（哈希仍为 `aca07ac2…`）。
2. **checkpoint 兼容性（初判，待阶段 2 回归验证）**：`ckpt_999424.pt`（step 999424，sha `b8a3be1d247eac45af0bbc38a207336d6f702c82fd0b19e5f787cc7addf08a1e`）与 `ckpt_500736.pt`（step 500736，sha `b02b0724ebbb90916e9d495cbc4ab73b30f6bb761cea993366798942b5477a3d`）`net_config` 相同（`transformer2, d_model=544, nhead=8, nlayers=5, ff=256, static=false, would_ko=true, value_atoms=51`），`card_emb.weight = (1269, 544)` = `vocab_size(1268)+1`，其中 0=PAD、1268=UNK。由于 ID↔卡名 1:1 未变，`card_emb` 行语义保持；`would_ko` 三列来自引擎观测而非 CSV，不受库替换影响。风险点集中在 `static=false` 路径之外的 CSV 派生输入（`max_dmg` 变动伤害、`tera` 标记、`effect_data`/`buff_data` 生成表）。
3. **引擎二进制当前缺失（阻塞级，新出现）**：本次会话开始时 `C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim\…\cg.dll` 可用（阶段 2 引擎导出即由它完成），随后该临时目录被清理，当前 `Temp/opencode` 已不存在；WSL2 侧 `python3` 无 `kaggle_environments`，但历史验证目录 `/home/swcmb/pkmn-linux-verify-20261020/pylibs/…/libcg.so` 仍在。§6.2「必须用真实 CABT 引擎开局校验」、§6.3 交叉评估、§7 WSL2 运行验证、以及导出命令都依赖引擎，需要先恢复（在 conda 环境 `pip install --no-deps kaggle-environments==1.33.0`，或重建 shim）。
4. 旧产物全部保留：未改动 `agent/deck.csv`、`runs/evaluation/**`、三个历史提交目录与 `submission_rl_transformer2*.tar.gz`、未触碰任何 checkpoint。

## 8. 需要你决定的分叉（阻塞后续阶段）

“替换本地卡牌库”在证据下有三种可执行语义，代价与风险不同，见对话中的选项。产出文件：
`original_cards.csv`、`downloaded_cards.csv`、`schema_comparison.json`、`id_mapping_comparison.json`、`engine_all_card_data.json`、`engine_all_attack.json`、`engine_vs_csv_comparison.json`、`compare_databases.py`、`compare_three_way.py`、`classify_field_diffs.py`。

## 9. 执行结果（当日更新：方案 A 已落地，分叉已关闭）

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
