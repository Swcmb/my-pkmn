# 卡牌库迁移最终报告

日期：2026-10-20　项目：`D:\pkmn-kaggle`
提交包：`submission_rl_transformer2_db_migrated.tar.gz`
SHA-256 `8d4ec3ccdebd563c259efa12bfe1061493dcaff7c1f0154fcfff6f128d58ae71`　大小 75,126,979 B

导出命令由用户本人执行（AI 未运行、未间接运行）；本报告汇总其后所有验证。

## 1. 原数据库备份

| 项 | 值 |
|:---|:---|
| 原权威库 | `D:\pkmn-kaggle\EN_Card_Data.csv`（赛事导出，一招式一行，17 列） |
| 原库 SHA-256 | `aca07ac210be6fa23297174e4855eb900a54e267fa73b31e1397bf84bed89dc5` |
| 备份位置 | `runs/card_database_migration/20261020/original_cards.csv` |
| 备份哈希 | 与原库逐字节一致 ✅ |
| 引擎内部库 | 独立于 CSV：`sdk_cg.api.all_card_data()` 由官方 CABT 二进制提供（1431 卡 / 1755 招式） |
| 派生数据 | `rl/card_features.py` 运行时特征矩阵（缓存不落盘）、`rl/buff_data.py`、`rl/effect_data.py`（按依赖单独评估，未盲目覆盖） |

## 2. 新数据库

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

## 3. 新旧 schema 与 ID 语义差异

- ID 集合：`ids_equal_as_sets = true`，新增 0、删除 0；同 ID 卡名 **1267/1267 一致** → ID 是稳定语义键，未重新分配。
- 与本机引擎（1.33.0，1431 卡）：共享 1267 个 ID 中 1265 张逐字段完全一致，仅 2 处版本差
  （id 15 `Team Rocket's Energy` 能量枚举；id 979 `Koraidon ex` 招式表）。
  引擎另有 164 张 ID 1268–1431 的新卡不在任何一份 CSV 中（140 宝可梦 / 8 支援者 / 6 物品 / 5 特殊能量 / 3 竞技场 / 2 道具）。
- 字段差异分类（旧→新）：表示差异为主 —— HP `n/a→0` 206、Retreat `n/a→0` 246、能量列缺失→`COLORLESS` 191、
  变动伤害 `20×/n/a→基值` 333；真实语义差异 —— 能量类型 39、进化前名称 5、招式条数 250（含 `Tera` 等 0 伤招式是否计入）。
- 特征矩阵实际位移：501/1268 张卡的向量有变，只落在 dim12 `type_C`(195)、dim22 `type_A`(2)、
  dim45 `n_attacks`(75)、dim46 `max_dmg`(250)；SPECIAL_TAGS 维 38–41 **零变化**。

## 4. 旧 checkpoint 兼容性结论

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

## 5. 牌组验证与独立评估

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

## 6. Windows 与 WSL2 真实运行验证

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

## 7. 提交包内容与一致性

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

## 8. 未解决项与下一步

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

## 9. 0.5M 环境步微调配置建议（未执行，等待用户决定）

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

## 10. 需要用户决定的两项

1. 微调前是否先扩库到引擎真值 1431 卡（涉及 `card_emb` 权重迁移与重训代价）。
2. `best_deck.csv` 是否加入 `rl/decks_train.py` 的 `TRAIN_TOP50` 池，以便微调锁定提交牌组。
