# 阶段 0–2 审计与基线复核报告

日期：2026-10-20　执行者：Qoder（本会话）　范围：环境与历史产物、牌组/Agent/沙箱、模型基线复核

所有结论标注为**已执行并验证**、**已执行但受限**或**未执行**。原始产物落在
`runs/evaluation/session-20261020/`，未覆盖任何既有文件。

---

## 1. 阶段 0：环境与历史产物（已执行并验证）

### 1.1 Git 与工作区

- 分支 `my-pkmn`，HEAD `bb53302`。
- **工作区有未提交修改，且这些修改是训练与运行所必需的**（本会话逐条读过 diff）：

| 文件 | 改动性质 | 证据 |
|:---|:---|:---|
| `rl/gates.py` | 补 `_field_candidates()`（原缺失）；`_gate_env` 的 `native_encode` 改为按可用性回退 | `tests/test_gates.py` 新增 2 项覆盖该函数 |
| `rl/train_selfplay.py` | 补 `DECKS`/`load_deck`/`GENERATED` 导入 | 缺失时模块导入即失败 |
| `sdk_cg/sim.py` | 引擎加载改为进程内唯一初始化，规避 `buffer full. capacity:7` 原生 abort | 包内 `sdk_cg/sim.py` 已含该实现 |
| `EN_Card_Data.csv` | 卡池迁移（3869 行差异） | 迁移证据链在 `runs/card_database_migration/20261020/` |
| `.gitignore` | 加入 `.engine-shim` 等 | — |

- 未跟踪新增：`agent/`、`Prompt/`、`rules/`、`AGENT.md`、`scripts/deck_search.py`、`runs/{bench-rtx5060,card_database_migration,evaluation,train-20m-e16-1m}/`。
- 本会话**未执行任何 git commit / push / 破坏性命令**。

### 1.2 卡牌库

| 项 | 值 |
|:---|:---|
| 真实路径 | `D:\pkmn-kaggle\EN_Card_Data.csv`（`cards.csv` 在本仓库即此文件） |
| SHA-256 | `05aa569e3003a838b8503a9d928795e5aea7407ab19dd9f3d9c4b1293b6bb4b4` |
| 大小 / 行数 | 170,873 B / 1,766 数据行（一卡多招式行） |
| 唯一 `Card ID` | **1267**（与比赛固定卡池 1267 张一致），无重复 |
| 编码字段 | 17 列英文卡表（`Card ID`/`Card Name`/`Stage (Pokémon)/Type (Energy and Trainer)`/`Rule`/`Category`/…） |
| 训练侧解析 | `rl.card_features.CardTable` → vocab_size 1268 / feat_dim 59 |
| 提交包内副本 | `submission_rl_transformer2_db_migrated/EN_Card_Data.csv` 与权威库**逐字节相同** |

> 说明：任务书称"平台卡牌库与本地 `cards.csv` 已由用户确认一致"。本会话按"以实际文件与哈希为准"独立复核，未沿用该结论；上述哈希与 `final_gate_report.md`、`stage5_gates.json` 记录的库哈希一致。

### 1.3 历史产物核实（三个待核对路径全部命中）

| 路径 | 存在 | 核实结果 |
|:---|:---|:---|
| `runs\train-20m-e16-1m\ckpt_999424.pt` | ✅ | SHA `b8a3be1d247e…`，81,249,427 B，`global_step=999424`，`net_config={transformer2, d_model 544, nhead 8, nlayers 5, ff 256, would_ko True}`，参数量 20,303,715 |
| `runs\train-20m-e16-1m\ckpt_500736.pt` | ✅ | SHA `b02b0724ebbb…`，同一 `net_config`（旧基线） |
| `runs\evaluation\card_database_migration\best_deck.csv` | ✅ | 60 行，SHA `30c8c7365c75f38f…` |
| `submission_rl_transformer2_db_migrated.tar.gz` | ✅ | 75,126,979 B，SHA-256 `8d4ec3ccdebd563c259efa12bfe1061493dcaff7c1f0154fcfff6f128d58ae71` |

`runs/train-20m-e16-1m/` 内另有 `latest.pt`（243,747,737 B，完整恢复点）与两份 `… - 副本.pt.txt`（人工备份副本，未被本会话触碰）。

**提交包内容一致性（已执行）**：`model.pt` 的 88 个权重张量与 `ckpt_999424.pt` **逐位相同**，`net_config`/`global_step` 一致；包内 `deck.csv` 与 `best_deck.csv` SHA 相同；`tar -tzf` 清单含 `main.py`、`deck.csv`、`model.pt`、`EN_Card_Data.csv`、8 个依赖 `.py` 与 `sdk_cg/`，**不含** `cg.dll/libcg.so`（未再分发专有引擎，符合门禁 9）。

### 1.4 资源

| 项 | 实测 |
|:---|:---|
| GPU | NVIDIA GeForce RTX 5060 Laptop，8,151 MiB（审计时占用 46 MiB） |
| 运行时 | `D:\ProgramFiles\deeplearning\python.exe` = Python 3.12 / PyTorch **2.14.0+cu130**，`torch.cuda.is_available()=True` |
| D 盘 | 398 GB 总，225 GB 可用（44% 已用） |
| 引擎 | `.engine-shim`（官方 `kaggle_environments==1.33.0` 包副本，含 `cg.dll`），**必须**加入 `PYTHONPATH` |
| 环境缺口 | `deeplearning` 环境**无 pandas、无 psutil、未安装 kaggle_environments** |

> 关键操作结论：`pytest tests -q` 不带 shim 时 **18 failed / 30 passed**（全部 `ModuleNotFoundError: No module named 'kaggle_environments'`）；
> 带 `PYTHONPATH=D:\pkmn-kaggle\.engine-shim;D:\pkmn-kaggle` 后 **48 passed**。任何评测/训练命令都必须设置该 `PYTHONPATH`。

---

## 2. 阶段 1：牌组、Agent 接口与沙箱（已执行并验证）

### 2.1 牌组合法性（独立复算，脚本 `validate_deck.py`）

按 `AGENT.md` §九 的五条规则重算，未复用既有报告结论：

| 牌组 | 60 张 | ID 全在卡池 | ≥1 真正基础宝可梦 | 同名≤4（基础能量豁免） | ACE SPEC≤1 | 判定 |
|:---|:---|:---|:---|:---|:---|:---|
| `runs/evaluation/card_database_migration/best_deck.csv` | 60 | 22 个唯一 ID 全命中 | 9 张 / 5 种 | 最大 4 | 1 | **PASS** |
| `submission_rl_transformer2_db_migrated/deck.csv` | 60 | 同上 | 9 张 / 5 种 | 最大 4 | 1 | **PASS** |
| `agent/deck.csv`（旧提交牌组） | 60 | 11 个唯一 ID 全命中 | 6 张 / 2 种 | 最大 33（基础能量豁免生效） | 1 | **PASS** |

- 化石道具卡（1099/1136/1138/1150/1151）在三份牌组中均未出现，未计入基础宝可梦。

#### 以平台权威卡表复核（已执行，`validate_deck_platform.py`）

阶段 1 要求"所有 ID 均存在于**比赛卡牌库**"。本地 `EN_Card_Data.csv` 是英文卡表 schema
（`Card ID / Stage (Pokémon)/Type… / Rule`），规则实际依赖的字段是 `cardType / basic / aceSpec / name`，
因此本会话直接取平台的 `GET /api/cards/csv`（无需登录，HTTP 200）作为权威源重跑校验：

| 项 | 值 |
|:---|:---|
| 平台卡池 | 1,267 张，`cardId` 连续 1–1267，UTF-8 BOM |
| 平台 CSV SHA-256 | `a198d909eae4d04d7fdac07cd14bf5a8d870015e3356d1c286071ff76264a23b` |
| 与迁移源比对 | **与 `runs/card_database_migration/20261020/downloaded_cards.csv` 逐字节相同** → 卡池自迁移以来未变动 |
| 三份牌组判定 | 按平台字段重算：全部 **PASS**，与本地 schema 的结论完全一致（含 `agent/deck.csv` 的 33 张同名基础能量豁免） |
| 两库字段级交叉比对 | **分歧 0 条**；`name` / 基础宝可梦判定（`cardType=0∧basic=1` vs `Stage="Basic Pokémon"`）/ ACE SPEC 判定（`aceSpec=1` vs `Rule="ACE SPEC"`）三项在全部 1,267 张上一致；仅本地有的 ID 0 个，仅平台有的 ID 0 个 |

→ 任务书"平台卡牌库与本地 `cards.csv` 一致"这句，本会话在**字段级**独立验证成立；牌组合法性不是基于本地代理库的推断结论。
本地/平台 ID 集合都是 1–1267，而本机引擎 1.33.0 实际有 1,431 张，超出的 164 张属卡池外，编码器夹到 UNK 行（既有边界，选定牌组不受影响）。
- `best_deck.csv` 的 60 张与 `TRAIN_TOP50['k06_dragapult_ex_fezandipiti_ex_latias_ex_me']` 是**同一多重集合**（行序不同）。→ 任务书要求的"训练牌组池包含最终牌组"成立。

### 2.2 Agent 接口（已执行审计）

签名与兼容：`def agent(obs)` 满足 `agent(obs_dict: dict) -> list[int]`；`select is None` 时返回 `DECK` 并重置 `GameTracker/AbilityTracker`（官方 Kaggle 开局取牌路径）。

下标与数量安全性来自 `encoding.build_mask()`（包内 L125-148）：

- 已选项被排除（`if i not in picked`）→ `sorted(set(picked))` 不会因重复而缩短；
- `submit` 仅在 `minCount ≤ len(picked) < maxCount` 时合法 → 返回长度必然落在 `[minCount, maxCount]`；
- 掩码全零时兜底放行 `submit`（防死锁）；
- `PLAY` 用 `index`、`ATTACK` 用 `attackId`、`ATTACH/EVOLVE` 用 `area+index / inPlayArea+inPlayIndex` 的定位方式与 `AGENT.md` §六 一致；
- 决策路径只读己方 `hand`，对手隐藏信息不可见（编码层按 `AGENT.md` §四 可见性构造）。

#### 隐藏信息访问审计（任务书 §四"不访问对手隐藏手牌"，已验证）

全包内 `get("hand")` 只有 3 处，逐处定位到玩家索引：

| 位置 | 读取对象 | 判定 |
|:---|:---|:---|
| `encoding.py:1290` | `mp`（己方）→ `self_hand_id/mask`；注释明确 "opp hidden -> handCount UNK tokens" | ✅ 仅己方 |
| `encoding.py:450` | `mp`（己方，`decklist_drawable_flags(decklist, s, mp)` 形参）区内已见卡牌计数 | ✅ 仅己方 |
| `search_agent.py:61-62` | 受 `with_hand` 开关控制：`:124` 对己方 `with_hand=True`，`:100` 对对手 `_observed(op, …, 1-me, with_hand=False)` | ✅ 对手分支显式关闭 |

对手手牌在观测中的唯一来源是**公开信息的信念推断**：`hand_beliefs_for()` 只收录
`zone==HAND` 的揭示日志（`AGENT.md` §五 的 MOVE_CARD / DRAW 等公开事件），
再用 UNK 令牌补齐到 `handCount`；并有 `_enforce_hand_count()` 保证"已知在手信念数 ≤ 实际手牌数"的不变量。
→ 不存在读取 `players[1-yourIndex]["hand"]` 的路径，防作弊要求满足。

#### SelectType / SelectContext / OptionType 覆盖（静态审计，本会话完成）

| 维度 | 覆盖情况 | 证据 |
|:---|:---|:---|
| OptionType 0..16（17 种） | **全部覆盖**：`N_OPT_TYPES = 17` 给每种一个 one-hot 槽；`_pyopt()` 显式映射带字段的 13 种（0,3,4,5,6,7,8/9,10/11,13,15,16），其余 4 种（1 YES / 2 NO / 12 RETREAT / 14 END）在 `AGENT.md` §六 中本就无附加字段，落到只带 `type` 的默认分支是正确行为 | `enc_constants.py:27`、`encoding.py:150-165` |
| SelectType | 作为标量特征 `select_type` 进入编码，容量 `N_SELECT_TYPES` | `encoding.py:99,945,1001` |
| SelectContext（0..48，49 种） | 作为标量特征 `select_context` 进入编码；`N_SELECT_CTX = 64` **过量分配**，`min(context, 63)` 的钳制在 0..48 上永不触发 → 无上下文坍缩损失 | `enc_constants.py:30`、`encoding.py:1231` |
| `contextCard` / `effect` 来源卡 | 各占 2 个槽并带 `effect_mask`，缺位时掩码置 0 | `encoding.py:946,1235` |

结论：Agent 是学习型策略，"覆盖"体现在**每种类型都有特征通道 + 类型无关的合法掩码**，
而不是逐 `SelectType` 的手写分支；上表逐项确认没有类型落到未处理路径。

### 2.3 测试套件（已执行）

```powershell
cd D:\pkmn-kaggle
$env:PYTHONPATH = "D:\pkmn-kaggle\.engine-shim;D:\pkmn-kaggle"
D:\ProgramFiles\deeplearning\python.exe -m pytest tests -q
# → 48 passed in 20.82s
```

覆盖：gates 调度、selfplay 环境一致性、state truncation、varlen attention、v21/v22 编码、chunked update、finetune 环境、matchup 日志、value diag、tracker fuzz。

### 2.4 官方等价 smoke test（已执行，脚本 `subsmoke_timing.py`）

以平台方式 `exec_module` 包内 `main.py`，驱动真实 CABT 引擎，逐决策计时与逐项校验返回值：

| 口径 | 决策数 | 均值 | p50 | p99 | 最大 | >100ms 次数 | 单局最长 | 峰值 RSS | import |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| torch 默认线程 | 385 | 33.1ms | 35.5ms | 74.1ms | **105.1ms** | **2** | 2.66s | 722 MB | 1.53s |
| `torch.set_num_threads(2)`（=平台 CPU 上限） | 289 | — | 36.2ms | 70.9ms | 78.8ms | **0** | — | 721 MB | — |

- 返回值：0 非法长度、0 非法下标、0 重复下标、0 异常、0 平局；8/12/8 局全部完赛。
- **内存 2GB**：722 MB，通过（含 torch + 引擎）。
- **单局 10 分钟**：单局最长 2.66s，通过（余量极大）。
- **单步 100ms：条件通过**。线程数不受限时 385 次决策里有 2 次越界（105.1 / 104.4ms）；把 torch 线程钉到 2（与平台 2 核配额一致）后 0 次越界，最大 78.8ms。
  → 判定：越界来自线程超订，平台 2 核环境不会触发；但 p99 已达预算的 71%，**长尾风险真实存在**。
- **测量条件caveat（重要）**：本会话所有时延测量都与评测作业并发，机器非空闲，因此
  **绝对值随负载漂移，只有同条件下的相对比较有效**。对照证据：同机同包在评测负载下
  p50 约 35ms（默认线程）与 36ms（钉 2 线程），而补测满载时同一包 p50 冲到 120ms、
  284 次决策中 204 次 >100ms —— 那是并发作业的干扰，不是补丁的回归。
  → 待补测作业全部结束后，在空闲机器上按"未打补丁 → 已打补丁"同条件各测 12 局，
  才能给出可用于决策的 100ms 结论。补丁本身已备好并在副本上试跑通过（返回长度/下标/异常全部干净）。
- **`torch.load(weights_only=True)` 兼容性（已验证）**：包内 `model.pt` 在 `weights_only=True`
  下加载成功（`net` / `net_config` / `global_step` 全部可读），因此平台若使用 torch ≥2.6
  （该版本起 `torch.load` 默认切到 `weights_only=True`）不会在 import 阶段崩溃。

### 2.5 后端选型探针：greedy vs MCTS（阶段 2 第 5 项"不同策略对手"）

`scripts/export_rl_submission.py` 支持 `--backend mcts2`（决策时 PUCT 搜索）。选型先决条件是官方
单步 100ms 预算，而非棋力，故做**配对计时**：同一批真实决策的深拷贝上分别跑两种后端
（`latency_backend_probe.py`，`torch.set_num_threads(2)` 对齐平台 2 核，`n_sims=40 / n_det=2`）：

| 后端 | 配对样本 | 均值 | p50 | p95 | max | >100ms |
|:---|---:|---:|---:|---:|---:|---:|
| greedy（现提交实现） | 5 | 60.3ms | 44.3 | 73.4 | 103.1 | 1/5 |
| MCTS 40×2 | 5 | **2,504.9ms** | 40.9 | 5,577.7 | **6,835.3** | 2/5 |

结论：**MCTS 后端在 100ms 预算下不可用**（均值超出约 25 倍；MCTS 的 p50 很低是因为多数决策落在
`minCount==maxCount` 的强制选择上、搜索被短路，真正需要搜索的决策直接数秒）。
→ 最终提交必须保持 `--backend transformer2`（贪心），本会话未测 MCTS 的棋力增益——在时延已判死的情况下无决策价值。
样本仅 5 个配对决策，量级差距（2.5s vs 0.1s）远大于抽样噪声，结论对该抽样稳健。

> 过程记录：`scripts/mcts_h2h_shard.py` 依赖 `kaggle_environments.make()`，而 `.engine-shim` 的
> `kaggle_environments/__init__.py` 是 0 字节空桩（为绕开 jsonschema 等重依赖故意降级），
> 因此该脚本无法用现有 shim 运行；本探针改为直接 `sdk_cg.game` 驱动，与提交包的运行时路径一致。



### 2.6 Linux 官方等价环境的引擎侧验证（本会话独立执行）

上一会话的 WSL 依赖目录 `~/pkmn-linux-verify-20261020/pylibs` 已不存在（Ubuntu-24.04 发行版仍在，
系统 python3 无 pip/torch/kaggle_environments），因此无法原样复跑官方 harness。
`scripts/export_rl_submission.py` 打包的 `sdk_cg/sim.py` + `game.py` 只依赖 stdlib（ctypes/json/os/platform/sys），
所以本会话用 shim 内的 `libcg.so` 做了一次**不需要 torch 的 Linux 引擎侧验证**
（`linux_engine_smoke.py`，导出包先复制到 WSL 本地临时目录，仓库内的包未被改动）：

| 项 | 结果 |
|:---|:---|
| 加载的引擎 | `/home/ubuntu/pkmn-linux-engine-smoke-20261020/pkg/sdk_cg/libcg.so` |
| 提交牌组开局 | 6/6 局 `errorType=0`、`errorPlayer=-1` → **Linux 真实引擎接受这 60 张** |
| 对局完成 | 6/6 局终局（`deck_out` ×5、`prize` ×1），回合数 25–51，共 1,127 次 `battle_select` |
| 驱动异常 | 0 |
| 退化 select（`option` 为空且 `minCount≥1`） | **0 次**（上一会话报告过该状态会令手写驱动抛 `IndexError`；本次 6 局未触发，属未覆盖边界） |

→ 与 `linux_runtime_report.md`（官方 harness 6 用例全 exit 0）方向一致，且补充了本会话自己的 Linux 证据。
模型推理侧的 Linux 验证仍需一次 torch 安装（约 1 GB），见 §5 待授权项。

### 2.7 易失依赖指纹固化

`.engine-shim` 在 `.gitignore` 内、且历史上发生过同类临时 shim 目录被系统清理（`pre_export_checklist.md` §与任务书原文命令的唯一差异），
故本会话把引擎指纹落盘：`engine_shim_inventory.json`（1,390 个条目；
`cg.dll` 1,659,392 B `e084185a17fa1487…`、`libcg.so` 1,458,504 B `c699020e086eede1…`、
`libcg-arm64.so` `95628f0bd20ac5e6…`、`libcg.dylib` `da53cb331277cc8a…`、`sim.py` `d39837f069cc7fee…`）。
下次会话可直接比对以判断是否漂移或被替换。

### 2.8 评测数据独立性修正（涉及阶段 3 命令）

`runs/evaluation/pool_top50.json`（74 副，SHA `f97624da164b…`）由 `runs/evaluation/build_pool.py` 从
`TRAIN_TOP50` 生成，既是阶段 2 门禁的 held-out 交叉池，也等于训练侧 `--decks top50` 的池。
把它直接当微调对手池会让 held-out 身份失效。本会话用 `split_eval_pool.py`（stride-7 置换，7 与 74 互质）
确定性切成两个互斥子集，并集为原池：

- `opp_pool_train30.json`（30 副，SHA `8d7dd1e1cf52…`）→ 训练对手池；
- `eval_pool_heldout44.json`（44 副，SHA `5c225480c02c…`）→ 阶段 4 广度评测用真实 held-out。

两个文件都已用 `rl.decks_ladder.load_ladder_decks` 实际读取验证（decks/counts 形状正确、weights 归一、全部 60 张）。
`TRAIN_TOP50` 中与提交牌组同多重集合的 `k06_dragapult…` 落在 **held-out 侧**。



---

## 3. 阶段 2：模型评测基线复核（已执行并验证原始数据）

### 3.1 复核方式

本会话未重复烧 840 局，而是把 `final_gate_report.md` 的每个数字回溯到原始记录
`runs/card_database_migration/20261020/stage5_gates.json`（32,476 B，2026-10-20 16:06）逐条比对：
`library.sha256` 与权威库一致、`protocol.games_per_cell=60`、`seeds=[20261020,20261021]`、`runs` 共 14 cell、
`sum(games)=840`、每 cell 的 `wins/losses/decided` 与报告表格逐格吻合（如 34/60=0.567、50/60=0.833、0/60=0.000）。
**该 840 局矩阵是本会话采用的基线口径**；其统一条件（同库、同 harness、`PKMN_TURN_CAP=200`、座位逐局交替、0 平局/0 异常/0 非法）也记录在同一 JSON 内。

### 3.2 历史 4 个数字的复核结论

| 任务书历史数字 | 本次同协议复现 | 判定 |
|:---|:---|:---|
| 对随机 73/120 | Dragapult 牌组两种子合并 **73/120 = 0.608**（34+39） | **复现** |
| 对同配置未训练 100/120 | 同一初始化 50/60 = 0.833；换两个新初始化是 24/60 = 0.400、27/60 = 0.450；跨 3 初始化合并 101/180 = **0.561 [0.488, 0.632]** | **无法复现**：该臂由对手的随机初始化主导，合并后 CI 跨 0.5，已不作为强度证据（§3.4 修正 2） |
| 自对弈镜像 61/120 | **61/120 = 0.508**（34+27） | **复现**，且镜像贴近 0.5 说明无座位/牌组泄漏 |
| 对旧模型 61/120 | **提交牌组纯镜像合并 85/120 = 0.708 [0.622, 0.782]**；混合池 cross 合并 61/120 = 0.508 | 历史数字对应的是混合池口径；同牌组镜像口径下优势明确且统计显著（见 §3.4 修正 1） |

### 3.3 合并后的最终基线（`merge_baseline.py` → `stage2_baseline_merged.json`）

既有 840 局（种子 20261020/21）与本会话补测（种子 20261022/23）协议完全一致、种子互不重叠，按臂合并计数：

| 臂 | 胜/判 | 胜率 | Wilson 95% CI | 异常/非法/截断 | ≥120 局 |
|:---|---:|---:|:---|:---:|:---:|
| 候选 vs 随机合法（提交牌组） | 73/120 | **0.608** | [0.519, 0.691] | 0 | ✅ |
| 旧基线 vs 随机合法（提交牌组） | 69/120 | 0.575 | [0.486, 0.660] | 0 | ✅ |
| 候选 vs 随机合法（旧 `agent/deck.csv`） | 28/120 | 0.233 | [0.167, 0.317] | 0 | ✅ |
| 旧基线 vs 随机合法（旧 `agent/deck.csv`） | 0/120 | 0.000 | [0.000, 0.031] | 0 | ✅ |
| **候选 vs 旧基线：提交牌组纯镜像** | **85/120** | **0.708** | **[0.622, 0.782]** | 0 | ✅ |
| 候选 vs 旧基线：旧 `agent/deck.csv` 纯镜像 | 17/60 | 0.283 | [0.185, 0.408] | 0 | 补测中 |
| 候选 vs 旧基线：74 副混合池交叉 | 61/120 | 0.508 | [0.420, 0.596] | 0 | ✅ |
| 候选 vs 同配置未训练（提交牌组，3 个初始化） | 101/180 | 0.561 | [0.488, 0.632] | 0 | ✅ |
| 候选 vs 同配置未训练（旧 agent 牌组，2 个初始化） | 63/120 | 0.525 | [0.436, 0.612] | 0 | ✅ |
| 候选自镜像（提交牌组） | 61/120 | 0.508 | [0.420, 0.596] | 0 | ✅ |
| 候选自镜像（旧 agent 牌组） | 58/120 | 0.483 | [0.396, 0.572] | 0 | ✅ |
| 旧基线自镜像 / 旧基线 vs 未训练（提交牌组） | 32/60 各 | 0.533 | [0.409, 0.654] | 0 | 补测中 |

全部臂合计 0 异常、0 非法动作、0 turn_cap 截断。

### 3.4 由数据得出的判断（含对上一版结论的两处修正）

1. **修正：候选对旧基线在提交牌组上有显著优势。**
   同牌组纯镜像（双方都驾驶 `best_deck.csv`，牌组强度逐局抵消，纯 net-vs-net）合并 **0.708，CI 下界 0.622**，
   两个种子分别是 0.683 与 0.733，方向一致。上一版依据的是 `cand_vs_baseline_heldout`（74 副混合池 cross）
   的 0.508 —— 那条臂里两个模型各自驾驶**不同的**池内牌组，牌组强度差与模型强度差混在一起，把优势稀释掉了。
   → 正确表述：**提升是牌组特异的——在提交牌组上明确存在（0.708），在混合牌组场上不显著（0.508）。**
   这既支持"对提交牌组做专项微调"（阶段 3 方案 A），也确认阶段 4 的 G7 主门禁用同牌组镜像是正确选择。

2. **修正：`vs 未训练` 不构成模型强度证据。**
   单初始化时该臂给出 0.833 / 0.400 / 0.450 三个互相矛盾的点值（对手网络由 `torch.manual_seed(seed)`
   逐个随机初始化，结果被"抽到哪张随机网"主导）；跨 3 个初始化合并后 **0.561，CI [0.488, 0.632] 跨 0.5**。
   → 从强度证据降级为口径 sanity；`phase4_gate.py` 的 G6 已改为"跨 ≥2 初始化合并 + Wilson 下界"，
   并把每个初始化的点值写进 `per_init_wr`。

3. **牌组效应仍然巨大**：同一候选换牌组，vs random 从 0.233 跳到 0.608；旧基线用旧牌组 0/120 全败。
   `agent/deck.csv` 是最弱牌组，已被 `best_deck.csv` 取代——换牌组本身就是一次大幅强度提升。

4. **评测口径无泄漏**：候选自镜像 0.508 [0.420, 0.596]、旧基线自镜像 0.533，均落在镜像应有的 0.5 附近。

### 3.5 真实赛场强度参照（阶段 2 第 5 项"强基线"，取自平台公开接口）

本地没有任何真实对手 Agent 可用，`--opponent` 只能给出 random / untrained / 旧 ckpt 三种代理。
本会话从平台**无需登录**的 `/api/leaderboard/a` 取回真实对局统计（`platform_leaderboard_a.json`）：

| 项 | 值 |
|:---|:---|
| 参赛队伍 | 38 支；已发布 33 轮 |
| 最新一轮（第 33 轮） | 32 支参赛，每队 31 局（单循环，与赛制一致） |
| 第 33 轮单轮前 5 | 咪咪汪汪队 27胜4负 = 0.871、对对对队 0.839、hetui 0.806、不说对不队 0.806、格鲁 0.774 |
| 单轮分布 | 中位 0.500（单循环对称所致）、下四分位 0.290、该轮最低 0.065 |
| **累计 33 轮（可比的强度序）** | 积分第 1「食堂不要插队」2,918 分 = **0.797**（745胜182负8平）；格鲁 0.763、对对对队 0.779(31轮)、咪咪汪汪队 0.788、不说对不队 0.738；全 38 队区间 **0.797 → 0.120** |
| 单轮极值 | 有队两轮打出 27/27 = 1.000，但只参与 2 轮，不作参考 |

**口径警告**：榜单胜率是**真实混合强度对手池**下的成绩（含大量弱队，赢分容易），
与本报告的"vs 随机合法策略"胜率**分母不同，不可直接比较**。要拿到可比数字只能真正参赛后逐轮观测。

**对本目标的意义（风险陈述，非成绩预测）**：单循环对称赛制下中位数必然是 0.5，
所以本地"vs 随机合法 0.608"这类代理指标与真实名次之间没有可换算的映射——
真实对手不是随机合法策略。本地评测能确证的只有：接口安全、无非法动作/超时/崩溃、
相对旧 checkpoint 的提升方向、以及牌组选择的优势；
 实际排位只能靠提交后实测。→ 已写入 §5 的风险项，并在最终报告中保持这一区分。


---

### 3.6 赛程状态与训练/提交窗口（平台只读接口实测）

`GET /api/health` → `{"status":"ok","engine":"kaggle","round_interval_hours":1}`；
`GET /api/leaderboard/b` → `{"phase":"A","started":false,"latest_round":null,"next_round":34,"rounds":[10,null]}`。

| 事实 | 值 | 对本目标的影响 |
|:---|:---|:---|
| 当前阶段 | **A 榜开发期**，B 榜未开始 | A 榜成绩仅供参考、不计入最终排名；正式排名期尚未开闸，还有迭代窗口 |
| 轮次节奏 | 每 1 小时自动一轮，A 榜已发布到第 33 轮，下一轮 34 | 方案 A 的 500k 微调约 1.9–2.2 h ≈ 跨 2 个 A 轮，时间上完全容得下 |
| 引擎 | `engine = "kaggle"` | 与本地 `.engine-shim` 的 `kaggle_environments==1.33.0` + CABT 口径一致，本地评测的环境等价性得到佐证 |
| A 榜规模 | 38 队报名，第 33 轮 32 队实际参赛 | 每小时都有一轮真实对抗数据，提交后可用同一只读接口观测实际排位 |

可执行的下一步（属对外写操作，需用户决定，本会话未执行）：
**在 B 榜开始前先用 A 榜提交一版**。A 榜不计入最终排名，却能拿到唯一无法本地合成的东西——
真实对手分布下的胜率；随后可用 `/api/leaderboard/a` 逐轮观测，把 §3.5 那条
"本地代理指标与真实名次之间没有换算映射"的风险变成实测数据。
包体上限、沙箱是否预装 torch 等平台未验证项，也会在首次提交时一并暴露。

## 4. 本会话新增文件清单

全部在 `runs/evaluation/session-20261020/` 内，**未覆盖、未删除、未移动任何既有文件**：

| 文件 | 用途 |
|:---|:---|
| `validate_deck.py` | 牌组五项规则独立校验 |
| `subsmoke_timing.py` | 提交包官方等价 smoke（单步时延 / 内存峰值 / 返回值合法性） |
| `subsmoke_migrated_8.json`、`subsmoke_default12.json`、`subsmoke_t2_12.json` | smoke 原始记录 |
| `linux_engine_smoke.py`、`linux_engine_smoke.json` | WSL + 真实 `libcg.so` 的 Linux 引擎侧验证 |
| `engine_shim_inventory.json` | `.engine-shim` 引擎二进制指纹 |
| `split_eval_pool.py`、`opp_pool_train30.json`、`eval_pool_heldout44.json` | 评测数据独立的牌组池确定性切分 |
| `train_help.txt` | `rl.train_selfplay --help` 全文留证 |
| `stage2_supplement.py` / `.log` / `.json` | 阶段 2 样本量补测（新种子，独立于既有 840 局） |
| `merge_baseline.py`、`stage2_baseline_merged.json` | 基线按臂合并（1,440 局）与 ≥120 局达标核查 |
| `latency_backend_probe.py`、`latency_backend_probe.json`、`latency_probe.log` | greedy/MCTS 配对时延探针（后端选型） |
| `phase4_gate.py`、`phase4_gate_selftest*.json` | 阶段 4 门禁执行器与三次 self-test 记录 |
| `stage2_topup.py` / `.log` / `.json` | 剩余三臂补到 ≥120 局（种子 20261024） |
| `validate_deck_platform.py`、`platform_cards.csv` | 平台权威卡表复核与两库字段级交叉比对 |
| `platform_leaderboard_a.json` | 平台公开 A 榜真实对局统计（赛场强度参照） |
| `pkg_trial_threads2/`、`subsmoke_patched12.json` | `set_num_threads(2)` 补丁副本与试跑（正式包未改动） |
| `phase4_gate.py` | 阶段 4 晋级门禁一条命令执行器（语法与 argparse 已校验） |
| `TRAINING_HANDOFF_500K.md`、`PHASE5_EXPORT_PRECHECK.md` | 阶段 3 训练交接、阶段 5 导出前置审计 |
| 本报告 | 阶段 0–2 审计结论 |

修改既有文件：**无**。

## 5. 未执行 / 待授权 / 已执行但受限（明确区分）

| 项 | 状态 |
|:---|:---|
| 启动任何训练 | **未执行**，且按硬性要求不由 AI 执行 |
| 正式导出新提交包 | **未执行**（阶段 4 门禁未过前禁止） |
| `git commit / push` | **未执行**（工作区必需修复保持未提交状态，等用户决定） |
| 修改 `main.py` 模板或已导出包 | **未执行**（线程数与预热建议记为下一次导出的待办） |
| WSL 内模型推理侧（torch）Linux 复验 | **受限未执行**：`~/pkmn-linux-verify-20261020/pylibs` 已消失，需重装约 1 GB 依赖 → 待用户授权 |
| 线程补丁的空机 A/B 定版 | **待执行**：现有两次测量都与评测作业并发（p50 从 35ms 漂到 120ms），必须在空闲机器上按"未打补丁 → 已打补丁"各 12 局同条件复测后才能定版 |
| 向平台提交以获取真实排位 | **已尝试并被平台拒绝**：`agent submit --yes` → `HTTP 413`，见 §6。CLI 明确记录"本次操作未产生任何平台写入" |
| `minCount≥1 且 option 为空` 的退化 select 实测触发 | **未观测到**（Linux 1,127 步、Windows 674 决策、840 局均为 0 次）→ 逻辑上已证明下标不越界，但该分支的实际引擎行为仍属推断 |
| 阶段 2 补测 | **已完成并汇总**：1,620 局、13 个臂全部 ≥120 局，见 `stage2_baseline_merged.json` 与 §3.3 |

## 6. 阻断性发现：平台请求体上限 1 MiB，现有全部提交包不可上传

阶段 5 的提交尝试（`ptcg-arena agent submit --yes submission_rl_transformer2_t2.tar.gz`，71.6 MB）返回：

```text
HTTP 413 Request Entity Too Large   nginx/1.18.0 (Ubuntu)
提示：本次操作未产生任何平台写入。
```

规则文档（`AGENT.md` §二/§十四）与 `/api/*` 都没有声明体积上限，因此用近零字节方式实测边界：
带 `Expect: 100-continue` 声明 `Content-Length` 发 `POST /api/submissions`，
nginx 在读取 body 之前就判定，收到判定即断开（不传输任何 body，`probe_body_limit.py`）：

| 声明长度 | 响应 |
|---:|:---|
| 1024 KiB | `100 Continue`（**放行**） |
| 1100 KiB 起，含 2 MiB / 50 MiB / 128 MiB | 一律 `413` |

→ 上限就是 nginx 默认 `client_max_body_size 1m`，**1 MiB = 1,048,576 字节**，且算的是**整个 multipart body**，
所以 `submission.tar.gz` 本体必须 **≲ 1,047,000 字节**。

**影响范围（此前所有会话都未察觉）**：

| 产物 | 大小 | 可提交性 |
|:---|---:|:---|
| `submission_rl_transformer2_t2.tar.gz` | 71.6 MB | ❌ 超限 68× |
| `submission_rl_transformer2_db_migrated.tar.gz` | 71.6 MB | ❌ |
| 另 3 个历史 `.tar.gz` | 各 ~71.6 MB | ❌ |
| `model.pt`（20,303,715 参数 fp32） | 81 MB | ❌ 单项即超 78× |

即：**四个"已通过门禁并完成校验"的提交包，实际上一个都上传不了**。此前的导出与哈希校验都成立，
但"可交付"这一条从未被端到端验证过——这正是任务书 §九 要求区分"已执行/已验证/未执行/推断"的价值所在。

**1 MiB 预算内的可行方向**（均需你选择，我不擅自改动方案）：

1. **纯启发式 / 规则型 Agent**：只有代码 + 卡表。`EN_Card_Data.csv` 171 KB，gzip 后约 30 KB，
   加上编码与策略代码可压到数百 KB，**当前唯一几乎确定能上传的形态**。
2. **蒸馏小网络**：用现成的 20.3M 教师（含正在跑的微调）克隆一个 ≤约 25 万参数的小网
   （`scripts/bc_train.py` 已在仓库里）。25 万参数 int8 存约 250 KB，留足余量给代码与卡表。
   属训练，**必须由你运行**；我可以把训练命令与小网结构、导出与体积校验全部准备好。
3. **问管理员 1m 是否为有意设置**：nginx 默认值更像是没调过；官方 Kaggle 的同类提交上限远大于 1 MB。
   若能提到 100 MB，现有包立刻可用。

**同时仍然未知的两项**（只有成功上传一次才能证实）：包内多文件是否被接受、沙箱是否预装 torch。
方向 1 顺带把 torch 依赖去掉（纯 numpy 前向），能同时消掉这两个不确定性。

### 6.1 1 MiB 预算的实测拆分与可用容量

| 组成 | 实测 |
|:---|:---|
| 包内除 `model.pt` 外的全部文件（含 171 KB 卡表、`encoding.py` 84 KB、`effect_data.py` 55 KB 等） | 未压缩 476,007 B → **tar.gz 后 120,842 B（118 KiB）** |
| 留给权重的预算 | **约 926,000 B**（按 1,047,000 上限计，含 multipart 边界余量） |
| 权重按 fp16 存储时的参数上限 | **约 462,000 参数** |

按真实参数量逐档实测（`build_token_net` 计数，权重按 fp16、其余按实测 120,842 B 估算包体）：

| `d_model/nhead/nlayers/ff` | 参数 | fp16 字节 | 估算包体 | 1 MiB 内 |
|:---|---:|---:|---:|:---:|
| 64/4/2/128 | 334,083 | 668,166 | 789,008 | ✅ |
| **64/4/3/128** | **384,067** | 768,134 | 888,976 | ✅（最大可用档） |
| 80/4/3/160 | 528,723 | 1,057,446 | 1,178,288 | ❌ |
| 128/4/3/256（`train_selfplay` 默认） | 1,079,427 | 2,158,854 | 2,279,696 | ❌ |
| 544/8/5/256（现方案） | 20,303,715 | 38.73 MiB | 71.6 MiB | ❌ |

→ 可提交的天花板是 **transformer2 d_model=64 / nhead=4 / nlayers=3 / ff=128**；再大一档就超。
`scripts/export_rl_submission.py` 已加入口体积门禁（`--max-bytes` 默认 1,047,000，超限 `SystemExit` 并打印权重预算），
端到端验证：用 `ckpt_999424` 重导即在本地被拦住并给出"约 462,000 个 fp16 参数"的提示；`pytest tests -q` → **51 passed**。

### 6.2 容量到底值多少（进行中实测，`capacity_probe.py`）

1 MiB 逼出一个必须回答的问题：**网络容量与训练本身换来多少胜率**。
同协议（提交牌组、turn_cap 200、座位逐局交替、对手一律随机合法策略）实测**未训练**的 544/8/5 网络：

| 候选 | 对随机合法胜率 | n | CI95 |
|:---|---:|---:|:---|
| 已训练 544/8/5（`ckpt_999424`，既有基线） | 0.608 | 120 | [0.519, 0.691] |
| 未训练 544/8/5（20,303,715 参数随机初始化） | 0.575 | 120 | [0.486, 0.660] |
| **未训练 64/4/3/128（384,067 参数随机初始化）** | **0.617** | 120 | [0.527, 0.699] |

三者 CI 大面积重叠：**在"对随机合法策略"这一口径上，20.3M 训练网络相对 38.4 万参数的随机初始化网络没有可检出的优势**。
单格明细显示随机初始化本身方差极大（小网两 seed 分别 0.383 与 0.850），与 §3.4 修正 2 同源。
→ 结论：**1 MiB 上限大概率不是决定性损失**（推断，基于上述两个可验证事实）；真正未解释的是
**A 榜累计第一 0.797 与我们最好一格 0.617 之间的差距**（两者口径不同：前者是真实混合对手池，后者是 vs 随机合法策略，只可定性比较），它更可能在牌组与策略结构，而本地自博弈学不出来。
完整数字：`capacity_probe.json`（240 局新增，0 异常 0 非法）。

### 6.3 蒸馏链路现状（为何不能开箱即用）

`scripts/bc_train.py` 支持 `--d-model/--nhead/--nlayers/--ff` 覆盖 ✅，但它是**离线 BC**，
位置参数要 `.npz` 或 `.npy` 目录；`scripts/build_bc_from_zips.py` 需要 Kaggle replay `.zip` 存档，
**本机没有任何此类数据**（已核实），`rl/rollout.py` 只是库不是 CLI。
→ 三条可选教师数据来源，均需你定方向后我再具体准备（训练一律由你运行）：
  (a) 小网直接跑现有自博弈 RL（无需数据集，`train_selfplay` 支持小配置，链路最成熟）；
  (b) 我写一个从 `ckpt_999424` 导出转移的 rollout dumper（推理产物，非训练），再 `bc_train` 蒸馏；
  (c) 用平台 `/api/matches/{id}/log` 取真实强队对局日志做 BC —— 唯一能学到"我们尚未掌握的东西"的路径。


