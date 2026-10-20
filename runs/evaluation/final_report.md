# Pokémon TCG 提交修复与牌组优化 — 最终报告

生成日期：2026-10-20（本地时钟）。本文件合并以下四份报告；原文件已删除，本文件为唯一版本：

1. 提交包运行阻塞审计（原 `package_audit.md`）
2. 牌组搜索与对照评估（原 `deck_search_report.md`）
3. 评估工具修复（原 `gates_fix.md`）
4. 训练决策（原 `training_decision.md`）

所有胜率/状态均来自本机实测；未验证项在各部分内单独标注。


---

# 第一部分：提交包运行阻塞审计

## 提交包运行阻塞审计（package_audit）

审计日期：2026-10-20（本地时钟）。全部数字来自本机实测；未执行的项目单独标为「未验证」。
本报告不覆盖 `submission_rl_transformer2.tar.gz` 与 `runs/train-20m-e16-1m/ckpt_*.pt`。

### 0. 结论

1. **阻塞已定位**：`would_ko=True` 的提交推理路径需要官方引擎的 search API
   （`AgentStart / SearchBegin / SearchStep / SearchEnd / SearchRelease`），原 tar 不含任何
   引擎二进制；干净环境 `from sdk_cg import api` 即 `FileNotFoundError`，首个决策崩。
2. **二进制来源已核实**：本机 shim 的 `libcg.so` / `cg.dll` 与 PyPI 官方
   `kaggle_environments==1.33.0` wheel 内的同名文件 **sha256 逐字节一致**。
3. **Linux 兼容性静态通过**：`libcg.so` 为 x86-64 ELF、依赖
   `libstdc++.so.6 / libgcc_s.so.1 / libc.so.6`、最高 `GLIBC_2.14`；WSL2 Ubuntu 24.04
   （glibc 2.39）满足。**未在 Linux 实跑**（原因见 §4）。
4. **分发许可不明确**：wheel 自带 Apache-2.0 LICENSE，但仓库 README 写引擎二进制
   proprietary；比赛规则页为 JS 渲染、API 需登录，未能取得再分发条款。因此**未把二进制
   放进候选包**。
5. **修复方案**：`sdk_cg/sim.py` 增加运行时解析——优先复用平台已加载的官方引擎，其次包内
   二进制，最后官方 `kaggle_environments` 安装路径；不复制、不再分发。
6. **干净目录验证通过**：候选包解包后 `import main`、60 张牌组请求、6 局真实引擎对局
   （两个座位各 3 局）、首决策/连续决策、`would_ko` 写回、`list[int]` 动作全部通过；
   0 异常、0 非法动作、1/6 胜（与既有对随机约 0.20 的基线同量级）。

### 1. `would_ko=True` 的完整执行路径

| 步骤 | 位置 | 行为 |
|---|---|---|
| 1 | `submission_rl_transformer2/main.py:28-30` | `_WK_ON = net_config["would_ko"]` → `import search_agent` |
| 2 | `main.py:56-57` | 每个决策 obs 调用 `annotate_would_ko(obs, DECK, _ENC)` |
| 3 | `search_agent.py:394-403` | `would_ko_flags` → `write_would_ko`，写 `would_ko/would_ko_prizes/would_ko_win` |
| 4 | `search_agent.py:335,348-371` | `from sdk_cg import api`；`search_begin/search_step/search_release/search_end` |
| 5 | `sdk_cg/api.py:517-639` | 绑定 `lib.AgentStart / SearchBegin / SearchStep / SearchEnd / SearchRelease` |
| 6 | `encoding.py:325-327`（包内同构） | 三列写入 `opt_attr[:, 11:14]` |

需要的引擎符号：`AgentStart、SearchBegin、SearchStep、SearchEnd、SearchRelease`。
`WouldKO / GetSerialObs / GetBinaryObs / AdvanceLog` **不需要**；它们属于训练侧的
`libcg_custom.so` / `libcg_wk.so`。

### 2. Linux 引擎二进制

| 项 | 值 |
|---|---|
| 文件 | `libcg.so`，1,458,504 B |
| sha256 | `c699020e086eede189c445c7abf9d891a60005ceec6eb91dccd5bbdcd3c0de22` |
| 架构 | ELF 64-bit LSB shared object, x86-64 |
| NEEDED | `libstdc++.so.6`、`libgcc_s.so.1`、`libc.so.6` |
| 最高 GLIBC 版本 | `GLIBC_2.14` |
| 搜索符号 | `AgentStart / SearchBegin / SearchStep / SearchEnd / SearchRelease` 全部存在 |
| 来源 | `kaggle_environments-1.33.0-py3-none-any.whl` 的 `envs/cabt/cg/libcg.so` |
| 一致性 | 与 wheel 内文件 sha256 完全一致；`cg.dll` 同样一致（`e084185a…`） |

wheel 路径：
`C:\Users\Swcmb\AppData\Local\Temp\opencode\kepkg\kaggle_environments-1.33.0-py3-none-any.whl`

### 3. 许可与分发判断

- **wheel 证据**：`kaggle_environments-1.33.0.dist-info/licenses/LICENSE` 为 Apache License 2.0
  全文（201 行）；`METADATA` 声明 `License-File: LICENSE`。wheel 内确实包含上述二进制。
- **仓库 README 证据**：`README.md:39-40` 写 “the engine binaries themselves are proprietary
  and not included -- get them from the competition page”。任务书提到的 `sdk_cg/README.md`
  实际不存在。
- **比赛规则**：`https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules` 返回的是
  JS 壳（6,029 B），HTML 内无规则正文；Kaggle API 需要登录。**再分发条款未取得**。
- **结论**：授权证据互相冲突，按「未声明」处理。**候选包不内嵌引擎二进制**，改用运行时
  依赖解析；本地验证使用的二进制仅来自本机官方 wheel，不再分发。

### 4. Linux 测试环境与执行边界

| 检查 | 结果 |
|---|---|
| WSL2 | 存在，默认发行版 `Ubuntu-24.04`，版本 2；glibc 2.39、Python 3.12.3 |
| Docker | 客户端 29.2.1 存在；`docker-desktop` 发行版存在，但 daemon 未运行 |
| Linux 实跑 | **未执行**。当前执行约束要求所有命令/Python 一律在 conda `deeplearning`（Windows）中；WSL 属于另一环境，未在其中执行 Python |
| Linux 静态验证 | 已完成：ELF 架构、NEEDED、GLIBC、搜索符号（§2） |

因此本报告只声称「Linux 二进制静态兼容」，不声称 Linux 运行时通过。

### 5. 候选包构建

产物：`submission_rl_transformer2_linux_nodist.tar.gz`
（75,155,277 B，sha256 `3acb3dea7995457bfb824bc50c3ab0db7af829923e3ef5f6d0b17e36408e1470`）

- 导出命令：
  `python scripts/export_rl_submission.py --ckpt runs/train-20m-e16-1m/ckpt_999424.pt --backend transformer2 --deck agent/deck.csv --out submission_rl_transformer2_linux_nodist.tar.gz --builddir submission_rl_transformer2_linux_nodist`
- 19 个成员，`.so/.dll/.dylib` 列表为空。
- 与原包逐成员比对：18 个文件中 **17 个 sha256 完全相同**（含 `model.pt`、`deck.csv`、
  `main.py`、`search_agent.py` 与全部数据表）；唯一差异是 `sdk_cg/sim.py`（运行时解析）。
- 原包未覆盖：`submission_rl_transformer2.tar.gz` 仍为 75,154,720 B，sha256
  `e5c3f89a1f6a44b825d1a48981838cc2dd8d12b69e6a77c5844037976955690f`。

#### 运行时解析逻辑（`sdk_cg/sim.py`）

1. 若平台已导入 `kaggle_environments.envs.cabt.cg.sim`，复用其 `lib` 句柄（不二次加载）；
2. 否则查包内 `sdk_cg/cg.dll|libcg.so|libcg.dylib|libcg-arm64.so`；
3. 否则查官方安装的 `kaggle_environments.envs.cabt.cg` 目录；
4. 全部缺失才抛 `FileNotFoundError`，错误信息给出两种修复方式。

### 6. 干净目录验证（Windows 官方 `cg.dll`）

脚本：`runs/evaluation/package_check.py`；输出：`runs/evaluation/package_audit_run.json`。

流程：临时目录解包 → 检查二进制列表 → `import main` → `agent({"select": None})` 牌组请求
→ 真实 `sdk_cg.game` 对局，主 agent 与随机合法对手交替座位 → 每步校验动作合法性
→ 检查 `would_ko` 写回。

| 局 | 主座位 | 主决策 | 对手决策 | 含 would_ko 决策 | 写入单元格 | 非法动作 | 异常 | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 0 | 18 | 35 | 10 | 11 | 0 | 无 | 负 |
| 1 | 1 | 6 | 19 | 0 | 0 | 0 | 无 | 负 |
| 2 | 0 | 15 | 23 | 5 | 6 | 0 | 无 | 胜 |
| 3 | 1 | 38 | 55 | 18 | 25 | 0 | 无 | 负 |
| 4 | 0 | 15 | 29 | 2 | 2 | 0 | 无 | 负 |
| 5 | 1 | 6 | 16 | 0 | 0 | 0 | 无 | 负 |

汇总：`import_main=True`、`wk_on=True`、牌组 60 张全 int、6 局全部首决策成功、
`bad_actions=0`、`errors=0`、主视角 1/6 胜。含 would_ko 的决策数在 0–18 之间，取决于该局
是否出现攻击选项；这与 `would_ko` 只标注攻击选项的实现一致。

### 7. 修改文件与测试

| 文件 | 改动 |
|---|---|
| `sdk_cg/sim.py` | 增加引擎路径解析；复用平台已加载句柄；缺失时给出明确错误 |
| `scripts/export_rl_submission.py` | **未改**（原样复制 `sdk_cg/`，自动带上解析逻辑） |
| `runs/evaluation/package_check.py` | 新增干净目录验证脚本 |
| `submission_rl_transformer2_linux_nodist.tar.gz` | 新候选包（独立命名，未覆盖原包） |

已运行：

- `python runs/evaluation/package_check.py --tar submission_rl_transformer2_linux_nodist.tar.gz --games 6`
  → `ok=True`，`binaries=[]`，6 局 0 异常 / 0 非法。
- `readelf -h/-d/-V/-Ws libcg.so` → §2 静态结论。
- wheel 内 `libcg.so` / `cg.dll` sha256 与本机 shim 比对 → 完全一致。

### 8. 剩余阻塞与未验证项

1. **平台是否提供官方 `kaggle_environments`**：候选包依赖它来解析 `libcg.so`。真实 Kaggle
   沙箱已不可用，**未验证**；若平台既不提供包、也不在 agent 目录放二进制，仍会在首个
   would_ko 决策失败。
2. **Linux 实跑**：**未执行**（执行环境限制，§4）。仅静态确认 ELF/依赖/GLIBC/符号。
3. **再分发许可**：wheel Apache-2.0 与 README proprietary 冲突，比赛规则未取得；当前候选包
   不含二进制，因此不触发再分发，但若未来要发布含二进制的包，需先取得明确授权。
4. **实际平台注入行为**：`main.py` 的 `sys.path.insert(0, _HERE)` 使包内 `sdk_cg` 优先；
   若平台只提供 `kaggle_environments` 而不直接写 agent 目录，解析逻辑仍可用；若平台把引擎
   放在其他路径，需要补充搜索路径（未验证）。

### 9. 牌组优化候选包（第二阶段联动）

在牌组评估（第二部分）选定 `official_dragapult` 后，用同一导出脚本生成：

- 路径：`submission_rl_transformer2_dragapult_nodist.tar.gz`
- 大小：75,155,298 B；sha256 `09b2929c5f9cf77759a087b09ad202bf05bdf32b36d9ac00e89ca2505b1d9d28`
- 19 个成员，`.so/.dll/.dylib` 列表为空。
- 与原包逐成员比对：唯一差异是 `deck.csv`（换成 dragapult 60 张）与 `sdk_cg/sim.py`
  （运行时解析）；`model.pt`、`main.py`、`search_agent.py` 等 16 个文件完全相同。
- 干净目录验证（`runs/evaluation/package_audit_run_dragapult.json`）：4 局、两座位各 2 局，
  `ok=True`、0 异常、0 非法动作；这 4 局主策略未遇到攻击选项，`would_ko` 写入为 0。
- 另用真实 dragapult 攻击 obs 直接调用 `annotate_would_ko`：2 个攻击选项、2 个
  `would_ko` 单元格写入（该局面 KO 率 0.0），确认代码路径对该牌组有效。

该包与 `submission_rl_transformer2_linux_nodist.tar.gz` 共享同一运行时解析修复；
两者都不含引擎二进制，都不覆盖原包。

---

# 第二部分：牌组搜索与对照评估

## 牌组搜索与对照评估（deck_search_report）

评估日期：2026-10-20（本地时钟）。所有胜率均来自本机官方引擎实跑；未验证项单独标注。
本报告不修改 `agent/deck.csv`，不覆盖原提交包与 checkpoint。

### 0. 结论

1. **当前提交牌组就是引擎自带的样例牌组**：`agent/deck.csv` 与
   `kaggle_environments/envs/cabt/cabt.py` 的 `deck` 列表 **60 张、顺序完全一致**。
   它只有 10 张宝可梦（6 张基础）、33 张基础水能量、17 张训练家，且不在
   `TRAIN_TOP15` / `TRAIN_TOP50` 中。这是“对随机合法策略约 0.20、败局多为
   `no_active`”的结构性原因。
2. **最佳新牌组是 `official_dragapult`**（`rl/decks.py` 的官方默认 Dragapult 列表）：
   16 张宝可梦（9 基础）、8 张能量、36 张训练家，引擎接受。
   同一 checkpoint `ckpt_999424.pt` 下，两种子合计：
   - 对随机合法策略：旧 **0.2200** [0.1682, 0.2824] → 新 **0.5400** [0.4708, 0.6077]
   - 对 `ckpt_500736.pt`：旧 **0.4450** [0.3778, 0.5143] → 新 **0.6900** [0.6228, 0.7500]
   - 自镜像：新 **0.5500** [0.4524, 0.6439]（旧 0.5300 [0.4329, 0.6249]）
   - 同模型直接对局：dragapult **0.57** vs 旧牌组 **0.44**（CI 均含 0.5，只能算略优）
3. **`k36_marnie_s_grimmsnarl_ex_35` 被否决**：它对随机 0.575、对 500736 0.57，数据很好，
   但同模型直接对局被当前牌组 **0.755** 克制（两种子：0.760 / 0.750；反向 k36 对当前牌组
   仅 0.280）。这是明确的非传递性，不能仅凭对随机/对旧 checkpoint 的数字替换提交牌组。
4. **建议**：把 `official_dragapult` 作为候选提交牌组，保留 `agent/deck.csv` 作为对照；
   新包已按独立名称导出。若要最终替换，建议再做第三种子确认（见 §8）。

### 1. 事实核对

| 项 | 结果 |
|---|---|
| 提交牌组 | 60 张，11 种，**10 宝可梦（6 基础）**、33 基础水能量、17 训练家 |
| 与引擎样例牌组 | 60 张顺序完全一致（官方 `cabt.py`） |
| 与训练池 | 不在 `TRAIN_TOP15` / `TRAIN_TOP50`；最接近的池牌组重叠 49/60 |
| 旧牌组对随机 | 100 局两种子均 0.22；平均 14.4–16.0 turn，远短于池牌组的 ~36 turn |
| `no_active` | 旧牌组短局的主要终局原因（审计报告 §4.5 已记录 38/40、40/40） |

### 2. 候选生成与引擎校验

脚本：`scripts/deck_search.py`。来源：

1. 当前提交牌组（对照组，`agent/deck.csv`）；
2. `rl/decks.py` 的 4 副官方默认牌组；
3. 现有 `pool_top15.json`（30 副）与 `pool_top50.json`（74 副）；
4. 5 副对照组定向变体：补基础宝可梦（Kyogre 2→4、Fezandipiti ex ×2）、
   降低能量（33→20/24/28）、增加训练家（Ultra Ball/Mega Signal/Switch/Boss's Orders）。

校验：

- 卡牌规则：恰好 60 张；卡牌 ID 存在；非基础能量每种 ≤4；ACE SPEC ≤1；至少 1 张基础宝可梦。
- 真实引擎：`sdk_cg.game.battle_start(deck, deck)` 必须返回 obs；返回空指针即拒绝。
- 组成门槛（仅用于短名单）：基础宝可梦 ≥8、能量 ≥4、宝可梦 ≥12；对照组豁免。

结果：**76 副牌组全部通过引擎校验**；60 副通过组成门槛；短名单 18 副进入小样本。
产物：

- `runs/evaluation/deck_search/candidates.json`（全部 76 副的组成、合法性、引擎结果）
- `runs/evaluation/deck_search/pools/<name>.json`（每副一个单牌组 pool，供 `eval_h2h.py` 直接复用）
- `runs/evaluation/deck_search/summary.md`（组成表）

#### 2.1 对照组与最终候选的组成

| 牌组 | 宝可梦（基础/进化） | 能量（基础/特殊） | 训练家 | ACE | 唯一卡 |
|---|---|---|---|---|---|
| `control_deckcsv`（当前） | 10 (6/4) | 33 (33/0) | 17 | 1 | 11 |
| `official_dragapult`（最终候选） | 16 (9/7) | 8 (8/0) | 36 | 1 | 22 |
| `pool_k36_marnie_grimmsnarl`（被否决） | 22 (13/9) | 10 (10/0) | 28 | 1 | 19 |

`official_dragapult` 列表：4 Dreepy、4 Drakloak、3 Dragapult ex、1 Fezandipiti ex、
1 Latias ex、2 Budew、1 Meowth ex；4 Basic {R} + 4 Basic {P} Energy；36 张训练家
（含 4 Ultra Ball、4 Buddy-Buddy Poffin、4 Crushing Hammer、3 Poke Pad、4 Crispin 等）。
独立文件：`runs/evaluation/deck_search/dragapult_deck.csv` / `dragapult_deck.json`
（`best_deck.csv` 同步指向它）。

### 3. 小样本筛选（先筛后排）

协议：`eval_h2h.py --opponent random --mode mirror`，每个候选两座位牌组相同，
候选座位严格 `gi % 2` 均衡；固定 `PKMN_TURN_CAP=200`；串行 `live=1`。

#### 3.1 第一轮：76 副 × 8 局（种子 20261019）

全部 76 副合法牌组都跑了 8 局（不只是短名单）。0 异常、0 非法动作、0 平局。
前 12 名：

| 排名 | 牌组 | 胜/局 | 胜率 | Wilson 95% CI | 平均 applies |
|---:|---|---:|---:|---|---:|
| 1 | `pool_meta_aichi_dragapult_dudunsparce_5th_19` | 7/8 | 0.875 | [0.530, 0.977] | 135 |
| 2 | `pool_meta_aichi_ns_zoroark_2nd_16` | 7/8 | 0.875 | [0.530, 0.977] | 110 |
| 3 | `pool_k05_fezandipiti_ex_04` | 6/8 | 0.750 | [0.409, 0.929] | 99 |
| 4 | `pool_k36_marnie_s_grimmsnarl_ex_35` | 6/8 | 0.750 | [0.409, 0.929] | 138 |
| 5 | `pool_meta2_fez_grimmsnarl_ex_28` | 6/8 | 0.750 | [0.409, 0.929] | 162 |
| 6 | `pool_meta3_fez_70` | 6/8 | 0.750 | [0.409, 0.929] | 90 |
| 7 | `pool_meta_aichi_ogerpon_meganium_3rd_17` | 6/8 | 0.750 | [0.409, 0.929] | 110 |
| 8 | `official_dragapult` | 5/8 | 0.625 | [0.306, 0.863] | 90 |
| 9 | `pool_k15_barbaracle_non_ex_14` | 5/8 | 0.625 | [0.306, 0.863] | 111 |
| 10 | `pool_k27_fezandipiti_ex_latias_ex_lillie_s_clefai_26` | 5/8 | 0.625 | [0.306, 0.863] | 115 |
| 11 | `pool_k33_dragapult_ex_dudunsparce_ex_fezandipiti__32` | 5/8 | 0.625 | [0.306, 0.863] | 125 |
| 12 | `pool_meta4_naic_clefairy_1st_72` | 5/8 | 0.625 | [0.306, 0.863] | 149 |

#### 3.2 第二轮：前 5 名 × 20 局（种子 20261020）

8 局样本的 CI 很宽（±0.3），必须复测。第二轮结果：

| 牌组 | 第二轮 | 两轮合并 | 合并胜率 |
|---|---:|---:|---:|
| `pool_k36_marnie_s_grimmsnarl_ex_35` | 11/20 | 17/28 | 0.607 |
| `pool_meta_aichi_dragapult_dudunsparce_5th_19` | 9/20 | 16/28 | 0.571 |
| `pool_k05_fezandipiti_ex_04` | 9/20 | 15/28 | 0.536 |
| `pool_meta_aichi_ns_zoroark_2nd_16` | 8/20 | 15/28 | 0.536 |
| `pool_meta_aichi_ogerpon_meganium_3rd_17` | 8/20 | 14/28 | 0.500 |

**教训**：第一轮 0.875 的 aichi_dragapult 在第二轮只有 0.45；8 局只能排序，不能当结论。
`official_dragapult` 第一轮 5/8，未进第二轮；它在正式 100 局矩阵中表现更稳（§4.3），
说明第一轮对它的低估同样是小样本噪声。

### 4. 正式 100 局矩阵

协议：`eval_h2h.py`，镜像模式（两座位同一副牌），候选座位 `gi % 2` 严格均衡；
随机策略从 `action_mask` 均匀抽取合法动作；固定 `PKMN_TURN_CAP=200`；串行 `live=1`；
每局 `seed+gi`。正式组均 0 平局、0 异常、0 非法动作（`illegal.total=0`，按决策计 0）。

#### 4.1 当前牌组（对照组）

| 对手 | 种子 | 胜/负/平 | 胜率 | Wilson 95% CI | seat0 / seat1 | 平均 applies / turn | 文件 |
|---|---:|---|---:|---|---|---|---|
| 随机 | 20261021 | 22/78/0 | 0.2200 | [0.1500, 0.3107] | 0.20 / 0.24 | 51.7 / 16.0 | `old_vs_random_20261021_100.json` |
| 随机 | 20261022 | 22/78/0 | 0.2200 | [0.1500, 0.3107] | 0.24 / 0.20 | 46.8 / 14.4 | `old_vs_random_20261022_100.json` |
| 500736 | 20261021 | 42/58/0 | 0.4200 | [0.3280, 0.5179] | 0.64 / 0.20 | 91.7 / 85.9 | `old_vs_baseline_20261021_100.json` |
| 500736 | 20261022 | 47/53/0 | 0.4700 | [0.3751, 0.5671] | 0.80 / 0.14 | 93.8 / 88.6 | `old_vs_baseline_20261022_100.json` |
| 自镜像 | 20261021 | 53/47/0 | 0.5300 | [0.4329, 0.6249] | 0.60 / 0.46 | 89.1 / 81.4 | `old_vs_selfmirror_20261021_100.json` |

合并（两种子）：对随机 **44/200 = 0.2200 [0.1682, 0.2824]**；
对 500736 **89/200 = 0.4450 [0.3778, 0.5143]**。

#### 4.2 `k36_marnie_s_grimmsnarl_ex_35`（对随机最好的候选，最终被否决）

| 对手 | 种子 | 胜/负/平 | 胜率 | Wilson 95% CI | seat0 / seat1 | 平均 applies / turn |
|---|---:|---|---:|---|---|---|
| 随机 | 20261021 | 49/51/0 | 0.4900 | [0.3942, 0.5865] | 0.58 / 0.40 | 135.8 / 46.1 |
| 随机 | 20261022 | 66/34/0 | 0.6600 | [0.5628, 0.7454] | 0.70 / 0.62 | 154.5 / 53.3 |
| 500736 | 20261021 | 55/45/0 | 0.5500 | [0.4524, 0.6439] | 0.82 / 0.28 | 138.9 / 84.9 |
| 500736 | 20261022 | 59/41/0 | 0.5900 | [0.4920, 0.6813] | 0.82 / 0.36 | 150.1 / 88.7 |
| 自镜像 | 20261021 | 50/50/0 | 0.5000 | [0.4038, 0.5962] | 0.78 / 0.22 | 172.6 / 78.4 |

合并：对随机 **115/200 = 0.5750 [0.5057, 0.6415]**；
对 500736 **114/200 = 0.5700 [0.5007, 0.6367]**。
数字上它是“对随机/对旧 checkpoint”最好的候选，但 §5 的直接对局否决了它。

#### 4.3 `official_dragapult`（最终候选）

| 对手 | 种子 | 胜/负/平 | 胜率 | Wilson 95% CI | seat0 / seat1 | 平均 applies / turn | 文件 |
|---|---:|---|---:|---|---|---|---|
| 随机 | 20261021 | 50/50/0 | 0.5000 | [0.4038, 0.5962] | 0.48 / 0.52 | 107.6 / 39.2 | `drag_vs_random_20261021_100.json` |
| 随机 | 20261022 | 58/42/0 | 0.5800 | [0.4821, 0.6720] | 0.54 / 0.62 | 107.8 / 39.4 | `drag_vs_random_20261022_100.json` |
| 500736 | 20261021 | 68/32/0 | 0.6800 | [0.5834, 0.7633] | 0.90 / 0.46 | 91.5 / 78.5 | `drag_vs_baseline_20261021_100.json` |
| 500736 | 20261022 | 70/30/0 | 0.7000 | [0.6042, 0.7811] | 0.86 / 0.54 | 90.5 / 78.8 | `drag_vs_baseline_20261022_100.json` |
| 自镜像 | 20261021 | 55/45/0 | 0.5500 | [0.4524, 0.6439] | 0.58 / 0.52 | 78.6 / 61.7 | `drag_vs_selfmirror_20261021_100.json` |

合并：对随机 **108/200 = 0.5400 [0.4708, 0.6077]**；
对 500736 **138/200 = 0.6900 [0.6228, 0.7500]**。

自镜像的座位分项 0.58/0.52，明显比 k36 的 0.78/0.22 平衡；总胜率 0.55（CI 含 0.5），
说明没有系统性的座位偏差。

#### 4.4 组间对比（同一 checkpoint `ckpt_999424.pt`）

| 指标 | 当前牌组 | k36 | dragapult |
|---|---:|---:|---:|
| 对随机（合并 200 局） | 0.2200 | 0.5750 | **0.5400** |
| 对 500736（合并 200 局） | 0.4450 | 0.5700 | **0.6900** |
| 自镜像（100 局） | 0.5300 | 0.5000 | **0.5500** |
| 平均对局长度（随机，applies） | 51.7 | 135.8 | 107.6 |

dragapult 在“对随机”上比 k36 低 0.035（合并 CI 重叠），但在“对 500736”上高 0.12，
且自镜像座位更平衡、直接对局不输当前牌组。综合选择 dragapult。

### 5. 直接对局与非传递性

同 checkpoint `ckpt_999424.pt`、两座位同网、`mode=cross`、座位严格均衡；
候选座位用 A 牌组，对手座位用 B 牌组。

| 候选座位牌组 | 对手座位牌组 | 局数 | 候选胜率 | Wilson 95% CI | seat0 / seat1 |
|---|---|---:|---:|---|---|
| 当前牌组 | k36 | 100（种子 21） | **0.7600** | [0.6677, 0.8331] | 0.80 / 0.72 |
| k36 | 当前牌组 | 100（种子 21） | **0.2800** | [0.2014, 0.3749] | 0.30 / 0.26 |
| 当前牌组 | k36 | 100（种子 22） | **0.7500** | [0.6570, 0.8245] | 0.68 / 0.82 |
| 当前牌组 | dragapult | 100（种子 22） | **0.4400** | [0.3467, 0.5377] | 0.74 / 0.14 |
| dragapult | 当前牌组 | 100（种子 22） | **0.5700** | [0.4722, 0.6627] | 0.80 / 0.34 |
| dragapult | 当前牌组 | 60（种子 21） | **0.5500** | [0.4249, 0.6691] | 0.70 / 0.40 |
| aichi_dragapult | 当前牌组 | 60（种子 21） | 0.3667 | [0.2562, 0.4932] | 0.50 / 0.23 |
| k05_fezandipiti | 当前牌组 | 60（种子 21） | 0.4167 | [0.3006, 0.5427] | 0.53 / 0.30 |
| ns_zoroark | 当前牌组 | 60（种子 21） | 0.5000 | [0.3774, 0.6226] | 0.87 / 0.13 |

结论：

- **k36 被当前牌组稳定克制**（当前牌组 0.755，两种子）；虽然 k36 对随机/对 500736 更好，
  但不能替换提交牌组。这是本任务最重要的非传递性证据。
- **dragapult 不输当前牌组**：两个方向 0.57 / 0.44，CI 均含 0.5，只能算“略优/持平”，
  但至少没有 k36 那样的反向克制。
- aichi、k05、zoroark 的直接对局均不优于当前牌组（zoroark 恰好 0.50）。
- 直接对局的座位效应很强（seat0 0.68–0.87，seat1 0.13–0.40），
  因此所有总胜率都必须在 50/50 座位混合下解读。

### 6. 牌组与模型策略的分离

同一 checkpoint `ckpt_999424.pt`，只改变牌组：

| 对照 | 当前牌组 | dragapult | 变化 |
|---|---:|---:|---:|
| 对随机 | 0.2200 | 0.5400 | +0.320 |
| 对 500736 | 0.4450 | 0.6900 | +0.245 |
| 自镜像 | 0.5300 | 0.5500 | +0.020（噪声内） |

因此“对随机从 0.22 到 0.54、对 500736 从 0.445 到 0.69”不能归因于模型能力变化：
模型完全相同，唯一变量是牌组。**但有一个必须声明的混杂因素**：dragapult 是
`rl/decks.py` 的官方默认列表，很可能在训练分布内；当前 `deck.csv` 是引擎样例，
明确不在 `TRAIN_TOP15/50`。所以提升混合了“牌组结构”与“模型对该牌组的熟悉度”，
本次没有做“同结构、训练外”牌组的消融来分离二者。

### 7. 产物

| 产物 | 路径 |
|---|---|
| 当前牌组（对照，未修改） | `agent/deck.csv` |
| 最佳新牌组（独立文件） | `runs/evaluation/deck_search/dragapult_deck.csv` / `.json` |
| 最佳新牌组（统一别名） | `runs/evaluation/deck_search/best_deck.csv` / `.json` |
| 全部候选与组成 | `runs/evaluation/deck_search/candidates.json`、`summary.md`、`pools/` |
| 小样本筛选 | `runs/evaluation/deck_search/screen/`、`screenB/` |
| 正式矩阵 | `runs/evaluation/deck_search/final/` |
| 新提交候选包（无引擎二进制） | `submission_rl_transformer2_dragapult_nodist.tar.gz` |
| 原提交包（未覆盖） | `submission_rl_transformer2.tar.gz` |
| 原 checkpoint（未覆盖） | `runs/train-20m-e16-1m/ckpt_999424.pt`、`ckpt_500736.pt` |

新包干净目录验证：`runs/evaluation/package_audit_run_dragapult.json`。
4 局（两座位各 2 局）`ok=True`、`binaries=[]`、0 异常、0 非法动作；这 4 局主策略
没有遇到攻击选项，所以 `would_ko` 写入为 0。另用真实 dragapult 攻击 obs 直接调用
`annotate_would_ko`，2 个攻击选项写入 2 个 `would_ko` 单元格（KO 率 0.0，该局面确实不 KO），
证明代码路径对该牌组有效。

### 8. 限制与未验证

1. **引擎 RNG 不可设置种子**：同一 seed、同一牌组、同一进程内重跑，逐局结果不同
   （审计报告 §2.5 已有证据）。本报告所有结论以两种子的合并 CI 为准，不把单次 100 局
   当作可复现实验。
2. **小样本噪声**：第一轮 8 局把 aichi_dragapult 估到 0.875，第二轮 20 局只有 0.45；
   8 局只用于排序。正式结论只取 100 局组。
3. **只测了一个候选 checkpoint**（`ckpt_999424.pt`）；没有让 `ckpt_500736.pt` 亲自
   驾驶新牌组（它只作为对手）。因此“新牌组对旧模型是否同样有效”未验证。
4. **训练分布混杂**：dragapult 很可能在训练池内，当前牌组明确不在；未做
   “同结构、训练外”消融，不能把提升全部归因于牌组结构。
5. **Linux 运行**：未执行（执行约束要求所有命令/Python 在 conda `deeplearning`，
   WSL 属于另一环境）；Linux 引擎仅做静态 ELF/GLIBC 验证。
6. **真实 Kaggle 沙箱**：比赛已结束，平台是否提供 `kaggle_environments` 或注入引擎
   二进制未验证；新包的无二进制方案依赖官方运行时包。
7. **第三种子未跑**：dragapult 的直接对局 CI 含 0.5，替换提交牌组前建议再跑一组
   100 局交叉确认；本次因时间限制未做。

### 9. 复现命令

环境（Windows，conda `deeplearning`）：

```bat
conda activate deeplearning
set PYTHONPATH=.;C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim
set PKMN_TURN_CAP=200
```

生成/校验候选：

```bat
python scripts\deck_search.py
```

单组正式对局（以 dragapult 对随机为例）：

```bat
python runs\evaluation\eval_h2h.py --opponent random --games 100 --seed 20261021 ^
  --pool-json runs\evaluation\deck_search\pools\official_dragapult.json ^
  --out runs\evaluation\deck_search\final\drag_vs_random_20261021_100.json
```

同模型交叉对局：

```bat
python runs\evaluation\eval_h2h.py --opponent selfmirror --mode cross --games 100 --seed 20261022 ^
  --pool-json runs\evaluation\deck_search\pools\official_dragapult.json ^
  --net-pool-json runs\evaluation\deck_search\pools\control_deckcsv.json ^
  --out runs\evaluation\deck_search\final\cross_drag_vs_control_20261022_100.json
```

导出新候选包：

```bat
python scripts\export_rl_submission.py --ckpt runs\train-20m-e16-1m\ckpt_999424.pt ^
  --backend transformer2 --deck runs\evaluation\deck_search\best_deck.csv ^
  --out submission_rl_transformer2_dragapult_nodist.tar.gz ^
  --builddir submission_rl_transformer2_dragapult_nodist
```

---

# 第三部分：评估工具修复

## 评估工具修复（gates_fix）

日期：2026-10-20。对象：`rl/gates.py` 的 `_field_candidates` 与依赖它的
`scripts/h2h_ckpt.py`；另记录为保证 runner 可运行而做的原生编码回退。

### 1. 缺陷与调用路径

- `rl/gates.py:235`（修复前）`field_winrate_fast` 调用 `_field_candidates(len(agent_pool), n_games)`，
  全仓库无定义。`scripts/h2h_ckpt.py:77` 通过 `field_winrate_fast(..., mirror=True)` 使用它。
- 因此 `scripts/h2h_ckpt.py` 在修复前必然 `NameError`；仓库自带 h2h runner 不可用。
- 该符号在基础 squash 提交 `6e7829e` 中同样不存在，无历史实现可回退。

### 2. 语义与最小实现

调用约定（`rl/gates.py`）：

- `mirror=True`：`sched = [(c, c) for c in cand][rank::world]`，`cand` 必须长度 `n_games`；
- `cross`：`sched = [(cand[g], seq[g]) for g in range(n_games)][rank::world]`。

因此 `_field_candidates(n_decks, n_games)` 的契约是：

1. 长度恰为 `n_games`；
2. 每个元素是 `[0, n_decks)` 的池索引；
3. 纯函数（同一参数每次 gate 完全相同，配对比较成立）；
4. 均匀覆盖池。

最小实现：`[g % n_decks for g in range(n_games)]`；`n_decks <= 0` 抛 `ValueError`。
这满足上述四条，且不引入随机性/外部状态。**无法从现有代码确定的部分**：原实现是否使用
stride、洗牌或 pilot-swap 块；本实现选择最简单的确定性轮转，并在回归测试中把这一语义固定。
如果未来要求与某个历史 gate 数值逐局对齐，需要原始实现或历史 JSON；当前仓库没有这些证据。

### 3. 原生编码回退（runner 可运行的必要条件）

`_gate_env` 原先硬编码 `native_encode=True`，本机官方引擎没有 `GetBinaryObs/AdvanceLog`，
且 `encode_native` 未构建，因此即使修好 `_field_candidates`，runner 仍会在首个 env 处失败。

修复：`_native_encode_available()` 检查 `encode_native` 模块与引擎导出；可用则保持
`native_encode=True`，不可用则回退 `native_encode=False`（Python 编码）。编码器语义相同
（仓库已有 parity 验证），差异是速度；`_gate_schedule`、座位分配、胜负判定、turn-cap
均未改动。该回退只影响本机不可用原生路径的情况，有原生环境的机器保持原行为。

### 4. 回归测试

`tests/test_gates.py` 新增：

- `test_field_candidates_cycle_pure_and_complete`：长度、纯函数、范围、全池覆盖、
  `30/100` 的前 32 项、`n_decks=1` 退化；
- `test_field_candidates_rejects_empty_pool`：`n_decks=0` 抛 `ValueError`。

结果：

```text
$ python -m pytest tests/test_gates.py -q
.....                                                                    [100%]
5 passed in 1.55s
```

（原有 3 个 `_gate_schedule` 测试保持不变，全部通过。）

### 5. Runner 实跑

```text
$ python scripts/h2h_ckpt.py runs/train-20m-e16-1m/ckpt_999424.pt \
    runs/train-20m-e16-1m/ckpt_500736.pt 4 \
    --pool-json runs/evaluation/pool_top15.json \
    --pool-weights runs/evaluation/pool_top15_w.npy --sample 30 --live 4 \
    --out runs/evaluation/h2h_ckpt_test --tag smoke
H2H ... games=4 ... wrA=0.7500 (3/4 draws=0 errs=0)

$ python scripts/h2h_ckpt.py ... 20 ... --tag smoke20
H2H ... games=20 ... wrA=0.4500 (9/20 draws=0 errs=0)
```

输出：`runs/evaluation/h2h_ckpt_test/smoke_r0of1.json`、`smoke20_r0of1.json`。
两次都是 0 异常、0 平局；4 局 0.75 与 20 局 0.45 的差异来自样本量，不能当作稳定胜率。

### 6. 未验证 / 限制

1. **未跑完整 200 局 gate**：本次只验证 runner 可运行；20 局不足以判断模型强弱。
2. **Python 编码回退较慢**：本机 runner 使用 `native_encode=False`；有 `libcg_custom.so` +
   `encode_native` 的训练机仍走原生路径。
3. **`_field_candidates` 的原始语义不可考**：仓库无历史实现；本实现是最小、确定性、
   均匀覆盖的版本。若要与旧 gate 数值逐局对齐，需要原始证据。
4. **`h2h_ckpt.py` 的 `--seed` 不控制 `field_winrate_fast` 的内部 seed**：脚本内部固定
   `seed=4321`，`--seed` 只影响别处/未使用；这不是本次改动，但报告数字时需注明。

---

# 第四部分：训练决策

## 训练决策（training_decision）

日期：2026-10-20。依据本文件第二部分（牌组搜索与对照评估）、第一部分（提交包运行阻塞审计）
与 `runs/evaluation/audit_report.md` 的实测结果。**本轮没有启动任何新训练。**

### 1. 决策

**暂不启动新一轮长训练；先把提交牌组定稿，再做一次短程、带对照的微调。**

理由：

1. 提交阻塞的修复已落地，但 Linux/Kaggle 真实运行时仍未验证；在部署路径未闭环前启动长训练，
   可能再次把算力花在无法提交的模型上。
2. 牌组是当前最大的可控变量：同一 `ckpt_999424.pt`，仅把提交牌组换成 dragapult，
   对随机从 0.2200 升到 0.5400，对 500736 从 0.4450 升到 0.6900。这个提升不需要训练。
3. 直接对局仍有非传递性：k36 对随机/旧 checkpoint 都更好，却被当前牌组 0.755 克制；
   dragapult 对当前牌组仅 0.57/0.44（CI 含 0.5）。在牌组选择没有通过第三种子确认前，
   训练目标（对谁更强）本身不稳定。
4. 评估工具已修复（`_field_candidates` + 回归测试 + `h2h_ckpt.py` 可运行），
   下一轮训练可以用仓库自带 runner 做门禁，不必再依赖临时脚本。

### 2. 现有训练接口能支持什么（已核对代码）

| 需求 | 支持情况 | 证据 |
|---|---|---|
| 把 dragapult / 当前提交牌组加入训练池 | **支持** | `rl/train_selfplay.py` 的 `--decks ladder_all` + `--ladder-decks-file`；`load_ladder_decks` 接受 `{"decks": [...], "counts": [...]}`，每副必须 60 张 |
| 按权重混合牌组 | **支持** | `load_ladder_decks` 的 `counts` 作为采样权重；`TwoSidedSelfPlayEnv._sample_decks` 支持 `deck_weights` |
| 自博弈 PPO（同一 net 两座位） | **支持** | `rl/train_selfplay.py` 文件头注释与 `--decks` 路径 |
| 冻结旧 checkpoint 作为对手 | **支持** | `--opponent-ckpt`、`--opponent-decks`（finetune 模式） |
| 随机合法策略作为训练对手 | **不支持（需改代码）** | 采集循环只有 self-play / 冻结 net 两种对手；没有 random-policy 分支。要加需在 `rl/vec_env_selfplay.py` / 采集路径新增动作源与值函数处理 |
| `would_ko=True` 训练 | **支持（默认）** | `--would-ko` 默认 True；`--native-encode` 与 would_ko 不兼容（代码显式报错） |
| 在 Windows 本机跑 PPO | **支持** | `D:\ProgramFiles\deeplearning` + `kaggle_environments` shim；GPU 可用 |

结论：**“把新牌组加入训练分布”可以直接做；“随机合法策略作为训练对手”需要先改采集代码，不能
在现有接口上直接宣称支持。**

### 3. 建议的下一轮（需用户批准后才启动）

#### 3.1 先做（不训练，1–2 小时）

1. 用第三个种子（如 20261023）复跑 dragapult vs 当前牌组的直接对局 100 局；
   若 dragapult 仍 ≥0.5，才把它定为提交牌组。
2. 把 dragapult 与当前牌组都加入一个候选 pool JSON，用 `h2h_ckpt.py` 做一次
   999424 vs 500736 的 200 局门禁，确认新牌组下模型进步仍在。
3. 确认 Kaggle 运行时是否提供 `kaggle_environments`（或在 agent 目录注入引擎）；
   若不能确认，提交包仍只能作为“本地可运行候选”，不能宣称平台可运行。

#### 3.2 再做（短程微调，不增加总步数掩盖问题）

- 初始化：`ckpt_999424.pt`。
- 牌组池：`TRAIN_TOP15` + 当前 `deck.csv` + dragapult，权重先按 1:1:1 或
  top15 占 0.5、两个提交候选各 0.25；不要只放 dragapult（会把模型过拟合到单副牌）。
- 步数：先做 **10–20M 步**的短程微调，用同一评估矩阵（随机/500736/自镜像/交叉对局）
  作为门禁；不建议直接上 3M–5M 的“加大步数”方案，也不建议在牌组未定稿时开长跑。
- 若要让随机合法策略进入训练：先单独实现并测试 random-opponent 分支（动作源、
  掩码、值目标、日志），再加进 PPO；这是一项代码改动，不是配置项。
- 训练后用 `h2h_ckpt.py` + `eval_h2h.py` 双门禁；禁止只用训练内 reward 判断。

### 4. 不做什么

- 不覆盖 `runs/train-20m-e16-1m/ckpt_999424.pt` / `ckpt_500736.pt`。
- 不覆盖 `submission_rl_transformer2.tar.gz`。
- 不把引擎二进制塞进对外发布的包（许可未明确；当前方案运行时解析官方安装）。
- 不以“步数从 1M 加到 3M–5M”替代牌组/部署修复。
- 不在 Linux/Kaggle 路径未验证时声称提交包已在平台通过。