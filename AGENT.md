# Global Agent Contract

## 0. 定位与分层

- **性质**：全局操作契约。适用于几乎每个任务、每个项目、每个读 `AGENTS.md` 的编码 agent。
- **内容边界**：只含全局行为。机器路径、工具、领域流程在 `D:/ai-configs/rules/`。
- **分层（每个事实只放一层，其余层引用）**：

```text
本文件（全局契约）
  → rules/README.md（索引）
    → 适用的规则文件（领域细节）
      → AI_GLOSSARY.md（术语）
```

## 1. 运行环境（硬性）

- **Conda 环境**：所有命令、脚本、Python 执行一律在 conda 的 `deeplearning` 环境中运行；不得在 base 或其他环境执行。执行前确认环境：`CONDA_DEFAULT_ENV=deeplearning`。
- **GPU**：涉及计算、深度学习、数据处理、模型训练/推理的任务，必须使用 GPU，执行前验证可用（`nvidia-smi` 或 `torch.cuda.is_available()` 为 True）。
- **不适用情形**：纯文本、文档、注释、配置文件类任务无计算对象，GPU 条款对其不适用；但若执行任何命令，仍受上一条环境约束。
- **冲突处理**：GPU 不可用或环境无法满足时，**停止并报告**，不得降级到 CPU 或 base 环境静默执行（同 §7.7 指令优先级）。

## 2. 语言规范

- 回复用户用**中文**；代码注释、项目文档用**中文**。
- **GitHub Issue 例外（硬性）**：issue 的标题和正文必须用**英文**，覆盖上一条——但仅限 issue 内容；总结该 issue 的回复仍是中文。
- 不翻译项目名、路径、命令、环境变量、产品名。
- 简洁直接（caveman style）；若 `JuliusBrussee/caveman` skill 可用则用，本仓库不要求也不安装。

## 3. 交互行为

### 3.1 先探索再提问（顺序固定）

1. 查当前工作目录和打开的项目：先列目录，再读相关文件。
2. 查本会话历史和已有 memory。
3. 读 `D:/ai-configs/rules/` 下适用文件。
4. 读已注册 `D:` 路径下明显相关的只读材料。
5. 本地源都失败，才用网络搜索和官方/公开文档。

仅在探索无法提供最低信息时才问。问时须：报告探索了什么、还缺什么，并提出最可能的默认选项待确认。

### 3.2 先想再做

实现前明确：重要假设、实质不确定性、存在的多种解释（列出而非默默选一个）、有意义的权衡、更简单的有效方案（有就指出）。关键歧义/矛盾/逻辑不一致阻塞安全实现时，停下问。**绝不关键歧义下猜测**。琐碎任务用判断力，省去不必要流程。

### 3.3 填空/组织类任务：先起草再问（硬触发）

- 触发词：`help me write / organize / fill in / do / generate / complete`；`write a / write one / organize / fill in / complete / polish / format`；以及以 `based on… / in the format of… / referring to…` 开头的组织或生成请求。
- 步骤：
  1. 先探索可用上下文；
  2. 立即产出完整草稿；
  3. 真正缺失字段用 `[to be filled]`；
  4. 草稿后列出剩余缺口；
  5. 草稿后才问必要问题。
- 仅当连有用的草稿骨架都无法确定，才先问。

### 3.4 显式操作请求直接用工具执行

多步工作：先给简短计划（含验证标准），然后执行，无需额外确认——除非关键歧义需授权。

### 3.5 交付简洁

给核心结果和重要决定。避免：重述请求、重复已建立上下文、回声本契约、不影响结果的流程细节。

## 4. 实现原则

### 4.1 简洁优先（最小实现）

最小实现完全解决任务。不要：超出请求的功能、单次使用代码的抽象、投机架构、无具体需求的灵活性/可配置性、不可能场景的错误处理。偏好最小连贯改动；方案不成比例地复杂时先简化。

自问：*资深工程师会觉得这对实际需求过于复杂吗？* 是则先简化。

### 4.2 手术式改动（只碰任务需要的）

- 写前读目标文件和相关上下文；**绝不盲目覆盖**。
- 改最小可能面；保持现有项目风格和约定。
- 不改进相邻代码/注释/格式，不重构无关的工作代码；授权目标之外的一切保持不变。
- 每条改动行都要有可追溯到用户请求的直接理由。
- 产生未使用代码时：移除**自己改动造成**的未使用 import/变量/函数；保留无关的既有死代码（相关时作注记）。
- 未经当前轮次对确切目标的明确授权，不删除文件/目录/内容（见 `rules/filesystem.md`）。

### 4.3 目标驱动执行与验证

*Done 意味着已执行并验证。* 实现前定义成功标准；把模糊目标转成可验证结果（如“加校验”→先测无效输入再让测试通过；“修 bug”→先复现再修；“重构 X”→改前改后测试都过；“改行为”→加/改测试证明预期行为）。

多步工作先声明计划：

```text
1. [步骤] → verify: [验证方式]
2. [步骤] → verify: [验证方式]
```

持续到验收标准满足；报告执行的命令/测试及其结果作为证据。完整测试流程见 `rules/testing.md`。

## 5. 规则引用机制

改代码或配置前：

1. 读 `D:/ai-configs/rules/README.md`；
2. 识别适用规则；
3. 只读那些规则文件；
4. 行动前遵守。

**不要读每个规则文件。**

Git/GitHub 工作前读：`rules/git.md`、`rules/Github-KB.md`。

## 6. 硬边界（不可协商）

### 6.1 DeepSeek Harness

绝不在原地修复 DSH。故障排查与修复属于固定链：

```text
user → DSH → conductor skill → Claude Code → DSH-Repair subagent → Web Profile
```

DSH 开单并移交；`DSH-Repair` 是唯一负责诊断和修复的组件。DeepSeek 官方产物保持只读。任何 DSH 故障、配置/依赖问题、插件故障或安装健康问题，遵循 `rules/dsh.md`，不要在此直接修复。

### 6.2 文件系统

- **绝不永久删除**任何东西，除非用户在当前轮次点名确切目标（`rules/filesystem.md`）。
- **绝不在可写根之外写**：`D:/ai-configs` 是 agent 配置的唯一真相源和唯一可写内容根；`D:/search` 只读。路径与安装目标见 `rules/environment.md`。

### 6.3 Agent 配置

绝不在 `D:/ai-configs` 之外安装 agent、命令、MCP、规则或技能。规范目标：`agents/` `commands/` `mcp/` `rules/` `skills/` `workflows/`。

不得安装非托管副本到：`~/.claude/`、`~/.agents/`、`~/.codex/`、`~/.cache/opencode/`、`~/.config/opencode/` 或项目目录。

`~/.claude/{agents,commands,mcp,rules,workflows}` 是指向本仓库的符号链接，故规范目标改动立即生效。

### 6.4 技能

`manage-skills` 是技能的唯一管理入口：安装、更新、移除、部署、采纳、打标签、组织技能都走它。

禁止：手工把技能复制进 agent 目录；用已废弃的 `npx skills add` CLI 管理技能。

开发与管理分开——编写和评估 `SKILL.md` 可直接做。

新技能流程：

```text
requirements → development → local verification → upload to GitHub → shared library → install and register with manage-skills → install verification
```

规范库路径、部署模型、安装语义、验证标准见 `rules/skills.md`。

### 6.5 第三方许可

绝不从仓库级标记推断许可证范围。文本与程序源代码**分开判断**；`GPL` 标记不能确立其旁文档、示例、模板或 vendored 文件的任何东西。

每个许可结论都带证据：精确定位的上游许可证文本 + 原文件的 `SPDX-License-Identifier` 或版权行；判断须指名路径或 glob。无证据则答案是 **“undeclared”**——停止复制和再分发，把问题返回。

`check-the-licence-before-you-copy` skill 执行此边界开启的署名义务；判断标准与证据格式见 `rules/license.md`。

### 6.6 自著项目许可

从零创作的新项目默认 **`GPL-3.0-or-later`**。创作携带五个动作：根 `COPYING`、手写源文件的逐文件 `SPDX-License-Identifier` 头、包清单 license 字段、版权与许可通知、程序从 stdin 读取时的交互式启动横幅。Vendored 第三方内容保留其上游标识符，不参与此默认。

选其他许可证是用户的决定：说明 copyleft 成本、在当前轮次请求明确授权、然后停在那里。`Apache-2.0` 代码仅在明确 `WITH Apache-2.0-exception` 下加入 GPLv3 项目；`GPL-2.0-only` 代码不兼容。流程见 `rules/license.md` §6。

### 6.7 指令优先级

绝不绕过更高优先级指令。请求与硬边界冲突时：

1. 停；
2. 报告冲突和支持证据；
3. 适用处请求明确授权。

## 7. 术语

`@D:/ai-configs/AI_GLOSSARY.md` 是术语真相源。遇不熟悉的 AI 术语查阅；**不要在其他地方重复术语定义**。

---

[PTCG-Arena · AI 对战比赛平台](http://115.159.47.81/)

用户名：`Ianwusb`

密码：`Wc_Mb37r-ryhdGf`

# 比赛规则与 Agent 接口规范

## 一、赛制与积分

- 赛程不设轮次上限，默认每 1 小时自动触发一轮，管理后台可调整倒计时。
- A 榜开发期轮数由管理员决定，仅作参考，不计入最终排名。
- 管理员结束后进入 B 榜正式排名期，起始轮次以管理员启动时为准，累积积分决定最终名次。
- 每轮为单循环赛，每对 Agent 对战 1 局。
- 每轮按名次赋分：
  - 第 1 名 100 分；
  - 最后一名 10 分；
  - 中间按本轮实际参赛队数线性递减；
  - 所有实际参赛队都能获得积分。
- 仅实际参赛（提交有效 Agent 并参加该轮）的队伍获得积分；未提交或未参赛不得分，也不上榜。
- B 榜总积分 = B 榜启动后各轮积分之和。
- 同分处理：比较最近一轮名次，依次往前；仍相同按团队名字典序。

## 二、提交物与函数签名

提交物 `submission.tar.gz` 内含：

- `main.py`：定义 `agent(obs_dict)`；
- `deck.csv`：60 行卡牌 ID。

函数签名：

```python
def agent(obs_dict: dict) -> list[int]:
    """
    obs_dict：当前观测，含 select / current / logs 三部分。
    返回：选中的动作下标列表，长度须满足
    obs_dict["select"]["minCount"] ≤ 长度 ≤ obs_dict["select"]["maxCount"]。
    """
```

说明：

- 开局选牌阶段平台不会调用 agent，而是直接用 `deck.csv` 组牌。
- 正常对局中 `obs_dict["select"]` 始终非 `None`。
- 为兼容官方 Kaggle 写法，可在函数开头对 `select is None` 返回任意列表（如空列表）。

## 三、obs_dict 顶层结构

```python
obs_dict = {
    "select":  { ... } | None,   # 本轮需决策的选择；正常对局中非 None
    "logs":    [ {...}, ... ],   # 对局日志
    "current": { ... },          # 当前局面
}
```

## 四、current 结构

```python
current = {
    "turn": int,             # 回合数
    "turnActionCount": int,  # 本回合已执行的动作数
    "yourIndex": int,        # 你的玩家索引（0 或 1）
    "firstPlayer": int,      # 先手玩家索引
    "supporterPlayed": bool, # 本回合是否已用支援者
    "stadiumPlayed": bool,   # 本回合是否已用竞技场
    "energyAttached": bool,  # 本回合是否已贴能量
    "retreated": bool,       # 本回合是否已撤退
    "result": int,           # -1 进行中；0 = 玩家0胜；1 = 玩家1胜
    "stadium": [ Card ],       # 竞技场卡（0 或 1 张）
    "looking": [ Card ] | None,# 正在查看的卡（如检索效果）
    "players": [ PlayerState, PlayerState ],
}
```

### PlayerState 字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `active` | `[Pokemon \| null]` | 出战区（无宝可梦时为 `[null]`） |
| `bench` | `[Pokemon]` | 备战区（最多 `benchMax` 只） |
| `benchMax` | `int` | 备战区上限（通常 5） |
| `deckCount` | `int` | 牌库剩余张数 |
| `discard` | `[Card]` | 弃牌堆 |
| `prize` | `[Card \| null]` | 奖赏卡（6 张，已取走的为 `null`） |
| `handCount` | `int` | 手牌张数 |
| `hand` | `[Card] \| null` | 手牌明细（仅自己可见，对手为 `null`） |
| `poisoned` | `bool` | 出战宝可梦中毒 |
| `burned` | `bool` | 出战宝可梦灼伤 |
| `asleep` | `bool` | 出战宝可梦睡眠 |
| `paralyzed` | `bool` | 出战宝可梦麻痹 |
| `confused` | `bool` | 出战宝可梦混乱 |

可见性：

- 只能看到自己的 `hand`（对手为 `null`）；
- `deckCount`、`handCount`、`discard`、`prize` 数量双方可见。

### Pokemon 字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id` | `int` | 卡牌 ID（对应 `cards.csv` 的 `cardId`） |
| `serial` | `int` | 卡牌序列号（区分同名多张） |
| `playerIndex` | `int` | 所属玩家 |
| `hp` | `int` | 当前 HP |
| `maxHp` | `int` | 最大 HP |
| `appearThisTurn` | `bool` | 本回合是否刚上场（进化/换上） |
| `energies` | `[int]` | 已贴能量类型列表（如 `[3,3]` = 2 个水） |
| `energyCards` | `[Card]` | 贴附的能量卡 |
| `tools` | `[Card]` | 贴附的道具卡 |
| `preEvolution` | `[Pokemon]` | 进化前形态（叠放在下面） |

注意：宝可梦对象里没有 `attacks` 字段；招式与伤害需用 `id` 去 `cards.csv` 查询。

### Card 字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id` | `int` | 卡牌 ID |
| `serial` | `int` | 卡牌序列号 |
| `playerIndex` | `int` | 所属玩家 |

`Card` 只有这三个字段，没有名称/HP/类型，需用 `id` 去 `cards.csv` 查名称与属性。

## 五、logs 结构

`logs` 是事件列表，每个元素含 `type`（`LogType`）及对应字段。

| type | 名称 | 主要字段 |
| --- | --- | --- |
| 0 | SHUFFLE | `playerIndex` |
| 1 | HAS_BASIC_POKEMON | `playerIndex`, `hasBasicPokemon` |
| 2 | TURN_START | `playerIndex` |
| 3 | TURN_END | `playerIndex` |
| 4 | DRAW | `playerIndex`, `cardId`, `serial` |
| 5 | DRAW_REVERSE | `playerIndex` |
| 6 | MOVE_CARD | `playerIndex`, `cardId`, `serial`, `fromArea`, `toArea` |
| 7 | MOVE_CARD_REVERSE | `playerIndex`, `fromArea`, `toArea` |
| 8 | SWITCH | `playerIndex`, `cardIdActive`, `serialActive`, `cardIdBench`, `serialBench` |
| 9 | CHANGE | 保留类型，常规对局中未出现 |
| 10 | PLAY | `playerIndex`, `cardId`, `serial` |
| 11 | ATTACH | `playerIndex`, `cardId`, `serial`, `cardIdTarget`, `serialTarget` |
| 12 | EVOLVE | `playerIndex`, `cardId`, `serial`, `cardIdTarget`, `serialTarget` |
| 13 | DEVOLVE | `playerIndex`, `cardId`, `serial`, `cardIdTarget`, `serialTarget` |
| 14 | MOVE_ATTACHED | `playerIndex`, `cardId`, `serial`, `cardIdBefore`, `serialBefore`, `cardIdAfter`, `serialAfter` |
| 15 | ATTACK | `playerIndex`, `cardId`, `serial`, `attackId` |
| 16 | HP_CHANGE | `playerIndex`, `cardId`, `serial`, `value`, `putDamageCounter` |
| 17 | POISONED | `playerIndex`, `cardId`, `serial`, `isRecover` |
| 18 | BURNED | `playerIndex`, `cardId`, `serial`, `isRecover` |
| 19 | ASLEEP | `playerIndex`, `cardId`, `serial`, `isRecover` |
| 20 | PARALYZED | `playerIndex`, `cardId`, `serial`, `isRecover` |
| 21 | CONFUSED | `playerIndex`, `cardId`, `serial`, `isRecover` |
| 22 | COIN | `playerIndex`, `head` |
| 23 | RESULT | `result`, `reason` |

`fromArea` / `toArea` / `area` 取值见 AreaType。

## 六、select 与 option 结构

```python
select = {
    "type": int,        # SelectType
    "context": int,     # SelectContext（具体场景）
    "minCount": int,    # 最少需选数量
    "maxCount": int,    # 最多可选数量
    "remainDamageCounter": int,
    "remainEnergyCost": int,
    "option": [ {...}, ... ],  # 所有合法动作
    "deck": [Card] | None,     # 某些选择（如卡组检索）给出的卡组
    "contextCard": Card | None,
    "effect": Card | None,
}
```

说明：

- `context` 用于判断当前选择在做什么（检索 / 弃牌 / 选奖赏卡 / 选能量 / 进化等）。
- 同样是 `SelectType.CARD`：
  - `context = 7(TO_HAND)`：检索；
  - `context = 8(DISCARD)`：弃牌；
  - `context = 11(TO_PRIZE)`：选奖赏卡。
- `effect`：触发本次选择的来源卡，结构为 `Card`（`{ id, playerIndex, serial }`）；若不是由某张卡触发则为 `null`。用其 `id` 去 `cards.csv` 查效果文本。

### option 字段

每个 `option` 是 dict，`type` 决定它带哪些字段（未列出即不出现）。

| type | 名称 | 含义 | 携带字段 |
| --- | --- | --- | --- |
| 0 | NUMBER | 选择数值 | `number` |
| 1 | YES | 是 | — |
| 2 | NO | 否 | — |
| 3 | CARD | 选卡 | `area`, `index`, `playerIndex` |
| 4 | TOOL_CARD | 选道具卡 | `area`, `index`, `playerIndex`, `toolIndex` |
| 5 | ENERGY_CARD | 选能量卡 | `area`, `index`, `playerIndex`, `energyIndex` |
| 6 | ENERGY | 选能量 | `area`, `index`, `playerIndex`, `energyIndex`, `count` |
| 7 | PLAY | 打出卡牌 | `index` |
| 8 | ATTACH | 贴能量 | `area`, `index`, `inPlayArea`, `inPlayIndex` |
| 9 | EVOLVE | 进化 | `area`, `index`, `inPlayArea`, `inPlayIndex` |
| 10 | ABILITY | 发动特性 | `area`, `index` |
| 11 | DISCARD | 弃卡 | `area`, `index` |
| 12 | RETREAT | 撤退 | — |
| 13 | ATTACK | 攻击 | `attackId` |
| 14 | END | 结束回合 | — |
| 15 | SKILL | 技能 | `cardId`, `serial` |
| 16 | SPECIAL_CONDITION | 特殊状态 | `specialConditionType` |

关键结论：

- `ATTACK` 只有 `attackId`，没有伤害 `number`；伤害用 `attackId` 查 `cards.csv`。
- `PLAY` 只有 `index`，没有 `cardId`；需用 `index` 在自己 `hand` 里定位卡牌。
- `ATTACH` / `EVOLVE` 用 `area + index` 定位来源卡，用 `inPlayArea + inPlayIndex` 定位场上宝可梦。
- 只有 `ENERGY`（type 6）带 `count`，对应几个能量单位。

## 七、枚举速查

### OptionType（option.type）

| 值 | 名称 | 含义 |
| --- | --- | --- |
| 0 | NUMBER | 选择数值 |
| 1 | YES | 是 |
| 2 | NO | 否 |
| 3 | CARD | 选卡 |
| 4 | TOOL_CARD | 选道具卡 |
| 5 | ENERGY_CARD | 选能量卡 |
| 6 | ENERGY | 选能量 |
| 7 | PLAY | 打出卡牌 |
| 8 | ATTACH | 贴能量 |
| 9 | EVOLVE | 进化 |
| 10 | ABILITY | 发动特性 |
| 11 | DISCARD | 弃卡 |
| 12 | RETREAT | 撤退 |
| 13 | ATTACK | 攻击 |
| 14 | END | 结束回合 |
| 15 | SKILL | 技能 |
| 16 | SPECIAL_CONDITION | 特殊状态 |

### AreaType（area / inPlayArea / fromArea / toArea）

| 值 | 名称 | 含义 |
| --- | --- | --- |
| 1 | DECK | 卡组 |
| 2 | HAND | 手牌 |
| 3 | DISCARD | 弃牌堆 |
| 4 | ACTIVE | 出战区 |
| 5 | BENCH | 备战区 |
| 6 | PRIZE | 奖赏区 |
| 7 | STADIUM | 竞技场 |
| 8 | ENERGY | 能量 |
| 9 | TOOL | 道具 |
| 10 | PRE_EVOLUTION | 进化前 |
| 11 | PLAYER | 玩家 |
| 12 | LOOKING | 查看中 |

### SelectType（select.type）与 OptionType 对应

| 值 | 名称 | 含义 | 会出现的 OptionType |
| --- | --- | --- | --- |
| 0 | MAIN | 主决策 | PLAY / ATTACH / EVOLVE / ABILITY / DISCARD / RETREAT / ATTACK / END |
| 1 | CARD | 选卡 | CARD |
| 2 | ATTACHED_CARD | 选已贴附卡 | TOOL_CARD / ENERGY_CARD |
| 3 | CARD_OR_ATTACHED_CARD | 选卡或已贴附卡 | CARD / TOOL_CARD / ENERGY_CARD |
| 4 | ENERGY | 选能量 | ENERGY |
| 5 | SKILL | 选技能 | SKILL |
| 6 | ATTACK | 选攻击 | ATTACK |
| 7 | EVOLVE | 进化 | EVOLVE |
| 8 | COUNT | 选数量 | NUMBER |
| 9 | YES_NO | 是否 | YES / NO |
| 10 | SPECIAL_CONDITION | 特殊状态 | SPECIAL_CONDITION |

### SelectContext（select.context）

| 值 | 名称 | 含义 |
| --- | --- | --- |
| 0 | MAIN | 主行动阶段 |
| 1 | SETUP_ACTIVE_POKEMON | 开场布置出战宝可梦 |
| 2 | SETUP_BENCH_POKEMON | 开场布置备战宝可梦 |
| 3 | SWITCH | 交换出战/备战 |
| 4 | TO_ACTIVE | 移动到出战区 |
| 5 | TO_BENCH | 移动到备战区 |
| 6 | TO_FIELD | 移动到场上 |
| 7 | TO_HAND | 加入手牌（检索） |
| 8 | DISCARD | 弃牌 |
| 9 | TO_DECK | 放回牌库 |
| 10 | TO_DECK_BOTTOM | 放回牌库底部 |
| 11 | TO_PRIZE | 放入奖赏区 |
| 12 | NOT_MOVE | 不移动 |
| 13 | DAMAGE_COUNTER | 放置伤害指示物 |
| 14 | DAMAGE_COUNTER_ANY | 任意放置伤害指示物 |
| 15 | DAMAGE | 造成伤害 |
| 16 | REMOVE_DAMAGE_COUNTER | 移除伤害指示物 |
| 17 | HEAL | 回复 HP |
| 18 | EVOLVES_FROM | 选择进化前形态 |
| 19 | EVOLVES_TO | 选择进化后形态 |
| 20 | DEVOLVE | 退化选择 |
| 21 | ATTACH_FROM | 选择贴附来源 |
| 22 | ATTACH_TO | 选择贴附目标 |
| 23 | DETACH_FROM | 选择卸下来源 |
| 24 | LOOK | 查看牌 |
| 25 | EFFECT_TARGET | 选择效果目标 |
| 26 | DISCARD_ENERGY_CARD | 弃能量卡 |
| 27 | DISCARD_TOOL_CARD | 弃道具卡 |
| 28 | SWITCH_ENERGY_CARD | 替换能量卡 |
| 29 | DISCARD_CARD_OR_ATTACHED_CARD | 弃手牌或贴附卡 |
| 30 | DISCARD_ENERGY | 弃能量（单位） |
| 31 | TO_HAND_ENERGY | 能量回手 |
| 32 | TO_DECK_ENERGY | 能量回牌库 |
| 33 | SWITCH_ENERGY | 替换能量（单位） |
| 34 | SKILL_ORDER | 特性结算顺序 |
| 35 | ATTACK | 选择招式 |
| 36 | DISABLE_ATTACK | 禁用招式 |
| 37 | EVOLVE | 选择进化 |
| 38 | DRAW_COUNT | 选择抽牌数量 |
| 39 | DAMAGE_COUNTER_COUNT | 选择伤害指示物数量 |
| 40 | REMOVE_DAMAGE_COUNTER_COUNT | 选择移除伤害指示物数量 |
| 41 | IS_FIRST | 是否先手 |
| 42 | MULLIGAN | 重抽（无基础宝可梦） |
| 43 | ACTIVATE | 是否发动 |
| 44 | FIRST_EFFECT | 先结算哪个效果 |
| 45 | MORE_DEVOLVE | 是否继续退化 |
| 46 | COIN_HEAD | 猜硬币正反面 |
| 47 | AFFECT_SPECIAL_CONDITION | 施加特殊状态 |
| 48 | RECOVER_SPECIAL_CONDITION | 解除特殊状态 |

### EnergyType（energies / energyType）

| 值 | 名称 | 含义 |
| --- | --- | --- |
| 0 | COLORLESS | 无色（任意） |
| 1 | GRASS | 草 |
| 2 | FIRE | 火 |
| 3 | WATER | 水 |
| 4 | LIGHTNING | 雷 |
| 5 | PSYCHIC | 超 |
| 6 | FIGHTING | 斗 |
| 7 | DARKNESS | 恶 |
| 8 | METAL | 钢 |
| 9 | DRAGON | 龙 |
| 10 | RAINBOW | 彩虹 |
| 11 | TEAM_ROCKET | 火箭 |

### pokemonType（宝可梦大类，不是 EnergyType）

| 值 | 含义 |
| --- | --- |
| 0 | 非宝可梦（道具 / 支援者 / 竞技场 / 能量卡） |
| 1 | 普通宝可梦 |
| 2 | 化石类宝可梦道具（以道具形态打出的基础宝可梦） |
| 3 | 宝可梦 ex |
| 4 | Mega 进化 ex |

### CardType（cards.csv 的 cardType）

| 值 | 名称 | 含义 |
| --- | --- | --- |
| 0 | POKEMON | 宝可梦 |
| 1 | ITEM | 道具 |
| 2 | TOOL | 宝可梦道具 |
| 3 | SUPPORTER | 支援者 |
| 4 | STADIUM | 竞技场 |
| 5 | BASIC_ENERGY | 基础能量 |
| 6 | SPECIAL_ENERGY | 特殊能量 |

## 八、攻击与能量（cards.csv 格式）

`cards.csv` 是完整卡池（1267 张）。

`attacks` 列格式：

```text
attackId:招式名(伤害)[能量需求列表]
```

多个招式用 `/` 分隔。

示例：

```text
2:High Jump Kick(100)[7,0,0]
```

表示：

- 招式 ID = 2；
- 名称 `High Jump Kick`；
- 伤害 100；
- 能量需求 `[7,0,0]`（1 个恶 + 2 个任意能量）。

说明：

- `attackId` 与 `option` 中 `ATTACK` 的 `attackId`、`logs` 中 `ATTACK` 的 `attackId` 一一对应。
- `[]` 内是能量类型列表（`EnergyType` 枚举），每个数字代表 1 个能量；
- `0`（无色）表示可用任意能量替代。

`skills` 列：

- 给出该卡的特性与效果文本；
- 格式为：`技能/特性名：效果文本`，多张技能用 `/` 分隔；
- 是判断一张卡能做什么的唯一依据；
- 训练家卡、道具卡、竞技场卡的效果以及宝可梦的特性都写在 `skills` 列；
- 用某张卡的 `cardId` 定位该行读取即可，无需外部 TCG 知识。

## 九、组牌规则

`deck.csv` 由平台在提交时校验，必须满足以下全部条件（任一不满足即提交失败）：

1. 恰好 60 行，每行一个整数 `cardId`（空行忽略）。
2. 所有 `cardId` 必须来自 `cards.csv`（不在卡池内的 ID 会被拒绝）。
3. 至少 1 张真正的「基础宝可梦」：
   - 必须是 `cardType = 0`（POKEMON）且 `basic = 1` 的卡；
   - 化石卡不算：Antique Root / Cover / Plume / Jaw / Sail Fossil（1099 / 1136 / 1138 / 1150 / 1151）虽标记 `basic = 1`，但 `cardType = 1`（ITEM）、`pokemonType = 2`，引擎不会把它们视为可开局的基础宝可梦；
   - 只要含 1 张真正的基础宝可梦即可正常开局。
4. 同名卡最多 4 张（按卡名 `name` 统计，不是按 `cardId`）。
   - 例：Scrafty 有两张不同印（`cardId 21` 与 `611`），混用后合计也不能超过 4 张。
   - 基础能量卡（`cardType = 5`）不受此限制，可放任意多张。
5. ACE SPEC 卡全牌组最多 1 张：
   - 按 `aceSpec = 1` 统计；
   - 即使卡名不同，合计也只能有 1 张（如 Precious Trolley、Master Ball 等）；
   - 多放会导致提交校验失败。

其他：

- 平台不额外限制能量数量、无禁用卡；
- 牌组不合法时提交会被拒绝并返回具体原因；
- 建议按 4 张同类基础宝可梦 + 8–15 张基础能量 + 策略卡组牌。

## 十、胜负规则

按官方宝可梦 TCG 规则，满足任一即结束对局：

1. 一方取完 6 张奖赏卡 → 该方胜。
2. 一方场上没有任何宝可梦（出战区与备战区均为空）→ 对方胜。
3. 一方回合开始需抽牌但牌库已空 → 该方负。

对局结果通过 `current.result` 读取：

- `-1`：进行中；
- `0`：玩家 0 胜；
- `1`：玩家 1 胜。

终局 `logs` 末尾的 `RESULT` 日志含 `reason`（结束原因）。

## 十一、返回值规则

- 返回 `list[int]`；
- 长度须满足 `minCount ≤ 长度 ≤ maxCount`；
- 多数选择 `minCount == maxCount`，即「恰好选 `maxCount` 个」；
- 每个元素是 `option` 数组的下标（0-based），且不能重复；
- 建议按升序返回；平台不强制排序；
- 返回非法下标、重复下标或错误长度，该局判负。

## 十二、超时与沙箱

- `agent` 在加载阶段 import 一次，之后每步决策单独调用；
- 单步决策限时：默认 100ms；
- 整局限时：默认 10 分钟；
- 超时判负；
- import / 初始化发生在对局开始前，不计入单步限时；
- 平台固定对局随机种子以保证可复现；
- agent 内若用 `random`，请自行管理种子（平台不会在每次调用前重置它）；
- 返回非法动作、崩溃、超时均判负。

## 十三、cg.api 兼容层

平台沙箱内预置 `cg.api` 模块，可把 dict 转成带字段与类型提示的对象：

```python
from cg.api import to_observation_class, OptionType, AreaType, SelectType

def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)   # dict -> Observation 数据类
    select = obs.select                    # 含 .type / .option / ...
    for i, opt in enumerate(select.option):
        if opt.type == OptionType.END:     # 枚举比较，避免手写数字
            return [i]
    return list(range(select.maxCount))
```

可导入：

- 枚举：`AreaType`、`EnergyType`、`CardType`、`SpecialConditionType`、`SelectType`、`SelectContext`、`OptionType`、`LogType`；
- 数据类：`Observation`、`State`、`PlayerState`、`Pokemon`、`Card`、`Option`、`SelectData`、`Log`、`CardData`、`Attack`、`Skill` 等；
- 函数：`to_observation_class`、`all_card_data`、`all_attack`。

### 数据类字段一览

#### Observation（to_observation_class 的返回类型）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `logs` | `List[Log]` | 对局日志 |
| `current` | `State \| None` | 当前局面 |
| `search_begin_input` | `str \| None` | 检索输入（一般不使用） |
| `select` | `SelectData \| None` | 本次选择 |

#### State（obs.current）

| 字段 | 类型 | 字段 | 类型 |
| --- | --- | --- | --- |
| `turn` | `int` | `supporterPlayed` | `bool` |
| `turnActionCount` | `int` | `stadiumPlayed` | `bool` |
| `yourIndex` | `int` | `energyAttached` | `bool` |
| `firstPlayer` | `int` | `retreated` | `bool` |
| `result` | `int` | `stadium` | `List[Card]` |
| `looking` | `List[Card] \| None` | `players` | `List[PlayerState]` |

#### PlayerState（current.players[i]）

| 字段 | 类型 | 字段 | 类型 |
| --- | --- | --- | --- |
| `active` | `List[Pokemon \| None]` | `handCount` | `int` |
| `bench` | `List[Pokemon]` | `hand` | `List[Card] \| None` |
| `benchMax` | `int` | `poisoned` | `bool` |
| `deckCount` | `int` | `burned` | `bool` |
| `discard` | `List[Card]` | `asleep` | `bool` |
| `prize` | `List[Card \| None]` | `paralyzed` | `bool` |
| `confused` | `bool` |  |  |

#### Pokemon / Card

| 类 | 字段 | 类型 | 含义 |
| --- | --- | --- | --- |
| Pokemon | `id` / `serial` | `int` | 卡牌 ID / 序列号 |
| Pokemon | `hp` / `maxHp` | `int` | 当前 / 最大 HP |
| Pokemon | `appearThisTurn` | `bool` | 本回合是否刚上场 |
| Pokemon | `energies` | `Dict[int, int]` | 已贴能量（数据类中为「类型 → 数量」字典，如 `{3: 2}`；原始 dict 为列表 `[3, 3]`） |
| Pokemon | `energyCards` / `tools` | `List[Card]` | 贴附的能量卡 / 道具卡 |
| Pokemon | `preEvolution` | `List[Pokemon]` | 进化前形态 |
| Card | `id` / `playerIndex` / `serial` | `int` | 卡牌 ID / 所属玩家 / 序列号 |

#### Option（select.option[i]）

| 字段 | 类型 | 字段 | 类型 |
| --- | --- | --- | --- |
| `type` | `int` | `cardId` | `int \| None` |
| `area` | `int \| None` | `serial` | `int \| None` |
| `index` | `int \| None` | `toolIndex` | `int \| None` |
| `playerIndex` | `int \| None` | `energyIndex` | `int \| None` |
| `inPlayArea` | `int \| None` | `count` | `int \| None` |
| `inPlayIndex` | `int \| None` | `specialConditionType` | `int \| None` |
| `attackId` | `int \| None` | `number` | `int \| None` |

#### SelectData（obs.select）

| 字段 | 类型 | 字段 | 类型 |
| --- | --- | --- | --- |
| `type` | `int` | `option` | `List[Option]` |
| `context` | `int` | `deck` | `List[Card] \| None` |
| `minCount` | `int` | `contextCard` | `Card \| None` |
| `maxCount` | `int` | `effect` | `Card \| None` |
| `remainDamageCounter` | `int` | `remainEnergyCost` | `int` |

#### Log（obs.logs[i]）

- 官方 `Log` 数据类仅声明 `type` 字段；
- 其余字段（`cardId`、`playerIndex`、`serial` 等）保存在 `log.data`（dict）中，并可通过属性访问：

```python
for lg in obs.logs:
    if lg.type == LogType.ATTACK:
        print(lg.data["attackId"])   # 也可写 lg.attackId（__getattr__ 转发到 data）
```

#### CardData（all_card_data() 的元素，对应 cards.csv 一行）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `cardId` / `name` | `int / str` | 卡牌 ID / 名称 |
| `cardType` | `int` | CardType 枚举 |
| `pokemonType` | `int` | 宝可梦大类 |
| `evolutionType` | `int` | 进化阶段标记 |
| `retreatCost` | `int` | 撤退能量数 |
| `hp` | `int` | HP |
| `weakness` / `resistance` | `int \| None` | 弱点 / 抗性 |
| `energyType` | `int` | EnergyType 枚举 |
| `basic` / `stage1` / `stage2` | `bool` | 进化阶段 |
| `ex` / `megaEx` / `tera` / `aceSpec` | `bool` | 特殊标识 |
| `evolvesFrom` | `int \| None` | 进化前卡牌 ID |
| `skills` | `List[Skill]` | 特性与效果（`Skill{name, text}`） |
| `attacks` | `List[int]` | 招式 ID 列表 |

## 十四、防作弊与沙箱

- 每个 Agent 在独立沙箱中运行。
- 资源限制：
  - 内存 2GB；
  - CPU 2 核；
  - 进程数上限 100。
- 网络禁用；
- 文件系统只读挂载；
- 每局固定随机种子，保证可复现；
- 单步超时 100ms；
- 单局上限 10 分钟；
- 超时判负。

## 十五、策略报告

- B 榜结束后 24 小时内，提交策略报告（Markdown / PDF / 链接）。
- 评审权重：
  - 稳定性 30%；
  - 独创性 30%；
  - B 榜排名 40%。