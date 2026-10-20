# Linux（WSL2）真实运行时验证报告 · 卡牌库迁移

日期：2026-10-20　目录：`runs/evaluation/card_database_migration/`
证据文件：`runs/card_database_migration/20261020/{wsl_stage4.log, wsl_stage4_rerun.log, wsl_stage4_results.json, wsl_stage4c.json, init_repro_results_windows.json, win_init_repro.py, wsl_stage4_run.py, wsl_stage4_cases.py, wsl_stage4b_harness.py, wsl_stage4c_instrumented.py}`

## 1. 运行环境（实测）

| 项 | 值 |
|:---|:---|
| 发行版 | Ubuntu-24.04（WSL2），内核 `Linux 6.6.87.2-microsoft-standard-WSL2 x86_64` |
| Python | 3.12.3（系统解释器） |
| PyTorch | 2.14.1+cpu（`~/pkmn-linux-verify-20261020/pylibs`） |
| glibc | 2.39（Ubuntu GLIBC 2.39-0ubuntu8.9） |
| 引擎 | 官方 `kaggle-environments==1.33.0`，`…/envs/cabt/cg/libcg.so`（1,458,504 B） |
| 暂存目录 | `~/pkmn-migration-verify-20261020/pkg`（干净解包） |
| 只读性 | 未改动 Windows 侧原提交包、checkpoint 与历史评估产物 |

## 2. 暂存包构造（迁移叠加）

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

## 3. 缺陷定位：单向句柄复用保护

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

### 修复（最小范围）

`sdk_cg/sim.py::_load_lib()` 改为**优先导入官方 sim 模块**，让官方模块完成它那唯一一次
`GameInitialize`，本模块复用其 `lib` 句柄；仅当官方包不可导入时才自行加载并初始化一次。
保留 `SearchBegin/SearchStep/SearchEnd/SearchRelease` 等全部真实声明与生命周期，
不伪造符号、不关闭 `would_ko`、失败时原始错误照常抛出。

## 4. 修复后验证（WSL2，官方 libcg.so）

六个用例各自独立进程，全部通过（`wsl_stage4_rerun.log` + `wsl_stage4_results.json`）：

| 用例 | 退出码 | 关键结果 |
|:---|:---|:---|
| 干净 `import main` | 0 | 模块可导入，`deck_len=60` |
| 官方环境先加载再导入 agent | 0 | `make("cabt")` 成功；同一引擎句柄 |
| 按文件路径加载 agent | 0 | `spec_from_file_location` 加载 `pkg/main.py`，`agent` 可调用 |
| 60 张牌组请求 + 引擎开局 | 0 | `deck_len=60`、`unknown_ids=[]`、`engine_accepted=true`、`errorType=0`、基础宝可梦 9 张 |
| 双座位真实对局 | 0 | 见下表（官方 harness 口径） |
| `would_ko=True` 真实 Search API | 0 | `SearchBegin/SearchStep/SearchEnd/SearchRelease` 齐备 |

### 4.1 官方 harness 真实对局（插桩，`wsl_stage4c.json`）

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

### 4.2 `would_ko` 真实正例

同一批 harness 对局内插桩 `search_agent.would_ko_flags`（引擎 Search API 路径，非构造数据）：

```text
calls=387  非空攻击选项集=53  正例 KO=21  正例终局胜=3
samples: {"ko":1.0,"prizes":0.0,"win":1.0} / {"ko":1.0,"prizes":1.0,"win":0.0} ×2
```

来源标注：正例取自真实引擎对局中候选 agent 自身产生的攻击 observation（`main.agent`
内部调用 `annotate_would_ko`），未使用任何伪造符号或静默降级。

### 4.3 Windows 侧同机制对照（`init_repro_results_windows.json`）

WSL 故障后在 Windows 用官方 `cg.dll` 复现同一句柄语义（子进程：先导变体 `sdk_cg.sim`，
再导官方 `kaggle_environments.envs.cabt.cg.sim`）：

| 变体 | 退出码 | 结果 |
|:---|:---|:---|
| 修复前 `_load_lib` | 1 | 官方模块 `sim.py:34 lib.GameInitialize()` 抛 C++ 异常 `OSError 0xe06d7363`（与 Linux 的 `buffer full. capacity:7` 同源） |
| 修复后 `_load_lib` | 0 | `loads=["cg.dll"]`、`game_initialize_calls=1`、`same_handle=true`、开局 `errorType=0` |

## 5. Windows 回归

- `PYTHONPATH=.engine-shim pytest tests/ -q` → **48 passed**（含 gates、selfplay、encoding、decks 等）。
- 阶段 2 兼容性回归（`ckpt_compat_report.json`）：迁移库下两个 checkpoint 严格加载通过、
  745 个真实观测 logits/value 与备份库逐位一致、采样动作 0 非法、`would_ko` 136/136 调用无错误。

## 6. 剩余边界

1. WSL2 发行版在本次会话末段无法启动：挂载 `F:\DevOffload\WSL\Ubuntu-24.04\ext4.vhdx` 报
   `Wsl/Service/CreateInstance/MountDisk/HCS/E_ACCESSDENIED`（外部环境故障，未做任何修复尝试）。
   本报告所需的全部 Linux 证据均在该故障之前采集并落盘；仅“Linux 侧 old/new 机制对照”这一
   非必需项尚未补做（Windows 侧等价对照已完成）。
2. 本机 glibc 2.39 与 Kaggle 官方运行镜像的 glibc 一致性：本机验证通过，Kaggle 镜像实测待提交后确认。
3. 最终提交包尚未导出（按任务要求由用户手动执行导出命令），因此本报告验证的是**与将要导出的包同构的暂存包**
   （同一 `main.py` 模板、同一迁移库、同一 `sdk_cg` 修复）。

## 7. 结论

在真实官方 CABT 运行时（WSL2 + `libcg.so` 1.33.0）下，迁移库暂存包的两处核心问题均已闭环：
引擎初始化在本进程内唯一（干净 `import main`、官方环境先加载、按路径加载三种顺序全部 exit 0），
60 张牌组被引擎接受，双座位各 3 局真实对局无异常无非法动作，`would_ko` 走真实 Search API
并捕获到 21 个正例。Linux 运行时验证达到本阶段要求。

## 8. 最终提交包的 Linux 复验（导出后补做，2026-10-20）

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
