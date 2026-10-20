# Linux 候选提交包真实运行验证报告

日期：2026-10-20
候选包：`submission_rl_transformer2_dragapult_nodist.tar.gz`
SHA-256：`09b2929c5f9cf77759a087b09ad202bf05bdf32b36d9ac00e89ca2505b1d9d28`
结论：**未无条件通过**。在“先加载 `kaggle_environments` / CABT 环境，再加载 agent”的顺序下，真实引擎双座位对局可以完成；在干净解释器中直接 `import main` 会在 `sdk_cg/sim.py` 的第二次 `GameInitialize()` 处触发原生崩溃（SIGABRT）。因此不能把 Linux 兼容性写成“Linux 运行通过”。

## 1. 测试环境

| 项 | 值 |
|---|---|
| 发行版 | Ubuntu 24.04.4 LTS |
| 内核 | Linux 6.6.87.2-microsoft-standard-WSL2 x86_64 |
| Python | 3.12.3（系统解释器） |
| PyTorch | 2.14.1+cpu |
| CUDA | 不可用（WSL CPU 推理） |
| kaggle-environments | 1.33.0 |
| glibc | 2.39 |
| 隔离依赖目录 | `~/pkmn-linux-verify-20261020/pylibs`（`pip --target`，未改系统 Python） |
| 解包目录 | `~/pkmn-linux-verify-20261020/pkg` |
| 原始包 | 只读复制；Windows 原始包、模型、checkpoint 未修改 |

`kaggle-environments` 通过官方 PyPI 安装；Tsinghua 镜像未返回该包，按规则回退到上游。安装内容包含官方 CABT 引擎：

```text
/home/swcmb/pkmn-linux-verify-20261020/pylibs/kaggle_environments/envs/cabt/cg/libcg.so
```

## 2. 包内容与引擎加载路径

压缩包共 19 个文件，`sdk_cg/` 只有 5 个 `.py`，**不包含** `cg.dll` / `libcg.so`，符合“运行时解析引擎依赖”的候选方案。`pkg/` 内没有原生库。

`kaggle_environments.envs.cabt.cg.sim` 的实际加载路径：

```text
libcg._name = /home/swcmb/pkmn-linux-verify-20261020/pylibs/kaggle_environments/envs/cabt/cg/libcg.so
```

动态依赖检查：

```text
file: ELF 64-bit LSB shared object, x86-64, dynamically linked, BuildID=...
ldd:
  linux-vdso.so.1
  libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6
  libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1
  libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
  libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6
  /lib64/ld-linux-x86-64.so.2
```

没有缺失依赖。`libcg.so` 需要 glibc 2.39 环境（WSL 提供），未发现 `GLIBC_... not found`。

## 3. import 顺序：一个真实的阻塞

### 3.1 干净解释器直接 `import main`：失败

命令（工作目录 `pkg/`，`PYTHONPATH=../pylibs`）：

```text
python3 -c "import sdk_cg.sim"
```

输出：

```text
buffer full. capacity:7
terminate called after throwing an instance of 'std::runtime_error'
  what():  buffer full. capacity:7
Fatal Python error: Aborted
...
  File "/home/swcmb/pkmn-linux-verify-20261020/pkg/sdk_cg/sim.py", line 58 in <module>
```

退出码 134（SIGABRT）。`sdk_cg/sim.py` 第 58 行是 `lib.GameInitialize()`。

原因（已从代码与调用顺序确认）：

1. `_load_lib()` 先检查 `sys.modules["kaggle_environments.envs.cabt.cg.sim"]`。
2. 干净进程中该模块不存在，随后 `_engine_candidates()` 导入 `kaggle_environments.envs.cabt.cg`；导入父包会加载 CABT 环境，环境再加载 `cg.sim` 并调用一次 `GameInitialize()`。
3. `sdk_cg/sim.py` 随后又用 `ctypes.cdll.LoadLibrary` 加载同一 `libcg.so`，因为 `_loaded_here=True`，再次调用 `GameInitialize()`。
4. 官方引擎禁止二次初始化，抛出 `buffer full. capacity:7` 并 abort。

这不是静态链接问题，是运行期引擎初始化顺序问题。

### 3.2 先加载 `kaggle_environments`，再加载 agent：通过

以下两种顺序都能 `import main` 并返回 60 张牌组：

```text
python3 -c "import kaggle_environments; import main; print(len(main.DECK))"
python3 -c "import kaggle_environments.envs.cabt.cg.sim as s; import main; print(len(main.DECK))"
```

输出 `60`。此时 `sdk_cg.sim._loaded_here == False`，说明它复用了官方 `cg.sim` 的句柄，没有二次初始化。

### 3.3 按 Kaggle runner 的文件路径加载：通过

用 `env.run(["/.../pkg/main.py", opponent])` 让 `kaggle_environments` 先创建环境、再按文件路径导入 agent，真实对局完成：

```text
ok=true, steps=78, status=["DONE","DONE"], reward=[1,-1]
```

这与“环境先加载、agent 后加载”的实际 runner 顺序一致，但不能消除 3.1 的独立 `import main` 崩溃。候选包若被任何先导入 agent 的 harness 加载，会直接 abort。

## 4. 牌组请求与合法性

- `main.DECK` 长度 60，内容与 `runs/evaluation/deck_search/dragapult_deck.csv` 完全一致。
- `agent({"select": None})` 返回 60 个整数，与 `main.DECK` 相同。
- 直接调用官方 `battle_start(DECK, DECK)`：`errorPlayer = -1`、`errorType = 0`，引擎接受该 60 张牌组。
- 首决策调用序号为 2（第一次调用是牌组请求），说明 `select is None` 与后续选择分支都被真实执行。

## 5. 双座位真实对局

环境：`kaggle_environments.make("cabt", configuration={"bo": 1})`，对手为合法脚本代理或同牌组代理；每个 agent 在同一进程中只驱动一个座位。

| 测试 | 座位 | 步数 | 最终状态 | 奖励（本 agent） | 决策数 | 合法动作 |
|---|---:|---:|---|---:|---:|---:|
| `linux_verify.py` 第 1 局 | 0 | 67 | DONE/DONE | -1 | 27 | 27/27 |
| `linux_verify.py` 第 2 局 | 1 | 21 | DONE/DONE | +1 | 4 | 4/4 |
| `test_runner_order.py`（文件路径加载） | 0 | 78 | DONE/DONE | +1 | — | — |
| `probe_would_ko.py`（脚本代理） | 0/1 | 42 | DONE/DONE | +1/-1 | — | — |

- 所有测试对局均以 `DONE/DONE` 结束，没有 `ERROR` / `INVALID` / `TIMEOUT`。
- 动作合法性检查：动作是整数列表、无重复、长度不超过 `maxCount`、索引落在 `option` 范围内；上述 agent 对局中 `action_bad = 0`。
- 对局长度差异较大（21–78 步），部分对局在模型尚未展开攻击前就结束；这不影响“真实引擎可运行”的判断，但说明该 CPU 测试样本不足以评估策略强度。

## 6. `would_ko` Search API 与特征写回

### 6.1 提交 agent 自身的调用

在 `linux_verify.py` 的两局中，`_WK_ON=True`，`annotate_would_ko` 被调用 31 次；共产生 12 个攻击选项 flag，全部 `ko=0.0`，写回 12/12 成功，Search 调用 `begin=12, step=12, release=12, end=12`，无异常。提交 agent 在测试对局中没有遇到可 KO 的攻击选项，因此没有正例。

### 6.2 真实引擎上的正例

为验证 `would_ko=True` 的 Search 路径，使用**脚本化合法代理**（不是提交模型）在同一官方引擎、同一 CABT 环境中驱动对局；在真实 main select 中捕获到攻击选项 `attackId=1044`，调用 `would_ko_flags` 并写回：

```text
seat=1, turn=18, n_attack=1, attack_ids=[1044]
flags = {"13": [1.0, 1.0, 1.0]}   # ko_rate, expected_prizes, win_rate
writeback_ok = true
search = begin 17 / begin_ok 17 / step 18 / step_ok 18 / release 17 / end 17
begin_err = [] ; step_err = []
```

这证明官方 `libcg.so` 上 `SearchBegin` / `SearchStep` / `SearchRelease` / `SearchEnd` 可以正常调用，`would_ko=1.0` 的结果能写回 option 的 `would_ko` 字段。该正例来自脚本代理，提交 agent 自身尚未在测试中触发同类正例。

## 7. 结论与阻塞

已确认：

1. WSL2 Ubuntu 24.04 可以安装并加载官方 `kaggle-environments 1.33.0` 与 `libcg.so`。
2. 官方引擎接受候选牌组（60 张，`errorPlayer=-1`）。
3. 在环境先加载的顺序下，提交包可以导入、请求牌组、完成双座位真实对局，返回合法动作。
4. `would_ko` Search API 与特征写回在真实 observation 上工作，正例 `ko=1.0`。

未通过 / 剩余阻塞：

1. **干净解释器直接 `import main` 会 SIGABRT**（`buffer full. capacity:7`，`sdk_cg/sim.py:58`）。这是运行期初始化顺序缺陷，不能把 Linux 兼容性写成无条件通过。
2. 提交 agent 自身在本次 CPU 测试中未出现可 KO 的攻击选项，`would_ko` 正例由脚本代理覆盖；提交 agent 的 `would_ko=True` 行为仍未在真实对局中直接观察到。
3. 未验证真实 Kaggle 线上 runner 的 agent 导入顺序；如果线上先导入 agent 再创建环境，会命中第 1 条崩溃。
4. 未验证 GPU / CUDA 推理路径；本报告只覆盖 WSL CPU。

## 8. 产物与命令

新增（均在 WSL `~/pkmn-linux-verify-20261020/`，不在 Windows 原始包内）：

```text
pylibs/                              # 隔离依赖
pkg/                                 # 解包后的候选包
linux_verify.py
linux_verify_result.json
linux_verify_mirror.py
linux_verify_mirror_result.json
probe_would_ko.py
probe_would_ko_result.json
probe_would_ko_multi.py
probe_would_ko_multi_result.json
debug_game.py / debug_short.py / debug_attacks.py / debug_options.py
test_runner_order.py
```

主要命令：

```bash
python3 -m pip install --target ./pylibs -i https://pypi.org/simple kaggle-environments
python3 -m pip install --target ./pylibs --index-url https://download.pytorch.org/whl/cpu torch
tar -xzf submission_rl_transformer2_dragapult_nodist.tar.gz -C pkg
PYTHONPATH=./pylibs:./pkg python3 linux_verify.py
PYTHONPATH=./pylibs:./pkg python3 probe_would_ko_multi.py
PYTHONPATH=./pylibs:./pkg python3 test_runner_order.py
```

未修改：

```text
Windows 原始 submission_rl_transformer2_dragapult_nodist/
Windows 原始 submission_rl_transformer2_dragapult_nodist.tar.gz
runs/train-20m-e16-1m/ckpt_999424.pt
runs/train-20m-e16-1m/ckpt_500736.pt
```
