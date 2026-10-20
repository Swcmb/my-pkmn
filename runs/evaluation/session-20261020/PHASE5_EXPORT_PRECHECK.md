# 阶段 5 预备：导出脚本审计与执行前置条件（尚未执行导出）

日期：2026-10-20　脚本：`scripts/export_rl_submission.py`（本会话逐行读过 L155-235 并跑过 `--help`）

现有已交付包（本会话已独立验证，保持不动）：
`submission_rl_transformer2_db_migrated.tar.gz`，SHA-256 `8d4ec3ccdebd563c259efa12bfe1061493dcaff7c1f0154fcfff6f128d58ae71`，
75,126,979 B，`model.pt` 与 `ckpt_999424.pt` 逐位一致，`deck.csv` == `best_deck.csv`。

---

## 1. 脚本参数与行为核实

| 项 | 核实结果 |
|:---|:---|
| 参数集 | `--ckpt --backend{transformer2,mcts2} --deck --csv --out --builddir --n-sims --n-det`（`--help` 实测） |
| `--deck` 默认 | `agent/deck.csv` —— **即阶段 2 测得 vs random 仅 0.233 的弱牌组。必须显式传 `--deck`，否则会把弱牌组打进包里** |
| `--csv` 默认 | 仓库根 `EN_Card_Data.csv`（迁移后权威库，SHA `05aa569e…`）✅ |
| `--out` / `--builddir` 默认 | `submission_rl_<backend>.tar.gz` / `submission_rl_<backend>` |
| 牌组校验 | 打包前强制 `len(deck.csv 非空行) == 60`，否则 `SystemExit`（L218-221）✅ |
| 模型瘦身 | 只保留 `net / net_config / global_step`，丢弃 optimizer+teacher（注释说明 307 MB 整包会被 Kaggle 以裸 400 拒绝）✅ |
| 顶层结构 | `sorted(os.listdir(builddir))` 平铺写入 tar，符合 Kaggle 要求 ✅ |
| 引擎二进制 | `transformer2` 且 `would_ko=True` 时打包 `decks.py / search_agent.py / sdk_cg/`，**只含 `.py`**（`ignore __pycache__`，不含 `cg.dll/libcg.so`）✅ |

## 2. 覆盖风险（两条，均为源码级事实）

1. **`--builddir` 已存在会被整目录删除**：`if os.path.exists(b): shutil.rmtree(b)`（L170-171）。
   传入任何已存在的目录都会连带其中的历史文件一起销毁。
2. **`--out` 指向的 tar 会被静默覆盖**：`tarfile.open(out, "w:gz")`（L225）无任何存在性检查。

→ 阶段 5 执行导出时，`--out` 与 `--builddir` **必须都是当前不存在的新名字**。
本会话已核实 `runs/` 与仓库根的四个既有 `.tar.gz` 与四个 `submission_rl_transformer2*/` 目录全部保留原样。

## 3. 待训练完成后使用的导出命令（占位 ckpt 路径，届时按实际产物替换）

```powershell
cd D:\pkmn-kaggle
$env:PYTHONPATH = "D:\pkmn-kaggle\.engine-shim;D:\pkmn-kaggle"

D:\ProgramFiles\deeplearning\python.exe scripts\export_rl_submission.py `
  --ckpt runs\finetune-500k-e8-20261020\ckpt_499712.pt `
  --backend transformer2 `
  --deck runs\evaluation\card_database_migration\best_deck.csv `
  --out submission_rl_transformer2_ft500k.tar.gz `
  --builddir submission_rl_transformer2_ft500k
```

执行前的存在性检查（必须全部返回 False 才继续）：

```powershell
Test-Path .\submission_rl_transformer2_ft500k.tar.gz
Test-Path .\submission_rl_transformer2_ft500k
```

导出后本会话已备好的验证工具：

| 步骤 | 工具 |
|:---|:---|
| 牌组五项规则 | `runs/evaluation/session-20261020/validate_deck.py` |
| 单步 100ms / 单局 10min / 内存 2GB / 返回值合法 | `runs/evaluation/session-20261020/subsmoke_timing.py <新builddir> 12`（带 `SMOKE_THREADS=2`） |
| `model.pt` 与源 ckpt 逐位比对、deck SHA 比对、`tar -tzf` 清单、SHA-256 与大小 | 本会话已用过的一次性检查，阶段 5 复用于新包 |

## 4. 尚存的两个待决项

1. **Linux 官方等价环境复验受阻**：`linux_runtime_report.md` 记录的 WSL 依赖目录
   `~/pkmn-linux-verify-20261020/pylibs` **现已不存在**（Ubuntu-24.04 发行版本身可用，但系统 python3
   无 pip、无 torch、无 kaggle_environments）。要由我重新独立复跑，需要在 WSL 内装约 1 GB 依赖。
   该安装属修改用户环境且体积大，**需用户授权后再做**。在此之前，Linux 侧结论只能引用既有报告
   （最终包 6 用例全 exit 0、官方 harness 双座位各 3 局完赛、`GameInitialize` 恰 1 次），
   记为**已有证据但本会话未复核**。
2. **`torch.set_num_threads(2)` 与预热前向**：阶段 1 实测默认线程下 385 次决策有 2 次越过 100ms
   （最大 105.1ms），钉 2 线程后 0 次。建议在导出脚本的 `TRANSFORMER2_HEAD` 模板里加两行
   （import 期设线程 + 一次空前向预热），这样下一次导出自动带上；**改动模板属修改现有脚本，
   等阶段 4 门禁通过、确需再导一版时一并做，并在改动前备份原脚本。**

## 5. `main.py` 退化 select 路径的结论（已验证，无需改动）

对"引擎给出 `minCount=1` 且 `option` 为空"的退化状态：`build_mask()` 在全零时兜底放行 `SUBMIT_ACTION`，
`agent()` 随即 `break` 并返回 `[]`。官方 harness 与训练 env 两条路径下均 0 异常（既有 840 局 +
本会话 674 次决策的 `illegal_len/illegal_index` 全 0），且掩码只把下标开放到 `i < len(option)`，
**下标越界不可能发生**。历史上观察到 `IndexError` 的位置是手写最小驱动循环，不是提交包。
→ 判定：现有实现是该退化状态下的唯一可行响应，不加投机性兜底。
