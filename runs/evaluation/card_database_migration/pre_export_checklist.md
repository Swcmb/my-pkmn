# 最终导出前门禁核对表（等待用户手动执行导出）

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

## 与任务书原文命令的**唯一差异**

任务书里的 `$env:PYTHONPATH` 指向 `C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim`。
该临时目录已在本会话中被系统清理（`Temp/opencode` 不复存在），因此按用户选定的方案，
改用仓库内永久 shim：`D:\pkmn-kaggle\.engine-shim`（官方 `kaggle_environments==1.33.0` 包副本，
包 `__init__` 链已降级为桩以避开 `jsonschema` 等重依赖；已写入 `.gitignore`，不入库）。
其余参数与任务书一致。

## 待用户执行的命令

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

## 已知边界（不影响门禁，但需记录）

1. 引擎 1.33.0 实际卡池 1431 张，本库覆盖 ID 1–1267；ID 1268–1431 的 164 张新卡在编码器中被夹到 UNK 行。
   选定牌组全部 ID ≤1267，不受影响。
2. `cards.csv` 与本机引擎在 ID 15、ID 979 两处存在版本差，以用户提供的 `cards.csv` 为权威来源。
3. WSL2 发行版在会话末段无法启动（挂载 `ext4.vhdx` 报 `E_ACCESSDENIED`，外部环境故障）。
   Linux 侧全部必需证据在故障前已采集落盘。
