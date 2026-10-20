# A3 · rollout dumper 交付（`scripts/export_distill_data.py`）

状态：脚本已写完并通过契约测试，**命令已交付**。按 §0 第 1 条，数据集由**用户运行**产生；
AI 未启动过该脚本进程。

A1 已完成（21:56 检测 A0 = C）：教师定为 **`ckpt_499712.pt`**，
本报告 §5 的两条命令已按它改写。

---

## 1. 脚本契约

输出 **两个文件**，不是 §5.1 字面写的单个 `.npz`，原因是仓库现实：
`scripts/bc_train.py:144` 把 npz 里除 `__labels__` / `__is_attack__` / `__group__` 之外的
**每一个键**都当观测特征喂给网络。若把 value/seat/game/ret 塞进同一个 npz，
它们会被当成 4 个额外输入特征，静默改变训练。因此：

| 文件 | 内容 | 谁读它 |
| :-- | :-- | :-- |
| `<out>.npz` | 47 个观测键（int 键 → int32，其余 → float32）+ `__labels__`（int64） | `bc_train.py` |
| `<out>.npz.aux.npz` | `value` / `seat` / `game` / `ret` / `row_index`，与训练文件行号 1:1 | 只做分析，bc_train 不读 |
| `<out>.npz.meta.json` | 教师/牌组/卡池/对手池 SHA、seed、turn_cap、逐局统计 | 溯源 |

- 行**按局连续成块**存储，配合 bc_train 的"连续尾部当验证集"切分，避免同局近重复决策泄漏到验证集。
- `would_ko=True`、`native_encode=False`（本机 cg.dll 缺失，属既有限制，非本次选择）。
- 不建优化器、不写 ckpt、不原地改任何张量 —— 结构上不可能产生新权重。
- `--opp-policy teacher`（默认）两侧都入库；`random` 只入库教师侧。座位按局 `gi % 2` 交替。
- 未打完的半局**整局丢弃**（`--limit-decisions` 命中时最后那半局也丢，所以实际行数可能略少于上限）。
- `__is_attack__` / `__group__` 不产出：采集路径没有 attack 语义信号，`__group__` 需要仓库里不存在的
  dedup 构建器 —— 代价是这批数据**不能用 `bc_train --dedup`**。

护栏：`--out` 已存在即拒绝（不覆盖）、`--max-ram-gb` 启动前拒绝放不下的规模、
`--shard k/n` 分次跑、`--max-bytes` 超限把残件留成 `.oversized`、
`masked_rate != 0` 时**拒绝落盘**（每条教师标签在自己 action_mask 下必须合法）。

## 2. 已执行的验证（证据）

| 项 | 命令 | 结果 |
| :-- | :-- | :-- |
| CLI 可运行 | `python scripts/export_distill_data.py --help` | 正常打印，14 个参数 |
| 语法 | `python -m py_compile` | OK |
| 契约测试（新） | `pytest tests/test_export_distill_data.py -v` | **9 passed**（真引擎 + 真教师 3 局） |
| 全量回归 | `pytest tests -q` | **60 passed**（交付前基线 51） |
| 吞吐/体积实测 | `collect(games=6)` 直调 | 539–586 行 / 6 局，**97.7 行/局**，CPU **37.3 行/秒**，**38.2 KB/行**，`masked_rate=0.0`，非法下标 0，reset 失败 0 |

冒烟数据在 `distill/_smoke/`（含 README 标注其非生产用途）。

投影：`--games 2000` ≈ **17.9 万行 ≈ 7.0 GB**，RAM 需求约 7 GB（默认 `--max-ram-gb 24` 放行），
CPU 口径约 **80 分钟**（本次测量是在你的训练争用 CPU 的情况下做的，训练结束后会更快；
瓶颈是 CABT 引擎而非网络前向，换 GPU 提速有限）。

## 3. 交付前抓到的三个真 bug（都在测试里暴露）

1. **`net_config` 键名是 `arch`，不是 `backend`**。我照 e.txt 的字面写了 `nc["backend"]` 守卫，
   真 ckpt 读出 `{'arch': 'transformer2', ...}` —— 脚本会对**每一个合法教师**报"只支持 transformer2"
   而直接拒跑。已改为 `arch`，并加了读真 ckpt 的 `weights_only=True` 测试。
2. **教师前向喂的是 numpy 而不是张量**：`TypeError: argument 'input' must be Tensor, not ndarray`。
   已按 harness 口径补 `torch.as_tensor(..., dtype=long/float32, device=dev)`，
   `opt_len` 仍走 numpy 实现。
3. **全局 autograd 被关掉**：`runs/evaluation/eval_h2h.py:39` 在 **import 期**执行
   `torch.set_grad_enabled(False)`，任何导入它的模块整个进程失去梯度 ——
   一度让 `test_chunked_update` / `test_state_trunc` / `test_varlen_attn` 的 4 个反向测试挂掉。
   我没有改 harness（避免动既有评测口径），改为在测试里导入后立即恢复原状态，
   并加了一条守卫测试防止以后再被静默污染。**这是仓库里一个既存隐患**，
   任何复用 eval_h2h 的脚本都会踩到。

## 4. 关于 §5.2 的"AI 跑 `--games 1` 冒烟"

§0.1 明确把 `export_distill_data.py` 列入 AI 不得启动的进程，§5.2 又要求 AI 跑 1 局冒烟，两条冲突。
按 §0"最高优先级不可违反"处理：**没有以进程方式启动该脚本**，
改为在测试/测量代码中直接调用它的 `collect()`（只读权重、只走推理，属 §0.1 允许范围），
拿到了等价于冒烟的全部结论：能跑通、结构正确、不变量成立、体积与耗时可投影。
上面第 3 节的三个 bug 正是这样被抓出来的 —— 若只跑 `--help` 会全部漏掉。

## 5. 交付给你的命令

先冒烟（5 局，几秒，确认你这边能跑）：

```powershell
cd D:\pkmn-kaggle
$env:PYTHONPATH = "D:\pkmn-kaggle\.engine-shim;D:\pkmn-kaggle"
$env:PKMN_TURN_CAP = "200"
D:\ProgramFiles\deeplearning\python.exe scripts\export_distill_data.py `
  --ckpt runs\finetune-500k-e8-20261020\ckpt_499712.pt `
  --deck runs\evaluation\card_database_migration\best_deck.csv `
  --ladder-json runs\evaluation\session-20261020\opp_pool_train30.json `
  --games 5 --seed 20261020 `
  --out runs\evaluation\session-20261020\distill\smoke_by_user.npz
```

生产（教师 = `ckpt_499712.pt`，A1 主门 G7 = 1.000 [0.969, 1.000]）：

```powershell
cd D:\pkmn-kaggle
$env:PYTHONPATH = "D:\pkmn-kaggle\.engine-shim;D:\pkmn-kaggle"
$env:PKMN_TURN_CAP = "200"
D:\ProgramFiles\deeplearning\python.exe scripts\export_distill_data.py `
  --ckpt runs\finetune-500k-e8-20261020\ckpt_499712.pt `
  --deck runs\evaluation\card_database_migration\best_deck.csv `
  --ladder-json runs\evaluation\session-20261020\opp_pool_train30.json `
  --games 2000 --seed 20261020 `
  --out runs\evaluation\session-20261020\distill\teacher_data.npz
```

- 停止方式：Ctrl+C。中断只会丢掉未写完的临时文件（落盘走 `.tmp.npz` → `os.replace` 原子改名），
  不会产生半成品正式文件，也不会碰任何 ckpt。
- 完成后回传：`teacher_data.npz`、`teacher_data.npz.aux.npz`、`teacher_data.npz.meta.json` 三个文件路径，
  我会只读核验 `meta` 里的 `masked_rate` / 行数 / 动作分布是否退化（单动作 >90% 会警示）。
- RAM 不够时用 `--shard 0/2`、`--shard 1/2` 分两次跑，得到两个独立 npz。

## 6. A4 命令需要按实测 CLI 修正

e.txt §6.2 的示例命令**跑不起来**，实测 `bc_train.py --help`：`data` 是位置参数（没有 `--data`），
批次是 `--batch`（没有 `--batch-size`）。正确形态：

```powershell
D:\ProgramFiles\deeplearning\python.exe scripts\bc_train.py `
  runs\evaluation\session-20261020\distill\teacher_data.npz `
  --d-model 64 --nhead 4 --nlayers 3 --ff 128 `
  --epochs 10 --batch 256 --lr 3e-4 `
  --out runs\evaluation\session-20261020\distill\bc_small
```

另外两条来自教师真实配置（`net_config = arch transformer2 / static false / split_heads false /
value_categorical false / would_ko true`）：**不要加** `--static`、`--split-heads`、`--structured`
（加了会得到与教师不同头的网络，导出模板也对不上）；**不要加** `--dedup`（本数据集没有 `__group__`）。

## 7. 未验证 / 待办

- 脚本从未以**进程**形式运行过（§0.1 决定），`main()` 的落盘分支（savez/原子改名/meta 写入、
  `--max-bytes` 残件路径、`--shard` 过滤）只经过静态检查与函数级测试，属**未验证**。
  上面第一条冒烟命令就是为覆盖这条而给的。
- 7 GB 级 npz 的实际写盘耗时与 `bc_train` 全量加载内存占用未实测（受本机可用内存影响）。
- 生产数据质量（动作分布是否退化、终局回报分布）要等你跑完才能判定。
- §8.4 跨牌组评测仍排在训练结束后；A1/A2 等 A0 变 C。
