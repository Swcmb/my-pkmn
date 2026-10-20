# 阶段 3：500,000 步微调训练执行说明（命令已备好，训练由用户亲自执行）

日期：2026-10-20　初始化 checkpoint：`runs\train-20m-e16-1m\ckpt_999424.pt`

> **本文件只准备命令。本会话未启动、也绝不会启动任何训练进程。**
> 下述每一条参数都经 `rl/train_selfplay.py` 源码与 `--help` 实测核实（help 全文留证于
> `runs/evaluation/session-20261020/train_help.txt`）。

---

## 1. 任务书历史参数的逐项核实

| 历史参数 | 是否存在 | 实际语义 | 与当前脚本/ckpt 是否兼容 |
|:---|:---|:---|:---|
| `--arch transformer2` | ✅（L193，默认即 `transformer2`） | 网络架构名 | ✅ 与 ckpt `net_config.arch` 相同 |
| `--d-model 544` | ✅（L248） | Transformer 宽度 | ✅ 但**必须显式给**：默认是 **128** |
| `--nhead 8` | ✅（L249） | 注意力头数 | ✅ 但默认是 **4** |
| `--nlayers 5` | ✅（L250） | 层数 | ✅ 但默认是 **3** |
| `--total-timesteps 500000` | ✅（L130，默认 18,000,000） | **全局环境步预算**（非梯度步数） | ✅ |
| `--num-envs 16` | ✅（L131，默认 64） | 并行环境数 | ✅ 见 §3 显存结论，**建议改 8** |
| `--update-epochs 1` | ✅（L170，默认即 1） | 每 rollout 的 PPO epoch | ✅ |
| `--num-steps 64` | ✅（L132，默认 256） | 每环境 rollout 长度 | ✅（batch = 16×64 = 1024） |
| `--lr 2e-5` | ✅（L133，默认 2.5e-4） | 峰值学习率 | ✅ 微调合理（较默认低 12.5×） |
| `--lr-schedule cosine` | ✅（L135，默认 cosine） | LR 曲线 | ✅ |
| `--anneal-lr` | ✅（L134，`default=True`） | 退火开关 | ✅ 冗余但无害 |
| `--gamma 0.997` | ✅（L157，默认即 0.997） | 折扣因子 | ✅ |
| `--gae-lambda 0.95` | ✅（L160，默认即 0.95） | GAE λ | ✅ |
| `--num-minibatches 16` | ✅（L161，默认即 16） | 每 rollout  minibatch 数 | ✅（minibatch = 64） |
| `--clip-coef 0.2` | ✅（L181，默认即 0.2） | PPO 裁剪 | ✅ |
| `--ent-coef 0.002` | ✅（L182，默认 0.01） | 熵系数 | ✅ 微调期降低探索合理 |
| `--vf-coef 0.5` | ✅（L183，默认即 0.5） | 价值损失权重 | ✅ |
| `--max-grad-norm 0.5` | ✅（L185，默认即 0.5） | 梯度裁剪 | ✅ |
| `--would-ko` | ✅（L195，`BooleanOptionalAction`，默认 True） | 引擎 KO 标注特征 | ✅ 必须与 ckpt 一致（ckpt `would_ko=True`） |
| `--decks top50` | ✅（L254，默认即 `top50`） | 通才池 | ⚠️ 见 §2 冲突 1 |
| `--gate-pairs mirror` | ✅（默认 mirror） | 晋级门禁牌组调度 | ⚠️ 见 §2 冲突 2 |
| `--promote-winrate 0.6` | ✅ | 教师网络晋级阈值开关 | ⚠️ 见 §2 冲突 2/3 |

`--native-encode` **不可用**：本机引擎未导出 `GetBinaryObs/AdvanceLog`，开启会失败；训练与评测一律用默认 Python 编码路径（与 ckpt 训练口径一致）。

---

## 2. 核实中发现的 4 个语义冲突（必须修正，否则训练与预期不符）

**冲突 1：微调模式缺 `--opponent-ckpt`，命令会直接拒绝启动。**
`rl/train_selfplay.py:522-524`：

```python
if args.finetune_deck or args.opponent_ckpt:
    if not (args.finetune_deck and args.opponent_ckpt):
        raise SystemExit("finetune mode needs BOTH --finetune-deck and --opponent-ckpt")
```

历史参数表里既无 `--finetune-deck` 也无 `--opponent-ckpt`，所以它其实是一套**通才自博弈**配置；而 `--decks top50` 是默认值，等于该项不起任何作用。任务书同时提到"独立微调"与 `--decks top50`，两者不能共存（微调模式下 `pool` 被强制替换为 ship deck，L632）。

**冲突 2：微调模式下 `--gate-decks-file` 与 `--gate-pairs` 均无效。**
`L658`：`if args.gate_decks_file and frozen_opp is None:` —— 有 `frozen_opp`（即微调模式）时**静默忽略**。
`L1420-1457`：微调分支 `if opp_net is not None:` 直接跳过 mirror h2h，用训练流里的 `ft_wr`（对静态对手的滚动胜率）做"守卫式晋级"，**`--promote-winrate` 的 0.6 数值在该分支里根本没被使用**，它只是"进入门禁模式"的开关。
→ 任务书要求核查的 `--finetune-deck` 与 `--gate-decks-file` 语义不匹配**确认成立**。

**冲突 3：`--promote-winrate` 在 500k 预算内永不触发。**
`L2083`：`if teacher_net is not None and global_step - last_promote >= args.teacher_refresh:`，而 `--teacher-refresh` 默认 **5,000,000**（L284）> 500,000 预算。
→ 整个训练期间教师网络永不刷新；`--promote-winrate` 是空操作。若确需中途门禁，必须同时把 `--teacher-refresh` 调到 ≤ 预算。

**冲突 4：`--warmup-steps` 默认值会把前 20% 预算用于 LR 预热。**
`L1523`：`warmup_iters = min(args.warmup_steps // batch, max(1, num_iters // 5))`，默认 `--warmup-steps 2,000,000` → 被钳到 `num_iters//5` = 488//5 = **97 次迭代（约 99,300 步，占预算 19.9%）**。对 500k 微调来说预热段过长。
→ 本方案显式设 `--warmup-steps 50000`：A（batch 512）→ 97 次迭代、49,664 步，占预算 9.9%；
B（batch 1024）→ 48 次迭代，同样 9.9%。两者都只有默认钳制值（19.9%）的一半。

**另需注意**：模型维度默认 128/4/3 与 ckpt 的 544/8/5 不符时，`load_net_compat()` 会抛
`RuntimeError: checkpoint/arch mismatch`（形状不符在 `strict=False` 下同样报错），
**失败是响亮的、不会静默降级**——但仍必须显式给全 `--d-model/--nhead/--nlayers/--ff`。

---

## 3. 最终建议方案

### 方案 A（推荐）：ship-deck 专项微调，E8 保守显存

理由（本会话补测后的最终基线，见 `AUDIT_PHASE0-2.md` §3.3/§3.4）：

- **同牌组纯镜像口径对模型差异是敏感的**：`ckpt_999424` 对旧基线 `ckpt_500736` 在提交牌组上
  合并 **85/120 = 0.708，CI [0.622, 0.782]**，两个种子分别 0.683 / 0.733。
  同一条臂在 74 副混合池 cross 口径下只有 0.508（CI 跨 0.5）→ 提升是**牌组特异**的。
- 提交使用固定 `deck.csv`，比赛中双方牌组本就不同，而我们要优化的是"用这副牌打牌"的能力。
  把 500k 步全部预算投在提交牌组上，与该证据方向一致；混合池那 0.508 说明通才续训的边际收益低。
- 教师 KL 锚点负责防止专项化过程毁掉已在混合池上成立的通用棋力，G7b 用 44 副 held-out 交叉臂盯着它。

```powershell
cd D:\pkmn-kaggle

$env:PYTHONPATH = "D:\pkmn-kaggle\.engine-shim;D:\pkmn-kaggle"
$env:PYTHONUNBUFFERED = "1"

D:\ProgramFiles\deeplearning\python.exe -m rl.train_selfplay `
  --init-from runs\train-20m-e16-1m\ckpt_999424.pt `
  --finetune-deck csv:runs\evaluation\card_database_migration\best_deck.csv `
  --opponent-ckpt runs\train-20m-e16-1m\ckpt_999424.pt `
  --ladder-decks-file runs\evaluation\session-20261020\opp_pool_train30.json `
  --arch transformer2 --d-model 544 --nhead 8 --nlayers 5 --ff 256 --would-ko `
  --total-timesteps 500000 `
  --num-envs 8 --num-steps 64 --num-minibatches 16 --update-epochs 1 `
  --lr 2e-5 --lr-schedule cosine --anneal-lr --lr-min-ratio 0.1 --warmup-steps 50000 `
  --gamma 0.997 --gae-lambda 0.95 --clip-coef 0.2 --ent-coef 0.002 --vf-coef 0.5 --max-grad-norm 0.5 `
  --teacher-kl-coef 0.005 --teacher-value-coef 0.005 `
  --save-every 250000 --log-every 1 --value-diag `
  --seed 20261020 `
  --out runs\finetune-500k-e8-20261020
```

参数取值依据（仅列关键项）：

| 参数 | 取值 | 依据 |
|:---|:---|:---|
| `--finetune-deck csv:…best_deck.csv` | 用 `csv:` 直接读提交牌组 | `_resolve_deck()` 支持 `csv:<path>` 并断言恰好 60 张（L538-547）。虽然 `TRAIN_TOP50` 里有同多重集合的 `k06_dragapult…`，但行序不同；`csv:` 保证训练用的就是导出用的那份 |
| `--opponent-ckpt ckpt_999424.pt` | 冻结自身作为对手 | 对手难度与当前棋力对齐；`--opponent-decks` 与 `--ladder-decks-file` 二选一（L576-586），后者文件已存在 |
| `--ladder-decks-file runs\evaluation\session-20261020\opp_pool_train30.json` | 30 副 meta 牌组作对手场 | **评测数据独立要求（任务书 §六）触发的修正**：`runs/evaluation/pool_top50.json` 同时是阶段 2 门禁的 held-out 交叉池，若直接拿它当训练对手池，微调后该池不再独立。现用 `split_eval_pool.py` 按 stride-7 置换把 74 副确定性地切成 30 副训练对手（本文件，SHA `8d7dd1e1cf52…`）与 44 副 held-out 评测池（`eval_pool_heldout44.json`，SHA `5c225480c02c…`），两侧互斥、并集为原池，且都可被 `rl.decks_ladder.load_ladder_decks` 正常读取（weights 归一） |
| `--num-envs 8` | 而非 16 | E8/E16 通才吞吐几乎相同（82.7 vs 88.1 sps，+6.6%），但显存峰值 3,335 vs 6,163 MiB。微调模式额外常驻冻结对手网络（+约 81 MB 权重与对手前向激活），E16 的 24.4% 余量不足以稳妥吸收；E32 在通才模式下已实测 CUDA OOM |
| `--teacher-kl-coef / --teacher-value-coef 0.005` | 锚定 ckpt_999424 | 默认 0.0 = 无锚；专项化有遗忘风险，0.005 是既有方案留的建议值（**未在本模型做过 A/B**，见 §6 风险） |
| `--teacher-refresh` 不给（默认 5M） | 教师整轮不刷新 | 让锚点始终等于初始 checkpoint（冲突 3 的正面利用） |
| `--save-every 250000` | 中途留检查点 | 迭代按 1024 步推进，实际落盘在 ≈250,880 与终点 499,712；末次迭代由 `it == num_iters` 强制保存，不会出现"只有 latest.pt" |

### 方案 B（备选）：通才续训 E16

若判断"广度优先"（例如担心 A 榜对手牌组多样），去掉三个微调参数即可回到通才自博弈，此时 `--gate-*`/`--promote-winrate` 才有意义：

```powershell
D:\ProgramFiles\deeplearning\python.exe -m rl.train_selfplay `
  --init-from runs\train-20m-e16-1m\ckpt_999424.pt `
  --decks top50 `
  --arch transformer2 --d-model 544 --nhead 8 --nlayers 5 --ff 256 --would-ko `
  --total-timesteps 500000 `
  --num-envs 16 --num-steps 64 --num-minibatches 16 --update-epochs 1 `
  --lr 2e-5 --lr-schedule cosine --anneal-lr --lr-min-ratio 0.1 --warmup-steps 50000 `
  --gamma 0.997 --gae-lambda 0.95 --clip-coef 0.2 --ent-coef 0.002 --vf-coef 0.5 --max-grad-norm 0.5 `
  --teacher-refresh 250000 --gate-pairs mirror --promote-winrate 0.6 `
  --save-every 250000 --log-every 1 --seed 20261020 `
  --out runs\finetune-500k-e16-generalist-20261020
```

- 此方案下门禁会在 250,000 步触发一次：`n_gate = max(200, 5×74) = 370` 局 mirror 对局（L639），胜率 ≥0.6 才刷新教师。
- ship deck 只占采样 1/74 ≈ 1.4%，对提交牌组的针对提升有限。
- 显存口径 = 已实测的通才 E16（6,163 MiB，余 1,988 MiB），**无新增不确定性**；但 370 局门禁会占用额外墙钟。

---

## 4. 输出位置与预计产物

`--out` 由脚本 `os.makedirs(exist_ok=True)` 自动创建。两个目标目录**当前均不存在**（已核实），不会覆盖任何历史产物；`ckpt_999424.pt`、`latest.pt`、`best_deck.csv`、四个既有 `.tar.gz` 全部保持原样。

| 文件 | 说明 |
|:---|:---|
| `runs\finetune-500k-e8-20261020\latest.pt` | 唯一完整恢复点（net+opt+teacher+global_step+ft_best），约 **243 MB** |
| `…\ckpt_250880.pt`（近似名） | 中途 net-only 检查点，约 **81 MB** |
| `…\ckpt_499712.pt` | 终局 net-only 检查点 → **评估与导出用这个** |
| `…\matchup_r0.jsonl` | 每次迭代的牌组配对计数（仅作覆盖观察，不作效果结论） |
| `…\value_diag.jsonl` | 每 20 次迭代价值头分桶指标 |
| `…\promotion_gate.jsonl` | 仅方案 B 会产生（门禁胜率记录） |

迭代与步数换算：`batch = num_envs × num_steps`；`num_iters = total_timesteps // batch`。
A（8×64=512）→ 976 次迭代，终点 `global_step = 499,712`，优化器 step ≈ 976×16 = **15,616**。
B（16×64=1024）→ 488 次迭代，同终点，优化器 step ≈ **7,808**。
`--init-from` 使 `global_step` 从 0 计，日志步数是**本轮相对值**，与 999,424 不可混读。

---

## 5. 预计耗时与资源风险

吞吐基准来源：`runs/bench-20m/report.md`（同机 RTX 5060 Laptop 8,151 MiB、同 20.3M 参数网络、JSON/Python 编码、compile 关闭）。

| 方案 | 参考吞吐 | 纯训练估算 | 含启动/保存/门禁 |
|:---|:---|:---|:---|
| A（E8） | 82.7 环境步/s | 500,000/82.7 ≈ 6,046 s ≈ **1.68 h** | **≈ 1.9–2.2 h** |
| B（E16） | 88.1 环境步/s | ≈ 5,676 s ≈ **1.58 h** | **≈ 1.8–2.1 h**（另加 370 局门禁） |

风险与依据：

1. **显存（最高风险）**：微调模式常驻三份网络（live / teacher / frozen opponent）。通才 E16 实测峰值 6,163 MiB、E8 为 3,335 MiB；E32 在 `loss.backward()` 处两次 CUDA OOM（7,795 MiB，占 95.6%）。**A 方案选 E8 正是为此留 4.8 GB 余量。**
   若首轮前 20 次迭代显存稳定低于 6 GB，可中止后改 `--num-envs 16` 提速约 6%（收益很小，非必需）。
2. **系统内存**：`--obs-shm` 未开启（默认关），RAM 压力主要来自环境 JSON。基准中 E8 最低可用 RAM 8,050 MiB（总 32,492 MiB），E32 才逼近耗尽。E8/E16 安全。
3. **长跑降频未测**：基准最长单次运行 26.6 min，温度均值 86.2 °C、峰值 91 °C，未观察到降频；**小时级长跑的热行为属未测得项**，吞吐可能低于估算 10–20%。
4. **编码路径不可加速**：`--native-encode` 在本机不可用；`--compile` 会改变吞吐口径，使 §5 估算失效，故两个方案都不开。
5. **磁盘**：`runs/` 现有产物 + 新增约 0.6 GB；D 盘可用 225 GB，无风险。

---

## 6. 监控方法

新开一个 PowerShell 窗口观测（训练本身用上面的命令，不要用这些命令替代它）：

```powershell
# 1) 启动确认：必须出现这两行，否则说明微调模式没生效（历史上出现过静默退化为通才自博弈的缺陷）
#    [finetune] agent deck 'csv:…best_deck.csv' EVERY episode (random seat) vs FROZEN … on 74 mixture decks
#    [net] params=20,303,715
# 2) 实时跟踪日志（把输出重定向到文件时）
Get-Content .\train_e8.log -Wait -Tail 30

# 3) 显存 / 利用率 / 温度 / 功耗，每秒刷新
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv -l 1

# 4) 是否已落盘
Get-ChildItem .\runs\finetune-500k-e8-20261020 | Select-Object Name,Length,LastWriteTime
```

每行训练日志的含义（`--log-every 1`）：`step` 本轮累计环境步；`it/n_iters` 迭代进度；`sps` 该迭代环境步速率；
`avg` 全程均值速率；`tch_wr` 训练内对教师胜率（**不可代替独立门禁**）；`kl` 近似 KL；`ent` 策略熵；`clip` 裁剪触发率。

需要立刻警觉的信号：`sps` 持续 < 60、`kl` 连续 > 0.05、`ent` 相对初值骤降 > 30%、出现 `NaN/Inf`、或 `[ckpt] saved` 缺失。

---

## 7. 安全中止方式

1. 在训练窗口按一次 `Ctrl+C`。Windows 上 Python 通常在下一次迭代边界退出；`latest.pt` 只在保存点写入，中止不会破坏已落盘的 `ckpt_*.pt`。
2. 若 `Ctrl+C` 无效，再终止进程（先看 PID，确认是本次训练的 python）：

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*train_selfplay*' } |
  Select-Object ProcessId,CommandLine
Stop-Process -Id <上面查到的PID>
```

3. **不要**删除或改写 `runs\train-20m-e16-1m\` 下任何文件——那是当前候选与旧基线的唯一副本。
4. 恢复训练（如需）用 `--resume runs\finetune-500k-e8-20261020\latest.pt`（加载 net+optimizer+teacher+global_step，LR 曲线继续）；
   注意 `--total-timesteps` 在 `--resume` 下是**累计**预算。想干净重来就再次 `--init-from` 某个 `ckpt_*.pt`（优化器与 schedule 全新）。
   `--resume` 不能接 `ckpt_*.pt`：那类文件不含 `opt`。

---

## 8. 训练结束后请交给 AI 的产物

1. 训练窗口完整 stdout（或 `train_e8.log`），**尤其是开头的 `[cfg]`、`[finetune]`、`[net]` 三行**；
2. `runs\finetune-500k-e8-20261020\` 整个目录（`ckpt_*.pt` + `latest.pt` + `matchup_r0.jsonl` + `value_diag.jsonl`）；
3. 是否异常退出、退出码、以及中途有无 OOM/NaN；
4. 训练期间显存峰值读数（若截了 `nvidia-smi` 更好）。

拿到后我将执行阶段 4：核验 checkpoint 完整性与 `net_config` 匹配 → 与 `ckpt_999424` 同协议独立对战（含随机、未训练、镜像、held-out 牌组交叉，每格 ≥120 局）→ 逐项晋级门禁 →
通过后由我自主执行阶段 5 最终导出与 SHA-256 校验（导出属允许 AI 执行的范围）。

**若新 checkpoint 没有稳定超过 `ckpt_999424`，将保留旧模型作为提交版本，不强行晋级。**
