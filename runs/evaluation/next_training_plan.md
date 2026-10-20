# 下一轮短程微调方案（暂不启动训练）

日期：2026-10-20
初始化：`runs/train-20m-e16-1m/ckpt_999424.pt`（global step 999424，纯权重）
目标：在不扩大风险的前提下，验证“继续微调能否稳定超过当前候选”，而不是直接烧 10–20M 步。

## 1. 训练接口核查结果

### 1.1 计步方式

- `--total-timesteps` 是**全局环境步预算**（`rl/train_selfplay.py:130`）。
- 每次迭代采集 `num_envs × num_steps` 个环境步；`global_step += batch_size * world`（`:468-469`, `:1931`）。
- 默认 `num_envs=64, num_steps=256` → 每迭代 16,384 环境步；当前 1M 训练实际使用 `num_envs=16, num_steps=64` → 每迭代 1,024 环境步（`latest.pt` 的 `args` 可核实）。
- `ckpt_999424` 的 999,424 = 976 × 1,024，正好 976 次迭代；因此“10–20M 步”是约 9,766–19,531 次迭代的**环境步**，不是 10–20M 次梯度更新。

### 1.2 梯度更新频率

- 每次迭代收集一个 rollout（`batch_size` 条转移），再做一次 PPO 更新。
- `batch_size = num_envs × num_steps`；`minibatch_size = batch_size / num_minibatches`（`:468-469`）。
- 默认 `num_minibatches=16`、`grad_accum=1`、`update_epochs=1`（`:161-172`）→ **每个 rollout 16 次优化器 step**，1 个 epoch。
- 当前 1M 配置（batch 1024）下：976 次迭代 = 15,616 次优化器 step。
- 若扩到 10M 步（batch 1024）：约 9,766 次迭代 = 156,256 次优化器 step；20M 约 312,512 次。

### 1.3 checkpoint 与恢复

| 文件 | 内容 | 用途 |
|---|---|---|
| `ckpt_{global_step}.pt` | net、args、net_config、global_step（无优化器/teacher） | 评估/导出/冷启动 |
| `latest.pt` | net、optimizer、teacher、global_step、sched_anchor、ft_best（两网模式另含 net_b/opt_b/t_b/cg_state） | 唯一完整恢复点 |

- `--init-from <ckpt>`：严格加载权重，**优化器/LR schedule 全新**，teacher 从该权重复制（`:671-676`）。
- `--resume <latest.pt>`：加载 net+optimizer+teacher+global_step，LR schedule 从恢复点继续（`:729-752`）。
- `ckpt_999424.pt` **不含 `opt`**；虽然 `--resume` 遇到缺失 `opt` 会标 `opt=FRESH`，但语义上它是“净权重初始化”。本轮方案使用 `--init-from ckpt_999424.pt`，避免把旧优化器状态与新的牌组分布混在一起。
- checkpoint 保存：`latest.pt` 每次保存；`ckpt_{global_step}.pt` 在 `global_step % save_every` 触发（`:2123-2140`），默认 `save_every=500000`。
- 实测加载开销（warm cache，CPU）：`ckpt_999424.pt` 0.019–0.028 s；`latest.pt` 0.053–0.072 s。相对小时级训练可忽略；冷盘未测。

### 1.4 日志定义

- stdout 每次迭代（`--log-every 1`）：`step / it / sps / avg / tch_wr / pg / v_ce / ent / clip / kl`（`:2089-2121`）。
- `matchup_r{rank}.jsonl`：每次迭代写 `{"step", "cells"}`，按牌组对记录对局计数（`:1942-1949`）。
- `value_diag.jsonl`：`--value-diag` 默认开启，每 `--diag-every 20` 次迭代汇总价值头分桶指标（`:438-443`）。
- 训练日志的 `tch_wr` 是训练内 teacher 对局指标，**不能替代独立对战门禁**；本方案的门禁全部用 `h2h_ckpt.py` / `eval_h2h.py`。

### 1.5 牌组接口

- `resolve_deck_pool()` 只接受预定义名称（`rl/train_selfplay.py:55-90`）：
  - `top15` → `TRAIN_TOP15`，实际 **30 副**（不是 15 副）；
  - `sample` → `agent/deck.csv`（当前提交牌组）；
  - `official` → `rl/decks.DECKS` 的 4 副官方牌组，含 `dragapult`，**不含**当前牌组；
  - `all` → 4 副官方 + sample，**不含** TOP15。
- 已核实：当前 `agent/deck.csv` 与 `official_dragapult` 都**不在** `TRAIN_TOP15` / `TRAIN_TOP50` 中；`TRAIN_TOP15` 与 `TRAIN_TOP50` 也不含这两副目标牌组。
- 因此“TOP15 + 当前牌组 + official_dragapult”的混合分布**当前接口不能直接表达**。执行前必须先做一个小的接口扩展：
  - 在 `resolve_deck_pool()` 中支持 `json:<path>`（与 `--finetune-deck` 的 `_resolve_deck_set` 类似），读取 `{"decks": [[60 ints], ...]}`；
  - 或用新的命名池常量实现同样效果；
  - 扩展后先跑 1 次 ≤1k 步的 smoke，确认 pool 长度、权重长度、日志中的 `[decks]` 行与实际一致，再进入正式阶段。
- `--deck-weights-file` 已经是现成接口（`:643-652`）：`np.load` 后按 pool 顺序一一对应；权重会归一化。

### 1.6 随机策略对手

- `rl/train_selfplay.py` **没有随机合法策略对手分支**。训练侧只有：对称自博弈、frozen `--opponent-ckpt`（finetune）、teacher/旧策略锚点；`random` 只出现在 seed、座位和 minibatch shuffle。
- 因此不能宣称“随机策略已加入 PPO 训练”。本方案的随机合法策略只出现在**离线评估矩阵**（`eval_h2h.py --opponent random`）。
- 若将来要把随机对手加入训练，需要单独实现 + 单测 + 小规模 A/B；本轮不做。

### 1.7 编码路径

- 本机引擎不导出 `GetBinaryObs` / `AdvanceLog`，`native_encode_available=False`；`--native-encode` 会失败。
- 训练与门禁必须使用默认的 **Python 编码回退**；不要开启 `--native-encode`。所有结果只与 Python 编码口径比较。

## 2. “10–20M 步”的实际预算与耗时

实测来源：`runs/bench-20m/report.md`（E16，`num_envs=16, num_steps=64`，JSON/Python 编码，compile 关闭）。

| 预算（环境步） | 迭代数（batch 1024） | 优化器 step（×16） | E16 端到端 86.6 步/s | 说明 |
|---:|---:|---:|---:|---|
| 0.5M | ~488 | ~7,808 | ~1.6 h | 建议的第一阶段试跑 |
| 1M | ~976 | ~15,616 | ~3.2 h | 与当前 1M 训练同量级 |
| 2M | ~1,953 | ~31,248 | ~6.4 h | 第二阶段上限 |
| 5M | ~4,883 | ~78,128 | ~16.0 h | 可选第三阶段 |
| 10M | ~9,766 | ~156,256 | ~32.1 h | 不建议直接启动 |
| 20M | ~19,531 | ~312,500 | ~64.1 h | 当前证据不支持 |

- 当前 1M 训练的时间戳（500736→999424 约 1h42m）与 E16 实测吞吐一致；1M 全量约 3.2–3.4 h。
- E32 在 benchmark 中 CUDA OOM，不可用；E8 约 81.5 步/s，略慢。推荐 E16。
- 上述是**连续运行**估算；加门禁、保存、恢复、环境启动，实际多 10–20%。
- `latest.pt` 243 MB、`ckpt` 81 MB；按 250k/500k 保存，I/O 开销远小于训练本身。恢复的主要成本是重新加载 243 MB 优化器状态和启动环境，实测 warm load <0.1 s，冷盘需另测。

## 3. 牌组分布与采样权重

### 3.1 目标分布

| 组 | 牌组数 | 权重 | 单副权重 |
|---|---:|---:|---:|
| `TRAIN_TOP15` | 30 | 0.60 | 0.02 / 副 |
| `official_dragapult` | 1 | 0.25 | 0.25 |
| 当前提交牌组（`agent/deck.csv`） | 1 | 0.15 | 0.15 |
| 合计 | 32 | 1.00 | — |

### 3.2 设计理由

- 60% 保留 generalist 覆盖，避免把模型拉回单一 archetype；TOP15 的 30 副按均匀权重，不人为放大某一副。
- 25% 给候选提交牌组 `official_dragapult`，使模型有机会专门改善它；上限 25%，避免单牌组过拟合。
- 15% 保留当前 `agent/deck.csv`，作为“不遗忘对照牌组”和回退保险；它的权重低于候选牌组，但不为零。
- 不把任何单副牌组放到 >25%；如果后续要提高候选牌组权重，必须先通过分牌组门禁，且每次最多 +5pp。

### 3.3 防过拟合措施

1. 分牌组评估：dragapult、当前牌组、TOP15 中随机抽 5 副、以及 3 副**未参与训练**的 TOP50 牌组（held-out）分别跑独立对局。
2. 牌组交叉：dragapult vs 当前牌组双向各 100 局，确认候选牌组优势没有被训练分布改写。
3. 训练日志的 `matchup_r0.jsonl` 只用于观察采样覆盖，不作为效果结论；效果结论只看离线 h2h。
4. 若 dragapult 提升但当前牌组/held-out 下降 >3pp，立即停止并保留 `ckpt_999424.pt`。

### 3.4 执行前的接口产物（尚未创建/未测试）

```text
runs/finetune-20261020/pool_top15_deploy.json
runs/finetune-20261020/pool_top15_deploy_w.npy
```

- `pool_top15_deploy.json`：`{"decks": [30 副 TOP15..., dragapult, control], "counts": [60...], "names": [...]}`。
- `pool_top15_deploy_w.npy`：`[0.02] * 30 + [0.25, 0.15]`，`np.float64`。
- 由于 `resolve_deck_pool()` 目前不接受 `json:`，上述文件只有在完成 1.5 节的接口扩展后才能被训练读取。**在扩展和 smoke 通过前，不得宣称该分布已生效。**

## 4. 分阶段微调方案

### Stage 0：冻结基线（已完成）

- 当前候选：`ckpt_999424.pt`。
- 200 局混合池：117/200 = 58.5%，CI [51.6%, 65.1%]；0 异常/0 非法。
- 三种子牌组交叉：dragapult 视角 56.5%（22+23 合并），control 视角 42.5%。
- 这些数字是本方案的门禁基线。
- 混合池的 random/self-mirror 基线尚未单独测；Stage 1 前先补测同池同种子的 100 局 random 与 self-mirror，作为回归阈值。不要用 dragapult/control 单牌组的历史 random 数字替代混合池基线。

### Stage 1：0.5M 环境步试跑（建议首先执行）

> 下列命令中的 `--decks json:` 需要先完成 1.5 节的接口扩展；当前代码会报 `unknown --decks`。未完成扩展前，只能用 `--decks top15`/`official`/`sample` 的单一预定义池，不能执行本方案的三牌组分布。

```text
--init-from runs/train-20m-e16-1m/ckpt_999424.pt
--decks json:runs/finetune-20261020/pool_top15_deploy.json
--deck-weights-file runs/finetune-20261020/pool_top15_deploy_w.npy
--num-envs 16 --num-steps 64 --num-minibatches 16 --update-epochs 1
--total-timesteps 500000
--lr 5e-5 --warmup-steps 50000 --decay-steps 500000 --lr-min-ratio 0.1
--teacher-kl-coef 0.005 --teacher-value-coef 0.005
--teacher-refresh 5000000
--save-every 250000 --log-every 1 --value-diag
--out runs/finetune-20261020/stage1
```

- 预算：约 1.6 h（E16），约 488 次迭代、7,808 次优化器 step。
- `--init-from` 让 global_step 从 0 开始，日志中的 step 是**本轮相对步数**；不要与 999,424 混读。
- `--teacher-kl-coef 0.005` / `--teacher-value-coef 0.005` 是建议的稳定项；teacher 从 `ckpt_999424` 复制，`--teacher-refresh 5M` 大于本轮预算，不会中途换 teacher。
- 训练结束后，用 `ckpt_{step}.pt`（net-only）做评估；用 `latest.pt` 只在需要继续训练时恢复。

**Stage 1 中途检查（250k 步）**：

- 0 异常、0 非法动作；KL 没有连续 10 次 >0.05；entropy 没有快速塌缩。
- 与 `ckpt_999424` 的 200 局 h2h 快速门禁：胜率 <50% 或 0 非法不成立 → 立即停止，保留原 checkpoint。

**Stage 1 结束门禁（500k 步）**：

- 主门禁：`h2h_ckpt.py new vs ckpt_999424`，双牌组池，2 个种子 × 200 局 = 400 局；合并胜率 **≥55%** 且 Wilson 95% 下界 **>50%**。
- 安全：0 异常、0 非法动作、平局 ≤2%。
- 牌组保持：dragapult vs control 双向各 100 局，dragapult 视角 ≥54%，control 视角 ≤46%，且相对 Stage 0 不下降 >2pp。
- 保留项：random 100 局不低于 Stage 0 同池同种子 -3pp；self-mirror 100 局在 45–55% 且 0 异常；held-out TOP50 3 副不低于 Stage 0 -3pp。
- 若主门禁失败：停止，不进入 Stage 2；保留 `ckpt_999424.pt` 和 Stage 1 的中间 ckpt 作为证据。

### Stage 2：扩展到累计 2M 环境步（仅在 Stage 1 通过后）

- 从 Stage 1 的最佳 ckpt 继续：优先用 `--resume stage1/latest.pt`（保留优化器），或重新 `--init-from` Stage 1 最佳 net（更干净，但优化器重置）。
- 预算语义不同：`--resume` 时 `--total-timesteps` 是**累计**预算（Stage 2 设为 2,000,000，global_step 从 Stage 1 继续）；`--init-from` 时是**本轮新增**预算（设为 1,500,000，global_step 从 0 开始）。不要把两者混用。
- 追加 1.5M 步（累计 2M），约 4.8 h；每 500k 步跑一次 200 局快速门禁。
- 门禁阈值同 Stage 1；若两次连续 500k 窗口的 h2h 点估计没有上升（或低于 53%），停止。
- 累计 2M 仍无明显改善 → 停止，不进入 Stage 3。

### Stage 3：可选累计 5M（不建议现在启动）

- 仅当 Stage 2 的 2M 结果相对 0.5M/1M 呈单调改善，且分牌组/held-out 没有回归时，追加 3M（累计 5M，约 16 h）。
- 10M（约 32 h）和 20M（约 64 h）**不建议在当前证据下启动**；需要先把 5M 的门禁做扎实，再讨论。

## 5. 评估矩阵（每阶段结束）

| 项 | 命令/协议 | 用途 | 阈值 |
|---|---|---|---|
| 主改善门禁 | `h2h_ckpt.py new ckpt_999424 200` ×2 seeds，双牌组池 | 新模型是否超过当前候选 | 合并 ≥55%，CI 下界 >50% |
| 旧基线 | `eval_h2h.py --opponent baseline --mode mirror --games 100`，双牌组池 | 是否仍胜 `ckpt_500736` | ≥55%（Stage 0 为 58.5%） |
| 自镜像 | `eval_h2h.py --opponent selfmirror --mode mirror --games 100` | 管线/座位 sanity | 45–55%，0 异常 |
| 随机合法 | `eval_h2h.py --opponent random --mode mirror --games 100` | 回归下限（离线，非训练对手） | 不低于 Stage 0 -3pp |
| 牌组交叉 A | `eval_h2h.py --mode cross`，dragapult 候选、control 对手，100 局 | 候选牌组是否仍占优 | dragapult 视角 ≥54% |
| 牌组交叉 B | 反向，control 候选、dragapult 对手，100 局 | 对称检查 | control 视角 ≤46% |
| held-out | 3 副未参与训练的 TOP50 牌组，各 100 局 vs baseline | 防遗忘 | 不低于 Stage 0 -3pp |

- 所有门禁必须使用**独立对战**，不得用训练内 `tch_wr` 代替。
- `h2h_ckpt.py` 的座位 flip 使用字面量 `seed=4321`，`--seed` 主要改变池采样顺序；两个种子不是独立的座位随机化。若需要座位平衡的补充证据，用 `eval_h2h.py` 的严格 50/50 协议，但 `eval_h2h.py` 只能对固定 baseline，不能替代 ckpt-vs-ckpt 主门禁。
- `h2h_ckpt.py` 不输出非法动作计数；需要时用 `runs/evaluation/final_gate/run_h2h_instrumented.py` 的同类包装器补计。
- 编码口径固定为 Python 回退；不得把任何 `native_encode=True` 历史数字混入。

## 6. 中止与保留策略

立即停止的条件（任一成立）：

1. 任意门禁出现异常 >0 或非法动作 >0。
2. KL 连续 10 次 >0.05，或 entropy 相对 Stage 0 下降 >30%。
3. 250k/500k 快速门禁 <50%，或主门禁 400 局 <55%。
4. dragapult 提升但 control/held-out 下降 >3pp。
5. 实际吞吐低于 70 步/s，导致 0.5M 阶段超过 2.5 h（检查环境/显存后再决定）。
6. 训练日志出现 NaN/Inf 或 checkpoint 保存失败。

保留策略：

- 永远保留 `runs/train-20m-e16-1m/ckpt_999424.pt` 和 `latest.pt`，不覆盖。
- Stage 1/2/3 各自独立输出目录；失败的 ckpt 不删除，作为证据。
- 只有主门禁通过的 ckpt 才进入下一阶段；否则回退到 `ckpt_999424.pt`。

## 7. 未实现 / 未验证

1. **三牌组池接口未实现**：`resolve_deck_pool()` 目前不接受 `json:`；本方案第 3.4 节的文件和权重只是设计，尚未创建/测试。
2. **随机策略对手未加入 PPO 训练**：训练接口无该分支；随机合法策略只用于离线 `eval_h2h.py`。
3. **5e-5 / teacher KL 0.005 是建议值**：未在本模型上做 LR/系数的 A/B；首阶段必须以 KL、entropy 和门禁结果决定是否调整。
4. **held-out TOP50 牌组未选定**：执行前需要从 `TRAIN_TOP50` 中排除 TOP15 后固定 3 副，并记录名称/ID。
5. **Linux 提交包导入顺序阻塞未修复**：`import main` 先于 `kaggle_environments` 会 SIGABRT；微调不解决该问题。部署前必须修复或明确 runner 顺序，并重新跑 `linux_runtime_report.md` 的验证。
6. **训练吞吐只测到 E16/E8/E32**；若改变 `num_envs`/`num_steps`/compile 设置，耗时需重新实测。

## 8. 结论

- “10–20M 步”在当前硬件上是 **32–64 小时**的连续训练，不是一次短试跑；当前 58.5% 的边际优势不足以支持直接投入该预算。
- 建议先做 **0.5M 步（约 1.6 h）** 的 Stage 1，用 400 局独立 h2h 和分牌组门禁决定是否继续；只有 Stage 1 通过，才扩展到 2M（约 6.4 h）；5M 作为可选中止点；10–20M 暂不启动。
- 执行前先完成三牌组池的接口扩展与 1k 步 smoke；不要把“计划中的分布”写成“已生效的训练配置”。
