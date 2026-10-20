# 200 局 checkpoint 对比门禁报告

日期：2026-10-20
候选：`runs/train-20m-e16-1m/ckpt_999424.pt`（global step 999424）
基线：`runs/train-20m-e16-1m/ckpt_500736.pt`（global step 500736）
结论：**门禁通过，但优势中等**。候选 117/200 = **58.5%**，Wilson 95% CI **[51.6%, 65.1%]**；0 平局、0 异常、0 非法动作。CI 下界高于 50%，支持候选优于基线；点估计低于此前 100 局 `eval_h2h.py` 口径的 69.0%，原因是牌组池与座位/协议不同，不能直接混用。

## 1. 复核项

| 项 | 值 | 证据 |
|---|---|---|
| 候选模型 | `runs/train-20m-e16-1m/ckpt_999424.pt` | 脚本参数；JSON `a`/`step_a` |
| 候选 global step | 999424 | `torch.load(...).global_step` |
| 基线模型 | `runs/train-20m-e16-1m/ckpt_500736.pt` | JSON `b`/`step_b` |
| 基线 global step | 500736 | `torch.load(...).global_step` |
| 候选 SHA-256 | `b8a3be1d247eac45af0bbc38a207336d6f702c82fd0b19e5f787cc7addf08a1e` | `sha256sum` |
| 基线 SHA-256 | `b02b0724ebbb90916e9d495cbc4ab73b30f6bb761cea993366798942b5477a3d` | `sha256sum` |
| 两边 `net_config` | 完全相同：transformer2, d_model=544, ff=256, nhead=8, nlayers=5, value_atoms=51, would_ko=true | checkpoint `net_config` |
| 牌组池 | `runs/evaluation/final_gate/pool_control_plus_dragapult.json` | 新建，未覆盖旧文件 |
| 池内容 | 2 副 60 张：`control_deckcsv`（= `agent/deck.csv`）、`official_dragapult` | 池 JSON `names`/`counts` |
| 池权重 | `[0.5, 0.5]` | `pool_control_plus_dragapult_w.npy` |
| 采样 | `--sample 2`，两副牌都被选中；每副 100 局 | `_field_candidates(2,200)` = [0,1,0,1,…] |
| 座位分配 | `field_winrate_fast` 的随机 flip，seed 固定为 `4321`；200 局中座位 0 = 107、座位 1 = 93 | `np.random.default_rng(4321+1_000_003).integers(0,2,200)` |
| 运行参数 | `PKMN_TURN_CAP=200`，`--live 1`，`--rank 0 --world 1`，`--seed 777`（池采样） | 命令行/JSON |
| 设备 | CUDA，RTX 5060 Laptop，torch 2.14.0+cu130 | 运行环境 |
| 编码路径 | **Python 编码回退**（`native_encode_available=False`）；原生 `GetBinaryObs`/`AdvanceLog` 不在本机引擎导出 | `rl.gates._native_encode_available()` |

## 2. 主结果

命令：

```text
PYTHONPATH=".;C:/Users/Swcmb/AppData/Local/Temp/opencode/bench-shim" \
PKMN_TURN_CAP=200 \
python scripts/h2h_ckpt.py \
  runs/train-20m-e16-1m/ckpt_999424.pt \
  runs/train-20m-e16-1m/ckpt_500736.pt 200 \
  --pool-json runs/evaluation/final_gate/pool_control_plus_dragapult.json \
  --pool-weights runs/evaluation/final_gate/pool_control_plus_dragapult_w.npy \
  --sample 2 --seed 777 --live 1 --progress-every 50 \
  --out runs/evaluation/final_gate --tag h2h_cand_vs_base_200
```

| 指标 | 值 |
|---|---:|
| 对局数 | 200 |
| 候选胜 | 117 |
| 候选负 | 83 |
| 胜率（已决出） | 0.5850 |
| Wilson 95% CI | [0.5157, 0.6511] |
| 平局 / 截断 | 0 |
| 异常（spawn/step） | 0 |
| 耗时/吞吐 | 约 0.6 局/s（进度日志） |

`h2h_ckpt.py` 的 `errs` 同时计入 `draws`；本批 `errs=0, draws=0`。

## 3. 非法动作核验

`scripts/h2h_ckpt.py` → `field_winrate_fast` 不输出非法动作计数。为核实，另用
`runs/evaluation/final_gate/run_h2h_instrumented.py` 包一层
`TwoSidedSelfPlayEnv.step`，参数、池、种子与主运行完全一致，重跑 200 局：

| 指标 | 值 |
|---|---:|
| 完成局数 | 200 |
| step 调用数 | 18741 |
| illegal clamp | 0 |
| illegal forfeit | 0 |
| 异常 | 0 |
| 候选胜 | 118/200 = 59.0%（CI [52.1%, 65.6%]） |

- 两次运行相差 1 局（117 vs 118），来自 GPU/引擎的非严格确定性；均在主结果 CI 内。
- 动作 mask 在 `rl/policy.py:635` 由 `logits_value` 应用（`masked_fill(action_mask < 0.5, -1e9)`），策略不会直接选到被 mask 的动作；环境侧仍保留 clamp/forfeit 兜底。本批未触发兜底。
- 结论：**200 局内 0 异常、0 非法动作成立**（以 instrumented 重跑为直接证据）。

## 4. 与前序结果的关系

| 来源 | 协议 | 结果 | 可比性 |
|---|---|---|---|
| 本报告 | `h2h_ckpt.py`，双牌组池 50/50，200 局，随机座位 flip 107/93 | 58.5% (117/200) | 本轮主结果 |
| `runs/evaluation/h2h_ckpt/h2h_999424_vs_500736_r0of1.json` | 同一脚本，30 局，`sample=30` 的旧池，旧 runner 包装 | 66.7% (20/30)，CI [48.8%, 80.8%] | 小样本、不同池；仅方向参考 |
| `runs/evaluation/final_report.md` §0 | `eval_h2h.py` 对 `ckpt_500736`，两种子各 100 局，旧牌组 | 44.5% [37.8%, 51.4%] | 不同协议/牌组 |
| 同上 | `eval_h2h.py` 对 `ckpt_500736`，两种子各 100 局，`official_dragapult` | 69.0% [62.3%, 75.0%] | 不同协议/牌组 |

- 本轮池是 50/50 混合；若用前序两副牌的分牌组估计做线性预期：
  `(0.445 + 0.690) / 2 = 0.5675`，与实测 58.5% 接近。这个 56.75% 是**推断**，不是本轮直接测量；本轮 `h2h_ckpt.py` 不输出分牌组胜负。
- 因此“58.5% 低于此前 69.0%”不是模型退化，主要来自牌组池构成与协议差异。不能把 69.0% 与 58.5% 当作同一口径的前后对比。
- 旧 30 局 66.7% 的 CI 很宽，且池/座位不同，不能用来否定本轮 58.5%。

## 5. 编码口径

- 本轮 `native_encode_available=False`，所有 200 局走 **Python 编码回退**。
- 原生编码路径需要 `encode_native` + 引擎 `GetBinaryObs`/`AdvanceLog`；当前 bench-shim 的 `cg.dll` 不导出这两符号，强行开启会在第一局抛
  `AttributeError: function 'GetBinaryObs' not found`（前序报告已记录）。
- 本轮结果只对“Python 编码 + 当前引擎”有效；不能与任何 `native_encode=True` 的历史数字直接合并或比较。前序 `eval_h2h.py` 也使用 `native_encode=False`，但牌组池/座位协议不同，仍不能直接混算。

## 6. 门禁判定

| 检查 | 结果 |
|---|---|
| 200 局完成 | 是 |
| 0 异常 | 是（两次运行均为 0） |
| 0 非法动作 | 是（instrumented 运行 clamp=0, forfeit=0） |
| 候选胜率 > 50% 且 CI 下界 > 50% | 是（58.5%，CI 下界 51.6%） |
| 与旧基线相比有稳定改善 | 方向成立；幅度中等，且受池构成影响 |
| 可直接扩大训练预算 | 否；58.5% 只说明相对旧基线的边际优势，尚不足以支持 10–20M 步的大预算 |

**建议**：保留 `ckpt_999424.pt` 作为当前候选；微调先做小规模、可中止的试跑，并以本报告的分牌组/分座位补充评估作为门禁，而不是直接按 10–20M 步启动。

## 7. 产物

新增：

```text
runs/evaluation/final_gate/pool_control_plus_dragapult.json
runs/evaluation/final_gate/pool_control_plus_dragapult_w.npy
runs/evaluation/final_gate/h2h_cand_vs_base_200_r0of1.json
runs/evaluation/final_gate/h2h_cand_vs_base_200_instr_r0of1.json
runs/evaluation/final_gate/illegal_counts_200.json
runs/evaluation/final_gate/run_h2h_instrumented.py
runs/evaluation/final_gate_report.md
```

未修改：

```text
scripts/h2h_ckpt.py
rl/gates.py（本轮未新增改动；仓库已有未提交的 _field_candidates / native_encode 回退改动）
runs/train-20m-e16-1m/ckpt_999424.pt
runs/train-20m-e16-1m/ckpt_500736.pt
agent/deck.csv
```
