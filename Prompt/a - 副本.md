# Objective

- 评估 D:\pkmn-kaggle 已训练约 999K 步的自博弈 PPO 模型（runs/train-20m-e16-1m/ckpt_999424.pt），产出含 95% CI 的独立对战结果、value_diag 分局长短分析、通过校验的提交包，并写入 runs/evaluation/report.md。
- Requirements
  - 不启动新训练；复用现有评估与导出脚本，不改训练框架。
  - 每个对手目标 100 局，固定种子并交换先后手；报告胜率、95% CI、非法动作率、平局率、平均对局长度。
  - 对手池须含随机合法策略、初始模型、现有基线。
  - 核验 value_diag 的 sign_acc/mse_out 是否基于独立样本与真实终局标签，并分长局/短局统计。
  - 提交包须验证：main.py 接口、60 张牌 deck.csv、所有动作合法、平台沙箱兼容。
  - 评估失败或信息不足须明确指出原因；禁止编造胜率、假设参数存在、宣称未测试包可运行。
  - 输出 runs/evaluation/report.md，列出真实测试结果、候选模型路径、牌组与导出产物。

## Decisions

- 对手池用 rl/decks_train.py 的 TRAIN_TOP15/TRAIN_TOP50 构建 pool_top15.json/pool_top50.json 与全 1 权重 npy（仓库无 data/）。
- 因 field_winrate_fast/gated_winrate 缺非法动作率与平均长度，自建驱动 runs/evaluation/eval_h2h.py，复用 TwoSidedSelfPlayEnv + rl/policy.py + rl/encoding.py。
- 评估产出放 runs/evaluation/（未跟踪），不改 scripts/。
- PKMN_TURN_CAP=200；须在 import rl 前 pre-parse 设该环境变量。
- 对拍用 field_winrate_fast（内部固定 seed=4321，座位种子公式 seed+1_000_003+rank）；主评估用 gi % 2 严格均衡座位。
- 牌组按 stride 7 轮转；评估用 mirror 模式（同 h2h_ckpt.py），cross 为可选附加。
- 自建驱动照搬 field_winrate_fast 骨架：_spawn 返回 {"env","obs","seat","cs"}；双组批量前向按 b["seat"]==b["cs"] 分流（net 对候选组、field_net 对对手组）；acts = gnet.logits_value(batch, opt_len=bucketed_opt_len_np(np_obs["action_mask"], np_obs["opt_attr"]))[0].argmax(-1).cpu().numpy()；终局 t=info.get("terminal")，t[cs]>t[1-cs] 计 wins。
- 非法动作双路径计数：clamp 型 info["illegal_action"]=True（须在 step 前记录 acting seat）、提交型 env._forfeit；异常局单列 errs，不计 draws。
- 对局长度双指标：_Counted 子类重写 _apply 计真实 apply 次数（reset 内清零），以及终局 env._state()["turn"]。
- “初始模型”= 同 net_config 随机初始化 net（真 untrained）；runs/bench-20m/E8/checkpoint/ckpt_20480.pt 作额外数据点单列。
- 本机无 native（cg.dll 缺 GetBinaryObs 等导出），统一 native_encode=False；would_ko=True 在此路径走 search_agent.annotate_would_ko，编码器读三列 would_ko/would_ko_prizes/would_ko_win；训练 net_config 为 would_ko:True，评估须开启以免特征偏移。
- 自建 env 沿用 _gate_env 的 pinned 子类语义（_sample_decks 固定 self._deck_idx，reset 内 self._deck_idx = (0,1)），但传 native_encode=False；_gate_env 本身硬编码 native_encode=True，不可直接复用。
- net 加载沿用 h2h_ckpt.py::_load：torch.load(..., map_location="cpu") → build_token_net(ct, ck["net_config"]) → load_state_dict(ck["net"]) → net.eval().to(dev)；torch.set_grad_enabled(False)。
- 编码器：ct = get_card_table() → enc = TokenEncoder(ct)；batch 张量化用 torch.as_tensor(v, dtype=torch.long if k in enc.int_keys else torch.float32, device=device)。
- 候选牌组用 agent/deck.csv；mirror 两座位同副牌由池承载，cross 模式候选座位用 agent/deck.csv。
- env 的 would_ko 取 cand_would_ko or opp_would_ko；随机策略忽略多余键。
- 对拍方案：monkeypatch rl.gates._field_candidates 后跑 scripts/h2h_ckpt.py，池用 pool_top15.json + pool_top15_w.npy、--sample 30、seed 777。
- 自建驱动调度：mirror=True 时 sched=[(c,c) for c in cand]；_spawn seed = seed + rank*1_000_003 + gi（rank=0）。
- 运行环境：唯一可用解释器 D:\ProgramFiles\anaconda3\envs\default\python.exe（torch 2.14.0+cpu，无 CUDA），评估在 CPU 上执行并在报告标注。

## Work State

### Completed

- 读任务书与仓库骨架（Prompt/a.md、README.md、AGENT.md、rl/README.md、rl/value_diag.py、scripts/export_rl_submission.py、rl/env_selfplay.py、rl/policy.py、rl/encoding.py、rl/gates.py、rl/decks_train.py 等）。
- 盘点 checkpoint 与 runs/train-20m-e16-1m/：确认含 ckpt_999424.pt（81,249,427 B，候选）、ckpt_500736.pt（同尺寸，基线）、latest.pt（243,747,237 B）、matchup_r0.jsonl、value_diag.jsonl，另有 .txt 副本备份；额外数据点 runs/bench-20m/E8/checkpoint/ckpt_20480.pt、runs/bench-20m/E16/checkpoint/ckpt_40960.pt、runs/bench-rtx5060/ckpt_32768.pt。net_config 一致：{arch:'transformer2', d_model:544, nhead:8, nlayers:5, ff:256, static:False, would_ko:True, split_heads:False, value_categorical:False, value_atoms:51, value_vmax:1.0}。
- 确认 value_diag.jsonl（34 行，末行 step=983040，sign_acc≈0.88–0.94）样本与 PPO rollout buffer 同源（非独立），终局标签为真实 terminal outcome，截断回合跳过；diag_from_buffer 仅处理真实 terminal，按 k（steps-to-terminal）分桶 BUCKETS=(2,4,8,16,32,64,128,1<<30)，sign_acc 排除 draw；短局=小 k 桶，长局=大 k 桶。
- 构建并校验对手池：pool_top15.json（30 副，10443B）+ pool_top15_w.npy（368B）、pool_top50.json（74 副，25968B）+ pool_top50_w.npy（720B）；python scripts\validate_pool_decks.py runs\evaluation\pool_top15.json --all → [validate] ALL DECKS ENGINE-OK。
- 逐行确认 rl/gates.py field_winrate_fast 全文（221–325 行）：235 行 cand = _field_candidates(len(agent_pool), n_games) 无条件调用；244 行座位 flip = np.random.default_rng(seed + 1_000_003 + rank).integers(0, 2, ...)；246–253 行 _spawn；276–288 行双组批量前向；289–312 行 step 异常与终局判定；318–323 行 finally 关闭；324–325 行返回字典（含 errs）。
- 复核 scripts/h2h_ckpt.py 全文（94 行；_load 52–56、池采样 62–66、mirror 调用 77–79、errs>0.02*n ABORT 84–85）。
- 复核 rl/env_selfplay.py 1–240 行、rl/policy.py 全文（721 行）、runs/evaluation/build_pool.py。
- 确认工作树状态：rl/train_selfplay.py 已修改未提交；AGENT.md、Prompt/、agent/、rules/、runs/bench-rtx5060/、runs/evaluation/、runs/train-20m-e16-1m/ 未跟踪；HEAD 为 bb53302。
- 重新核验 runs/evaluation/ 目录：仅 build_pool.py + 四个 pool/权重数据文件，eval_h2h.py 仍不存在。
- 重新核验 runs/train-20m-e16-1m/ 文件清单与大小：ckpt_999424.pt 与 ckpt_500736.pt 均在位，latest.pt、matchup_r0.jsonl、value_diag.jsonl 及 .txt 副本齐全。
- 复读 rl/gates.py 180–329 行确认 field_winrate_fast 逐行语义（含 finally 中 close 与 errs 返回字段），并确认 build_pool.py 输出键为 decks/counts/names、权重为 np.ones(len(decks), dtype=np.float64)。

### Active

- 即将创建 runs/evaluation/eval_h2h.py：_Counted 子类、批量张量化、随机/net 两类对手、非法动作与长度统计、Wilson CI、座位 gi % 2 均衡。

### Blocked

- scripts/h2h_ckpt.py 不可直接运行（rl/gates.py:235 依赖未定义的 _field_candidates）；须 monkeypatch 后对拍，报告须记录此缺陷。
- scripts/h2h_ckpt.py 默认池 data/census_elite_v1.json/_w.npy 缺失 → 须显式传 --pool-json/--pool-weights 且 --sample ≤ 30。
- scripts/ft_gate_ckpt.py 默认 data/deck_d1145.csv/data/ft_opp_d1145.json 缺失。
- scripts/mcts_h2h_shard.py 依赖 from kaggle_environments import make，shim 包 __init__.py 为空，尚未实测。

### Next Move

1. 创建 runs/evaluation/eval_h2h.py（_Counted 子类、batch 张量化、随机/net 两类对手、非法动作与长度统计、Wilson CI、座位 gi % 2 均衡）。
2. 跑 4 局小样本冒烟，验证 CPU 速度与非法动作计数路径（含 would_ko=True 的 JSON 开销）。
3. 跑三个必测对手各 100 局：随机合法策略、随机初始化 net、基线 ckpt_500736.pt；汇总胜率 + Wilson 95% CI + 非法动作率 + 平局率 + 平均对局长度。
4. 交叉验证：monkeypatch rl.gates._field_candidates 后跑 h2h_ckpt.py --pool-json runs/evaluation/pool_top15.json --pool-weights runs/evaluation/pool_top15_w.npy --sample 30 对拍。
5. 用 scripts/export_rl_submission.py --backend transformer2 --ckpt runs/train-20m-e16-1m/ckpt_999424.pt 导出候选包，本地校验 main.py 接口、60 张牌、动作合法性、沙箱兼容性。
6. 用 runs/train-20m-e16-1m/value_diag.jsonl 桶级数据整理长局/短局（大 k vs 小 k）统计。
7. 写 runs/evaluation/report.md（真实结果、候选模型路径、牌组、导出产物、仓库缺陷、native 不可用证据、CPU 评估标注、value_diag 独立样本结论）。

### Relevant Files

- D:\pkmn-kaggle\runs\evaluation\eval_h2h.py: 待创建的自建评估驱动（当前不存在）。
- D:\pkmn-kaggle\rl\gates.py: field_winrate_fast 驱动骨架（221–325 行）与批量张量化范式；235 行 _field_candidates 未定义。
- D:\pkmn-kaggle\rl\env_selfplay.py: TwoSidedSelfPlayEnv 构造与 reset/step/_apply/_terminal_rewards/_encode 语义。
- D:\pkmn-kaggle\rl\policy.py: logits_value、build_token_net、bucketed_opt_len_np、TokenTransformer._encode。
- D:\pkmn-kaggle\scripts\h2h_ckpt.py: _load、池采样、mirror 调用、ABORT 阈值，作为对拍基准。
- D:\pkmn-kaggle\rl\value_diag.py: diag_from_buffer、BUCKETS、DiagAccumulator.summary。
- D:\pkmn-kaggle\scripts\export_rl_submission.py: 导出入口，强制 60 牌校验，含 _bundle_engine_sim()。
- D:\pkmn-kaggle\runs\evaluation\: 现含 build_pool.py + pool_top15.json/pool_top15_w.npy/pool_top50.json/pool_top50_w.npy。
- D:\pkmn-kaggle\agent\deck.csv: 候选牌组 60 行。
- D:\pkmn-kaggle\rl\decks_train.py: TRAIN_TOP15/TRAIN_TOP50。
- D:\pkmn-kaggle\scripts\validate_pool_decks.py: 池牌组合法性校验器。
- D:\pkmn-kaggle\rl\search_agent.py: would_ko_flags、write_would_ko、annotate_would_ko。
- D:\pkmn-kaggle\runs\train-20m-e16-1m\: ckpt_999424.pt、ckpt_500736.pt、value_diag.jsonl、matchup_r0.jsonl、latest.pt。
- D:\pkmn-kaggle\rl\encoding.py: TokenEncoder、int_keys。

### Important Context

- 默认解释器 D:\ProgramFiles\anaconda3\envs\default\python.exe；torch 2.14.0+cpu，无 CUDA，评估与导出均在 CPU 上执行（报告须标注）；h2h_ckpt.py 无 CUDA 时打印 WARNING: no CUDA -- ~50x slower。
- TRAIN_TOP15 实为 30 副（k01–k15 + _AICHI(7) + META2(8)）；TRAIN_TOP50 为 74 副。
- cabt 座位不对称：P2 胜率约 0.60 → 必须交换先后手。
- PKMN_TURN_CAP 在 import rl/env_selfplay.py 时读入模块级 _TURN_CAP；PKMN_TRUNC_PENALTY 默认 0.0。
- _load_engine_lib 按 path 缓存 lib；同进程加载两个引擎 lib 会因 STB_GNU_UNIQUE 共享 CardTable 而损坏。
- logits_value 在 n < MAX_OPTIONS 时 pad -1e9 并 masked_fill(action_mask<0.5, -1e9) → argmax 必落合法位。
- step 中 _apply 返回 True 时立即返回，不经过 _steps += 1/turn-cap，故 _Counted 须在 _apply 内计数。
- _apply 抛异常时设 self._forfeit = seat 并返回 True；clamp 路径不设 forfeit。
- 训练步数证据：matchup_r0.jsonl 末行 step=999424；ckpt 与 latest.pt 的 global_step 同为 999424 → 实际约 999K 步。
- 仓库为单个 squash 提交 6e7829e，未发布符号无法从 git 历史找回。
- field_winrate_fast 非 mirror 分支才用 opp_sequence(opp_weights, n_games, seed)；mirror 分支绕过权重，sched 取 [(c,c) for c in cand][rank::world]。
- field_winrate_fast 与 gated_winrate 座位种子公式不同（seed+1_000_003+rank vs seed+rank），不可混用对拍基线。
- TwoSidedSelfPlayEnv.__init__ 中 deck_weights 长度必须等于 decks 长度（assert）。
- h2h_ckpt.py 中 pool 由 rng.choice(len(pj["decks"]), size=sample, replace=False, p=w) 采样后同时传给两个座位参数（mirror）；field_winrate_fast 默认 seed=4321、live_n=128；h2h_ckpt.py CLI 参数为 --pool-json/--pool-weights/--sample/--seed/--rank/--world/--live/--progress-every/--out/--tag，--out 写 {tag}_r{rank}of{world}.json。
- 池 JSON 由 build_pool.py 生成（键 decks/counts/names，权重全 1 np.float64，counts 为 [[0]] 占位）；退出码非零即存在非 60 张牌组。
- 本 shell 为 PowerShell：列目录须用 Get-ChildItem。
- reset() 内部对 tracker/abilities 调 reset()，并清零 _picked/_steps/_forfeit，但 _deck_idx 由 _sample_decks 写入（pinned 子类须自行设置）。
- bucketed_opt_len_np 的 present 掩码含 opt_attr[..., OPT_PICKED] > 0.5，仅在 n < opt_tok.shape[1] 时真正切片；OPT_TRUNC 默认开启。
- reset(seed=...) 会 self.rng.seed(seed)，覆盖构造传入的 seed 流。
- rl/gates.py 191–215 行注释记载 field_winrate_fast 与 field_winrate 是不同仪器，序列不可混用；注释还记载 seat 为“explicit fixed-seed draw”、candidate deck 显式置于 seat0/seat1、native encode 曾为生产默认。
- h2h_ckpt.py 在 errs > 0.02*n 时 sys.exit("ABORT: ...")。
- native_encode=True 时 would_ko 须配合 PKMN_WK_NATIVE=1，否则 _ensure_engine 抛 ValueError("native_encode (JSON-free) is incompatible with would_ko; use --no-would-ko.") —— 本机走非 native 路径不受此限。
- h2h_ckpt.py 调用 field_winrate_fast 时固定 seed=4321（非 CLI --seed），CLI --seed 只用于池采样 rng；--out 记录的 seed 是采样种子，报告对拍需注明。
- field_winrate_fast 的 finally 块对剩余 live 逐个 close() 且吞异常；_spawn/step 异常在 nxt < 5 时直接 re-raise（系统性故障不隐藏），之后计入 draws 与 errs。
- runs/train-20m-e16-1m/ 内 latest.pt（243,747,237 B）与 latest - 副本.pt.txt、matchup_r0.jsonl 与 matchup_r0 - 副本.jsonl.txt、value_diag.jsonl 与 value_diag - 副本.jsonl.txt 并存，后者为备份副本。