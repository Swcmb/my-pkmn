# 报告修订：并入牌组消融结果 + 修正行号/措辞（每条替换断言恰好命中一次）
import io, sys

P = r"D:\pkmn-kaggle\runs\evaluation\audit_report.md"
s = io.open(P, encoding="utf-8").read()

REPL = []

REPL.append((
"""   0.5，说明这是这批训练策略对该引擎/该协议的共性，而非候选单独损坏；具体机制尚未确证（见 §4.2）。""",
"""   0.5，说明这是这批训练策略对该引擎/该协议的共性，而非候选单独损坏；具体机制尚未确证（见 §4.2）。
   另外，把评估池换成只含提交牌组 `deck.csv` 后，候选对随机只有 **0.20**（60 局，CI [0.118, 0.318]），
   说明提交牌组本身是独立风险项（§4.5）。"""))

REPL.append((
"""- 独立驱动（提交包 `main.py` + 官方引擎，镜像 deck.csv）对随机策略 40 局两跑为 8/32 与 6/34
  （合计约 0.17，牌组与评测不同，见 §4.5），方向一致（低于 0.5），但绝对值不可直接比。""",
"""- 牌组因素已单独测量：把评估池换成只含 `deck.csv` 的 60 局，候选对随机为 **0.20**（12/60，
  Wilson CI [0.118, 0.318]，seat0 0.267 / seat1 0.133，0 异常 / 0 非法）。独立驱动（提交包
  `main.py` + 官方引擎，镜像 deck.csv，40×2 局）为 8/32 与 6/34，与该 0.20 一致（合计 16/86 ≈ 0.186）。
  也就是说，43% 是「池牌组 + 镜像」下的数字，换成提交牌组会进一步跌到 0.20。"""))

old_45 = """- 该数字与 harness 的 0.43–0.45 **不可直接比较**（牌组不同、无 turn-cap、样本小）；因此只能写：
  「提交牌组存在结构性脆弱与分布外风险，量级未分离」。**未验证**：在 harness 中只换牌组
  （deck.csv vs 池）对胜率的影响；这一消融实验因权限/时间未执行。"""
new_45 = """- **牌组消融（已测）**：`runs/evaluation/ctl2_cand_vs_random_deckcsv_60.json`（60 局，仅含
  `deck.csv` 的镜像池，随机对手）：胜率 **0.20**，Wilson CI [0.118, 0.318]，seat0 0.267 / seat1 0.133，
  0 平局 / 0 异常 / 0 非法；平均 turn **16.0**（池牌组同对局为 36.8）。对局明显更短，与「10 张宝可梦
  被清场后因 `no_active` 判负」的结构性解释一致。
- 结论：提交牌组是「候选对随机偏弱」的**主要放大器**：池牌组 0.43–0.45 → 提交牌组 0.20。
  该差异与策略无关（两座位同牌组、牌力抵消），只反映牌组结构对随机对手的鲁棒性。
- **未测**：同一消融对基线/未训练对手的效果（原批次中这两组未完成）。"""
REPL.append((old_45, new_45))

REPL.append((
"""3. `deck.csv` 与池牌组在同一 harness 下的胜率差：**未测量**（原计划的消融实验未执行）。""",
"""3. 牌组消融只完成了「对随机对手」一组（0.20 vs 池牌组 0.43–0.45）；**对基线/未训练对手的牌组
   消融未测**。"""))

REPL.append((
"""4. `would_ko_flags` 内部 **L279 / L348 / L443** 等位置执行 `from sdk_cg import api`，然后
   `api.search_begin(api.to_observation_class(obs), **_determinize(...))`、`api.search_step(...)`、
   `api.search_release(...)`、`api.search_end()`。""",
"""4. `would_ko_flags`（L318-L390）在 **L335** 执行 `from sdk_cg import api`，随后 L348
   `api.search_begin(api.to_observation_class(obs), **_determinize(...))`、L351 `api.search_step(...)`、
   L367 `api.search_release(...)`、L371 `api.search_end()`（`annotate_would_ko` 在 L394，经
   `would_ko_flags` 调用同一路径）。"""))

REPL.append((
"""  effect_data.py, enc_constants.py, encoding.py, main.py, model.pt, policy.py, sdk_cg/(4 py), search_agent.py`；""",
"""  effect_data.py, enc_constants.py, encoding.py, main.py, model.pt, policy.py, sdk_cg/（5 个 py）, search_agent.py`；"""))

REPL.append((
"""  3. 训练数据（top50 池 74 副，`matchup_r0.jsonl` header `pool_size=74`）中没有任何随机策略样本，
     价值/策略头对「无结构对手」没有校准信号。""",
"""  3. 训练数据（top50 池 74 副，`matchup_r0.jsonl` header `pool_size=74`）中没有任何随机策略样本，
     价值/策略头对「无结构对手」没有校准信号。
  4. 牌组结构：提交牌组只有 10 张宝可梦，换成该牌组后对随机胜率从 0.43–0.45 跌到 0.20（§4.5）。"""))

REPL.append((
"""3. **牌组**：把 `deck.csv` 加入评估池并单独测量其胜率；当前牌组只有 10 张宝可梦、33 张能量，
   且不在训练池中（最接近 49/60），是独立于策略强度的风险项。""",
"""3. **牌组**：`deck.csv` 已测得对随机仅 0.20（池牌组 0.43–0.45）；应把它加入训练/评估分布，
   并补测它对基线/未训练对手的胜率。当前牌组只有 10 张宝可梦、33 张能量且不在训练池中
   （最接近 49/60），是独立于策略强度的风险项。"""))

REPL.append((
"""- 小样本先行：三组新对局各 10 局冒烟（0 异常 / 0 非法，方向与 30 局对照一致），随后全部跑 100 局。""",
"""- 小样本先行：三组新对局各 10 局冒烟（基线 vs 随机 0.6、基线 vs 未训练 0.0、候选自镜像 0.6；
  均 0 异常 / 0 非法），随后全部跑 100 局。"""))

REPL.append((
"""- 其余 100 局组：把 `--ckpt/--opponent/--seed/--out` 换成 §3.2 表中的组合""",
"""- 提交牌组消融（已完成 60 局，仅随机对手）：加
  `--pool-json runs\\evaluation\\pool_deckcsv_only.json --games 60 --opponent random ^
  --out runs\\evaluation\\ctl2_cand_vs_random_deckcsv_60.json`。
- 其余 100 局组：把 `--ckpt/--opponent/--seed/--out` 换成 §3.2 表中的组合"""))

for i, (old, new) in enumerate(REPL, 1):
    n = s.count(old)
    if n != 1:
        print("REPL %d: matched %d times -> ABORT" % (i, n))
        print("---- old ----")
        print(old[:200])
        sys.exit(1)
    s = s.replace(old, new)

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("OK: %d replacements applied; size=%d bytes" % (len(REPL), len(s.encode("utf-8"))))
