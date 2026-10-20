# 插入"已确认问题清单"（§5.7），放在 §6 之前
import io, sys
P = r"D:\pkmn-kaggle\runs\evaluation\audit_report.md"
s = io.open(P, encoding="utf-8").read()
anchor = "## 6. 决策：是否值得继续训练、优先修复什么"
block = """### 5.7 已确认问题清单（证据与代码位置）

| # | 问题 | 严重度 | 证据 | 代码位置 |
|---|---|---|---|---|
| 1 | 提交 tar 不含引擎二进制，干净环境首个 would_ko 决策即 `FileNotFoundError` | 阻塞 | `audit_artifacts/tarcheck.log`（19 成员，二进制列表为空） | `main.py` L29-L30；`search_agent.py` L335；`sdk_cg/sim.py` L29 |
| 2 | `untrained` 是单次随机初始化抽样，22% 不可复现（0.22–0.70） | 评估方法 | §3.4；`wk_untrained_*.json` | `eval_h2h.py` L100-L107 |
| 3 | 官方引擎内部 RNG 不可设置种子，同 seed 重跑结果不同 | 复现性 | `sm_rr.log`（`reproducible=false`）；0.10 vs 0.20 | `rl/env_selfplay.py` L162-L164（只 seed Python RNG） |
| 4 | 提交牌组只有 10 张宝可梦/33 张能量，不在训练池；对随机仅 0.20 | 高 | `audit_artifacts/deck_probe_out.txt`；`ctl2_cand_vs_random_deckcsv_60.json` | `agent/deck.csv` |
| 5 | `label` 硬编码，与真实 ckpt 不符 | 数据质量 | `ctl_base_selfmirror_30.json`（label 999424 / meta 500736） | `eval_h2h.py` L110/L113 |
| 6 | `_field_candidates` 未定义，仓库 runner 不可用 | 中 | grep 仅 L235 调用；`scripts/h2h_ckpt.py` 依赖它 | `rl/gates.py` L235 |
| 7 | 旧报告把 `would_ko` 依赖说成 `WouldKO` 导出 | 文档错误 | §5.1 符号表 + 实测导出清单 | `runs/evaluation/report.md` §7.3/§8.1 |

"""
assert s.count(anchor) == 1
s = s.replace(anchor, block + anchor)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("OK inserted 5.7")
