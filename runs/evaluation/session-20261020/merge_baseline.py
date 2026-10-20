# -*- coding: utf-8 -*-
"""合并 stage5_gates.json 与 stage2_supplement.json，产出阶段 2 最终基线表。

同一协议（同卡池 SHA / 同 harness / turn_cap 200 / 座位逐局交替 / Python 编码路径），
种子互不重叠（20261020-21 与 20261022-23），因此按臂合并计数是合法的。
输出每臂：合并胜/判、Wilson 95% CI、样本量、是否达到任务书要求的 >=120 局。
"""
import json
import math
import os

ROOT = r"D:\pkmn-kaggle"
SESS = os.path.join(ROOT, "runs", "evaluation", "session-20261020")
SRC = [os.path.join(ROOT, "runs", "card_database_migration", "20261020", "stage5_gates.json"),
       os.path.join(SESS, "stage2_supplement.json"),
       os.path.join(SESS, "stage2_topup.json")]

# (stage5 cell, deck) / (supplement cell, deck) -> 统一臂名
ARMS = [
    ("候选 vs 随机合法（提交牌组）", [("cand_vs_random", "dragapult_newlib")]),
    ("旧基线 vs 随机合法（提交牌组）", [("base_vs_random", "dragapult_newlib")]),
    ("候选 vs 随机合法（旧 agent 牌组）", [("cand_vs_random", "current_agent_deck")]),
    ("旧基线 vs 随机合法（旧 agent 牌组）", [("base_vs_random", "current_agent_deck")]),
    ("候选 vs 同配置未训练（提交牌组）", [("cand_vs_untrained", "dragapult_newlib")]),
    ("候选 vs 同配置未训练（旧 agent 牌组）", [("cand_vs_untrained", "current_agent_deck")]),
    ("旧基线 vs 同配置未训练（提交牌组）", [("base_vs_untrained", "dragapult_newlib")]),
    ("候选自镜像（提交牌组）", [("cand_selfmirror", "dragapult_newlib")]),
    ("旧基线自镜像（提交牌组）", [("base_selfmirror", "dragapult_newlib")]),
    ("候选自镜像（旧 agent 牌组）", [("cand_selfmirror", "current_agent_deck")]),
    ("候选 vs 旧基线：提交牌组纯镜像", [("cand_vs_baseline_mirror", "dragapult_newlib")]),
    ("候选 vs 旧基线：旧 agent 牌组纯镜像", [("cand_vs_baseline_mirror", "current_agent_deck")]),
    ("候选 vs 旧基线：74 副混合池交叉", [("cand_vs_baseline_heldout", "pool_top50")]),
]


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = w / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


rows = []
pools = []
found = []
for p in SRC:
    if os.path.exists(p):
        pools.append(json.load(open(p, encoding="utf-8")))
        found.append(p)

for name, keys in ARMS:
    w = n = dr = er = il = tc = 0
    per = []
    for d in pools:
        for r in d["runs"]:
            if (r["cell"], r["deck_name"]) in keys:
                w += r["wins"]; n += r["decided"]; dr += r["draws"]
                er += r["errs"]; il += r["illegal"]["total"]
                tc += r["draw_breakdown"]["turn_cap"]
                per.append({"seed": r["seed"], "wr": r["winrate"],
                            "wins": r["wins"], "decided": r["decided"]})
    lo, hi = wilson(w, n)
    rows.append({"arm": name, "cells": len(per), "wins": w, "decided": n, "draws": dr,
                 "wr": round(w / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)],
                 "errs": er, "illegal": il, "turn_cap": tc,
                 "n_ge_120": n >= 120, "per_cell": per})

out = {"sources": [os.path.relpath(p, ROOT) for p in found],
       "protocol_shared": {"library_sha256_prefix": "05aa569e", "turn_cap": 200,
                           "encoding": "python TokenEncoder + 引擎 would_ko",
                           "seat_balance": "逐局交替",
                           "seeds": [20261020, 20261021, 20261022, 20261023, 20261024]},
       "total_games": sum(r["decided"] + r["draws"] for r in rows),
       "arms": rows}
dst = os.path.join(SESS, "stage2_baseline_merged.json")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"{'臂':<34} {'cells':>5} {'胜/判':>10} {'wr':>7} {'CI95':>18} {'非0异常':>7} {'≥120':>6}")
for r in rows:
    ci = f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]"
    bad = r["errs"] + r["illegal"] + r["turn_cap"]
    print(f"{r['arm']:<34} {r['cells']:>5} {r['wins']:>4}/{r['decided']:<5} "
          f"{(r['wr'] or 0):>7.3f} {ci:>18} {bad:>7} {str(r['n_ge_120']):>6}")
print("\n合计对局(不含重复计入混合臂):", out["total_games"], "->", os.path.relpath(dst, ROOT))
short = [r["arm"] for r in rows if not r["n_ge_120"]]
print("未达 120 局的臂:", short if short else "无")
