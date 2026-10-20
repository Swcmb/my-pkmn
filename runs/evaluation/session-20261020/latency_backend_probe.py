# -*- coding: utf-8 -*-
"""决策时延配对探针：同一批真实决策上比较 greedy 与 MCTS（阶段 1 沙箱约束 + 阶段 2 不同策略对手）。

平台限制单步 100ms、CPU 2 核。mcts2 后端每步跑 n_sims × n_det 次搜索，
若超预算则无论棋力如何都判负 —— 选型先决条件是时延。

不用 kaggle_environments.make()：仓库内的 `.engine-shim` 把包 `__init__` 链降级成空桩
（绕开 jsonschema 等重依赖），`make` 不可导入；这里直接用 `sdk_cg.game` 驱动真实引擎，
对局由 greedy 推进，同一决策的 obs 深拷贝一份分别计时两种后端（配对比较，消除局面难度差异）。
"""
import copy
import json
import os
import sys
import time

ROOT = r"D:\pkmn-kaggle"
PKG = os.path.join(ROOT, "submission_rl_transformer2_db_migrated")
sys.path.insert(0, PKG)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, ".engine-shim"))

import torch
torch.set_num_threads(2)                       # 对齐平台 2 核配额

from card_features import get_card_table
from encoding import TokenEncoder, GameTracker, AbilityTracker
from policy import build_token_net
import search_agent as SA
import sdk_cg.game as game

CKPT = os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_999424.pt")
DECK = [int(x) for x in open(os.path.join(
    ROOT, "runs", "evaluation", "card_database_migration", "best_deck.csv")) if x.strip()]
NSIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NDET = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 40      # 最多配对测多少个决策

cards = get_card_table(os.path.join(PKG, "EN_Card_Data.csv"))
enc = TokenEncoder(cards)
ck = torch.load(CKPT, map_location="cpu", weights_only=True)
net = build_token_net(cards, ck["net_config"])
net.load_state_dict(ck["net"]); net.eval()
WK = bool(ck.get("net_config", {}).get("would_ko"))

lat = {"greedy": [], "mcts": []}


def measure(obs):
    """同一 obs 的两份深拷贝上分别计时，配对消除局面难度差。返回 greedy 的选牌用于推进对局。"""
    picks = None
    for mode in ("greedy", "mcts"):
        o = copy.deepcopy(obs)
        tr, ab = GameTracker(), AbilityTracker()
        tr.update(o); ab.note_turn((o.get("current") or {}).get("turn"))
        t = time.perf_counter()
        if mode == "mcts":
            SA.mcts_select(o, net, enc, DECK, tr, ab.slots, n_sims=NSIMS, n_det=NDET)
        else:
            if WK:
                SA.annotate_would_ko(o, DECK, enc)
            picks = SA._net_greedy_select(o, net, enc, DECK, tr, ab.slots)
        lat[mode].append((time.perf_counter() - t) * 1000.0)
    return picks


tr, ab = GameTracker(), AbilityTracker()
tr.reset(); ab.reset()
obs, _sd = game.battle_start(list(DECK), list(DECK))
n = 0
while n < CAP:
    cur = obs.get("current") or {}
    if cur.get("result", -1) >= 0:
        break
    sel = obs.get("select") or {}
    if cur.get("yourIndex") == 0 and (sel.get("option") or []):
        # 用被测的 greedy 本身出牌推进：它正是提交包的策略，合法性已由 840 局与 smoke 验证
        picks = measure(obs)
    else:
        picks = list(range(min(sel.get("minCount", 0), len(sel.get("option") or []))))
    n += 1
    try:
        obs = game.battle_select([int(x) for x in picks])
    except Exception as e:                       # noqa: BLE001
        print("[drive stop] %s: %s" % (type(e).__name__, e)); break

out = {"n_sims": NSIMS, "n_det": NDET, "threads": 2, "paired_decisions": len(lat["greedy"]),
       "engine": "sdk_cg.game + .engine-shim libcg (make() 不可用：shim __init__ 为空桩)", "modes": {}}
for mode, v in lat.items():
    if not v:
        continue
    s = sorted(v)
    out["modes"][mode] = {"samples": len(s), "mean_ms": round(sum(v) / len(v), 1),
                          "p50_ms": round(s[len(s) // 2], 1),
                          "p95_ms": round(s[max(int(len(s) * 0.95) - 1, 0)], 1),
                          "max_ms": round(s[-1], 1),
                          "over_100ms": sum(1 for x in v if x > 100.0),
                          "fits_100ms": all(x <= 100.0 for x in v)}
    m = out["modes"][mode]
    print(f"{mode:>7}: 配对 {m['samples']:>3}  均值 {m['mean_ms']:>8.1f}ms  p50 {m['p50_ms']:>8.1f}  "
          f"p95 {m['p95_ms']:>9.1f}  max {m['max_ms']:>10.1f}  >100ms {m['over_100ms']}/{m['samples']}")

dst = os.path.join(ROOT, "runs", "evaluation", "session-20261020", "latency_backend_probe.json")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("[saved]", dst)
