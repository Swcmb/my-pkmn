# -*- coding: utf-8 -*-
"""Linux 引擎侧 smoke（阶段 1/5 的官方等价环境验证，不依赖 torch）。

把导出包复制到 WSL 本地临时目录（**不动仓库内的包**），把 shim 里的 libcg.so 放进
sdk_cg/ 作为包内二进制，然后用真实引擎：
  1) 以提交牌组 battle_start，读 errorType / errorPlayer；
  2) 双座位驱动完整对局（合法下标随机选择），确认引擎接受、能终局、无崩溃；
  3) 统计退化 select（options 为空）出现次数——上一会话的 IndexError 就来自这里。
"""
import ctypes
import json
import os
import random
import shutil
import sys

SRC_PKG = "/mnt/d/pkmn-kaggle/submission_rl_transformer2_db_migrated"
SHIM_SO = "/mnt/d/pkmn-kaggle/.engine-shim/kaggle_environments/envs/cabt/cg/libcg.so"
WORK = os.path.expanduser("~/pkmn-linux-engine-smoke-20261020")
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20261020

if os.path.exists(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)
PKG = os.path.join(WORK, "pkg")
shutil.copytree(SRC_PKG, PKG, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy(SHIM_SO, os.path.join(PKG, "sdk_cg", "libcg.so"))
sys.path.insert(0, PKG)

from sdk_cg import game as g           # noqa: E402
from sdk_cg.sim import lib             # noqa: E402

DECK = [int(x) for x in open(os.path.join(PKG, "deck.csv")).read().split()]
assert len(DECK) == 60, len(DECK)
rng = random.Random(SEED)

res = {"engine_lib": getattr(lib, "_name", "?"), "deck_len": len(DECK),
       "games": 0, "finished": 0, "start_errors": [], "degenerate_select": 0,
       "illegal_like": 0, "reasons": {}, "turns": [], "steps": []}

REASON = {1: "prize", 2: "deck_out", 3: "no_active", 4: "card_effect"}


def pick(sel):
    """按 JSON select 给一个结构合法的下标集合：随机取 max(1,minCount) 个不重复下标。"""
    opts = sel.get("option") or []
    mn, mx = sel.get("minCount", 1), sel.get("maxCount", 1)
    if not opts:
        return None                       # 退化 select：无任何合法下标
    k = max(mn, 1)
    k = min(k, len(opts), max(mx, 1))
    return sorted(rng.sample(range(len(opts)), k))


for gi in range(GAMES):
    seat = gi % 2
    obs, sd = g.battle_start(list(DECK), list(DECK))
    res["start_errors"].append({"game": gi, "errorType": int(sd.errorType),
                                "errorPlayer": int(sd.errorPlayer)})
    n, finished = 0, False
    while True:
        cur = obs.get("current") or {}
        if cur.get("result", -1) >= 0:
            finished = True
            res["turns"].append(cur.get("turn", -1))
            break
        if n >= 4000:
            break
        sel = obs.get("select") or {}
        picks = pick(sel)
        if picks is None:
            res["degenerate_select"] += 1
            break
        try:
            obs = g.battle_select(picks)
        except Exception as e:                    # noqa: BLE001
            res["illegal_like"] += 1
            res.setdefault("err_samples", []).append(f"{type(e).__name__}: {e}")
            break
        n += 1
    for lg in reversed(obs.get("logs") or []):
        if lg.get("type") == 23:
            r = REASON.get(lg.get("reason"), str(lg.get("reason")))
            res["reasons"][r] = res["reasons"].get(r, 0) + 1
            break
    res["steps"].append(n)
    res["games"] += 1
    res["finished"] += int(finished)

res["errorTypes"] = sorted({e["errorType"] for e in res["start_errors"]})
res["verdict"] = {
    "引擎接受提交牌组": res["errorTypes"] == [0],
    "全部对局终局": res["finished"] == res["games"],
    "无驱动异常": res["illegal_like"] == 0,
    "无退化select": res["degenerate_select"] == 0,
}
print(json.dumps(res, ensure_ascii=False, indent=2))
with open("/mnt/d/pkmn-kaggle/runs/evaluation/session-20261020/linux_engine_smoke.json",
          "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
