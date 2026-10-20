#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4c：官方 harness 下带插桩的真实对局。

记录：每局结果（reward/result）、引擎 GameInitialize 次数、would_ko Search API 调用与正例、
动作合法性（harness 侧非法动作会直接报错）。同时 dump 一次 env 状态结构，作为口径说明。
"""
import json
import os
import random
import sys

PKG = os.environ.get("VERIFY_PKG", "/home/swcmb/pkmn-migration-verify-20261020/pkg")
PYLIBS = os.environ.get("VERIFY_PYLIBS", "/home/swcmb/pkmn-linux-verify-20261020/pylibs")
sys.path.insert(0, PYLIBS)
sys.path.insert(0, PKG)

import ctypes                                              # noqa: E402
LOADS, INITS = [], []
_orig = ctypes.cdll.LoadLibrary


def _load(name, *a, **k):
    lib = _orig(name, *a, **k)
    LOADS.append(os.path.basename(name))
    try:
        real = lib.GameInitialize
        lib.__dict__["_gi_real"] = real
        lib.__dict__["GameInitialize"] = lambda *aa, **kk: (INITS.append(os.path.basename(name)), real(*aa, **kk))[1]
    except Exception:                                       # noqa: BLE001
        pass
    return lib


ctypes.cdll.LoadLibrary = _load

import search_agent as sa                                   # noqa: E402
WK = {"calls": 0, "nonempty": 0, "positive_ko": 0, "positive_win": 0, "samples": []}
_orig_flags = sa.would_ko_flags


def counting_flags(obs, deck, enc, *a, **kw):
    r = _orig_flags(obs, deck, enc, *a, **kw)
    WK["calls"] += 1
    if r:
        WK["nonempty"] += 1
    for _k, v in (r or {}).items():
        ko, prizes, win = (list(v) + [0, 0, 0])[:3]
        if ko > 0:
            WK["positive_ko"] += 1
            if len(WK["samples"]) < 3:
                WK["samples"].append({"ko": ko, "prizes": prizes, "win": win})
        if win > 0:
            WK["positive_win"] += 1
    return r


sa.would_ko_flags = counting_flags
sa.annotate_would_ko.__globals__["would_ko_flags"] = counting_flags

import main                                                 # noqa: E402
from kaggle_environments import make                        # noqa: E402


def read_deck():
    return [int(x) for x in open(os.path.join(PKG, "deck.csv")) if x.strip()]


def random_agent(opp_deck):
    def fn(obs):
        sel = obs.get("select")
        if sel is None:
            return list(opp_deck)
        opts = sel.get("option") or []
        if not opts:
            return []
        k = max(sel.get("minCount", 1) or 1, 1)
        hi = max(k, min(sel.get("maxCount", 1) or 1, len(opts)))
        return sorted(random.sample(range(len(opts)), min(hi, len(opts))))
    return fn


games, dump = [], None
for seat in (0, 1):
    for g in range(3):
        env = make("cabt")
        opp = random_agent(read_deck())
        order = [main.agent, opp] if seat == 0 else [opp, main.agent]
        err = None
        try:
            env.run(order)
        except Exception as e:                               # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        rewards = None
        try:
            rewards = [s.reward for s in env.state]
        except Exception:                                    # noqa: BLE001
            try:
                rewards = [env.info[i].get("reward") for i in range(2)]
            except Exception:                                 # noqa: BLE001
                rewards = None
        games.append({"agent_seat": seat, "game": g, "error": err, "rewards": rewards,
                      "steps": len(env.steps) if getattr(env, "steps", None) else 0})
        if dump is None:
            dump = {"state_keys": [list(vars(s).keys())[:14] for s in env.state][:1],
                    "reward_list": [s.reward for s in env.state],
                    "info": {k: type(v).__name__ for k, v in list(env.info.items())[:8]}
                    if isinstance(env.info, dict) else str(type(env.info))}

out = {"games": games, "loads": LOADS, "game_initialize_calls": len(INITS),
      "would_ko": WK, "errors": [g["error"] for g in games if g["error"]],
      "state_dump": dump, "engine_lib_path": [l for l in LOADS]}
print(json.dumps(out, ensure_ascii=False, default=str))
