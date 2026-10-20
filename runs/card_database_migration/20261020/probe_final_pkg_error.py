# -*- coding: utf-8 -*-
"""定位最终包在最小自写循环里出现的 engine IndexError：
在失败点 dump select 的 type/context/minCount/maxCount/option 数与 agent 返回值，
判定是包（agent 动作）不合规，还是自写驱动的多选提交语义与引擎要求不一致。
"""
import json
import os
import random
import sys

ROOT = r"D:\pkmn-kaggle"
WORK = os.path.join(ROOT, "runs", "card_database_migration", "20261020", "final_pkg_verify")
sys.path.insert(0, WORK)

import search_agent as sa                      # noqa: E402
from sdk_cg import game                        # noqa: E402
import importlib.util                          # noqa: E402

spec = importlib.util.spec_from_file_location("pkg_main", os.path.join(WORK, "main.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import decks as _d                             # noqa: E402

agent_deck = list(m.DECK)
opp_deck = list(_d.DECKS[list(_d.DECKS)[1]])


def run(seat, seed=777, cap=200):
    rng = random.Random(seed)
    d0, d1 = (agent_deck, opp_deck) if seat == 0 else (opp_deck, agent_deck)
    obs, sd = game.battle_start(d0, d1)
    trace = []
    steps = 0
    while obs.get("select") is not None and steps < cap:
        sel = obs["select"]
        acting = (obs.get("current") or {}).get("yourIndex")
        opts = sel.get("option") or []
        if acting == seat:
            pick = list(m.agent(obs))
            who = "agent"
        else:
            k = max(sel.get("minCount", 1) or 1, 1)
            hi = max(k, min(sel.get("maxCount", 1) or 1, len(opts)))
            pick = sorted(rng.sample(range(len(opts)), min(hi, len(opts)))) if opts else []
            who = "random"
        record = {"step": steps, "who": who, "type": sel.get("type"),
                  "context": sel.get("context"), "minCount": sel.get("minCount"),
                  "maxCount": sel.get("maxCount"), "n_options": len(opts), "pick": pick}
        try:
            obs = game.battle_select(pick)
            if who == "agent" and len(trace) < 3:
                trace.append(record)
        except Exception as e:                  # noqa: BLE001
            record["error"] = f"{type(e).__name__}"
            trace.append(record)
            break
        steps += 1
    try:
        result = (obs.get("current") or {}).get("result")
    except Exception:                            # noqa: BLE001
        result = None
    try:
        game.battle_finish()
    except Exception:                            # noqa: BLE001
        pass
    return {"seat": seat, "steps": steps, "result": result, "trace": trace[:6]}


out = [run(0), run(1)]
print(json.dumps(out, ensure_ascii=False, indent=1)[:3000])
with open(os.path.join(ROOT, "runs", "card_database_migration", "20261020",
                       "final_pkg_engine_error_trace.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
