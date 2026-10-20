# 提交包端到端验证 v2：agent-vs-random / random-vs-random 对照 / 终局原因 / 引擎可复现性
import importlib.util, json, os, random, shutil, sys, time, collections

PKG = r"C:\Users\Swcmb\AppData\Local\Temp\opencode\subsmoke2\pkg"
SRC = r"D:\pkmn-kaggle\submission_rl_transformer2"
DLL = r"C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim\kaggle_environments\envs\cabt\cg\cg.dll"
MODE = sys.argv[1] if len(sys.argv) > 1 else "agent"     # agent | rr
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 20261019
REPLAY = len(sys.argv) > 4 and sys.argv[4] == "replay"   # 第二次同种子重跑，比对逐局序列

if not os.path.exists(os.path.join(PKG, "main.py")):
    if os.path.exists(os.path.dirname(PKG)):
        shutil.rmtree(os.path.dirname(PKG))
    shutil.copytree(SRC, PKG, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(DLL, os.path.join(PKG, "sdk_cg", "cg.dll"))
sys.path.insert(0, PKG)

import sdk_cg.sim as _sim                                 # noqa: E402
import sdk_cg.game as game                                # noqa: E402
from encoding import build_mask, SUBMIT_ACTION            # noqa: E402

main = None
if MODE == "agent":
    spec = importlib.util.spec_from_file_location("submain", os.path.join(PKG, "main.py"))
    main = importlib.util.module_from_spec(spec)
    sys.modules["submain"] = main
    spec.loader.exec_module(main)
    assert main._WK_ON is True

DECK = [int(x) for x in open(os.path.join(PKG, "deck.csv")).read().split()]
REASON = {1: "0_prize", 2: "deck_out", 3: "no_active", 4: "card_effect"}


def one_run(seed, games):
    rng = random.Random(seed)
    st = {"wins": 0, "losses": 0, "draws": 0, "errors": 0, "illegal_returns": 0,
          "wk_decisions": 0, "wk_cells": 0, "reasons": collections.Counter(),
          "seq": [], "seat": collections.Counter(), "win_by_seat": collections.Counter()}

    def rlegal_select(sel, picked):
        mask = build_mask(sel, set(picked))
        legal = [i for i, m in enumerate(mask) if m]
        return rng.choice(legal) if legal else None

    def scripted_select(sel):
        picked = []
        for _ in range(sel.get("maxCount", 1) + 1):
            a = rlegal_select(sel, picked)
            if a is None or a == SUBMIT_ACTION:
                break
            picked.append(a)
            if len(picked) >= sel.get("maxCount", 1):
                break
        return sorted(set(picked))

    for gi in range(games):
        sub_seat = gi % 2
        st["seat"][sub_seat] += 1
        if main is not None:
            main.agent({"select": None})
        obs, sd = game.battle_start(list(DECK), list(DECK))
        napp, outcome, turns = 0, None, -1
        while True:
            cur = obs["current"]
            if cur["result"] >= 0:
                r = cur["result"]
                outcome = "draw" if r == 2 else ("win" if r == sub_seat else "loss")
                turns = cur.get("turn", -1)
                for lg in reversed(obs.get("logs") or []):
                    if lg.get("type") == 23:
                        st["reasons"][REASON.get(lg.get("reason"), lg.get("reason"))] += 1
                        break
                break
            if napp >= 3000:
                st["errors"] += 1
                outcome, turns = "err", -1
                break
            seat = cur["yourIndex"]
            sel = obs["select"]
            if main is not None and seat == sub_seat:
                try:
                    picks = main.agent(obs)
                except Exception as e:
                    st["errors"] += 1
                    print("[err] agent: %s: %s" % (type(e).__name__, e), flush=True)
                    outcome, turns = "err", -1
                    break
                src = sel.get("option") or []
                st["wk_decisions"] += 1
                st["wk_cells"] += sum(1 for o in src if isinstance(o, dict) and "would_ko" in o)
                for p in picks or []:
                    if not (isinstance(p, int) and 0 <= p < len(src)):
                        st["illegal_returns"] += 1
            else:
                picks = scripted_select(sel)
            try:
                obs = game.battle_select([int(x) for x in picks])
            except Exception as e:
                st["errors"] += 1
                print("[err] select: %s: %s" % (type(e).__name__, e), flush=True)
                outcome, turns = "err", -1
                break
            napp += 1
        if outcome == "win":
            st["wins"] += 1
            st["win_by_seat"][sub_seat] += 1
        elif outcome == "loss":
            st["losses"] += 1
        elif outcome == "draw":
            st["draws"] += 1
        st["seq"].append([napp, turns])
    st["reasons"] = dict(st["reasons"])
    st["seat"] = dict(st["seat"]); st["win_by_seat"] = dict(st["win_by_seat"])
    return st


t0 = time.time()
a = one_run(SEED, GAMES)
el = time.time() - t0
res = {"mode": MODE, "games": GAMES, "seed": SEED, "elapsed_s": el,
       "games_per_s": GAMES / max(el, 1e-9), "run1": a}
if REPLAY:
    b = one_run(SEED, GAMES)
    res["run2"] = b
    res["reproducible"] = (a["seq"] == b["seq"])
print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(PKG), "subsmoke2_%s.json" % MODE), "w",
                    encoding="utf-8"), ensure_ascii=False, indent=1)
