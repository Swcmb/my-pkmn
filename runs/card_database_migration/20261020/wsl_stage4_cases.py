#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4 用例驱动：WSL2 + 官方 kaggle_environments 1.33.0（libcg.so）运行暂存包。

用法：python3 wsl_stage4_cases.py <pkg_dir> <pylibs_dir> <case>
case: import_clean | env_first | by_path | deck_request | games | would_ko
每例一个独立进程，退出码即结论，stdout 末行是 JSON 摘要。
"""
import json
import os
import random
import sys

PKG = os.path.abspath(sys.argv[1])
PYLIBS = os.path.abspath(sys.argv[2])
CASE = sys.argv[3]
sys.path.insert(0, PYLIBS)
sys.path.insert(0, PKG)

out = {"case": CASE, "pkg": PKG}


def load_main():
    import main
    out["main_file"] = main.__file__
    out["deck_len"] = len(main.DECK)
    import sdk_cg.sim as s
    out["engine_lib"] = getattr(s.lib, "_name", None) or str(s.lib)
    out["sdk_sim_module"] = s.__name__
    return main


def read_deck(path):
    return [int(x) for x in open(path) if x.strip()]


def random_legal_pick(sel, rng):
    opts = sel.get("option") or []
    if not opts:
        return []
    k = max(sel.get("minCount", 1) or 1, 1)
    hi = max(k, min(sel.get("maxCount", 1) or 1, len(opts)))
    return sorted(rng.sample(range(len(opts)), min(hi, len(opts))))


def play_game(main_mod, agent_seat, decks, seed, turn_cap=300):
    from sdk_cg import game
    rng = random.Random(seed)
    obs, sd = game.battle_start(list(decks[agent_seat]), list(decks[1 - agent_seat]))
    if obs is None:
        return {"accepted": False, "errorPlayer": sd.errorPlayer, "errorType": sd.errorType}
    steps = illegal = 0
    exceptions = []
    result = None
    try:
        while obs.get("select") is not None and steps < turn_cap:
            seat = (obs["current"] or {}).get("yourIndex")
            sel = obs["select"]
            n_opts = len(sel.get("option") or [])
            try:
                if seat == agent_seat:
                    pick = main_mod.agent(obs)
                    if not isinstance(pick, list) or any(
                            (not isinstance(p, int)) or p < 0 or p >= n_opts for p in pick):
                        illegal += 1
                        pick = random_legal_pick(sel, rng)
                else:
                    pick = random_legal_pick(sel, rng)
                obs = game.battle_select(list(pick))
            except Exception as e:                                # noqa: BLE001
                exceptions.append(f"{type(e).__name__}: {e}")
                break
            steps += 1
        result = (obs.get("current") or {}).get("result")
    finally:
        try:
            game.battle_finish()
        except Exception:                                          # noqa: BLE001
            pass
    return {"accepted": True, "agent_seat": agent_seat, "steps": steps, "result": result,
            "illegal_picks": illegal, "exceptions": exceptions[:5]}


def opponent_deck(main_mod):
    """引擎合法的对手牌组：优先用包内 decks.py 的内置样本，否则镜像候选牌组。"""
    try:
        import decks as _d
        name = next((n for n in _d.DECKS if _d.DECKS[n] != main_mod.DECK), None)
        if name:
            return list(_d.DECKS[name]), f"decks.py:{name}"
    except Exception:                                              # noqa: BLE001
        pass
    return list(main_mod.DECK), "mirror_of_candidate"


def case_import_clean():
    load_main()
    out["ok"] = True


def case_env_first():
    import kaggle_environments                                    # noqa: F401
    from kaggle_environments import make
    env = make("cabt")
    out["env_type"] = type(env).__name__
    load_main()
    out["ok"] = True


def case_by_path():
    import importlib.util
    spec = importlib.util.spec_from_file_location("staged_agent", os.path.join(PKG, "main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out["has_agent"] = callable(getattr(m, "agent", None))
    out["deck_len"] = len(m.DECK)
    import sdk_cg.sim as s
    out["engine_lib"] = getattr(s.lib, "_name", None)
    out["ok"] = True


def case_deck_request():
    main = load_main()
    deck = main.agent({"select": None})
    from sdk_cg import api, game
    cards = {c.cardId: c.name for c in api.all_card_data()}
    basics = {c.cardId for c in api.all_card_data() if c.basic}
    unknown = [i for i in deck if i not in cards]
    opp, _ = opponent_deck(main)
    obs, sd = game.battle_start(list(deck), list(opp))
    out.update({"deck_len": len(deck), "unknown_ids": unknown[:10],
                "engine_accepted": obs is not None,
                "errorPlayer": sd.errorPlayer, "errorType": sd.errorType,
                "basic_pokemon_in_deck": len([i for i in deck if i in basics]),
                "first_names": [cards.get(i) for i in deck[:6]]})
    if obs is not None:
        game.battle_finish()


def _run_games(count_per_seat=3, seed0=20261020):
    main = load_main()
    opp, opp_src = opponent_deck(main)
    decks = [list(main.DECK), opp]
    out["opponent_deck_source"] = opp_src
    games = []
    for seat in (0, 1):
        for k in range(count_per_seat):
            games.append(play_game(main, seat, decks, seed=seed0 + 10 * seat + k))
    out["games"] = games
    out["n_games"] = len(games)
    out["all_accepted"] = all(g.get("accepted") for g in games)
    out["total_steps"] = sum(g.get("steps", 0) for g in games)
    out["illegal_picks"] = sum(g.get("illegal_picks", 0) for g in games)
    out["exceptions"] = [e for g in games for e in g.get("exceptions", [])]
    out["results"] = [g.get("result") for g in games]
    return main


def case_games():
    _run_games()


def case_would_ko():
    import search_agent as sa
    pos = {"calls": 0, "nonempty": 0, "positive_ko": 0, "positive_win": 0, "sample": None}
    orig = sa.would_ko_flags

    def wrapper(obs, deck, enc, *a, **kw):
        r = orig(obs, deck, enc, *a, **kw)
        pos["calls"] += 1
        if r:
            pos["nonempty"] += 1
            for _k, v in r.items():
                ko, prizes, win = (list(v) + [0, 0, 0])[:3]
                if ko > 0:
                    pos["positive_ko"] += 1
                    if pos["sample"] is None:
                        pos["sample"] = {"ko": ko, "prizes": prizes, "win": win}
                if win > 0:
                    pos["positive_win"] += 1
        return r

    sa.would_ko_flags = wrapper
    sa.annotate_would_ko.__globals__["would_ko_flags"] = wrapper
    main = _run_games()
    out["would_ko"] = pos
    out["annotate_would_ko_present"] = callable(getattr(sa, "annotate_would_ko", None))
    import sdk_cg.sim as s
    out["search_symbols"] = [n for n in ("SearchBegin", "SearchStep", "SearchEnd", "SearchRelease")
                             if hasattr(s.lib, n)]


DISPATCH = {"import_clean": case_import_clean, "env_first": case_env_first,
            "by_path": case_by_path, "deck_request": case_deck_request,
            "games": case_games, "would_ko": case_would_ko}

DISPATCH[CASE]()
print(json.dumps(out, ensure_ascii=False, default=str))
