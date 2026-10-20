
import ctypes, json, os, random, sys
PKG = sys.argv[1]
LOADS, INITS = [], []
_orig = ctypes.cdll.LoadLibrary
def _load(name, *a, **k):
    lib = _orig(name, *a, **k)
    LOADS.append(os.path.basename(name))
    try:
        real = lib.GameInitialize
        lib.__dict__["GameInitialize"] = lambda *aa, **kk: (INITS.append(1), real(*aa, **kk))[1]
    except Exception:
        pass
    return lib
ctypes.cdll.LoadLibrary = _load
sys.path.insert(0, PKG)
import importlib.util
res = {"pkg": PKG}
if sys.argv[2] == "import_main":
    import sdk_cg.sim  # noqa: F401  # 引擎封装可由包内路径正常加载
    spec = importlib.util.spec_from_file_location("staged_main", os.path.join(PKG, "main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    import search_agent as sa
    WK = {"calls": 0, "nonempty": 0, "positive_ko": 0}
    orig = sa.would_ko_flags
    def w(*a, **k):
        r = orig(*a, **k)
        WK["calls"] += 1
        if r:
            WK["nonempty"] += 1
        for _i, v in (r or {}).items():
            if list(v)[0] > 0:
                WK["positive_ko"] += 1
        return r
    sa.would_ko_flags = w
    sa.annotate_would_ko.__globals__["would_ko_flags"] = w
    from sdk_cg import game
    deck = list(m.DECK)
    opp_deck = list(__import__("decks").DECKS[list(__import__("decks").DECKS)[1]])
    out = {"deck_len": len(deck), "games": [], "first_decision": None, "consecutive": None}
    rng = random.Random(20261020)
    for seat in (0, 1):
        obs, sd = game.battle_start(deck if seat == 0 else opp_deck,
                                   opp_deck if seat == 0 else deck)
        if obs is None:
            out["games"].append({"seat": seat, "accepted": False, "errorType": sd.errorType})
            continue
        steps = illegal = 0
        exc = []
        first = None
        chain = []
        while obs.get("select") is not None and steps < 250:
            acting = (obs.get("current") or {}).get("yourIndex")
            sel = obs["select"]
            n_opts = len(sel.get("option") or [])
            try:
                if acting == seat:
                    pick = m.agent(obs)
                    if first is None:
                        first = list(pick)
                    elif len(chain) < 5:
                        chain.append(list(pick))
                    if not isinstance(pick, list) or any((not isinstance(p, int)) or p < 0 or p >= n_opts for p in pick):
                        illegal += 1
                        pick = None
                else:
                    pick = None
                if pick is None:
                    opts = sel.get("option") or []
                    if not opts:
                        break
                    k = max(sel.get("minCount", 1) or 1, 1)
                    hi = max(k, min(sel.get("maxCount", 1) or 1, len(opts)))
                    pick = sorted(rng.sample(range(len(opts)), min(hi, len(opts))))
                obs = game.battle_select(list(pick))
            except Exception as e:
                exc.append(f"{type(e).__name__}: {e}")
                break
            steps += 1
        try:
            result = (obs.get("current") or {}).get("result")
        except Exception:
            result = None
        try:
            game.battle_finish()
        except Exception:
            pass
        out["games"].append({"seat": seat, "accepted": True, "steps": steps, "result": result,
                             "illegal_shape": illegal, "exceptions": exc[:3]})
        out["first_decision"] = out["first_decision"] or first
        out["consecutive"] = out["consecutive"] or chain
    out["would_ko"] = WK
    out["loads"] = LOADS
    out["game_initialize_calls"] = len(INITS)
    res.update(out)
print(json.dumps(res, ensure_ascii=False, default=str))
