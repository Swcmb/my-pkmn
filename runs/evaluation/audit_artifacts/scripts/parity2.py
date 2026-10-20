# 前向路径对拍 v2：monkeypatch main._select，在真实 agent 决策时同时算两种口径
import importlib.util, json, os, random, sys

PKG = r"C:\Users\Swcmb\AppData\Local\Temp\opencode\subsmoke2\pkg"
sys.path.insert(0, PKG)
import numpy as np, torch                                   # noqa: E402
torch.set_grad_enabled(False)
import sdk_cg.game as game                                  # noqa: E402
from encoding import build_mask, SUBMIT_ACTION              # noqa: E402
from policy import bucketed_opt_len_np                      # noqa: E402

spec = importlib.util.spec_from_file_location("submain", os.path.join(PKG, "main.py"))
main = importlib.util.module_from_spec(spec)
sys.modules["submain"] = main
spec.loader.exec_module(main)
NET, ENC, DECK = main._NET, main._ENC, main.DECK
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 12
st = {"decisions": 0, "mismatch": 0, "wk_nonzero": 0, "opt_len_hist": {}, "examples": []}


def harness_action(o):
    batch = {k: torch.as_tensor(np.asarray(v)[None],
            dtype=(torch.long if k in ENC.int_keys else torch.float32)) for k, v in o.items()}
    ol = bucketed_opt_len_np(batch["action_mask"], batch["opt_attr"])
    return int(NET.logits_value(batch, opt_len=ol)[0].argmax(-1).item()), ol, o


_orig = main._select


def spy(obs, picked):
    a = _orig(obs, picked)
    o = ENC.encode(obs, set(picked), self_deck=DECK, tracker=main._TRACKER,
                   ability_slots=main._ABILITY.slots)
    ah, ol, o2 = harness_action(o)
    st["decisions"] += 1
    st["opt_len_hist"][ol] = st["opt_len_hist"].get(ol, 0) + 1
    if np.abs(np.asarray(o2["opt_attr"])[:, 11:14]).sum() > 0:
        st["wk_nonzero"] += 1
    if a != ah:
        st["mismatch"] += 1
        if len(st["examples"]) < 5:
            st["examples"].append({"sub": a, "harness": ah, "opt_len": ol,
                                   "sel_type": (obs.get("select") or {}).get("type")})
    return a


main._select = spy
rng = random.Random(20261019)


def scripted(sel):
    picked, mc = [], sel.get("maxCount", 1)
    for _ in range(mc + 1):
        mask = build_mask(sel, set(picked))
        leg = [i for i, m in enumerate(mask) if m]
        a = rng.choice(leg) if leg else None
        if a is None or a == SUBMIT_ACTION:
            break
        picked.append(a)
        if len(picked) >= mc:
            break
    return sorted(set(picked))


res = {"win": 0, "loss": 0, "draw": 0, "err": 0}
for gi in range(GAMES):
    main.agent({"select": None})
    obs, _sd = game.battle_start(list(DECK), list(DECK))
    napp = 0
    try:
        while obs["current"]["result"] < 0 and napp < 1500:
            seat = obs["current"]["yourIndex"]
            picks = main.agent(obs) if seat == 0 else scripted(obs["select"])
            obs = game.battle_select([int(x) for x in picks])
            napp += 1
        r = obs["current"]["result"]
        res["draw" if r == 2 else ("win" if r == 0 else "loss")] += 1
    except Exception as e:
        res["err"] += 1
        print("[err]", type(e).__name__, e, flush=True)
        break
st["game_result_seat0_subagent"] = res
st["mismatch_rate"] = st["mismatch"] / max(st["decisions"], 1)
st["wk_nonzero_rate"] = st["wk_nonzero"] / max(st["decisions"], 1)
print(json.dumps(st, ensure_ascii=False, indent=1), flush=True)
json.dump(st, open(os.path.join(os.path.dirname(PKG), "parity.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
