
import ctypes, json, os, sys
PKG = sys.argv[1]
LOADS, INITS = [], []
_orig = ctypes.cdll.LoadLibrary
def _load(name, *a, **k):
    lib = _orig(name, *a, **k)
    LOADS.append(os.path.basename(name))
    try:
        real = lib.GameInitialize
        lib.__dict__["GameInitialize"] = lambda *aa, **kk: (INITS.append(os.path.basename(name)), real(*aa, **kk))[1]
    except Exception:
        pass
    return lib
ctypes.cdll.LoadLibrary = _load
sys.path.insert(0, PKG)
import sdk_cg.sim as ours
from kaggle_environments.envs.cabt.cg import sim as official
import sdk_cg.game as g
deck = [int(x) for x in open(os.path.join(PKG, "deck.csv")) if x.strip()]
obs, sd = g.battle_start(deck, deck)
res = {"loads": LOADS, "game_initialize_calls": len(INITS),
       "same_handle": ours.lib is official.lib,
       "battle_accepted": obs is not None, "errorType": sd.errorType}
if obs is not None:
    g.battle_finish()
print(json.dumps(res))
