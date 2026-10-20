import os
import sys

def _agent_dir():
    candidates = list(sys.path) + ["/kaggle_simulations/agent", os.getcwd()]
    for d in candidates:
        if d and os.path.exists(os.path.join(d, "deck.csv")):
            return d
    return os.getcwd()

_HERE = _agent_dir()
sys.path.insert(0, _HERE)
import torch
from card_features import get_card_table
from encoding import SUBMIT_ACTION
from encoding import TokenEncoder, GameTracker, AbilityTracker
from policy import build_token_net

_DEVICE = torch.device("cpu")
_CARDS = get_card_table(os.path.join(_agent_dir(), "EN_Card_Data.csv"))
_ENC = TokenEncoder(_CARDS)
_CK = torch.load(os.path.join(_agent_dir(), "model.pt"), map_location="cpu")
_NET = build_token_net(_CARDS, _CK.get("net_config", {}))
_NET.load_state_dict(_CK["net"])
_NET.eval()
# net trained with the engine-sim would_ko feature? -> annotate it at inference too, so opt_attr's
# would_ko column is populated exactly as in training (train==test). Else it would read all-zero.
_WK_ON = bool(_CK.get("net_config", {}).get("would_ko"))
if _WK_ON:
    import search_agent as _SA2

# our true 60-card decklist (threaded as self_deck) + per-game reveal/ability memory.
with open(os.path.join(_agent_dir(), "deck.csv")) as f:
    DECK = [int(line) for line in f if line.strip()]
_TRACKER = GameTracker()
_ABILITY = AbilityTracker()

@torch.no_grad()
def _select(obs, picked):
    o = _ENC.encode(obs, set(picked), self_deck=DECK, tracker=_TRACKER,
                    ability_slots=_ABILITY.slots)
    t = {k: torch.as_tensor(v[None], dtype=(torch.long if k in _ENC.int_keys else torch.float32),
                            device=_DEVICE) for k, v in o.items()}
    logits, _ = _NET.logits_value(t)
    return int(logits.argmax(-1).item())

def agent(obs):
    sel = obs.get("select")
    if sel is None:                       # engine asking for the deck == start of a new game
        _TRACKER.reset(); _ABILITY.reset()
        return DECK
    # update memories ONCE per decision obs (NOT per buffered pick) -- exactly what the
    # training env's learner does (decision-obs-only), so reveal/ability memory is train==test.
    _ABILITY.note_turn((obs.get("current") or {}).get("turn"))
    _TRACKER.update(obs)
    if _WK_ON:                            # engine-sim KO per attack option, ONCE per decision (matches env)
        _SA2.annotate_would_ko(obs, DECK, _ENC)
    picked = []
    max_count = sel.get("maxCount", 1)
    for _ in range(max_count + 1):        # buffer single picks into a full selection
        a = _select(obs, picked)
        if a == SUBMIT_ACTION:
            break
        picked.append(a)
        if len(picked) >= max_count:
            break
    _ABILITY.record(sel, picked)          # remember OUR ability picks for later decisions this turn
    return sorted(set(picked))
