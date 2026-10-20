# -*- coding: utf-8 -*-
"""阶段 3b：新库下的牌组交叉评估（只变量牌组，模型/编码/环境参数固定）。

复用 runs/evaluation/eval_h2h.py 的 run_one 协议：严格均衡先后手（cs = gi % 2）、
would_ko=True、live=1 串行（JSON 引擎 battle_ptr 是进程级单例）、Wilson 95% CI、
非法动作/平局/异常分列、对局长度统计。

矩阵：D=新库校验后的 Dragapult，C=当前 agent/deck.csv，I=fixture iono
  · D→C, C→D, D→I, I→D, C→I, I→C（双向 = 谁当候选方各跑一次）
  · 每副牌组 vs 随机合法（mirror 模式）给绝对强度
两个独立种子。kind="selfmirror" 保证两侧同 checkpoint，仅牌组不同。

运行：PYTHONPATH=D:\\pkmn-kaggle\\.engine-shim PKMN_TURN_CAP=200 python runs/card_database_migration/20261020/deck_cross_eval.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

GAMES = int(os.environ.get("DECK_EVAL_GAMES", "40"))
SEEDS = [int(s) for s in os.environ.get("DECK_EVAL_SEEDS", "20261020,20261021").split(",")]
EVAL_H = os.path.join(ROOT, "runs", "evaluation", "eval_h2h.py")
CKPT = os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_999424.pt")
OUT = os.path.join(HERE, "deck_cross_eval.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_h2h():
    spec = importlib.util.spec_from_file_location("eval_h2h", EVAL_H)
    m = importlib.util.module_from_spec(spec)
    sys.modules["eval_h2h"] = m
    spec.loader.exec_module(m)
    return m


def main():
    import torch
    from rl.card_features import get_card_table
    from rl.encoding import TokenEncoder
    torch.set_grad_enabled(False)

    h = load_h2h()
    h._install_sdk_cg_shim_alias()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ct = get_card_table()
    enc = TokenEncoder(ct)
    cand_net, cand_step, cand_cfg = h.load_net(CKPT, ct, dev)

    decks = {
        "dragapult_newlib": [int(x) for x in open(os.path.join(
            ROOT, "runs", "evaluation", "deck_search", "dragapult_deck.csv")) if x.strip()],
        "current_agent_deck": [int(x) for x in open(os.path.join(ROOT, "agent", "deck.csv")) if x.strip()],
        "fixture_iono": list(__import__("rl.decks", fromlist=["DECKS"]).DECKS["iono"]),
    }
    deck_sha = {"dragapult_newlib": sha(os.path.join(ROOT, "runs", "evaluation", "deck_search", "dragapult_deck.csv")),
                "current_agent_deck": sha(os.path.join(ROOT, "agent", "deck.csv")),
                "fixture_iono": "rl/decks.py:_DECKS_SPEC['iono']"}

    cells = [(a, b) for a in decks for b in decks if a != b]
    runs = []
    for seed in SEEDS:
        for a, b in cells:
            print(f"[run] seed={seed} {a}(cand) vs {b}(opp) games={GAMES}", flush=True)
            r = h.run_one("selfmirror", GAMES, seed, [decks[a]], [decks[b]], dev, ct, enc,
                          1, "cross", cand_net, CKPT)
            r.update({"cell": f"{a} vs {b}", "cand_deck": a, "opp_deck": b, "seed": seed,
                      "cand_deck_sha": deck_sha[a], "opp_deck_sha": deck_sha[b]})
            runs.append(r)
            print("      wr=%s ci95=%s draws=%d errs=%d illegal=%d len=%.1f %.2f g/s"
                  % (r["winrate"], r["winrate_ci95"], r["draws"], r["errs"],
                     r["illegal"]["total"], r["length"]["mean_applies"], r["games_per_s"]), flush=True)
        for a in decks:
            print(f"[run] seed={seed} {a} vs random_legal (mirror)", flush=True)
            r = h.run_one("random", GAMES, seed, [decks[a]], [decks[a]], dev, ct, enc,
                          1, "mirror", cand_net, CKPT)
            r.update({"cell": f"{a} vs random", "cand_deck": a, "opp_deck": "random_legal",
                      "seed": seed, "cand_deck_sha": deck_sha[a]})
            runs.append(r)
            print("      wr=%s ci95=%s draws=%d errs=%d illegal=%d"
                  % (r["winrate"], r["winrate_ci95"], r["draws"], r["errs"],
                     r["illegal"]["total"]), flush=True)

    import platform
    out = {"library": {"path": os.path.join(ROOT, "EN_Card_Data.csv"), "sha256": sha(os.path.join(ROOT, "EN_Card_Data.csv")),
                       "vocab_size": ct.vocab_size, "feat_dim": ct.feat_dim},
           "candidate_ckpt": {"path": CKPT, "sha256": sha(CKPT), "global_step": cand_step,
                              "net_config": cand_cfg},
           "engine": {"kaggle_environments_pkg": "1.33.0",
                      "shim": os.path.join(ROOT, ".engine-shim")},
           "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                   "device": str(dev), "turn_cap": os.environ.get("PKMN_TURN_CAP"),
                   "would_ko": True, "native_encode": False, "live": 1,
                   "platform": platform.platform()},
           "protocol": {"games_per_cell": GAMES, "seeds": SEEDS, "bidirectional": True,
                        "seat_balance": "cs = gi % 2 逐局严格交替",
                        "model_fixed": "两侧同 ckpt_999424，仅牌组变量"},
           "runs": runs}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("[saved]", OUT, flush=True)


if __name__ == "__main__":
    main()
