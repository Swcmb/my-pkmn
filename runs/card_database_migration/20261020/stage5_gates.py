# -*- coding: utf-8 -*-
"""阶段 5：迁移库上的统一评估门禁重验（复用 eval_h2h.run_one 协议，不重写对战逻辑）。

矩阵（同一 checkpoint / 编码路径 / 环境参数，按任务要求分列）：
  cand = ckpt_999424（候选）, base = ckpt_500736（旧基线）
  decks: dragapult_newlib（新库校验通过）、current_agent_deck（agent/deck.csv）
  held-out 牌组池：runs/evaluation/pool_top50.json（对手座位轮转）
每个 cell 记录：ckpt 路径+SHA+step、牌组文件+SHA、新库 SHA、种子、Python/PyTorch/引擎版本、
编码路径（python encode + would_ko 引擎标注，native_encode=False）、座位分配、
平局/异常/非法动作计数、对局长度与耗时。

运行：PYTHONPATH=D:\\pkmn-kaggle\\.engine-shim PKMN_TURN_CAP=200 python runs/card_database_migration/20261020/stage5_gates.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

CAND = os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_999424.pt")
BASE = os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_500736.pt")
POOL_HELDOUT = os.path.join(ROOT, "runs", "evaluation", "pool_top50.json")
DRAG = os.path.join(ROOT, "runs", "evaluation", "deck_search", "dragapult_deck.csv")
CUR = os.path.join(ROOT, "agent", "deck.csv")
LIB = os.path.join(ROOT, "EN_Card_Data.csv")
OUT = os.path.join(HERE, "stage5_gates.json")
GAMES = int(os.environ.get("GATE_GAMES", "60"))
SEEDS = [int(s) for s in os.environ.get("GATE_SEEDS", "20261020,20261021").split(",")]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def read_deck(p):
    return [int(x) for x in open(p) if x.strip()]


def load_h2h():
    spec = importlib.util.spec_from_file_location("eval_h2h", os.path.join(ROOT, "runs", "evaluation", "eval_h2h.py"))
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
    cand_net, cand_step, cand_cfg = h.load_net(CAND, ct, dev)
    base_net, base_step, base_cfg = h.load_net(BASE, ct, dev)
    held = [list(d) for d in json.load(open(POOL_HELDOUT, encoding="utf-8"))["decks"]]
    decks = {"dragapult_newlib": read_deck(DRAG), "current_agent_deck": read_deck(CUR)}
    deck_sha = {"dragapult_newlib": sha(DRAG), "current_agent_deck": sha(CUR)}

    cells = []
    for dname, deck in decks.items():
        for seed in SEEDS:
            cells.append(("cand_vs_random", dname, seed,
                          ("random", GAMES, seed, [deck], [deck], "mirror", cand_net, CAND, cand_step)))
            cells.append(("cand_selfmirror", dname, seed,
                          ("selfmirror", GAMES, seed, [deck], [deck], "mirror", cand_net, CAND, cand_step)))
        cells.append(("cand_vs_untrained", dname, SEEDS[0],
                      ("untrained", GAMES, SEEDS[0], [deck], [deck], "mirror", cand_net, CAND, cand_step)))
        cells.append(("base_vs_random", dname, SEEDS[0],
                      ("random", GAMES, SEEDS[0], [deck], [deck], "mirror", base_net, BASE, base_step)))
    # held-out 牌组交叉：候选与对手都用 held-out 池轮转
    for seed in SEEDS:
        cells.append(("cand_vs_baseline_heldout", "pool_top50", seed,
                      ("baseline", GAMES, seed, held, held, "cross", cand_net, CAND, cand_step)))

    runs = []
    for label, dname, seed, spec in cells:
        kind, games, sd, pool, net_pool, mode, cnet, ck, step = spec
        print(f"[gate] {label} deck={dname} seed={sd} games={games}", flush=True)
        r = h.run_one(kind, games, sd, pool, net_pool, dev, ct, enc, 1, mode, cnet, ck)
        r.update({"cell": label, "deck_name": dname, "seed": seed,
                  "ckpt_path": ck, "ckpt_sha256": sha(ck), "ckpt_step": step,
                  "net_config": (cand_cfg if ck == CAND else base_cfg),
                  "deck_sha256": deck_sha.get(dname, sha(POOL_HELDOUT)),
                  "deck_source": dname if dname in deck_sha else POOL_HELDOUT})
        runs.append(r)
        print("     wr=%s ci95=%s draws=%d errs=%d illegal=%d len=%.1f %.2f g/s"
              % (r["winrate"], r["winrate_ci95"], r["draws"], r["errs"],
                 r["illegal"]["total"], r["length"]["mean_applies"], r["games_per_s"]), flush=True)

    out = {"library": {"path": LIB, "sha256": sha(LIB), "vocab_size": ct.vocab_size,
                       "feat_dim": ct.feat_dim},
           "engine": {"kaggle_environments": "1.33.0", "shim": os.path.join(ROOT, ".engine-shim"),
                      "would_ko": True, "native_encode": False,
                      "encoding_path": "python TokenEncoder + 引擎 would_ko 标注"},
           "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                   "device": str(dev), "turn_cap": os.environ.get("PKMN_TURN_CAP"),
                   "live": 1, "platform": platform.platform()},
           "protocol": {"harness": "runs/evaluation/eval_h2h.py run_one（原协议，未重写）",
                        "seat_balance": "cs = gi % 2 逐局交替", "games_per_cell": GAMES,
                        "seeds": SEEDS},
           "runs": runs}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("[saved]", OUT, flush=True)


if __name__ == "__main__":
    main()
