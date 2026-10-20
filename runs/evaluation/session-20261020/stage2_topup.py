# -*- coding: utf-8 -*-
"""把阶段 2 剩余 3 个臂补到 >=120 局（与 stage2_supplement.py 同协议、用新种子 S3=20261024）。"""
import hashlib
import importlib.util
import json
import os
import sys

ROOT = r"D:\pkmn-kaggle"
sys.path.insert(0, ROOT)

CAND = os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_999424.pt")
BASE = os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_500736.pt")
DRAG = os.path.join(ROOT, "runs", "evaluation", "deck_search", "dragapult_deck.csv")
CUR = os.path.join(ROOT, "agent", "deck.csv")
OUT = os.path.join(ROOT, "runs", "evaluation", "session-20261020", "stage2_topup.json")
GAMES = int(os.environ.get("TOPUP_GAMES", "60"))
S3 = 20261024


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_h2h():
    spec = importlib.util.spec_from_file_location(
        "eval_h2h", os.path.join(ROOT, "runs", "evaluation", "eval_h2h.py"))
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
    ct = get_card_table(os.path.join(ROOT, "EN_Card_Data.csv"))
    enc = TokenEncoder(ct)
    _cn, cand_step, cand_cfg = h.load_net(CAND, ct, dev)
    base_net, base_step, base_cfg = h.load_net(BASE, ct, dev)
    cand_net = _cn
    decks = {"dragapult_newlib": [int(x) for x in open(DRAG) if x.strip()],
             "current_agent_deck": [int(x) for x in open(CUR) if x.strip()]}
    deck_sha = {"dragapult_newlib": sha(DRAG), "current_agent_deck": sha(CUR)}

    cells = [
        ("base_vs_untrained", "dragapult_newlib", "untrained", base_net, BASE, base_step, base_cfg),
        ("base_selfmirror", "dragapult_newlib", "selfmirror", base_net, BASE, base_step, base_cfg),
        ("cand_vs_baseline_mirror", "current_agent_deck", "baseline", cand_net, CAND, cand_step, cand_cfg),
    ]
    runs = []
    for label, dname, kind, net, ck, step, cfg in cells:
        deck = decks[dname]
        print(f"[topup] {label} deck={dname} seed={S3} games={GAMES}", flush=True)
        r = h.run_one(kind, GAMES, S3, [deck], [deck], dev, ct, enc, 1, "mirror", net, ck)
        r.update({"cell": label, "deck_name": dname, "seed": S3, "ckpt_path": ck,
                  "ckpt_sha256": sha(ck), "ckpt_step": step, "net_config": cfg,
                  "deck_sha256": deck_sha[dname], "deck_source": dname})
        runs.append(r)
        print("     wr=%s ci95=%s draws=%d errs=%d illegal=%d"
              % (r["winrate"], r["winrate_ci95"], r["draws"], r["errs"],
                 r["illegal"]["total"]), flush=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"library": {"path": os.path.join(ROOT, "EN_Card_Data.csv"),
                                   "sha256": sha(os.path.join(ROOT, "EN_Card_Data.csv"))},
                       "protocol": {"harness": "runs/evaluation/eval_h2h.py run_one（与基线同协议）",
                                    "games_per_cell": GAMES, "seed": S3,
                                    "note": "补测 stage2_supplement 未达 120 局的三个臂"},
                       "runs": runs}, f, ensure_ascii=False, indent=2, default=str)
    print("[saved]", OUT, flush=True)


if __name__ == "__main__":
    main()
