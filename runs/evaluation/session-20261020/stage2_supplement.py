# -*- coding: utf-8 -*-
"""阶段 2 基线补测：把样本不足的 cell 提到 >=120 局，并补上原矩阵缺失的两个对照。

与 stage5_gates.py 完全同一协议（同一 eval_h2h.run_one、同一卡池、同一编码路径、
PKMN_TURN_CAP=200、live=1、座位逐局交替），只使用**新种子**，因此与既有 840 局相互独立、
可直接合并计数。输出写在本会话目录，不覆盖 stage5_gates.json。

补充的两个对照：
  cand_vs_baseline_mirror —— 两座位同用提交牌组，纯 net-vs-net，是"候选是否真的更强"最干净的检验；
                              原矩阵只有 held-out 池的 cross 模式。
  base_vs_untrained / base_selfmirror —— 旧基线自身的有效性 sanity。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys

ROOT = r"D:\pkmn-kaggle"
sys.path.insert(0, ROOT)

CAND = os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_999424.pt")
BASE = os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_500736.pt")
DRAG = os.path.join(ROOT, "runs", "evaluation", "deck_search", "dragapult_deck.csv")
CUR = os.path.join(ROOT, "agent", "deck.csv")
LIB = os.path.join(ROOT, "EN_Card_Data.csv")
OUT = os.path.join(ROOT, "runs", "evaluation", "session-20261020", "stage2_supplement.json")
GAMES = int(os.environ.get("SUPP_GAMES", "60"))
S1, S2 = 20261022, 20261023          # 与既有 20261020/20261021 不重叠


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def read_deck(p):
    return [int(x) for x in open(p) if x.strip()]


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
    ct = get_card_table(LIB)
    enc = TokenEncoder(ct)
    cand_net, cand_step, cand_cfg = h.load_net(CAND, ct, dev)
    base_net, base_step, base_cfg = h.load_net(BASE, ct, dev)

    decks = {"dragapult_newlib": read_deck(DRAG), "current_agent_deck": read_deck(CUR)}
    deck_sha = {"dragapult_newlib": sha(DRAG), "current_agent_deck": sha(CUR)}

    # (label, deck_name, seed, kind, mode, which_net)
    cells = [
        ("cand_vs_untrained",        "dragapult_newlib",   S1, "untrained",  "mirror", "cand"),
        ("cand_vs_untrained",        "dragapult_newlib",   S2, "untrained",  "mirror", "cand"),
        ("cand_vs_untrained",        "current_agent_deck", S1, "untrained",  "mirror", "cand"),
        ("base_vs_random",           "dragapult_newlib",   S1, "random",     "mirror", "base"),
        ("base_vs_random",           "current_agent_deck", S1, "random",     "mirror", "base"),
        ("base_selfmirror",          "dragapult_newlib",   S1, "selfmirror", "mirror", "base"),
        ("base_vs_untrained",        "dragapult_newlib",   S1, "untrained",  "mirror", "base"),
        ("cand_vs_baseline_mirror",  "dragapult_newlib",   S1, "baseline",   "mirror", "cand"),
        ("cand_vs_baseline_mirror",  "dragapult_newlib",   S2, "baseline",   "mirror", "cand"),
        ("cand_vs_baseline_mirror",  "current_agent_deck", S1, "baseline",   "mirror", "cand"),
    ]

    runs = []
    for label, dname, seed, kind, mode, who in cells:
        deck = decks[dname]
        net, ck, step = (cand_net, CAND, cand_step) if who == "cand" else (base_net, BASE, base_step)
        print(f"[supp] {label} deck={dname} seed={seed} games={GAMES}", flush=True)
        r = h.run_one(kind, GAMES, seed, [deck], [deck], dev, ct, enc, 1, mode, net, ck)
        r.update({"cell": label, "deck_name": dname, "seed": seed,
                  "ckpt_path": ck, "ckpt_sha256": sha(ck), "ckpt_step": step,
                  "net_config": cand_cfg if who == "cand" else base_cfg,
                  "deck_sha256": deck_sha[dname], "deck_source": dname})
        runs.append(r)
        print("     wr=%s ci95=%s draws=%d errs=%d illegal=%d len=%.1f %.2f g/s"
              % (r["winrate"], r["winrate_ci95"], r["draws"], r["errs"],
                 r["illegal"]["total"], r["length"]["mean_applies"], r["games_per_s"]), flush=True)
        with open(OUT, "w", encoding="utf-8") as f:      # 增量落盘，中途中断也保留已完成 cell
            json.dump(out_obj(runs), f, ensure_ascii=False, indent=2, default=str)

    print("[saved]", OUT, flush=True)


def out_obj(runs):
    return {"library": {"path": LIB, "sha256": sha(LIB)},
            "engine": {"kaggle_environments": "1.33.0", "shim": os.path.join(ROOT, ".engine-shim"),
                       "would_ko": True, "native_encode": False,
                       "encoding_path": "python TokenEncoder + 引擎 would_ko 标注"},
            "env": {"python": sys.version.split()[0], "device": "cuda",
                    "turn_cap": os.environ.get("PKMN_TURN_CAP"), "live": 1,
                    "platform": platform.platform()},
            "protocol": {"harness": "runs/evaluation/eval_h2h.py run_one（与 stage5_gates 同协议）",
                         "seat_balance": "cs = gi % 2 逐局交替", "games_per_cell": GAMES,
                         "seeds_new_independent": [S1, S2],
                         "note": "种子与 stage5_gates.json 的 20261020/20261021 不重叠，可合并计数"},
            "runs": runs}


if __name__ == "__main__":
    main()
