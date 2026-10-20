# -*- coding: utf-8 -*-
"""容量价值实测：未训练网络 / 小配置网络 对随机合法策略的胜率。

为什么现在要测：平台请求体上限实测为 1 MiB，1 MiB 内最大可容纳的网络是
transformer2 d_model=64/nhead=4/nlayers=3/ff=128（384,067 参数，估算包体 889 KB）；
现方案 20,303,715 参数放不进去。要不要为此蒸馏、以及蒸馏到什么程度，
取决于"训练与容量到底换来多少胜率"。

三格对照，全部同协议（run_one、turn_cap 200、座位逐局交替、提交牌组、Python 编码）：
  untrained_544_vs_random : 现架构随机初始化    —— 训练带来的增益上界
  small_untrained_vs_random: 小架构随机初始化   —— 小网不作弊时的起点
  random_vs_random        : 双随机合法         —— 该口径的地板（应约 0.5）
候选座位固定驾驶 best_deck.csv。种子用 20261050/51，与既有基线不重叠。
"""
import hashlib
import importlib.util
import json
import os
import sys

ROOT = r"D:\pkmn-kaggle"
sys.path.insert(0, ROOT)

DECK = os.path.join(ROOT, "runs", "evaluation", "card_database_migration", "best_deck.csv")
LIB = os.path.join(ROOT, "EN_Card_Data.csv")
BIG = os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_999424.pt")
OUT = os.path.join(ROOT, "runs", "evaluation", "session-20261020", "capacity_probe.json")
GAMES = int(os.environ.get("CAP_GAMES", "60"))
SEEDS = [20261050, 20261051]
SMALL = {"arch": "transformer2", "d_model": 64, "nhead": 4, "nlayers": 3, "ff": 128,
         "static": False, "would_ko": True, "split_heads": False,
         "value_categorical": False, "value_atoms": 51, "value_vmax": 1.0}


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
    from rl.policy import build_token_net
    torch.set_grad_enabled(False)
    h = load_h2h()
    h._install_sdk_cg_shim_alias()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ct = get_card_table(LIB)
    enc = TokenEncoder(ct)
    _c, big_step, big_cfg = h.load_net(BIG, ct, dev)
    deck = [int(x) for x in open(DECK) if x.strip()]

    def untrained(cfg, seed):
        torch.manual_seed(seed)
        n = build_token_net(ct, cfg)
        return n.eval().to(dev)

    runs = []
    arms = [
        ("untrained_544_vs_random", big_cfg),
        ("small_untrained_vs_random", SMALL),
    ]
    for label, cfg in arms:
        for sd in SEEDS:
            net = untrained(cfg, sd)
            print(f"[cap] {label} seed={sd} games={GAMES}", flush=True)
            r = h.run_one("random", GAMES, sd, [deck], [deck], dev, ct, enc, 1, "mirror", net, BIG)
            r["cell"] = label
            r["deck_sha256"] = hashlib.sha256(open(DECK, "rb").read()).hexdigest()
            r["candidate"] = f"untrained {cfg['arch']} d{cfg['d_model']}/h{cfg['nhead']}/L{cfg['nlayers']}"
            r["n_params"] = sum(p.numel() for p in untrained(cfg, sd).parameters())
            runs.append(r)
            print("     wr=%s ci95=%s errs=%d illegal=%d"
                  % (r["winrate"], r["winrate_ci95"], r["errs"], r["illegal"]["total"]), flush=True)
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump({"library_sha256": hashlib.sha256(open(LIB, "rb").read()).hexdigest(),
                           "deck": DECK, "turn_cap": os.environ.get("PKMN_TURN_CAP"),
                           "games_per_cell": GAMES, "seeds": SEEDS,
                           "small_target_cfg": SMALL,
                           "big_step": big_step,
                           "note": "候选座位固定驾驶提交牌组；对手一律随机合法策略。"
                                   "用于判断 1 MiB 上限下容量取舍",
                           "runs": runs}, f, ensure_ascii=False, indent=2, default=str)
    print("[saved]", OUT, flush=True)


if __name__ == "__main__":
    main()
