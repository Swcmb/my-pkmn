"""从 rl/decks_train.py 导出对手牌组池（JSON + 权重 npy）。

池来源为本仓唯一真实牌组常量：
  TRAIN_TOP15 -> pool_top15.json   (30 副: k01-k15 + _AICHI + META2)
  TRAIN_TOP50 -> pool_top50.json   (74 副)
权重为均匀权重（h2h_ckpt.py 读入后会自行归一化）。
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from rl.decks_train import TRAIN_TOP15, TRAIN_TOP50  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


def dump(name, pool):
    names = list(pool.keys())
    decks = [list(pool[n]) for n in names]
    bad = [(n, len(d)) for n, d in zip(names, decks) if len(d) != 60]
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
        json.dump({"decks": decks, "counts": [[0] for _ in decks], "names": names}, f)
    np.save(os.path.join(OUT, name + "_w.npy"), np.ones(len(decks), dtype=np.float64))
    print(f"[pool] {name}: {len(decks)} decks, bad_len={bad}")
    return bad


if __name__ == "__main__":
    b1 = dump("pool_top15", TRAIN_TOP15)
    b2 = dump("pool_top50", TRAIN_TOP50)
    sys.exit(1 if (b1 or b2) else 0)
