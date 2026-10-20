# 牌组归属核查：agent/deck.csv 是否在训练池 / 评估池内；并生成只含 deck.csv 的池
import json, os, sys
sys.path.insert(0, r"D:\pkmn-kaggle")
from collections import Counter
from rl.decks_train import TRAIN_TOP15, TRAIN_TOP50           # noqa: E402

DECK = [int(x) for x in open(r"D:\pkmn-kaggle\agent\deck.csv").read().split()]
cnt = Counter(DECK)

pools = {"TRAIN_TOP15": [list(d) for d in TRAIN_TOP15],
         "TRAIN_TOP50": [list(d) for d in TRAIN_TOP50]}
for f in ("pool_top15.json", "pool_top50.json"):
    pools[f] = [list(d) for d in json.load(open(os.path.join(r"D:\pkmn-kaggle\runs\evaluation", f),
                                                encoding="utf-8"))["decks"]]
for name, ds in pools.items():
    exact = any(Counter(d) == cnt for d in ds)
    order = any(list(d) == DECK for d in ds)
    print("%-14s n=%-3d exact_multiset=%s exact_order=%s" % (name, len(ds), exact, order))

out = os.path.join(r"D:\pkmn-kaggle\runs\evaluation", "pool_deckcsv_only.json")
json.dump({"decks": [DECK], "counts": [60] * 1, "names": ["deck.csv"]},
          open(out, "w", encoding="utf-8"), ensure_ascii=False)
print("wrote", out)

# 最接近的池牌组（按重叠张数）
best = []
for d in pools["pool_top50.json"]:
    c = Counter(d)
    ov = sum(min(cnt[k], c[k]) for k in set(cnt) | set(c))
    best.append((ov, len(set(c) & set(cnt))))
best.sort(reverse=True)
print("pool_top50 中与 deck.csv 重叠最高的 5 个 (重叠张数, 共同种类):", best[:5])
