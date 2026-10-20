# -*- coding: utf-8 -*-
"""把评测池 pool_top50.json 确定性地切成训练对手池(30) 与 held-out 评测池(44)。

动机：d.txt 阶段 3 要求"保证评测数据独立"。原 handoff 直接用 pool_top50.json 作
--ladder-decks-file，而同一文件又是 stage5 门禁的 held-out 交叉池 —— 微调后该池已被模型
见过，held-out 身份失效。这里按索引等距抽样分配（不依赖胜率排序），两侧都覆盖池的
强度分布，且切分可复现、可审计。
"""
import json
import os

ROOT = r"D:\pkmn-kaggle"
SRC = os.path.join(ROOT, "runs", "evaluation", "pool_top50.json")
OUT_TRAIN = os.path.join(ROOT, "runs", "evaluation", "session-20261020", "opp_pool_train30.json")
OUT_HELD = os.path.join(ROOT, "runs", "evaluation", "session-20261020", "eval_pool_heldout44.json")

d = json.load(open(SRC, encoding="utf-8"))
decks, counts, names = d["decks"], d["counts"], d.get("names", [])
N = len(decks)
assert all(len(x) == 60 for x in decks)

TRAIN_N = 30
# stride-7 置换（7 与 74 互质 -> 整池一个循环），取前 30 个作训练对手：
# 等距覆盖牌池顺序而非按强度排序，train/held 两侧都拿到跨池的均匀切片，切分可复现。
_perm = [(i * 7) % N for i in range(N)]
train_idx = sorted(_perm[:TRAIN_N])
held_idx = [i for i in range(N) if i not in set(train_idx)]


def dump(idx):
    return {"decks": [decks[i] for i in idx],
            "counts": [counts[i] for i in idx],
            "names": [names[i] if i < len(names) else f"pool#{i}" for i in idx],
            "source": SRC, "indices": idx,
            "note": "stride-7 等距切分，train/held 互斥且并集为原池"}


with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    json.dump(dump(train_idx), f)
with open(OUT_HELD, "w", encoding="utf-8") as f:
    json.dump(dump(held_idx), f)

import hashlib
for p in (SRC, OUT_TRAIN, OUT_HELD):
    print(os.path.basename(p), hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])
print(f"原池 {N} 副 -> 训练对手 {len(train_idx)} 副, held-out {len(held_idx)} 副")
print("train names:", [names[i] for i in train_idx][:6], "...")
print("held  names:", [names[i] for i in held_idx][:6], "...")
assert not (set(train_idx) & set(held_idx))
assert len(train_idx) + len(held_idx) == N
