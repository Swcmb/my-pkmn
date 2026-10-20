# 牌组构成核查：用官方引擎 AllCard 解析 agent/deck.csv
import collections, os, sys

PKG = r"C:\Users\Swcmb\AppData\Local\Temp\opencode\subsmoke2\pkg"
sys.path.insert(0, PKG)
from sdk_cg import api                                   # noqa: E402

cards = {c.cardId: c for c in api.all_card_data()}
deck = [int(x) for x in open(os.path.join(PKG, "deck.csv")).read().split()]
cnt = collections.Counter(deck)
print("total=%d unique=%d" % (len(deck), len(cnt)))
kinds = collections.Counter()
n_pkmn = 0
for cid, n in sorted(cnt.items()):
    c = cards.get(cid)
    if c is None:
        print(cid, n, "UNKNOWN CARD ID"); continue
    kinds[str(c.cardType)] += n
    extras = []
    if c.cardType == 0:
        n_pkmn += n
        extras.append("hp=%s basic=%s stage1=%s stage2=%s ex=%s" %
                      (c.hp, c.basic, c.stage1, c.stage2, c.ex))
        extras.append("attacks=%s" % (c.attacks,))
        extras.append("skills=%s" % [s.name for s in (c.skills or [])])
    print("%4d x%-3d %-28s type=%s %s" % (cid, n, (c.name or "")[:28], c.cardType,
                                          " ".join(str(x) for x in extras)))
print("\ncardType 分布:", dict(kinds), " (0=Pokemon 1=Item 2=Tool 3=Supporter 4=Stadium 5=BasicEnergy 6=SpecialEnergy)")
print("Pokemon 张数 =", n_pkmn)
