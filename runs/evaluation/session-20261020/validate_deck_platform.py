# -*- coding: utf-8 -*-
"""以平台权威卡表（GET /api/cards/csv）复核牌组合法性，并与本地 EN_Card_Data.csv 的判定交叉比对。

d.txt §一 说明卡池"已由用户确认一致"，但阶段 1 要求"所有 ID 均存在于比赛卡牌库"，
判合法必须以平台字段（cardType / basic / aceSpec / name）为准；本地 CSV 是另一套 schema
（`Stage (Pokémon)/Type…` / `Rule`），两者是否给出同一结论需要独立验证，而不是假设。
"""
import csv
import hashlib
import io
import sys
from collections import Counter

PLATFORM = "runs/evaluation/session-20261020/platform_cards.csv"
LOCAL = "EN_Card_Data.csv"


def load_platform(path=PLATFORM):
    lib = {}
    for r in csv.DictReader(io.open(path, encoding="utf-8-sig")):
        cid = int(r["cardId"])
        lib[cid] = {"name": r["name"].strip(), "cardType": int(r["cardType"]),
                    "basic": int(r["basic"]), "aceSpec": int(r["aceSpec"])}
    return lib


def load_local(path=LOCAL):
    """本地英文卡表：一卡多招式行，字段是文本描述，需要映射到规则语义。"""
    lib = {}
    for r in csv.DictReader(io.open(path, encoding="utf-8-sig")):
        cid = int(r["Card ID"])
        lib.setdefault(cid, {"name": r["Card Name"].strip(),
                             "stage": r["Stage (Pokémon)/Type (Energy and Trainer)"].strip(),
                             "rule": (r["Rule"] or "").strip()})
    return lib


def validate_platform(deck_path, lib):
    ids = [int(x) for row in csv.reader(io.open(deck_path, encoding="utf-8-sig")) for x in row if x.strip()]
    res = {"deck": deck_path, "n": len(ids), "checks": []}

    def chk(rule, ok, detail):
        res["checks"].append({"rule": rule, "pass": bool(ok), "detail": detail})

    chk("恰好60张", len(ids) == 60, f"{len(ids)} 行")
    unknown = sorted({i for i in ids if i not in lib})
    chk("ID存在于平台卡池", not unknown, f"未知={unknown}" if unknown else f"{len(set(ids))} 个唯一 ID 全命中")

    basics = {i for i in ids if lib.get(i, {}).get("cardType") == 0 and lib.get(i, {}).get("basic") == 1}
    n_basic_cards = sum(c for i, c in Counter(ids).items() if i in basics)
    fossil_like = sorted({i for i in ids if lib.get(i, {}).get("cardType") == 1
                          and "Fossil" in lib.get(i, {}).get("name", "")})
    chk("cardType=0且basic=1的基础宝可梦", n_basic_cards >= 1,
        f"{n_basic_cards} 张 / {len(basics)} 种" + (f"；化石道具 {fossil_like}" if fossil_like else ""))

    cnt = Counter(lib[i]["name"] for i in ids if i in lib)
    exempt = {lib[i]["name"] for i in ids if lib.get(i, {}).get("cardType") == 5}
    over = {n: c for n, c in cnt.items() if c > 4 and n not in exempt}
    chk("同名≤4（cardType=5豁免）", not over, f"超限={over}" if over else f"最大同名={max(cnt.values())}")

    ace = sum(1 for i in ids if lib.get(i, {}).get("aceSpec") == 1)
    chk("aceSpec合计≤1", ace <= 1, f"{ace} 张")

    res["pass"] = all(c["pass"] for c in res["checks"])
    return res


def cross_check(plat, local):
    """两库对同一牌组是否给出同样的合法性判定；并列出字段级分歧。"""
    diffs = []
    for cid, p in plat.items():
        l = local.get(cid)
        if l is None:
            diffs.append({"cardId": cid, "kind": "本地缺此 ID", "platform": p})
            continue
        if p["name"] != l["name"]:
            diffs.append({"cardId": cid, "kind": "name 不一致", "platform": p["name"], "local": l["name"]})
        p_basic = (p["cardType"] == 0 and p["basic"] == 1)
        l_basic = (l["stage"] == "Basic Pokémon")
        if p_basic != l_basic:
            diffs.append({"cardId": cid, "kind": "基础宝可梦判定不一致",
                          "platform_basic": p_basic, "local_basic": l_basic, "name": p["name"]})
        p_ace = (p["aceSpec"] == 1)
        l_ace = (l["rule"] == "ACE SPEC")
        if p_ace != l_ace:
            diffs.append({"cardId": cid, "kind": "ACE SPEC 判定不一致",
                          "platform": p_ace, "local": l_ace, "name": p["name"]})
    only_local = sorted(set(local) - set(plat))
    return {"field_diffs": diffs[:60], "n_diffs": len(diffs),
            "ids_only_in_local": only_local[:20], "n_only_local": len(only_local),
            "n_only_platform": len(set(plat) - set(local))}


if __name__ == "__main__":
    plat = load_platform()
    local = load_local()
    b = hashlib.sha256(open(PLATFORM, "rb").read()).hexdigest()
    print(f"平台卡池 {len(plat)} 张  SHA-256 {b}")
    print(f"本地卡池 {len(local)} 张  SHA-256 "
          f"{hashlib.sha256(open(LOCAL,'rb').read()).hexdigest()[:16]}…\n")

    for p in sys.argv[1:] or ["runs/evaluation/card_database_migration/best_deck.csv",
                              "agent/deck.csv",
                              "submission_rl_transformer2_db_migrated/deck.csv"]:
        r = validate_platform(p, plat)
        print(f"### {p} -> {'PASS' if r['pass'] else 'FAIL'}")
        for c in r["checks"]:
            print(f"  [{'x' if c['pass'] else '!'}] {c['rule']}: {c['detail']}")
        print()

    cc = cross_check(plat, local)
    print("### 两库字段级交叉比对")
    print(f"分歧条数 {cc['n_diffs']}（前 60 条列出）| 仅本地有的 ID {cc['n_only_local']} | "
          f"仅平台有的 ID {cc['n_only_platform']}")
    for d in cc["field_diffs"]:
        print("  ", d)
