# -*- coding: utf-8 -*-
"""三方卡牌库对比：旧 EN_Card_Data.csv vs 新 cards.csv vs 本机引擎 all_card_data()。

目的：判断新 CSV 是引擎卡牌库的子集、更新版，还是另一份数据；
区分真实语义差异与编码表示差异（n/a vs 0 等）。
只读，不修改任何输入。
"""
import csv
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ENERGY_ENUM = {0: "COLORLESS", 1: "GRASS", 2: "FIRE", 3: "WATER", 4: "LIGHTNING",
               5: "PSYCHIC", 6: "FIGHTING", 7: "DARKNESS", 8: "METAL", 9: "DRAGON",
               10: "RAINBOW", 11: "TEAM_ROCKET"}
SYM2ID = {"C": 0, "G": 1, "R": 2, "W": 3, "L": 4, "P": 5, "F": 6, "D": 7, "M": 8, "A": 9}
ATT_RE = re.compile(r"(\d+):(.+?)\((-?\d+)\)\[([0-9,\s]*)\]")


def to_int(s, default=0):
    m = re.search(r"-?\d+", str(s if s is not None else ""))
    return int(m.group()) if m and str(s).strip() not in ("", "n/a") else default


def load_old():
    g = defaultdict(list)
    with open(os.path.join(HERE, "original_cards.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            m = re.search(r"\d+", r["Card ID"] or "")
            if m:
                g[int(m.group())].append(r)
    out = {}
    for cid, rows in g.items():
        h = rows[0]
        atk = []
        for r in rows:
            mv = (r.get("Move Name") or "").strip()
            if mv and mv != "n/a":
                atk.append({"name": mv, "damage": to_int(r.get("Damage"))})
        out[cid] = {
            "name": (h.get("Card Name") or "").strip(),
            "hp": to_int(h.get("HP")),
            "energy": (lambda s: ENERGY_ENUM.get(SYM2ID.get((re.findall(r"\{(.*?)\}", s or "") or [""])[0], -1), None))(h.get("Type")),
            "prev": "" if (h.get("Previous stage") or "").strip() in ("", "n/a") else h["Previous stage"].strip(),
            "retreat": to_int(h.get("Retreat")),
            "attacks": atk,
        }
    return out


def load_new():
    out = {}
    with open(os.path.join(HERE, "downloaded_cards.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            cid = int(r["cardId"])
            out[cid] = {
                "name": (r.get("name") or "").strip(),
                "hp": to_int(r.get("hp")),
                "energy": (r.get("energyType") or "").strip() or None,
                "prev": (r.get("evolvesFrom") or "").strip(),
                "retreat": to_int(r.get("retreatCost")),
                "cardTypeName": (r.get("cardTypeName") or "").strip(),
                "attacks": [{"name": m.group(2).strip(), "damage": int(m.group(3))}
                            for m in ATT_RE.finditer(r.get("attacks") or "")],
                "flags": {k: r.get(k) for k in ("basic", "stage1", "stage2", "ex", "megaEx", "tera", "aceSpec")},
            }
    return out


def load_engine():
    cards = json.load(open(os.path.join(HERE, "engine_all_card_data.json"), encoding="utf-8"))
    attacks = {a["attackId"]: a for a in json.load(open(os.path.join(HERE, "engine_all_attack.json"), encoding="utf-8"))}
    out = {}
    for c in cards:
        out[c["cardId"]] = {
            "name": c["name"],
            "hp": c["hp"],
            "energy": ENERGY_ENUM.get(c["energyType"], str(c["energyType"])),
            "prev": c["evolvesFrom"] or "",
            "retreat": c["retreatCost"],
            "cardTypeName": {0: "POKEMON", 1: "ITEM", 2: "TOOL", 3: "SUPPORTER", 4: "STADIUM",
                             5: "BASIC_ENERGY", 6: "SPECIAL_ENERGY"}.get(c["cardType"], str(c["cardType"])),
            "attacks": [{"name": attacks[a]["name"], "damage": attacks[a]["damage"]} for a in c["attacks"]],
            "flags": {k: ("1" if c[k] else "0") for k in ("basic", "stage1", "stage2", "ex", "megaEx", "tera", "aceSpec")},
        }
    return out, len(cards)


def cmp_pair(a, b, keys=("name", "hp", "energy", "prev", "retreat", "cardTypeName")):
    diffs = defaultdict(int)
    examples = []
    ids = sorted(set(a) & set(b))
    for cid in ids:
        for k in keys:
            va, vb = a[cid].get(k), b[cid].get(k)
            if va != vb:
                diffs[k] += 1
                if len(examples) < 20 and k == "name":
                    examples.append({"id": cid, "a": va, "b": vb})
        if a[cid]["attacks"] != b[cid]["attacks"]:
            diffs["attacks"] += 1
        if a[cid].get("flags") and b[cid].get("flags") and a[cid]["flags"] != b[cid]["flags"]:
            diffs["flags"] += 1
    return ids, dict(diffs), examples


def main():
    old, new = load_old(), load_new()
    eng, n_eng = load_engine()
    res = {"engine_card_count": n_eng, "old_count": len(old), "new_count": len(new)}

    # 1) 新 CSV vs 引擎
    ids, diffs, ex = cmp_pair(new, eng)
    res["new_vs_engine"] = {
        "shared_ids": len(ids),
        "only_in_engine": sorted(set(eng) - set(new))[:20],
        "only_in_engine_count": len(set(eng) - set(new)),
        "only_in_new": sorted(set(new) - set(eng))[:20],
        "only_in_new_count": len(set(new) - set(eng)),
        "diff_counts": diffs, "name_examples": ex,
    }
    # 2) 旧 CSV vs 引擎
    ids, diffs, ex = cmp_pair({k: {**v, "flags": None} for k, v in old.items()}, eng)
    res["old_vs_engine"] = {"shared_ids": len(ids), "only_in_engine_count": len(set(eng) - set(old)),
                            "only_in_old_count": len(set(old) - set(eng)), "diff_counts": diffs,
                            "name_examples": ex}
    # 3) 旧 vs 新（同 ID）
    ids, diffs, ex = cmp_pair(old, new)
    res["old_vs_new"] = {"shared_ids": len(ids), "added": len(set(new) - set(old)),
                         "removed": len(set(old) - set(new)), "diff_counts": diffs,
                         "name_examples": ex}
    # 4) 引擎中 1268+ 的卡是什么（新卡 or 重编号）
    tail = sorted(set(eng) - set(new))
    res["engine_extra_cards_sample"] = [{"id": i, "name": eng[i]["name"], "type": eng[i]["cardTypeName"]}
                                        for i in tail[:30]]
    res["engine_id_gap_check"] = {"engine_ids_contiguous_1_to_max": max(eng) == len(eng),
                                  "engine_max_id": max(eng), "new_max_id": max(new), "old_max_id": max(old)}

    print(json.dumps(res, ensure_ascii=False, indent=2)[:6000])
    with open(os.path.join(HERE, "engine_vs_csv_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
