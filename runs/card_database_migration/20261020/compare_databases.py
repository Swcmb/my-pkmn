# -*- coding: utf-8 -*-
"""卡牌库迁移对比：旧 EN_Card_Data.csv（赛事导出，一招式一行）
vs 新 cards.csv（引擎 all_card_data 导出，一卡一行）。

输出：
  schema_comparison.json      字段级 schema 与逐卡字段差异
  id_mapping_comparison.json  ID 集合映射与名称一致性
不修改任何输入文件，只读。
"""
import csv
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "original_cards.csv")
NEW = os.path.join(HERE, "downloaded_cards.csv")

# 引擎 EnergyType 枚举（sdk_cg/api.py）
ENERGY_ENUM = {0: "COLORLESS", 1: "GRASS", 2: "FIRE", 3: "WATER", 4: "LIGHTNING",
               5: "PSYCHIC", 6: "FIGHTING", 7: "DARKNESS", 8: "METAL", 9: "DRAGON",
               10: "RAINBOW", 11: "TEAM_ROCKET"}
SYM2ID = {"C": 0, "G": 1, "R": 2, "W": 3, "L": 4, "P": 5, "F": 6, "D": 7, "M": 8, "A": 9}


def parse_old(path):
    with open(path, encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        header = list(rdr.fieldnames or [])
        groups = defaultdict(list)
        for r in rdr:
            m = re.search(r"\d+", r["Card ID"] or "")
            if m:
                groups[int(m.group())].append(r)
    cards = {}
    for cid, rows in groups.items():
        head = rows[0]
        attacks = []
        for r in rows:
            mv = (r.get("Move Name") or "").strip()
            if mv and mv != "n/a":
                attacks.append({
                    "name": mv,
                    "cost": (r.get("Cost") or "").strip(),
                    "damage": (r.get("Damage") or "").strip(),
                })
        cards[cid] = {
            "name": (head.get("Card Name") or "").strip(),
            "stage": (head.get("Stage (Pokémon)/Type (Energy and Trainer)") or "").strip(),
            "rule": (head.get("Rule") or "").strip(),
            "category": (head.get("Category") or "").strip(),
            "prev": (head.get("Previous stage") or "").strip(),
            "hp": (head.get("HP") or "").strip(),
            "type": (head.get("Type") or "").strip(),
            "weakness": (head.get("Weakness") or "").strip(),
            "resistance": (head.get("Resistance (Type)") or "").strip(),
            "retreat": (head.get("Retreat") or "").strip(),
            "attacks": attacks,
        }
    return cards, header


ATT_RE = re.compile(r"(\d+):(.+?)\((-?\d+)\)\[([0-9,\s]*)\]")


def parse_new(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        header = list(rdr.fieldnames or [])
        raw = list(rdr)
    cards, dup, bad = {}, [], []
    for r in raw:
        cid_s = (r.get("cardId") or "").strip()
        if not re.fullmatch(r"\d+", cid_s):
            bad.append({"cardId": cid_s, "name": r.get("name")})
            continue
        cid = int(cid_s)
        if cid in cards:
            dup.append(cid)
        atks = [{"attackId": int(m.group(1)), "name": m.group(2).strip(),
                 "damage": int(m.group(3)),
                 "energies": [int(x) for x in re.findall(r"\d+", m.group(4))]}
                for m in ATT_RE.finditer(r.get("attacks") or "")]
        cards[cid] = {
            "name": (r.get("name") or "").strip(),
            "cardType": (r.get("cardType") or "").strip(),
            "cardTypeName": (r.get("cardTypeName") or "").strip(),
            "pokemonType": (r.get("pokemonType") or "").strip(),
            "energyType": (r.get("energyType") or "").strip(),
            "evolutionType": (r.get("evolutionType") or "").strip(),
            "hp": (r.get("hp") or "").strip(),
            "retreatCost": (r.get("retreatCost") or "").strip(),
            "weakness": (r.get("weakness") or "").strip(),
            "resistance": (r.get("resistance") or "").strip(),
            "basic": (r.get("basic") or "").strip(),
            "stage1": (r.get("stage1") or "").strip(),
            "stage2": (r.get("stage2") or "").strip(),
            "ex": (r.get("ex") or "").strip(),
            "megaEx": (r.get("megaEx") or "").strip(),
            "tera": (r.get("tera") or "").strip(),
            "aceSpec": (r.get("aceSpec") or "").strip(),
            "evolvesFrom": (r.get("evolvesFrom") or "").strip(),
            "attacks": atks,
            "skills_raw": (r.get("skills") or "").strip(),
        }
    quality = {"row_count": len(raw), "unique_ids": len(cards),
               "duplicate_cardId": sorted(set(dup)), "unparseable_rows": bad[:20],
               "unparseable_count": len(bad)}
    return cards, header, quality


def old_type_syms(s):
    return sorted(re.findall(r"\{(.*?)\}", s or ""))


def compare_field(cid, ov, nv):
    """返回该卡在新旧库之间可对应字段的差异列表。"""
    out = []
    if ov["name"] != nv["name"]:
        out.append(("name", ov["name"], nv["name"]))
    ohp = re.search(r"\d+", ov["hp"])
    ohp = ohp.group() if ohp else ""
    if ohp != nv["hp"]:
        out.append(("hp", ov["hp"], nv["hp"]))
    # 属性：旧 {G} -> 新 energyType 名
    osym = old_type_syms(ov["type"])
    oname = ENERGY_ENUM.get(SYM2ID.get(osym[0], -1), "") if osym else ""
    if oname and oname != nv["energyType"]:
        out.append(("energyType", ov["type"], nv["energyType"]))
    # 阶段
    ostage = ov["stage"]
    nstage = ("Basic" if nv["basic"] == "1" else "Stage1" if nv["stage1"] == "1"
              else "Stage2" if nv["stage2"] == "1" else "")
    if "Pokémon" in ostage and nstage:
        key = ostage.split()[0].replace("Pokémon", "").strip() or "Basic"
        norm_o = {"Basic": "Basic", "Stage": None}.get(key)
        if key == "Basic" and nstage != "Basic":
            out.append(("stage", ostage, nstage))
    # 进化前
    oprev = "" if ov["prev"] in ("", "n/a") else ov["prev"]
    if oprev != nv["evolvesFrom"]:
        out.append(("evolvesFrom", ov["prev"], nv["evolvesFrom"]))
    # 招式数量
    if len(ov["attacks"]) != len(nv["attacks"]):
        out.append(("attack_count", len(ov["attacks"]), len(nv["attacks"])))
    else:
        for a, b in zip(ov["attacks"], nv["attacks"]):
            if a["name"] != b["name"]:
                out.append(("attack_name", a["name"], b["name"]))
            od = re.search(r"\d+", a["damage"])
            od = od.group() if od else ""
            if od and od != str(b["damage"]):
                out.append(("attack_damage", a["damage"], b["damage"]))
    return out


def main():
    old, old_hdr = parse_old(OLD)
    new, new_hdr, quality = parse_new(NEW)

    old_ids, new_ids = set(old), set(new)
    shared = sorted(old_ids & new_ids)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)

    field_change_counter = defaultdict(int)
    name_changed, per_card_diff = [], {}
    for cid in shared:
        d = compare_field(cid, old[cid], new[cid])
        for f in d:
            field_change_counter[f[0]] += 1
        if d:
            name_changed.append(cid)  # 有任意字段差异的卡
            if len(per_card_diff) < 40:
                per_card_diff[str(cid)] = {"old_name": old[cid]["name"],
                                           "new_name": new[cid]["name"],
                                           "diffs": [list(map(str, x)) for x in d]}

    schema = {
        "old_file": "original_cards.csv (repo EN_Card_Data.csv)",
        "new_file": "downloaded_cards.csv (E:/Files/Downloads/cards.csv)",
        "old_header": old_hdr,
        "new_header": new_hdr,
        "new_fields_absent_in_old": sorted(set(new_hdr) - set(old_hdr)),
        "old_fields_absent_in_new": sorted(set(old_hdr) - set(new_hdr)),
        "row_layout": {"old": "one row per attack, grouped by Card ID",
                       "new": "one row per card, attacks/skills packed in columns"},
        "new_quality": quality,
        "counts": {"old_unique_cards": len(old), "new_unique_cards": len(new),
                   "shared_ids": len(shared), "added_ids": len(added),
                   "removed_ids": len(removed)},
        "field_change_counts": dict(field_change_counter),
        "cards_with_any_diff": len(name_changed),
        "sample_diffs": per_card_diff,
        "schema_compatible_with_python_readers": {
            "rl/card_features.py": False, "rl/decks.py": False,
            "note": "readers expect the old column names (Card ID / Card Name / ...);"
                    " the new engine-format header lacks them",
        },
    }

    idmap = {
        "id_range_old": [min(old_ids), max(old_ids)] if old_ids else None,
        "id_range_new": [min(new_ids), max(new_ids)] if new_ids else None,
        "ids_equal_as_sets": old_ids == new_ids,
        "added_ids": added[:50], "added_count": len(added),
        "removed_ids": removed[:50], "removed_count": len(removed),
        "name_identical_for_shared_ids": sum(
            1 for c in shared if old[c]["name"] == new[c]["name"]),
        "shared_id_count": len(shared),
        "first_name_mismatches": [
            {"cardId": c, "old": old[c]["name"], "new": new[c]["name"]}
            for c in shared if old[c]["name"] != new[c]["name"]][:50],
    }

    with open(os.path.join(HERE, "schema_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    with open(os.path.join(HERE, "id_mapping_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(idmap, f, ensure_ascii=False, indent=2)

    print(json.dumps({"counts": schema["counts"],
                      "field_change_counts": schema["field_change_counts"],
                      "ids_equal_as_sets": idmap["ids_equal_as_sets"],
                      "name_identical": idmap["name_identical_for_shared_ids"],
                      "quality": quality["duplicate_cardId"] + quality["unparseable_rows"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
