# -*- coding: utf-8 -*-
"""分类 old→new 的字段差异：真实语义变化 vs 表示差异（n/a vs 0、列缺失、招式顺序）。
输出 migration_field_diff.md + field_diff_details.json
"""
import csv
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ATT_RE = re.compile(r"(\d+):(.+?)\((-?\d+)\)\[([0-9,\s]*)\]")
SYM = {"C": "COLORLESS", "G": "GRASS", "R": "FIRE", "W": "WATER", "L": "LIGHTNING",
       "P": "PSYCHIC", "F": "FIGHTING", "D": "DARKNESS", "M": "METAL", "A": "DRAGON"}


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
                atk.append((mv, (r.get("Cost") or "").strip(), (r.get("Damage") or "").strip()))
        out[cid] = {"head": h, "attacks": atk, "rows": len(rows)}
    return out


def load_new():
    out = {}
    with open(os.path.join(HERE, "downloaded_cards.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["cardId"])] = r
    return out


def main():
    old, new = load_old(), load_new()
    cat = defaultdict(list)
    for cid in sorted(set(old) & set(new)):
        o, n = old[cid], new[cid]
        oh, nh = o["head"], n

        # energy：旧 Type 首符号 vs 新 energyType
        os = re.findall(r"\{(.*?)\}", oh.get("Type") or "")
        oname = SYM.get(os[0]) if os else None
        ntype = (nh.get("energyType") or "").strip()
        obase = (oh.get("Stage (Pokémon)/Type (Energy and Trainer)") or "").strip()
        if oname != ntype:
            # 归类：旧列在训练家/能量卡上写 n/a 或空 → 表示差异
            if not os and ntype in ("COLORLESS", ""):
                cat["energy_repr_missing"].append((cid, oh.get("Type"), ntype, obase))
            else:
                cat["energy_REAL"].append((cid, oh.get("Name" if False else "Card Name"), oh.get("Type"), ntype))

        # HP
        ohpv = (oh.get("HP") or "").strip()
        nhpv = (nh.get("hp") or "").strip()
        on = re.search(r"\d+", ohpv)
        on = on.group() if on else ""
        if on != nhpv:
            if on == "" and nhpv == "0":
                cat["hp_repr_na_vs_0"].append(cid)
            else:
                cat["hp_REAL"].append((cid, oh.get("Card Name"), ohpv, nhpv))

        # retreat
        ort = (oh.get("Retreat") or "").strip()
        nrt = (nh.get("retreatCost") or "").strip()
        onr = re.search(r"\d+", ort)
        onr = onr.group() if onr else ""
        if onr != nrt:
            (cat["retreat_repr"] if (onr == "" and nrt == "0") else cat["retreat_REAL"]).append((cid, ort, nrt))

        # 进化前
        opv = (oh.get("Previous stage") or "").strip()
        opv_norm = "" if opv == "n/a" else opv
        npv = (nh.get("evolvesFrom") or "").strip()
        if opv_norm != npv:
            cat["evolvesFrom_REAL"].append((cid, oh.get("Card Name"), opv, npv))

        # 招式：旧逐行 (name, cost, damage) vs 新打包 id:name(dmg)[energies]
        old_names = [a[0] for a in o["attacks"]]
        new_atk = [(m.group(2).strip(), int(m.group(3))) for m in ATT_RE.finditer(nh.get("attacks") or "")]
        new_names = [a[0] for a in new_atk]
        if old_names != new_names:
            if sorted(old_names) == sorted(new_names):
                cat["attacks_order_only"].append(cid)
            elif len(old_names) != len(new_names):
                cat["attacks_count_REAL"].append((cid, oh.get("Card Name"), old_names, new_names))
            else:
                cat["attacks_name_set_REAL"].append((cid, oh.get("Card Name"), old_names, new_names))
        else:
            od = [a[2] for a in o["attacks"]]
            nd = [str(a[1]) for a in new_atk]
            od_norm = [re.search(r"-?\d+", d or "").group() if re.search(r"-?\d+", d or "") else "" for d in od]
            mismatch = [(a, b) for a, b in zip(od_norm, nd) if a != b]
            if mismatch:
                cat["attack_damage_REAL"].append((cid, oh.get("Card Name"), od, nd, mismatch))
        # 卡类别（旧 Stage 列 vs 新 cardTypeName）
        # 新库新增的布尔标记在旧库中由 Rule/Category 文本表达 → 单独统计
        rules = (oh.get("Rule") or "").strip()
        flags = "".join([("ex" if nh.get("ex") == "1" else ""), ("megaEx" if nh.get("megaEx") == "1" else ""),
                         ("ACE" if nh.get("aceSpec") == "1" else "")])
        cat.setdefault("rule_mapping", defaultdict(int))
        cat["rule_mapping"][f"rule={rules or '-'}|flags={flags or '-'}"] += 1

    summary = {k: (len(v) if isinstance(v, list) else dict(v)) for k, v in cat.items()
               if not isinstance(v, defaultdict)}
    summary = {k: v for k, v in summary.items()}
    detail_example_limits = {k: (v[:12] if isinstance(v, list) else dict(v)) for k, v in cat.items()
                             if not isinstance(v, defaultdict)}
    out = {"summary": summary, "examples": detail_example_limits,
           "rule_mapping": dict(cat["rule_mapping"])}
    print(json.dumps(out, ensure_ascii=False, indent=2)[:7000])
    with open(os.path.join(HERE, "field_diff_details.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
