# -*- coding: utf-8 -*-
"""方案 A 适配器：引擎格式 cards.csv → 项目权威库 EN_Card_Data.csv 的旧列契约。

规则（每处偏离都写在这里）：
* 以新库为权威重建：Card Name、Stage/Rule/Previous stage、HP、Type、Weakness、
  Resistance、Retreat、Move Name/Cost/Damage（招式逐条展开，一招式一行）。
* 新 schema 无法表达的列按稳定 ID 从前库继承：Expansion、Collection No.、
  Category（Ancient/Future/Tera(...)/Trainer's 等版本标签）。
* RAINBOW / TEAM_ROCKET 这类“一张卡提供多种能量”的旧式写法（`{A}`、
  `{Team Rocket}{Team Rocket}`）新库退化成单枚举，会改变既有 one-hot 语义，
  因此这两个枚举沿用前库 Type 字符串。
* 伤害取新库（与引擎 all_attack 基值一致）；旧库的 `20×` 变动写法不再保留。
* 无招式的卡输出单行 n/a，与旧库惯例一致。

只写迁移目录，不碰仓库权威库；--apply 才做原子替换（同卷临时文件 + os.replace）。
"""
from __future__ import annotations

import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OLD = os.path.join(HERE, "original_cards.csv")
NEW = os.path.join(HERE, "downloaded_cards.csv")
OUT = os.path.join(HERE, "EN_Card_Data.migrated.csv")
TARGET = os.path.join(ROOT, "EN_Card_Data.csv")

COLUMNS = ["Card ID", "Card Name", "Expansion", "Collection No.",
           "Stage (Pokémon)/Type (Energy and Trainer)", "Rule", "Category",
           "Previous stage", "HP", "Type", "Weakness", "Resistance (Type)",
           "Retreat", "Move Name", "Cost", "Damage", "Effect Explanation"]

# 引擎 EnergyType 枚举名 → 旧库括号符号
ENUM2SYM = {"COLORLESS": "C", "GRASS": "G", "FIRE": "R", "WATER": "W",
            "LIGHTNING": "L", "PSYCHIC": "P", "FIGHTING": "F", "DARKNESS": "D",
            "METAL": "M", "DRAGON": "A"}
# 保留旧写法的枚举（单枚举无法表达多能量）
INHERIT_TYPE = {"RAINBOW", "TEAM_ROCKET"}
# cardType 名称 → 旧 Stage 列取值
CARDTYPE2STAGE = {"POKEMON": None, "ITEM": "Item", "TOOL": "Pokémon Tool",
                  "SUPPORTER": "Supporter", "STADIUM": "Stadium",
                  "BASIC_ENERGY": "Basic Energy", "SPECIAL_ENERGY": "Special Energy"}
ATT_RE = re.compile(r"(\d+):(.+?)\((-?\d+)\)\[([0-9,\s]*)\]")


def braced(idx_str):
    """能量枚举名 → '{X}'；COLORLESS 在费用里写作 ●。"""
    sym = ENUM2SYM.get((idx_str or "").strip())
    return "{" + sym + "}" if sym else ""


def cost_str(energy_names):
    """新库 energies 是枚举索引，转旧式费用串：{F}● 形态。"""
    out = []
    for n in energy_names:
        if n == "COLORLESS":
            out.append("●")
        else:
            s = ENUM2SYM.get(n)
            if s:
                out.append("{" + s + "}")
    return "".join(out)


ENUM_IDX2NAME = {0: "COLORLESS", 1: "GRASS", 2: "FIRE", 3: "WATER", 4: "LIGHTNING",
                 5: "PSYCHIC", 6: "FIGHTING", 7: "DARKNESS", 8: "METAL", 9: "DRAGON",
                 10: "RAINBOW", 11: "TEAM_ROCKET"}


def main(apply=False):
    # 前库：按 ID 取表头行（继承 Expansion/Collection No./Category/Type 特殊写法）
    inherited = {}
    with open(OLD, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            cid = int(re.search(r"\d+", r["Card ID"]).group())
            if cid not in inherited:
                inherited[cid] = r

    with open(NEW, encoding="utf-8-sig", newline="") as f:
        new_rows = list(csv.DictReader(f))

    out_rows = []
    for r in new_rows:
        cid = int(r["cardId"])
        inh = inherited.get(cid, {})
        card_type = (r.get("cardTypeName") or "").strip()
        flags = {k: (r.get(k) or "0").strip() == "1"
                 for k in ("basic", "stage1", "stage2", "ex", "megaEx", "tera", "aceSpec")}
        if card_type == "POKEMON":
            stage = ("Basic Pokémon" if flags["basic"] else
                     "Stage 1 Pokémon" if flags["stage1"] else
                     "Stage 2 Pokémon" if flags["stage2"] else "Basic Pokémon")
        else:
            stage = CARDTYPE2STAGE.get(card_type, "Item")
        rule = ("ACE SPEC" if flags["aceSpec"] else
                "Mega Pokémon ex" if flags["megaEx"] else
                "Pokémon ex" if flags["ex"] else "n/a")
        prev = (r.get("evolvesFrom") or "").strip() or "n/a"
        etype = (r.get("energyType") or "").strip()
        if etype in INHERIT_TYPE and inh:
            ftype = (inh.get("Type") or "n/a").strip() or "n/a"
        else:
            ftype = braced(etype) or "n/a"
        weak = ENUM_IDX2NAME.get(int(r["weakness"])) if (r.get("weakness") or "").strip().isdigit() else None
        resp = ENUM_IDX2NAME.get(int(r["resistance"])) if (r.get("resistance") or "").strip().isdigit() else None
        hp = str(int(r["hp"] or 0))
        retreat = str(int(r["retreatCost"] or 0))
        category = (inh.get("Category") or "n/a").strip() or "n/a"
        base = {
            "Card ID": str(cid), "Card Name": r["name"],
            "Expansion": (inh.get("Expansion") or "").strip(),
            "Collection No.": (inh.get("Collection No.") or "").strip(),
            "Stage (Pokémon)/Type (Energy and Trainer)": stage, "Rule": rule,
            "Category": category, "Previous stage": prev, "HP": hp, "Type": ftype,
            "Weakness": (braced(weak) or "n/a"), "Resistance (Type)": (braced(resp) or "n/a"),
            "Retreat": retreat,
        }
        atks = list(ATT_RE.finditer(r.get("attacks") or ""))
        if not atks:
            out_rows.append({**base, "Move Name": "n/a", "Cost": "n/a",
                             "Damage": "n/a", "Effect Explanation": ""})
            continue
        for m in atks:
            names = [ENUM_IDX2NAME.get(int(x)) for x in re.findall(r"\d+", m.group(4))]
            energies = [n for n in names if n]
            out_rows.append({**base, "Move Name": m.group(2).strip(),
                             "Cost": cost_str(energies) or "n/a",
                             "Damage": m.group(3), "Effect Explanation": ""})

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"写出 {OUT}: {len(out_rows)} 行 / {len(new_rows)} 张卡")

    # ---- 校验：项目读取入口必须能吃下迁移库 ----
    sys.path.insert(0, ROOT)
    from rl.card_features import CardTable
    old_t, new_t = CardTable(OLD), CardTable(OUT)
    import numpy as np
    assert old_t.vocab_size == new_t.vocab_size, (old_t.vocab_size, new_t.vocab_size)
    assert old_t.feat_dim == new_t.feat_dim, (old_t.feat_dim, new_t.feat_dim)
    d = np.abs(new_t.matrix - old_t.matrix)
    changed = np.flatnonzero(d.max(axis=1) > 1e-6)
    print(f"vocab_size={new_t.vocab_size} feat_dim={new_t.feat_dim}（与原库一致）")
    print(f"特征向量发生变化的卡数: {len(changed)} / {new_t.vocab_size}"
          f"，单维最大绝对变化 {d.max():.4f}")
    names_ok = sum(1 for cid in old_t.names if old_t.names[cid] == new_t.names.get(cid))
    print(f"ID→卡名保持: {names_ok}/{len(old_t.names)}")
    for cid in changed[:8]:
        print("  变化示例", cid, new_t.names.get(cid),
              "old:", old_t.matrix[cid].round(3).tolist()[:12],
              "new:", new_t.matrix[cid].round(3).tolist()[:12])

    # ---- 牌组引用完整性 ----
    for deck_path in (os.path.join(ROOT, "agent", "deck.csv"),
                      os.path.join(ROOT, "runs", "evaluation", "deck_search", "dragapult_deck.csv")):
        if not os.path.exists(deck_path):
            continue
        ids = [int(x.strip()) for x in open(deck_path) if x.strip()]
        missing = [i for i in ids if i >= new_t.vocab_size or new_t.names.get(i) is None]
        print(f"{os.path.relpath(deck_path, ROOT)}: {len(ids)} 张, 库内缺失 {len(missing)} {missing[:5]}")

    if apply:
        tmp = TARGET + ".migrating"
        with open(tmp, "wb") as f:
            f.write(open(OUT, "rb").read())
        os.replace(tmp, TARGET)
        print("已原子替换权威库:", TARGET)
    else:
        print("（未覆盖权威库；加 --apply 执行原子替换）")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
