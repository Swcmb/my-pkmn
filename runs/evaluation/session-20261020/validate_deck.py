"""牌组合法性独立校验（阶段 1）。

依据 AGENT.md §九 的比赛组牌规则，逐条复核 deck csv：
1. 恰好 60 张；2. 所有 cardId 在卡池内；3. 至少 1 张真正的基础宝可梦（化石道具卡排除）；
4. 按卡名统计同名 ≤4（基础能量豁免）；5. ACE SPEC 全牌组 ≤1。
"""
import csv
import sys
from collections import Counter

CSV = "EN_Card_Data.csv"
# AGENT.md §九 明确点名的化石卡：标记 basic=1 但 cardType=ITEM，引擎不视为可开局基础宝可梦
FOSSIL_ITEM_IDS = {1099, 1136, 1138, 1150, 1151}


def load_library(path=CSV):
    """cardId -> {name, stage, rule}。同一卡的多行（多个招式）合并为一条卡记录。"""
    lib = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        cid = int(r["Card ID"])
        rec = lib.setdefault(cid, {"name": r["Card Name"].strip(),
                                   "stage": r["Stage (Pokémon)/Type (Energy and Trainer)"].strip(),
                                   "rule": (r["Rule"] or "").strip()})
    return lib


def validate(deck_path, lib):
    ids = [int(x) for row in csv.reader(open(deck_path, encoding="utf-8-sig")) for x in row if x.strip()]
    out = {"deck": deck_path, "count": len(ids), "checks": []}

    def chk(name, ok, detail):
        out["checks"].append({"rule": name, "pass": bool(ok), "detail": detail})

    chk("恰好60张", len(ids) == 60, f"{len(ids)} 行")

    unknown = sorted({i for i in ids if i not in lib})
    chk("ID全部存在于卡池", not unknown, f"未知ID={unknown}" if unknown else f"全部 {len(set(ids))} 个唯一ID 命中")

    basics = {i for i in ids if lib.get(i, {}).get("stage") == "Basic Pokémon"
              and i not in FOSSIL_ITEM_IDS}
    fossils_in = sorted({i for i in ids if i in FOSSIL_ITEM_IDS})
    chk("至少1张真正基础宝可梦", bool(basics),
        f"基础宝可梦 {sum(c for i, c in Counter(ids).items() if i in basics)} 张 / {len(basics)} 种"
        + (f"；化石道具卡 {fossils_in}（不计入）" if fossils_in else ""))

    name_cnt = Counter(lib[i]["name"] for i in ids if i in lib)
    over = {n: c for n, c in name_cnt.items() if c > 4
            and lib_name_stage(n, lib) != "Basic Energy"}
    chk("同名≤4（基础能量豁免）", not over, f"超限={over}" if over else f"最大同名数={max(name_cnt.values()) if name_cnt else 0}")

    ace = sum(1 for i in ids if lib.get(i, {}).get("rule") == "ACE SPEC")
    chk("ACE SPEC≤1", ace <= 1, f"{ace} 张")

    out["pass"] = all(c["pass"] for c in out["checks"])
    return out


def lib_name_stage(name, lib):
    for rec in lib.values():
        if rec["name"] == name:
            return rec["stage"]
    return ""


if __name__ == "__main__":
    lib = load_library()
    print(f"卡池: {len(lib)} 张唯一卡  ({CSV})")
    for p in sys.argv[1:]:
        res = validate(p, lib)
        print(f"\n### {p} -> {'PASS' if res['pass'] else 'FAIL'}  ({res['count']} 张)")
        for c in res["checks"]:
            print(f"  [{'x' if c['pass'] else '!'}] {c['rule']}: {c['detail']}")
