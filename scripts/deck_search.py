"""生成并校验合法 60 张候选牌组（任务书第二阶段）。

来源：当前提交牌组（对照组）、rl.decks 官方默认牌组、现有 top15/top50 牌池，
以及围绕对照组的定向变体（宝可梦数量 / 能量比例 / 训练家支持）。
每副牌组先过卡牌规则，再过真实引擎 BattleStart；输出单牌组 pool JSON 供
runs/evaluation/eval_h2h.py 直接复用。

  PYTHONPATH=.;<kaggle_environments shim> python scripts/deck_search.py
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rl.decks import DECKS as OFFICIAL_DECKS  # noqa: E402


def load_cards(path):
    info = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        cid = (r.get("Card ID") or "").strip()
        if not cid.isdigit():
            continue
        info[int(cid)] = {
            "name": r["Card Name"].strip(),
            "stage": r["Stage (Pokémon)/Type (Energy and Trainer)"].strip(),
            "rule": (r.get("Rule") or "").strip(),
            "type": (r.get("Type") or "").strip(),
        }
    return info


def kind(info, cid):
    stage = info[cid]["stage"]
    if stage == "Basic Energy":
        return "basic_energy"
    if stage == "Special Energy":
        return "special_energy"
    if stage == "Basic Pokémon":
        return "basic_pokemon"
    if stage == "Pokémon Tool":
        return "tool"
    if "Pokémon" in stage:
        return "evolution_pokemon"
    return {
        "Item": "item", "Supporter": "supporter", "Stadium": "stadium",
    }.get(stage, "other")


def stats(deck, info):
    c = Counter(deck)
    out = Counter({k: 0 for k in (
        "basic_energy", "special_energy", "basic_pokemon", "evolution_pokemon",
        "item", "supporter", "tool", "stadium", "other", "ace_spec")})
    for cid, n in c.items():
        out[kind(info, cid)] += n
        if info[cid]["rule"] == "ACE SPEC":
            out["ace_spec"] += n
    out["pokemon"] = out["basic_pokemon"] + out["evolution_pokemon"]
    out["energy"] = out["basic_energy"] + out["special_energy"]
    out["trainers"] = out["item"] + out["supporter"] + out["tool"] + out["stadium"]
    out["same_basic_energy"] = max(
        (n for cid, n in c.items() if kind(info, cid) == "basic_energy"), default=0)
    out["unique"] = len(c)
    return dict(out)


def rule_errors(deck, info):
    errs = []
    if len(deck) != 60:
        errs.append(f"{len(deck)} cards != 60")
    unknown = [c for c in deck if c not in info]
    if unknown:
        errs.append(f"unknown card ids {sorted(set(unknown))[:5]}")
        return errs
    c = Counter(deck)
    for cid, n in c.items():
        if kind(info, cid) != "basic_energy" and n > 4:
            errs.append(f"{n} copies of {cid} ({info[cid]['name']})")
    ace = sum(n for cid, n in c.items() if info[cid]["rule"] == "ACE SPEC")
    if ace > 1:
        errs.append(f"{ace} ACE SPEC cards")
    if sum(n for cid, n in c.items() if kind(info, cid) == "basic_pokemon") < 1:
        errs.append("no Basic Pokémon")
    return errs


def engine_check(deck):
    import sdk_cg.game as game
    from sdk_cg.sim import Battle
    try:
        obs, sd = game.battle_start(deck, deck)
        ok = obs is not None
        if ok:
            game.battle_finish()
        else:
            Battle.battle_ptr = None
        return ok, int(sd.errorPlayer), int(sd.errorType), ""
    except Exception as e:                                    # noqa: BLE001
        return False, None, None, f"{type(e).__name__}: {e}"


def score(s):
    """仅用于排序预筛，不代替实测胜率。"""
    return (2 * min(s["basic_pokemon"], 14) + min(s["pokemon"], 24)
            + min(s["trainers"], 40) - abs(s["energy"] - 9)
            + (6 if 26 <= s["trainers"] <= 42 else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "EN_Card_Data.csv"))
    ap.add_argument("--control", default=os.path.join(ROOT, "agent", "deck.csv"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "runs", "evaluation", "deck_search"))
    ap.add_argument("--pool", default=os.path.join(ROOT, "runs", "evaluation", "pool_top15.json"))
    ap.add_argument("--pool-wide", default=os.path.join(ROOT, "runs", "evaluation", "pool_top50.json"))
    args = ap.parse_args()

    info = load_cards(args.csv)
    os.makedirs(os.path.join(args.out_dir, "pools"), exist_ok=True)

    control = [int(x) for x in open(args.control) if x.strip()]
    sources = [("control_deckcsv", "agent/deck.csv", control)]
    for name, deck in OFFICIAL_DECKS.items():
        sources.append((f"official_{name}", "rl/decks.py", list(deck)))

    seen = {tuple(sorted(Counter(d).items())) for _, _, d in sources}
    for pool_path in (args.pool, args.pool_wide):
        pj = json.load(open(pool_path, encoding="utf-8"))
        for i, deck in enumerate(pj["decks"]):
            key = tuple(sorted(Counter(deck).items()))
            if key in seen:
                continue
            seen.add(key)
            arch = pj["names"][i] if i < len(pj.get("names", [])) else f"d{i}"
            sources.append((f"pool_{arch}_{i:02d}", pool_path, list(deck)))

    # 对照组定向变体：补充基础宝可梦 / 降低能量 / 增加训练家。
    # 全部保持 Secret Box 为唯一 ACE SPEC，副本数不超过 4。
    def variant(n_energy_out, adds):
        out = list(control)
        for _ in range(n_energy_out):
            out.remove(3)
        for cid, n in adds:
            out.extend([cid] * n)
        assert len(out) == 60, (len(out), n_energy_out, adds)
        return out

    variants = {
        # 只补基础宝可梦：Kyogre 2->4 / Fezandipiti ex x2
        "control_pkm12_e31": variant(2, [(721, 2)]),
        "control_fez2_e31": variant(2, [(140, 2)]),
        "control_pkm14_e28": variant(5, [(721, 2), (140, 2), (1262, 1)]),
        # 补宝可梦 + 训练家支持，降低能量
        "control_pkm14_e24": variant(9, [(721, 2), (140, 2), (1121, 2), (1145, 2), (1262, 1)]),
        "control_pkm14_e20": variant(13, [(721, 2), (140, 2), (1121, 2), (1145, 2),
                                          (1262, 1), (1123, 2), (1182, 2)]),
    }

    for name, deck in variants.items():
        sources.append((name, "control variant", deck))

    rows = []
    for name, source, deck in sources:
        s = stats(deck, info)
        errs = rule_errors(deck, info)
        engine_ok = False
        if not errs:
            engine_ok, ep, et, emsg = engine_check(deck)
            if not engine_ok:
                errs.append(f"engine rejected (player={ep}, type={et}) {emsg}")
        rows.append({
            "name": name, "source": source, "deck": list(deck), "stats": s,
            "legal": not errs, "errors": errs, "engine_ok": engine_ok,
            "score": score(s) if not errs else -1,
        })

    # 候选门槛：引擎接受 + 至少 8 张基础宝可梦 + 至少 4 张能量 + 至少 12 张宝可梦。
    # 当前提交牌组作为对照组必须保留（它的基础宝可梦只有 6 张，正是待验证的弱点）。
    def eligible(r):
        s = r["stats"]
        return (r["legal"] and
                (r["name"] == "control_deckcsv" or
                 (s["basic_pokemon"] >= 8 and s["energy"] >= 4 and s["pokemon"] >= 12)))

    cands = [r for r in rows if eligible(r)]
    cands.sort(key=lambda r: (-r["score"], r["name"]))
    for r in rows:
        if r["legal"] and not eligible(r):
            r["note"] = "filtered: basic_pokemon<8 or energy<4 or pokemon<12"

    # 短名单：对照组 + 全部对照组变体（分离宝可梦/能量/训练家三个轴）
    # + 可用的官方默认牌组 + 每个池 archetype 的最高分一副。
    short, used = [], set()

    def add(r):
        if r and r["name"] not in used:
            used.add(r["name"])
            short.append(r)

    add(next((r for r in cands if r["name"] == "control_deckcsv"), None))
    ctrl = sorted((r for r in cands if r["name"].startswith("control_")
                   and r["name"] != "control_deckcsv"), key=lambda r: -r["score"])
    for r in ctrl:
        add(r)
    for r in cands:
        if r["name"].startswith("official_"):
            add(r)
    best_by_arch = {}
    for r in cands:
        if r["name"].startswith("pool_"):
            arch = r["name"].split("_")[1]
            if arch not in best_by_arch:
                best_by_arch[arch] = r
    for r in best_by_arch.values():
        add(r)
    short = short[:18]

    with open(os.path.join(args.out_dir, "candidates.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    for r in rows:
        if r["legal"]:
            pool = {"decks": [r["deck"]], "counts": [60], "names": [r["name"]]}
            with open(os.path.join(args.out_dir, "pools", f"{r['name']}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(pool, f)

    lines = ["# 候选牌组（引擎校验）", "",
             "| 名称 | 来源 | 宝可梦(基础/进化) | 能量(基础/特殊) | 训练家 | ACE | 引擎 | 分数 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["stats"]
        lines.append(
            f"| {r['name']} | {os.path.basename(r['source'])} | "
            f"{s['pokemon']} ({s['basic_pokemon']}/{s['evolution_pokemon']}) | "
            f"{s['energy']} ({s['basic_energy']}/{s['special_energy']}) | "
            f"{s['trainers']} | {s.get('ace_spec', 0)} | "
            f"{'OK' if r['engine_ok'] else 'FAIL'} | {r['score']} |")
    lines += ["", f"通过筛选 {len(cands)} / {len(rows)}；短名单 {len(short)} 副。"]
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[deck_search] {len(rows)} decks, {len(cands)} eligible, shortlist {len(short)}")
    for r in short:
        s = r["stats"]
        print(f"  {r['name']:<34} P{s['pokemon']:>2} B{s['basic_pokemon']:>2} "
              f"E{s['energy']:>2} T{s['trainers']:>2} score={r['score']}")
    print(f"wrote {args.out_dir}/candidates.json, pools/, summary.md")


if __name__ == "__main__":
    main()
