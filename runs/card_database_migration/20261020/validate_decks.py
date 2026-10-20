# -*- coding: utf-8 -*-
"""阶段 3a：在新卡牌库下核对/重建牌组候选，并用真实 CABT 引擎做开局校验。

候选来源（全部保留历史文件，只读）：
  · runs/evaluation/deck_search/dragapult_deck.csv  旧 Dragapult 牌组
  · agent/deck.csv                                   当前提交牌组
  · rl/decks.DECKS['iono'] / ['mega_lucario']        合法补充候选
校验两层：Python 层（60 张、ID 存在于新库、同名 ≤4、ACE SPEC ≤1、≥1 基础宝可梦、
进化链引用完整）+ 引擎层（battle_start 真实开局，逐候选镜像与交叉两种配对）。

运行：PYTHONPATH=D:\\pkmn-kaggle\\.engine-shim python runs/card_database_migration/20261020/validate_decks.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "runs", "evaluation", "card_database_migration")
ENERGY_STAGES = {"Basic Energy", "Special Energy"}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def python_check(deck, table, eng_names, ace_by_name, basic_energy_names, label):
    errs, notes = [], []
    if len(deck) != 60:
        errs.append(f"张数 {len(deck)} != 60")
    unknown = [i for i in deck if table.names.get(i) is None]
    if unknown:
        errs.append(f"新库缺失 ID {sorted(set(unknown))[:10]}")
    unknown_eng = [i for i in deck if i not in eng_names]
    if unknown_eng:
        errs.append(f"引擎缺失 ID {sorted(set(unknown_eng))[:10]}")
    from collections import Counter
    names = Counter(table.names[i] for i in deck if table.names.get(i))
    over4 = {n: c for n, c in names.items() if c > 4 and n not in basic_energy_names}
    if over4:
        errs.append(f"同名超 4 张 {over4}")
    ace = sum(c for n, c in names.items() if n in ace_by_name)
    if ace > 1:
        errs.append(f"ACE SPEC 共 {ace} 张 > 1")
    basics = [i for i in deck if table.features(i) is not None and
              _is_basic(table, i)]
    if not basics:
        errs.append("无基础宝可梦")
    # 进化链：stage1/stage2 的 evolvesFrom 必须在牌组里有对应前置名
    stage_names = {table.names[i] for i in deck if table.names.get(i)}
    for i in deck:
        nm = table.names.get(i)
        if nm is None:
            continue
        pre = _evolves_from(eng_names, i)
        if pre and pre not in stage_names:
            notes.append(f"{nm} 的前置 {pre} 不在牌组")
    return {"label": label, "cards": len(deck), "errors": errs,
            "evolution_notes": sorted(set(notes))[:10],
            "distinct_names": len(names), "ace_spec_total": ace,
            "basics": len(basics)}


def _is_basic(table, cid):
    from rl.card_features import STAGES
    f = table.features(cid)
    idx = 3 + STAGES.index("Basic Pokémon")
    return bool(f[idx] > 0.5)


def _evolves_from(eng_names, cid):
    return eng_names.get(cid, {}).get("evolvesFrom")


def engine_check(decks_by_label, eng_all_ids):
    """真实 CABT 开局：镜像 + 全部交叉配对。"""
    from sdk_cg import game
    labels = list(decks_by_label)
    results = {}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j < i:
                continue
            d0, d1 = decks_by_label[a], decks_by_label[b]
            try:
                obs, sd = game.battle_start(list(d0), list(d1))
                ok = obs is not None
                results[f"{a} x {b}"] = {"accepted": ok,
                                         "errorPlayer": sd.errorPlayer,
                                         "errorType": sd.errorType}
                if ok:
                    game.battle_finish()
            except Exception as e:              # noqa: BLE001
                results[f"{a} x {b}"] = {"accepted": False, "exception": repr(e)}
    return results


def main():
    from rl.card_features import get_card_table
    from rl.decks import DECKS
    from sdk_cg import api

    table = get_card_table()
    cards = {c.cardId: {"name": c.name, "evolvesFrom": c.evolvesFrom,
                        "aceSpec": c.aceSpec, "cardType": int(c.cardType)}
             for c in api.all_card_data()}
    ace_by_name = {v["name"] for v in cards.values() if v["aceSpec"]}
    basic_energy_names = {v["name"] for v in cards.values() if v["cardType"] == 5}

    cand = {}
    src = {}
    dp = os.path.join(ROOT, "runs", "evaluation", "deck_search", "dragapult_deck.csv")
    if os.path.exists(dp):
        cand["dragapult_newlib"] = [int(x) for x in open(dp) if x.strip()]
        src["dragapult_newlib"] = {"path": dp, "sha256": sha256(dp)}
    ac = os.path.join(ROOT, "agent", "deck.csv")
    if os.path.exists(ac):
        cand["current_agent_deck"] = [int(x) for x in open(ac) if x.strip()]
        src["current_agent_deck"] = {"path": ac, "sha256": sha256(ac)}
    for name in ("iono", "mega_lucario", "mega_abomasnow"):
        cand[f"fixture_{name}"] = list(DECKS[name])
        src[f"fixture_{name}"] = {"path": "rl/decks.py:_DECKS_SPEC", "fixture": name}

    checks = {k: python_check(v, table, cards, ace_by_name, basic_energy_names, k)
              for k, v in cand.items()}
    eng = engine_check(cand, set(cards))
    report = {"library": {"path": os.path.join(ROOT, "EN_Card_Data.csv"),
                          "sha256": sha256(os.path.join(ROOT, "EN_Card_Data.csv")),
                          "vocab_size": table.vocab_size,
                          "unique_cards": len([1 for i in range(1, table.vocab_size) if table.names.get(i)])},
              "engine": {"cards": len(cards), "attacks": len(api.all_attack())},
              "deck_sources": src, "python_checks": checks, "engine_checks": eng}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, "deck_engine_checks.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    bad_python = [k for k, v in checks.items() if v["errors"]]
    bad_engine = [k for k, v in eng.items() if not v["accepted"]]
    print(json.dumps({"python_rejects": bad_python, "engine_rejects": bad_engine,
                      "out_dir": OUT_DIR, "out_dir_pre_existing": os.listdir(OUT_DIR)},
                     ensure_ascii=False, indent=2))
    for k, v in checks.items():
        print(f"[py] {k:22s} 60={v['cards'] == 60} errs={v['errors']} notes={v['evolution_notes']}")


if __name__ == "__main__":
    main()
