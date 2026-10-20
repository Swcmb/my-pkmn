# -*- coding: utf-8 -*-
"""阶段 2：迁移库下的解析 + checkpoint 兼容性回归。

对照实验：同一批真实对局观测，分别用
  · 备份库 runs/card_database_migration/20261020/original_cards.csv（旧）
  · 仓库现库 EN_Card_Data.csv（迁移后）
构造 CardTable/TokenEncoder/build_token_net，加载两个 checkpoint，
比较 logits/value 是否逐位一致，并统计
  · 严格 load_state_dict 结果
  · 采样动作合法性
  · 引擎新卡（cardId ≥ 1268）被夹到 UNK 的比例
  · would_ko 真实 Search API 路径（MAIN select 的 ATTACK 选项）

运行：PYTHONPATH=D:\\pkmn-kaggle\\.engine-shim python runs/card_database_migration/20261020/verify_ckpt_compat.py
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from rl.card_features import CardTable          # noqa: E402
from rl.encoding import TokenEncoder            # noqa: E402
from rl.policy import build_token_net, obs_to_tensors  # noqa: E402
from rl.search_agent import would_ko_flags      # noqa: E402
from rl.decks import DECKS                      # noqa: E402
from sdk_cg import game                         # noqa: E402

OLD_CSV = os.path.join(HERE, "original_cards.csv")
CKPTS = {"cand_e16_1m": "runs/train-20m-e16-1m/ckpt_999424.pt",
         "base_500k": "runs/train-20m-e16-1m/ckpt_500736.pt"}
N_GAMES = int(os.environ.get("VERIFY_GAMES", "6"))
SEEDS = [int(s) for s in os.environ.get("VERIFY_SEEDS", "20261020,20261021").split(",")]
TURN_CAP = int(os.environ.get("PKMN_TURN_CAP", "400"))


def collect_games(seed: int, n_games: int):
    """随机合法走子采集真实观测；返回 [(obs, deck_ids_of_agent_seat)]。"""
    rng = random.Random(seed)
    names = list(DECKS)
    out, results = [], []
    for g in range(n_games):
        d0, d1 = DECKS[names[g % len(names)]], DECKS[names[(g + 1) % len(names)]]
        obs, sd = game.battle_start(d0, d1)
        if obs is None:
            results.append({"game": g, "error": f"battle_start failed errType={sd.errorType}"})
            continue
        decks = {0: d0, 1: d1}
        steps = 0
        while obs.get("select") is not None and steps < TURN_CAP:
            sel = obs["select"]
            opts = sel.get("option") or []
            k = max(sel.get("minCount", 1) or 1, 1)
            hi = max(k, min(sel.get("maxCount", 1) or 1, len(opts)))
            if not opts:
                break
            pick = rng.sample(range(len(opts)), min(hi, len(opts))) if hi >= k else [0]
            you = obs.get("current", {}).get("yourIndex")
            snapshot = copy.deepcopy(obs)
            seat = you if you in (0, 1) else 0
            out.append((snapshot, decks[seat], seat))
            try:
                obs = game.battle_select(sorted(pick))
            except Exception as e:                      # noqa: BLE001
                results.append({"game": g, "step": steps, "select_error": repr(e)})
                break
            steps += 1
        try:
            result = (obs.get("current") or {}).get("result")
            results.append({"game": g, "steps": steps, "result": result})
        finally:
            game.battle_finish()
    return out, results


def run_pair(obs_batch, decks_batch):
    """旧库/新库各建一套 encoder+net，逐 checkpoint 比较前向输出。"""
    tables = {"old": CardTable(OLD_CSV), "new": CardTable(os.path.join(ROOT, "EN_Card_Data.csv"))}
    report = {}
    for tag, path in CKPTS.items():
        # 本地自训 ckpt（仓库产物）：需要 net_config/global_step 等 Python 对象，故 weights_only=False
        ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
        cfg = dict(ck.get("net_config") or {})
        per_lib = {}
        for lib, table in tables.items():
            net = build_token_net(table, cfg)
            res = net.load_state_dict(ck["net"], strict=True)
            enc = TokenEncoder(table)
            outs = [enc.encode(o) for (o, _d, _s) in obs_batch]
            logits_all, value_all, illegal = [], [], 0
            with torch.no_grad():
                for e in outs:
                    t = obs_to_tensors(e, "cpu")
                    batch = {k: v[None] for k, v in t.items()}
                    logits, value = net.logits_value(batch)
                    a, _lp, _ent, _v = net.get_action_and_value(batch)
                    mask = batch["action_mask"]
                    if int(mask.gather(1, a[:, None]).squeeze(1).item()) == 0:
                        illegal += 1
                    logits_all.append(logits.numpy().astype(np.float64))
                    value_all.append(value.numpy().astype(np.float64))
            per_lib[lib] = {"logits": np.concatenate(logits_all),
                            "value": np.concatenate(value_all),
                            "load_strict_ok": not (res.missing_keys or res.unexpected_keys),
                            "missing": list(res.missing_keys)[:5],
                            "unexpected": list(res.unexpected_keys)[:5],
                            "illegal_sampled": illegal, "n": len(outs),
                            "vocab": table.vocab_size, "feat_dim": table.feat_dim}
        o, n = per_lib["old"], per_lib["new"]
        dl = np.abs(o["logits"] - n["logits"])
        dv = np.abs(o["value"] - n["value"])
        report[tag] = {
            "ckpt": path, "global_step": ck.get("global_step"),
            "net_config": cfg,
            "old_vs_new_logits_max_absdiff": float(dl.max()),
            "old_vs_new_value_max_absdiff": float(dv.max()),
            "bit_identical_logits": bool((dl == 0).all()),
            "old": {k: v for k, v in o.items() if k not in ("logits", "value")},
            "new": {k: v for k, v in n.items() if k not in ("logits", "value")},
        }
    return report


def unk_exposure(obs_batch):
    """统计被夹到 UNK 的引擎新卡（id ≥ 1268）出现比例。"""
    total = masked = 0
    cards_seen = set()
    for obs, _d, _s in obs_batch:
        cur = obs.get("current") or {}
        for p in cur.get("players") or []:
            for pk in (p.get("active") or []) + (p.get("bench") or []):
                cid = (pk or {}).get("pokemon", {}).get("id")
                if cid is None:
                    continue
                cards_seen.add(cid)
                total += 1
                if cid >= 1268:
                    masked += 1
            for h in (p.get("hand") or []):
                cid = (h or {}).get("id")
                if cid is not None:
                    cards_seen.add(cid)
                    total += 1
                    if cid >= 1268:
                        masked += 1
    return {"card_slots_scanned": total, "ids_ge_1268": masked,
            "distinct_in_play_ids": len(cards_seen),
            "distinct_ids_ge_1268": sorted({c for c in cards_seen if c >= 1268})[:20]}


def would_ko_path(obs_batch, enc_new):
    """真实 Search API：对 MAIN select 的 ATTACK 选项求 would_ko 三元组。"""
    stats = {"main_selects": 0, "with_attack_opts": 0, "annotated_nonzero": 0,
             "errors": [], "sample": None}
    for obs, deck, _seat in obs_batch:
        sel = obs.get("select") or {}
        if sel.get("type") != 0:
            continue
        stats["main_selects"] += 1
        opts = sel.get("option") or []
        if not any(o.get("attackId") is not None for o in opts):
            continue
        stats["with_attack_opts"] += 1
        try:
            flags = would_ko_flags(obs, deck, enc_new, n_var=2)
        except Exception as e:                  # noqa: BLE001
            stats["errors"].append(repr(e))
            if len(stats["errors"]) > 3:
                traceback.print_exc()
            continue
        if flags:
            stats["annotated_nonzero"] += 1
            if stats["sample"] is None:
                stats["sample"] = {str(k): [round(float(x), 4) for x in v] for k, v in list(flags.items())[:4]}
    return stats


def main():
    obs_batch, results = [], []
    for s in SEEDS:
        g, r = collect_games(s, N_GAMES // len(SEEDS))
        obs_batch += g
        results += r
    report = {"games": results, "n_observations": len(obs_batch)}
    report["ckpt_compat"] = run_pair(obs_batch, [d for _o, d, _s in obs_batch])
    report["unk_exposure"] = unk_exposure(obs_batch)
    tables_new = CardTable(os.path.join(ROOT, "EN_Card_Data.csv"))
    report["would_ko"] = would_ko_path(obs_batch, TokenEncoder(tables_new))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    with open(os.path.join(HERE, "ckpt_compat_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
