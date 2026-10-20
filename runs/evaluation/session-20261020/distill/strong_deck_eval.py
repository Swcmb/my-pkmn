# -*- coding: utf-8 -*-
"""A5.5 §8.4：真实对手口径评测（AI 执行，只读权重、不写 ckpt、不训练）。

候选（默认教师 ckpt_999424；A5 小网出来后换成小网）分别与各副牌对打：

  forward：候选驾驶 best_deck.csv        vs 对手驾驶该副牌
  reverse：候选驾驶该副牌               vs 对手驾驶 best_deck.csv

两条一起看才能分辨"牌组优势"和"模型优势"。牌组来源：
  strong_decks/*_reconstructed.csv  —— A5.5 从平台日志**推断**还原的头部强队牌组（非官方 deck）
  deck_search/dragapult_deck.csv    —— 本项目的旧牌组参照（§8.4.4 要求的 Dragapult 对比）

口径：PKMN_TURN_CAP=200、native_encode=False（本机既有限制）、would_ko=True、
座位按局 gi%2 严格均衡；每队每方向 ≥ --games 局。

结果与"对随机合法 0.608"**并列呈现、不混算**（对手分布不同，不可合并比较）。

  PKMN_TURN_CAP=200 PYTHONPATH=D:\\pkmn-kaggle\\.engine-shim;D:\\pkmn-kaggle \\
  python runs/evaluation/session-20261020/distill/strong_deck_eval.py [--games 60] [--quick]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import sys

ROOT = r"D:\pkmn-kaggle"
sys.path.insert(0, ROOT)
SESS = os.path.join(ROOT, "runs", "evaluation", "session-20261020")
STRONG = os.path.join(SESS, "strong_decks")
SHIP_DECK = os.path.join(ROOT, "runs", "evaluation", "card_database_migration", "best_deck.csv")
DRAG_DECK = os.path.join(ROOT, "runs", "evaluation", "deck_search", "dragapult_deck.csv")
DEFAULT_CAND = os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_999424.pt")
RANDOM_REF_WR, RANDOM_REF_CI = 0.608, [0.519, 0.691]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def wilson(w, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    ph = w / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_h2h():
    spec = importlib.util.spec_from_file_location(
        "eval_h2h", os.path.join(ROOT, "runs", "evaluation", "eval_h2h.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["eval_h2h"] = m
    spec.loader.exec_module(m)
    return m


def read_deck(path):
    ids = [int(x) for row in __import__("csv").reader(open(path, encoding="utf-8-sig"))
           for x in row if x.strip()]
    if len(ids) != 60:
        raise SystemExit(f"ERROR: {path} 有 {len(ids)} 张，应为 60")
    return ids


def merge(runs):
    w = sum(r["wins"] for r in runs)
    n = sum(r["decided"] for r in runs)
    g = sum(r["games"] for r in runs)
    return {"wins": w, "decided": n, "games": g, "draws": sum(r["draws"] for r in runs),
            "errs": sum(r["errs"] for r in runs),
            "illegal": sum(r["illegal"]["total"] for r in runs),
            "turn_cap": sum(r["draw_breakdown"]["turn_cap"] for r in runs),
            "wr": (w / n) if n else None, "ci95": list(wilson(w, n)),
            "mean_turn": float(sum(r["length"]["mean_turn"] * r["games"] for r in runs) / max(g, 1)),
            "seat0": {"games": sum(r["seat_balance"]["games_seat0"] for r in runs),
                      "wins": sum(r["seat_balance"]["wins_seat0"] for r in runs)},
            "seat1": {"games": sum(r["seat_balance"]["games_seat1"] for r in runs),
                      "wins": sum(r["seat_balance"]["wins_seat1"] for r in runs)},
            "per_seed": [{"seed": r["seed"], "wins": r["wins"], "decided": r["decided"]}
                         for r in runs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CAND)
    ap.add_argument("--games", type=int, default=60, help="每队每方向、每种子的局数")
    ap.add_argument("--seeds", default="20261211,20261212")
    ap.add_argument("--quick", action="store_true", help="只验证跑通，结论无效")
    ap.add_argument("--resume", action="store_true",
                    help="沿用 --out-json 里已完成的格，只补跑缺的（崩溃后用）")
    ap.add_argument("--out-json", default=os.path.join(SESS, "distill", "strong_deck_eval.json"))
    ap.add_argument("--out-md", default=os.path.join(SESS, "distill", "strong_deck_eval.md"))
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    if not a.quick and len(seeds) < 2:
        raise SystemExit("ERROR: 非 quick 至少 2 个种子")
    games = 4 if a.quick else a.games
    if a.quick:
        seeds = seeds[:1]
    if not os.path.exists(a.ckpt):
        raise SystemExit(f"ERROR: 候选 ckpt 不存在：{a.ckpt}")

    decks = [(os.path.basename(p).replace("_reconstructed.csv", ""), p, "A5.5 还原（推断，非官方 deck）")
             for p in sorted(glob.glob(os.path.join(STRONG, "*_reconstructed.csv")))]
    if not decks:
        raise SystemExit("ERROR: 没有还原牌组，先跑 reconstruct_decks.py")
    if os.path.exists(DRAG_DECK):
        decks.append(("dragapult_旧参照", DRAG_DECK, "本项目旧牌组参照，非还原"))

    conf = {}
    rj = os.path.join(STRONG, "reconstructed.json")
    if os.path.exists(rj):
        d = json.load(open(rj, encoding="utf-8"))
        for k, v in d.items():
            name = f"team{k}_{v['team_name']}"
            hi = sum(1 for c in v["cards"] if c["confidence"] == "高")
            lo = sum(1 for c in v["cards"] if c["confidence"] == "低")
            conf[name] = {"team": v["team_name"], "games_observed": v["games_observed"],
                          "n_kinds": v["check"]["n_distinct"],
                          "conf_high": hi, "conf_low": lo,
                          "energy_budget": v.get("energy_budget")}

    import torch
    from rl.card_features import get_card_table
    from rl.encoding import TokenEncoder
    torch.set_grad_enabled(False)
    h = load_h2h()
    h._install_sdk_cg_shim_alias()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ct = get_card_table(os.path.join(ROOT, "EN_Card_Data.csv"))
    enc = TokenEncoder(ct)
    net, step, cfg = h.load_net(a.ckpt, ct, dev)
    h.CAND_CKPT = os.path.abspath(a.ckpt)
    ship = read_deck(SHIP_DECK)

    def build_ev(results, done):
        return {"candidate_ckpt": os.path.abspath(a.ckpt), "candidate_sha256": sha(a.ckpt),
                "candidate_global_step": step, "candidate_net_config": cfg,
                "ship_deck": os.path.abspath(SHIP_DECK), "ship_deck_sha256": sha(SHIP_DECK),
                "seeds": seeds, "games_per_seed_per_direction": games, "quick": a.quick,
                "turn_cap": os.environ.get("PKMN_TURN_CAP"),
                "native_encode": False, "would_ko": True, "device": str(dev),
                "cells_done": done, "cells_total": len(decks) * 2,
                "random_reference": {"wr": RANDOM_REF_WR, "ci95": RANDOM_REF_CI,
                                     "note": "对随机合法策略的既有基线，并列呈现，不与本节混算"},
                "reconstruction_confidence": conf}

    def save(results, done):
        """每格落一次盘：一次 CUDA 抖动不该把已跑完的 11/12 格一起带走。"""
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump({"evidence": build_ev(results, done), "results": results},
                      f, ensure_ascii=False, indent=1)
        write_md(a.out_md, build_ev(results, done), results)

    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_md)), exist_ok=True)
    done_cells = 0
    results = []
    if a.resume and os.path.exists(a.out_json):
        prev = json.load(open(a.out_json, encoding="utf-8"))
        results = prev.get("results", [])
        done_cells = len(results)
        print(f"[strong-eval] --resume：沿用已有 {done_cells} 格结果", flush=True)
    finished = {(r["deck_label"], r["direction"]) for r in results}

    for di, (label, path, kind) in enumerate(decks):
        deck = read_deck(path)
        for direction in ("forward", "reverse"):
            if (label, direction) in finished:
                print(f"[strong-eval] skip {label} {direction}（已在结果里）", flush=True)
                continue
            pool = [ship] if direction == "forward" else [deck]
            npool = [deck] if direction == "forward" else [ship]
            runs = []
            for sd in seeds:
                print(f"[strong-eval] {label} {direction} seed={sd} games={games}", flush=True)
                try:
                    runs.append(h.run_one("random", games, sd, pool, npool, dev, ct, enc,
                                          1, "cross", net, a.ckpt))
                except Exception as e:
                    # 单格异常（如瞬时 CUDA 故障）记进 cells_error，整轮继续
                    print(f"    !! {label} {direction} seed={sd} 失败："
                          f"{type(e).__name__}: {str(e)[:160]}", flush=True)
                    torch.cuda.empty_cache()
            if not runs:
                results.append({"deck_label": label, "direction": direction, "cells_error":
                                "所有种子均失败，本格无数据", "wr": None, "ci95": [0.0, 1.0],
                                "decided": 0, "games": 0, "draws": 0, "errs": 0,
                                "illegal": 0, "turn_cap": 0, "mean_turn": 0.0,
                                "seat0": {"games": 0, "wins": 0}, "seat1": {"games": 0, "wins": 0},
                                "per_seed": [], "deck_path": os.path.abspath(path),
                                "deck_sha256": sha(path), "deck_kind": kind,
                                "candidate_pilots": ("best_deck" if direction == "forward"
                                                     else "该副牌"),
                                "opponent_policy": "random_legal"})
                done_cells += 1
                save(results, done_cells)
                continue
            m = merge(runs)
            m.update({"deck_label": label, "direction": direction, "deck_path": os.path.abspath(path),
                      "deck_sha256": sha(path), "deck_kind": kind,
                      "seeds_ok": [r["seed"] for r in runs],
                      "candidate_pilots": ("best_deck" if direction == "forward" else "该副牌"),
                      "opponent_policy": "random_legal"})
            results.append(m)
            done_cells += 1
            save(results, done_cells)
            print(f"    -> wr={'-' if m['wr'] is None else round(m['wr'], 4)} "
                  f"CI[{m['ci95'][0]:.4f},{m['ci95'][1]:.4f}] decided={m['decided']} "
                  f"errs={m['errs']} illegal={m['illegal']} turn_cap={m['turn_cap']}", flush=True)

    save(results, done_cells)
    print(f"\n[saved] {done_cells}/{len(decks) * 2} 格 -> {a.out_json}")
    print("[saved]", a.out_md)
    if done_cells < len(decks) * 2:
        print(f"WARNING: 仅完成 {done_cells}/{len(decks) * 2} 格，报告不完整")
    if a.quick:
        print("[quick] 仅供脚本验证，不作为结论")


def fmt(x, n=4):
    return "-" if x is None else str(round(x, n))


def write_md(path, ev, results):
    L = ["# §8.4 真实对手口径评测", "",
         f"候选：`{os.path.basename(ev['candidate_ckpt'])}`（step={ev['candidate_global_step']}，"
         f"sha `{ev['candidate_sha256'][:12]}…`），arch="
         f"{ev['candidate_net_config'].get('d_model')}/"
         f"{ev['candidate_net_config'].get('nhead')}/"
         f"{ev['candidate_net_config'].get('nlayers')}/"
         f"{ev['candidate_net_config'].get('ff')}",
         "",
         f"- 口径：`PKMN_TURN_CAP={ev['turn_cap']}`、`native_encode=False`（本机既有限制）、"
         f"`would_ko=True`、座位按局 gi%2 均衡；种子 {ev['seeds']}，"
         f"每队每方向每种子 {ev['games_per_seed_per_direction']} 局",
         f"- 对手动作策略：**random legal**（本节只隔离牌组差异，不引入第二个模型变量）",
         f"- 候选自家牌组：`{os.path.basename(ev['ship_deck'])}`（sha `{ev['ship_deck_sha256'][:12]}…`）",
         "",
         "> **并列，不混算**：下表是「候选 vs 某副具体牌组」，"
         f"与既有「候选 vs 随机合法策略 {ev['random_reference']['wr']} "
         f"CI{ev['random_reference']['ci95']}」分布不同，两者不可相加、不可互相换算。",
         "",
         "> **还原牌组性质**：标 `A5.5 还原` 的牌组来自平台日志词频推断，"
         "平台不公开 deck 列表、无法证伪；能量张数部分是先验。结论必须连同置信度读。",
         "",
         "## 逐队结果", "",
         "| 牌组 | 方向 | 候选驾驶 | 胜率 | Wilson 95% CI | decided | games | draws | errs | 非法 | turn_cap | 平均回合 | 先手 w/g | 后手 w/g |",
         "| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |"]
    for m in results:
        L.append(f"| {m['deck_label']} | {m['direction']} | {m['candidate_pilots']} | "
                 f"{fmt(m['wr'])} | [{fmt(m['ci95'][0])}, {fmt(m['ci95'][1])}] | {m['decided']} | "
                 f"{m['games']} | {m['draws']} | {m['errs']} | {m['illegal']} | {m['turn_cap']} | "
                 f"{fmt(m['mean_turn'], 1)} | {m['seat0']['wins']}/{m['seat0']['games']} | "
                 f"{m['seat1']['wins']}/{m['seat1']['games']} |")
    L += ["", "## 还原置信度（A5.5）", "",
          "| 队 | 观测局数 | 卡种数 | 高置信卡 | 低置信卡 | 能量预算 |",
          "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for k, c in ev["reconstruction_confidence"].items():
        L.append(f"| {c['team']} | {c['games_observed']} | {c['n_kinds']} | {c['conf_high']} | "
                 f"{c['conf_low']} | {c.get('energy_budget')} |")
    L += ["", "## 结论口径", "",
          "- forward（候选驾驶 `best_deck` vs 该副牌）低 → 该副牌在**牌组层面**压制自家牌组。",
          "- reverse（候选驾驶该副牌 vs `best_deck`）高 → 同一模型换上该副牌更强，"
          "两条同时成立才构成「换牌组」的证据；只中一条属牌组与模型的匹配效应。",
          "- 单队样本量按上表 decided 读，CI 宽的行不足以支撑结论。",
          "- 换牌组必须走与 A5 同等级门禁，本节只是候选依据（§8.5）。",
          "", "## 未覆盖", "",
          "- 对手是随机合法而非对手模型：本节测的是**牌组对牌组**的先验强弱，"
          "不等于线上对真实对手的胜率。",
          "- 还原牌组非官方 deck，能量与低置信卡可能失真。", ""]
    open(path, "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
