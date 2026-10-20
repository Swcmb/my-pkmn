# -*- coding: utf-8 -*-
"""阶段 4：训练后评测与晋级门禁（一条命令跑完矩阵并逐项判定）。

用法（训练完成后，由 AI 执行）：
  PKMN_TURN_CAP=200 PYTHONPATH=D:\\pkmn-kaggle\\.engine-shim;D:\\pkmn-kaggle \\
  python runs/evaluation/session-20261020/phase4_gate.py --ckpt <新ckpt> [--quick]

门禁项与 d.txt §七 一一对应；参考值从既有基线 JSON 读取，不硬编码历史 0.6 阈值：
  G1 牌组合法            —— validate_deck 五项规则
  G2 Agent 接口兼容      —— 提交包 smoke：返回长度/下标合法、0 异常
  G3 关键测试通过        —— pytest tests -q
  G4 门禁对局干净        —— 全 cell：0 非法动作、0 异常、0 turn_cap 截断
  G5 对随机无退步        —— 新模型 vs random ≥ 基线 vs random - 3pp
  G6 对未训练稳定优势    —— 合并胜率 Wilson 95% 下界 > 0.5
  G7 对旧 checkpoint 提升 —— 提交牌组 mirror 合并胜率 >= --promote-wr 且 Wilson 下界 > 0.5
  G8 样本量充分          —— 每个合并 arm 的 decided >= 120（--quick 模式放宽）

--quick：每 cell 只跑 4 局，用于验证脚本本身可跑通（不用于结论）。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys

ROOT = r"D:\pkmn-kaggle"
sys.path.insert(0, ROOT)

BASELINE_JSON = [os.path.join(ROOT, "runs", "card_database_migration", "20261020", "stage5_gates.json"),
                 os.path.join(ROOT, "runs", "evaluation", "session-20261020", "stage2_supplement.json")]
SHIP_DECK = os.path.join(ROOT, "runs", "evaluation", "card_database_migration", "best_deck.csv")
DRAG = os.path.join(ROOT, "runs", "evaluation", "deck_search", "dragapult_deck.csv")
DEFAULT_BASE = os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_999424.pt")
HELDOUT_POOL = os.path.join(ROOT, "runs", "evaluation", "session-20261020",
                                          "eval_pool_heldout44.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def wilson(w, n, z=1.96):
    """Wilson score 95% 区间 (lo, hi)；n=0 时给全域。"""
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


def read_baseline_random_ref():
    """基线 ckpt_999424 在提交牌组上 vs random 的合并胜率（既有两 seed + 补测新 seed）。"""
    w = n = 0
    for p in BASELINE_JSON:
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        for r in d["runs"]:
            if r["cell"] == "cand_vs_random" and r["deck_name"] == "dragapult_newlib":
                w += r["wins"]
                n += r["decided"]
    return (w / n if n else None), n


def read_baseline_heldout_ref():
    """既有 cand_vs_baseline_heldout（74 副池、cross 模式）的合并胜率，仅作 G7b 的量级参照。"""
    w = n = 0
    for p in BASELINE_JSON:
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        for r in d["runs"]:
            if r["cell"] == "cand_vs_baseline_heldout":
                w += r["wins"]
                n += r["decided"]
    return (w / n if n else None), n


def merge(runs):
    w = sum(r["wins"] for r in runs)
    n = sum(r["decided"] for r in runs)
    return {"wins": w, "decided": n, "draws": sum(r["draws"] for r in runs),
            "errs": sum(r["errs"] for r in runs),
            "illegal": sum(r["illegal"]["total"] for r in runs),
            "turn_cap": sum(r["draw_breakdown"]["turn_cap"] for r in runs),
            "wr": (w / n) if n else None, "ci95": list(wilson(w, n)),
            "cells": len(runs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seeds", default="20261024,20261025")
    ap.add_argument("--promote-wr", type=float, default=0.55,
                    help="对旧 checkpoint 的最低合并胜率（主门 G7 = 提交牌组纯镜像）。"
                         "标定依据：本会话基线上 ckpt_999424 对 ckpt_500736 在该口径是 0.708 [0.622,0.782]，"
                         "说明此臂对真实差异敏感；一次 500k 专项微调的期望增量小于一次换牌组，"
                         "故取 0.55 并同时要求 Wilson 下界 > 0.5，历史 0.6 阈值无对应口径证据")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "evaluation", "session-20261020",
                                                  "phase4_gate.json"))
    a = ap.parse_args()

    games = 4 if a.quick else a.games
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.quick:
        seeds = seeds[:1]
    import torch
    from rl.card_features import get_card_table
    from rl.encoding import TokenEncoder
    torch.set_grad_enabled(False)
    h = load_h2h()
    h._install_sdk_cg_shim_alias()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ct = get_card_table(os.path.join(ROOT, "EN_Card_Data.csv"))
    enc = TokenEncoder(ct)
    cand_net, cand_step, cand_cfg = h.load_net(a.ckpt, ct, dev)
    # 记录 ckpt 顶层键：两网微调产物含 net_b，load_net 只取 net，键位留档可由人工判别是否取错网。
    # 用 weights_only=True：只读键名，不启用任意对象反序列化。
    try:
        CAND_KEYS = sorted(torch.load(a.ckpt, map_location="cpu", weights_only=True).keys())
    except Exception:                                     # 含优化器等非张量对象时退回标记
        CAND_KEYS = ["<unreadable-under-weights_only>"]
    base_net, base_step, _ = h.load_net(a.base, ct, dev)
    # eval_h2h 的 baseline/selfmirror 对手走模块全局常量（且是相对路径，依赖 CWD）。
    # 不显式改写就会比错对象：kind="baseline" 永远加载 ckpt_500736，而不是这里指定的 base。
    h.CAND_CKPT = os.path.abspath(a.ckpt)
    h.BASE_CKPT = os.path.abspath(a.base)
    ship = [int(x) for x in open(SHIP_DECK) if x.strip()]
    assert len(ship) == 60, f"ship deck 有 {len(ship)} 张，应为 60"

    cells = []
    for seed in seeds:
        cells.append(("vs_random", ("random", games, seed, [ship], [ship], "mirror", cand_net, a.ckpt)))
        cells.append(("vs_untrained", ("untrained", games, seed, [ship], [ship], "mirror", cand_net, a.ckpt)))
        cells.append(("vs_old_ckpt", ("baseline", games, seed, [ship], [ship], "mirror", cand_net, a.ckpt)))
    cells.append(("self_mirror", ("selfmirror", games, seeds[0], [ship], [ship], "mirror", cand_net, a.ckpt)))
    # 交叉臂：候选驾驶提交牌组 vs 旧模型驾驶 held-out 44 副牌组 —— 真实比赛里双方牌组本就不同，
    # 同牌组镜像只是最纯净的技巧对照；这一臂防止"只在自家牌组上变强"。
    # 池用 split_eval_pool.py 切出的 held-out 侧（训练对手用另一侧 30 副），二者互斥。
    held = [list(d) for d in json.load(open(HELDOUT_POOL, encoding="utf-8"))["decks"]]
    for seed in seeds:
        cells.append(("vs_old_heldout", ("baseline", games, seed, [ship], held, "cross", cand_net, a.ckpt)))

    runs = []
    for label, spec in cells:
        kind, g, sd, pool, npool, mode, net, ck = spec
        print(f"[gate] {label} seed={sd} games={g}", flush=True)
        r = h.run_one(kind, g, sd, pool, npool, dev, ct, enc, 1, mode, net, ck)
        if label == "vs_old_ckpt":      # 回读校验：对手确实是 --base 那个 ckpt
            assert os.path.abspath(r["meta"]["ckpt"]) == os.path.abspath(a.base), \
                f"baseline 对手解析错：{r['meta']} != {a.base}"
            assert r["meta"]["step"] == base_step, \
                f"baseline 对手 step 不符：{r['meta']['step']} != {base_step}"
        r["opp_ckpt"] = a.base if label == "vs_old_ckpt" else ck
        r["cell"] = label
        r["deck_sha256"] = sha(SHIP_DECK)
        r["ckpt_path"] = a.ckpt
        r["ckpt_sha256"] = sha(a.ckpt)
        r["ckpt_step"] = cand_step
        runs.append(r)
        print("     wr=%s ci95=%s draws=%d errs=%d illegal=%d turncap=%d"
              % (r["winrate"], r["winrate_ci95"], r["draws"], r["errs"],
                 r["illegal"]["total"], r["draw_breakdown"]["turn_cap"]), flush=True)

    agg = {c: merge([r for r in runs if r["cell"] == c]) for c in
           ("vs_random", "vs_untrained", "vs_old_ckpt", "vs_old_heldout", "self_mirror")}

    # G1 牌组合法性
    sys.path.insert(0, os.path.join(ROOT, "runs", "evaluation", "session-20261020"))
    import validate_deck as vd
    lib = vd.load_library()
    g1 = vd.validate(SHIP_DECK, lib)

    # G2 smoke（提交包路径：新模型导出前仍用旧包结构，故此项在阶段 5 后复跑）
    g2 = None
    if not a.quick:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "runs", "evaluation", "session-20261020",
                                                         "subsmoke_timing.py"),
                            os.path.join(ROOT, "submission_rl_transformer2_db_migrated"), "6"],
                           capture_output=True, text=True, env=dict(os.environ, SMOKE_THREADS="2"))
        try:
            g2 = json.loads(p.stdout)
        except Exception:
            g2 = {"parse_error": p.stdout[-400:] + p.stderr[-400:]}

    g3 = None
    if not a.quick:
        t = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                           cwd=ROOT, capture_output=True, text=True)
        g3 = {"returncode": t.returncode, "tail": t.stdout.strip().splitlines()[-1:]}

    rnd_ref, rnd_ref_n = read_baseline_random_ref()
    HELDOUT_REF = read_baseline_heldout_ref()      # (wr, n) —— G7b 的量级参照
    min_n = 0 if a.quick else 120

    gates = {
        "G1_牌组合法": {"pass": g1["pass"], "detail": g1["checks"]},
        "G2_agent接口": {"pass": True if a.quick else (
                            bool(g2) and g2.get("illegal_len", 1) == 0
                            and g2.get("illegal_index", 1) == 0 and g2.get("errors", 1) == 0),
                         "skipped": bool(a.quick), "detail": g2},
        "G3_测试": {"pass": True if a.quick else (bool(g3) and g3["returncode"] == 0),
                    "skipped": bool(a.quick), "detail": g3},
        "G4_门禁干净": {"pass": all(agg[c]["illegal"] == 0 and agg[c]["errs"] == 0
                                    and agg[c]["turn_cap"] == 0 for c in agg),
                        "detail": {c: {k: agg[c][k] for k in ("illegal", "errs", "turn_cap")} for c in agg}},
        "G5_对随机不退步": {"pass": rnd_ref is None or agg["vs_random"]["wr"] is None
                            or agg["vs_random"]["wr"] >= rnd_ref - 0.03,
                            "detail": {"new": agg["vs_random"]["wr"], "ref_base": rnd_ref,
                                       "ref_n": rnd_ref_n}},
        "G6_对未训练优势": {"pass": agg["vs_untrained"]["ci95"][0] > 0.5,
                            "detail": {"wr": agg["vs_untrained"]["wr"],
                                       "ci95": agg["vs_untrained"]["ci95"],
                                       "per_init_wr": [r["winrate"] for r in runs
                                                        if r["cell"] == "vs_untrained"],
                                       "note": "untrained 对手由 --seeds 逐个随机初始化；实测同一候选"
                                               "对不同初始化的胜率在 0.40-0.83 间摆动，"
                                               "故必须跨 >=2 个初始化合并判定，单格结果只作 sanity"}},
        "G7_对旧ckpt提升": {"pass": agg["vs_old_ckpt"]["wr"] is not None
                            and agg["vs_old_ckpt"]["wr"] >= a.promote_wr
                            and agg["vs_old_ckpt"]["ci95"][0] > 0.5,
                            "detail": {"wr": agg["vs_old_ckpt"]["wr"],
                                       "ci95": agg["vs_old_ckpt"]["ci95"],
                                       "threshold": a.promote_wr,
                                       "base_ckpt": a.base, "base_step": base_step}},
        "G7b_heldout非劣": {"pass": agg["vs_old_heldout"]["ci95"][0] > 0.40,
                             "detail": {"wr": agg["vs_old_heldout"]["wr"],
                                        "ci95": agg["vs_old_heldout"]["ci95"],
                                        "margin": "Wilson 下界 > 0.40（判断值）：主提升门是 G7 的同牌组镜像，"
                                                  "本臂用于排除仅在自家牌组上变强",
                                        "ref_note": "同型对照参照：既有 74 副池上 cand vs old 交叉 = "
                                                    f"{HELDOUT_REF[0]:.3f}（{HELDOUT_REF[1]} 局，池与本臂不同，"
                                                    "仅作量级参考）" if HELDOUT_REF[0] is not None else None,
                                        "pool": HELDOUT_POOL, "n_decks": len(held)}},
        "G8_样本量": {"pass": min_n == 0 or all(agg[c]["decided"] >= min_n
                                                for c in ("vs_random", "vs_untrained", "vs_old_ckpt",
                                                          "vs_old_heldout")),
                      "detail": {c: agg[c]["decided"] for c in agg}},
        "sanity_自镜像接近0.5": {"pass": agg["self_mirror"]["ci95"][0] <= 0.5 <= agg["self_mirror"]["ci95"][1],
                                  "detail": {"wr": agg["self_mirror"]["wr"],
                                             "ci95": agg["self_mirror"]["ci95"]}},
    }

    out = {"candidate": {"path": a.ckpt, "sha256": sha(a.ckpt), "step": cand_step,
                         "net_config": cand_cfg, "deck": SHIP_DECK, "deck_sha256": sha(SHIP_DECK),
                         "ckpt_keys": CAND_KEYS},
           "reference": {"base_ckpt": a.base, "base_sha256": sha(a.base), "base_step": base_step,
                         "random_ref_wr": rnd_ref, "random_ref_n": rnd_ref_n},
           "protocol": {"harness": "runs/evaluation/eval_h2h.py run_one（与基线同协议）",
                        "games_per_cell": games, "seeds": seeds,
                        "turn_cap": os.environ.get("PKMN_TURN_CAP"),
                        "ship_deck": "best_deck.csv（提交牌组）",
                        "quick": a.quick},
           "aggregates": agg, "gates": gates,
           "promote": (not a.quick) and all(v["pass"] for k, v in gates.items())}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    print("\n### 阶段 4 晋级门禁逐项")
    for k, v in gates.items():
        mark = "SKIP" if v.get("skipped") else ("PASS" if v["pass"] else "FAIL")
        print(f"  [{mark}] {k}")
    print(f"\nPROMOTE = {out['promote']}   (candidate step={cand_step})")
    print("[saved]", a.out, flush=True)


if __name__ == "__main__":
    main()
