# -*- coding: utf-8 -*-
"""A5：小网（蒸馏产物）门禁 G-a…G-f。AI 执行，全程只读权重、不写 ckpt、不训练。

一条命令跑完并逐项判定：
  PKMN_TURN_CAP=200 PYTHONPATH=D:\\pkmn-kaggle\\.engine-shim;D:\\pkmn-kaggle \\
  python runs/evaluation/session-20261020/distill/small_gate.py --ckpt <bc小网ckpt> [--quick]

阈值全部按 Prompt/e.txt §7 原文，不做放宽：
  G-a 对随机合法   ≥120 局合并，胜率 ≥0.608 且 Wilson 下界 >0.5
  G-b 对教师纯镜像 ≥120 局合并，胜率 ≥0.45
  G-c 自镜像 sanity ≥100 局合并，胜率 0.45–0.55，0 异常
  G-d held-out 交叉 ≥120 局合并，候选驾驶 best_deck、教师驾驶 held-out 44 副，Wilson 下界 >0.40
  G-e 非法动作 / 异常 / turn_cap 截断 全部为 0
  G-f 单步时延：p99<100ms、SMOKE_THREADS=2 下 >100ms 为 0、峰值 RSS<2GB
      —— G-f 量的是**提交包**，需要 A6 先导出，所以只有给了 --pkg 才执行；
         缺省时如实标 未执行，绝不拿"没跑"充当通过。

另两条硬约束的实现：
  * 种子不得与历史重叠：启动时扫描既有评测 JSON 里出现过的 seed，撞上就直接拒跑。
  * 每个 arm 至少跨 2 个种子合并（单种子的 vs-untrained 数字不作强度结论）。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys

ROOT = r"D:\pkmn-kaggle"
sys.path.insert(0, ROOT)
SESS = os.path.join(ROOT, "runs", "evaluation", "session-20261020")
SHIP_DECK = os.path.join(ROOT, "runs", "evaluation", "card_database_migration", "best_deck.csv")
HELDOUT_POOL = os.path.join(SESS, "eval_pool_heldout44.json")
TRAIN_POOL = os.path.join(SESS, "opp_pool_train30.json")
DEFAULT_TEACHER = os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_999424.pt")
TEACHER_VS_RANDOM_REF = 0.608          # e.txt §1 参考基线（120 局，提交牌组）
TEACHER_VS_RANDOM_CI = [0.519, 0.691]  # 同一条目的 Wilson 95% 区间，用于并列呈现噪声带
HIST_JSONS = [os.path.join(SESS, "*.json"),
              os.path.join(ROOT, "runs", "card_database_migration", "20261020", "*.json")]


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


def historical_seeds():
    """扫既有评测 JSON 里用过的 seed，供重叠守卫使用。

    跳过 distill/ 目录：本脚本自己的输出里也带 seed 字段，不排除的话第二次用同一批种子
    会被自己上一次的报告判成"与历史重叠"。
    """
    seen = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "seed" and isinstance(v, int):
                    seen.add(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for pat in HIST_JSONS:
        for p in glob.glob(pat):
            if os.sep + "distill" + os.sep in p:
                continue
            try:
                walk(json.load(open(p, encoding="utf-8")))
            except Exception:
                continue
    return seen


def merge(runs):
    w = sum(r["wins"] for r in runs)
    n = sum(r["decided"] for r in runs)
    return {"wins": w, "decided": n, "games": sum(r["games"] for r in runs),
            "draws": sum(r["draws"] for r in runs),
            "errs": sum(r["errs"] for r in runs),
            "illegal": sum(r["illegal"]["total"] for r in runs),
            "turn_cap": sum(r["draw_breakdown"]["turn_cap"] for r in runs),
            "max_steps": sum(r["draw_breakdown"]["max_steps"] for r in runs),
            "wr": (w / n) if n else None, "ci95": list(wilson(w, n)),
            "seeds": sorted({r["seed"] for r in runs}),
            "per_seed": [{"seed": r["seed"], "wins": r["wins"], "decided": r["decided"],
                          "wr": (r["wins"] / r["decided"]) if r["decided"] else None}
                         for r in runs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="A4 蒸馏产物（小网 ckpt）")
    ap.add_argument("--teacher", default=DEFAULT_TEACHER)
    ap.add_argument("--games", type=int, default=60, help="每种子每 arm 的局数")
    ap.add_argument("--seeds", default="20261201,20261202,20261203")
    ap.add_argument("--pkg", default="", help="A6 导出的提交包；给了才跑 G-f 时延门")
    ap.add_argument("--quick", action="store_true", help="只验证脚本可跑通，结论无效")
    ap.add_argument("--out", default=os.path.join(SESS, "distill", "small_gate.json"))
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):
        raise SystemExit(f"ERROR: 候选 ckpt 不存在：{a.ckpt}")
    seeds = [int(s) for s in a.seeds.split(",")]
    if not a.quick:
        clash = sorted(set(seeds) & historical_seeds())
        if clash:
            raise SystemExit(f"ERROR: 种子与历史评测重叠：{clash}；换一批新 seed")
        if len(seeds) < 2:
            raise SystemExit("ERROR: 非 quick 模式至少需要 2 个种子（单种子数字不作结论）")
    games = 4 if a.quick else a.games
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
    teacher_net, teacher_step, teacher_cfg = h.load_net(a.teacher, ct, dev)
    try:
        cand_keys = sorted(torch.load(a.ckpt, map_location="cpu", weights_only=True).keys())
    except Exception:
        cand_keys = ["<unreadable-under-weights_only>"]

    # eval_h2h 的 baseline/selfmirror 对手走模块全局常量，不改写就会比错对象。
    h.CAND_CKPT = os.path.abspath(a.ckpt)
    h.BASE_CKPT = os.path.abspath(a.teacher)

    ship = [int(x) for x in open(SHIP_DECK, encoding="utf-8-sig") if x.strip()]
    assert len(ship) == 60, f"ship deck 有 {len(ship)} 张"
    held = [list(d) for d in json.load(open(HELDOUT_POOL, encoding="utf-8"))["decks"]]

    runs = []
    for arm, kind, pool, npool, mode in (
            ("G_a_vs_random", "random", [ship], [ship], "mirror"),
            ("G_b_vs_teacher_mirror", "baseline", [ship], [ship], "mirror"),
            ("G_c_self_mirror", "selfmirror", [ship], [ship], "mirror"),
            ("G_d_vs_teacher_heldout", "baseline", [ship], held, "cross")):
        for sd in seeds:
            print(f"[small-gate] {arm} seed={sd} games={games}", flush=True)
            r = h.run_one(kind, games, sd, pool, npool, dev, ct, enc, 1, mode, cand_net, a.ckpt)
            if kind == "baseline":
                assert os.path.abspath(r["meta"]["ckpt"]) == os.path.abspath(a.teacher), \
                    f"教师对手解析错：{r['meta']} != {a.teacher}"
                assert r["meta"]["step"] == teacher_step, \
                    f"教师对手 step 不符：{r['meta']['step']} != {teacher_step}"
            if kind == "selfmirror":
                assert os.path.abspath(r["meta"]["ckpt"]) == os.path.abspath(a.ckpt), r["meta"]
                assert r["meta"]["step"] == cand_step, r["meta"]
            r["cell"], r["seed"] = arm, sd
            runs.append(r)

    merged = {arm: merge([r for r in runs if r["cell"] == arm])
              for arm in ("G_a_vs_random", "G_b_vs_teacher_mirror", "G_c_self_mirror",
                          "G_d_vs_teacher_heldout")}

    min_dec = 0 if a.quick else 120
    gates = []

    def gate(gid, ok, detail):
        gates.append({"id": gid, "pass": bool(ok), "detail": detail})

    m = merged["G_a_vs_random"]
    gate("G-a", m["wr"] is not None and m["wr"] >= TEACHER_VS_RANDOM_REF and m["ci95"][0] > 0.5
         and m["decided"] >= min_dec,
         f"wr={None if m['wr'] is None else round(m['wr'], 4)} 需 ≥{TEACHER_VS_RANDOM_REF}；"
         f"Wilson 下界={round(m['ci95'][0], 4)} 需 >0.5；decided={m['decided']} 需 ≥{min_dec}；"
         f"噪声带参照：教师同口径 {TEACHER_VS_RANDOM_REF} CI{TEACHER_VS_RANDOM_CI}")
    m = merged["G_b_vs_teacher_mirror"]
    gate("G-b", m["wr"] is not None and m["wr"] >= 0.45 and m["decided"] >= min_dec,
         f"wr={None if m['wr'] is None else round(m['wr'], 4)} 需 ≥0.45；decided={m['decided']} 需 ≥{min_dec}")
    m = merged["G_c_self_mirror"]
    gate("G-c", m["wr"] is not None and 0.45 <= m["wr"] <= 0.55 and m["errs"] == 0
         and m["decided"] >= (0 if a.quick else 100),
         f"wr={None if m['wr'] is None else round(m['wr'], 4)} 需落在 0.45–0.55；errs={m['errs']} 需 0；"
         f"decided={m['decided']} 需 ≥{100 if not a.quick else 0}")
    m = merged["G_d_vs_teacher_heldout"]
    gate("G-d", m["wr"] is not None and m["ci95"][0] > 0.40 and m["decided"] >= min_dec,
         f"Wilson 下界={round(m['ci95'][0], 4)} 需 >0.40；wr={None if m['wr'] is None else round(m['wr'], 4)}；"
         f"decided={m['decided']} 需 ≥{min_dec}")
    tot_illegal = sum(r["illegal"]["total"] for r in runs)
    tot_errs = sum(r["errs"] for r in runs)
    tot_tcap = sum(r["draw_breakdown"]["turn_cap"] for r in runs)
    gate("G-e", tot_illegal == 0 and tot_errs == 0 and tot_tcap == 0,
         f"非法动作={tot_illegal} 异常={tot_errs} turn_cap 截断={tot_tcap}，全部需 0")

    gf = {"id": "G-f", "ran": False, "pass": None, "detail": "未执行：G-f 量的是提交包，需要 A6 先导出（--pkg）"}
    if a.pkg:
        env = dict(os.environ, SMOKE_THREADS="2")
        p = subprocess.run([sys.executable, os.path.join(SESS, "subsmoke_timing.py"),
                            os.path.abspath(a.pkg), "50", str(seeds[0])],
                           capture_output=True, text=True, env=env)
        tail = p.stdout.strip().splitlines()[-6:]
        parsed = {}
        for ln in tail:
            for k in ("p99", "rss_mb", "over100", "illegal", "decisions"):
                if k in ln:
                    num = "".join(c for c in ln.split(":", 1)[-1] if c.isdigit() or c == ".")
                    if num:
                        parsed[k] = float(num)
        ok = (p.returncode == 0 and parsed.get("over100", 1) == 0
              and parsed.get("p99", 1e9) < 100.0 and parsed.get("rss_mb", 1e9) < 2048.0)
        gf = {"id": "G-f", "ran": True, "pass": bool(ok), "pkg": os.path.abspath(a.pkg),
              "parsed": parsed, "rc": p.returncode, "tail": tail,
              "detail": f"SMOKE_THREADS=2，p99<100ms、>100ms 次数 0、峰值 RSS<2048MB；解析={parsed}"}
    gates.append(gf)

    ev = {"candidate_ckpt": os.path.abspath(a.ckpt), "candidate_sha256": sha(a.ckpt),
          "candidate_global_step": cand_step, "candidate_net_config": cand_cfg,
          "candidate_ckpt_top_level_keys": cand_keys,
          "candidate_params": sum(v.numel() for v in cand_net.parameters()),
          "teacher_ckpt": os.path.abspath(a.teacher), "teacher_sha256": sha(a.teacher),
          "teacher_global_step": teacher_step, "teacher_net_config": teacher_cfg,
          "teacher_params": sum(v.numel() for v in teacher_net.parameters()),
          "ship_deck_sha256": sha(SHIP_DECK), "heldout_pool": os.path.abspath(HELDOUT_POOL),
          "train_pool": os.path.abspath(TRAIN_POOL),
          "heldout_pool_sha256": sha(HELDOUT_POOL), "train_pool_sha256": sha(TRAIN_POOL),
          "seeds": seeds, "games_per_seed": games, "quick": a.quick,
          "turn_cap": os.environ.get("PKMN_TURN_CAP"), "native_encode": False,
          "would_ko": True, "device": str(dev),
          "seed_overlap_guard": "passed" if not a.quick else "skipped (quick)"}
    promote = (not a.quick) and all(g["pass"] for g in gates if g.get("pass") is not None) \
        and gf.get("ran")
    report = {"evidence": ev, "merged": merged, "gates": gates,
              "promote": bool(promote),
              "skipped": [g["id"] for g in gates if g.get("pass") is None],
              "note": ("quick 模式只验证脚本可跑通，任何数字都不构成结论" if a.quick else
                       "G-f 未跑时 promote 必为 False：门禁集不完整")}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("\n=== A5 小网门禁 ===")
    for arm, m in merged.items():
        print(f"  {arm:<24} wr={'-' if m['wr'] is None else format(round(m['wr'], 4), '.4f')} "
              f"CI[{m['ci95'][0]:.4f},{m['ci95'][1]:.4f}] decided={m['decided']} "
              f"games={m['games']} draws={m['draws']} errs={m['errs']} "
              f"illegal={m['illegal']} turn_cap={m['turn_cap']}")
        for ps in m["per_seed"]:
            print(f"      seed {ps['seed']}: {ps['wins']}/{ps['decided']} "
                  f"wr={'-' if ps['wr'] is None else round(ps['wr'], 3)}")
    for g in gates:
        print(f"  [{'PASS' if g['pass'] else ('SKIP' if g['pass'] is None else 'FAIL')}] "
              f"{g['id']}: {g['detail']}")
    print(f"\npromote={promote}  candidate params={ev['candidate_params']:,} "
          f"arch={cand_cfg.get('d_model')}/{cand_cfg.get('nhead')}/{cand_cfg.get('nlayers')}"
          f"/{cand_cfg.get('ff')}")
    print("[saved]", a.out)
    if a.quick:
        print("[quick] 以上全部仅供脚本自检，不作为结论")


if __name__ == "__main__":
    main()
