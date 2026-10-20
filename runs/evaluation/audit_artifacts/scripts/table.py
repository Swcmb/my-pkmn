# 汇总 runs/evaluation 下全部对局 JSON 为 Markdown 表（口径一律取 meta，不取 label）
import glob, json, os

EV = r"D:\pkmn-kaggle\runs\evaluation"
ORDER = ["eval_baseline_100.json", "eval_random_100.json", "eval_untrained_100.json",
         "ctl2_base_vs_random_100.json", "ctl2_base_vs_untrained_100.json",
         "ctl2_cand_selfmirror_100.json", "ctl2_base_selfmirror_100.json",
         "ctl2_cand_vs_untrained_100_seedB.json", "ctl2_cand_vs_baseline_100_seedB.json",
         "ctl2_cand_vs_random_100_seedB.json",
         "ctl_base_random_30.json", "ctl_base_untrained_30.json",
         "ctl_base_selfmirror_30.json", "ctl_selfmirror_30.json", "smoke_random.json"]
sweep = sorted(glob.glob(os.path.join(EV, "wk_untrained_*.json")))
rows = []
for f in ORDER + [os.path.basename(p) for p in sweep]:
    p = os.path.join(EV, f)
    if not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8"))
    for r in d.get("results", []):
        m = r.get("meta") or {}
        sb = r.get("seat_balance") or {}
        il = r.get("illegal") or {}
        ln = r.get("length") or {}
        db = r.get("draw_breakdown") or {}
        pers = os.path.basename(d.get("cand_ckpt", "?"))
        opp = m.get("ckpt") or ("random_legal" if r["opponent"] == "random"
                                else "untrained(init_seed=%s)" % m.get("init_seed"))
        if isinstance(opp, str) and opp.endswith(".pt"):
            opp = "%s@%s" % (os.path.basename(opp), m.get("step"))
        ci = r.get("winrate_ci95") or [None, None]
        rows.append(dict(
            file=f, pers=pers, pstep=d.get("cand_step"), opp=opp, g=r.get("games"),
            seed=d.get("seed"), mode=d.get("mode"), w=r.get("wins"), l=r.get("losses"),
            dr=r.get("draws"), tc=db.get("turn_cap"), er=r.get("errs"),
            wr=r.get("winrate"), lo=ci[0], hi=ci[1],
            w0=sb.get("wr_seat0"), w1=sb.get("wr_seat1"),
            n0=sb.get("games_seat0"), n1=sb.get("games_seat1"),
            ill=il.get("total"), illc=il.get("clamp_candidate"), illf=il.get("forfeit_candidate"),
            ap=ln.get("mean_applies"), tn=ln.get("mean_turn"),
            el=r.get("elapsed_s"), gps=r.get("games_per_s")))


def fmt(x, nd=3):
    return "-" if x is None else (("%." + str(nd) + "f") % x)


print("| 视角 ckpt | 对手 | 局 | seed | mode | 胜/负/平(截断) | 异常 | 非法(候选) | 胜率 | 95%CI | seat0胜率(n) | seat1胜率(n) | 平均apply/turn | 耗时s(g/s) | 文件 |")
print("|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| %s@%s | %s | %s | %s | %s | %s/%s/%s(%s) | %s | %s(%s/%s) | %s | [%s, %s] | %s (%s) | %s (%s) | %s / %s | %s (%s) | %s |" % (
        r["pers"].replace("ckpt_", ""), r["pstep"], r["opp"], r["g"], r["seed"], r["mode"],
        r["w"], r["l"], r["dr"], r["tc"], r["er"], r["ill"], r["illc"], r["illf"],
        fmt(r["wr"], 4), fmt(r["lo"], 4), fmt(r["hi"], 4),
        fmt(r["w0"], 3), r["n0"], fmt(r["w1"], 3), r["n1"],
        fmt(r["ap"], 1), fmt(r["tn"], 1), fmt(r["el"], 1), fmt(r["gps"], 2), r["file"]))
