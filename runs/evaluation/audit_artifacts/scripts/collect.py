# 汇总补充：draw 明细、合并 CI、复制证据产物到 runs/evaluation/audit_artifacts/
import json, math, os, shutil

EV = r"D:\pkmn-kaggle\runs\evaluation"
TMP = r"C:\Users\Swcmb\AppData\Local\Temp\opencode"
ART = os.path.join(EV, "audit_artifacts")
os.makedirs(ART, exist_ok=True)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    s = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (c - s) / d, (c + s) / d


print("cand vs baseline 合并 120/200: %.4f CI %s" % (120 / 200, wilson(120, 200)))
print("cand vs random   合并  88/200: %.4f CI %s" % (88 / 200, wilson(88, 200)))
d = json.load(open(os.path.join(EV, "ctl2_cand_vs_untrained_100_seedB.json"), encoding="utf-8"))
print("seedB untrained draw_breakdown:", d["results"][0]["draw_breakdown"],
      "seat:", d["results"][0]["seat_balance"])
d = json.load(open(os.path.join(EV, "ctl2_cand_selfmirror_100.json"), encoding="utf-8"))
print("cand selfmirror seat:", d["results"][0]["seat_balance"])
d = json.load(open(os.path.join(EV, "ctl2_base_selfmirror_100.json"), encoding="utf-8"))
print("base selfmirror seat:", d["results"][0]["seat_balance"])

copies = ["subsmoke2_agent.json", "subsmoke2_rr.json", "parity.json", "ctl2.log",
          "untrained_sweep.log", "audit_subs.log", "sm_agent.log", "sm_rr.log",
          "subsmoke2_6.log", "deck_probe_out.txt", "tarcheck.log", "table.md"]
# 缺失的源文件先跳过；table.md/deck_probe_out/tarcheck.log 由本步骤之后的命令补
for c in copies:
    src = os.path.join(TMP, c)
    if not os.path.exists(src):
        src = os.path.join(TMP, "subsmoke2", c)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(ART, c))
        print("copied", c, "<-", src)
print("artifact dir:", ART)
