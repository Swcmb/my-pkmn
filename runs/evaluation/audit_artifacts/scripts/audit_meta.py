# 一次性审计脚本：核对 ckpt 真实步数/net_config、已有 JSON 的 meta 与 label、提交包与训练侧代码一致性
import hashlib, json, os, sys, torch

ROOT = r"D:\pkmn-kaggle"
sys.path.insert(0, ROOT)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]

print("== checkpoints ==")
for n in ("ckpt_999424.pt", "ckpt_500736.pt"):
    p = os.path.join(ROOT, "runs", "train-20m-e16-1m", n)
    ck = torch.load(p, map_location="cpu")
    print(n, "global_step=", ck.get("global_step"), "net_config=", ck.get("net_config"))

print("\n== eval/ctl JSONs: meta vs label ==")
evdir = os.path.join(ROOT, "runs", "evaluation")
for f in sorted(os.listdir(evdir)):
    if not f.endswith(".json"):
        continue
    try:
        d = json.load(open(os.path.join(evdir, f), encoding="utf-8"))
    except Exception as e:
        print(f, "PARSE-ERR", e); continue
    res = d.get("results")
    if not isinstance(res, list):
        continue
    top = {k: d.get(k) for k in ("cand_ckpt", "cand_step", "mode", "games", "seed", "turn_cap")}
    for r in res:
        m = r.get("meta", {})
        print("%-32s label=%-28s winrate=%s seat_wr=%s/%s ckpt_meta=%s step=%s init_seed=%s top=%s" % (
            f, r.get("label"), r.get("winrate"),
            (r.get("seat_balance") or {}).get("wr_seat0"),
            (r.get("seat_balance") or {}).get("wr_seat1"),
            m.get("ckpt", "-"), m.get("step", "-"), m.get("init_seed", "-"),
            {k: top.get(k) for k in ("cand_ckpt", "cand_step", "mode", "seed")}))

print("\n== submission vs rl source identity ==")
pairs = [
    ("submission_rl_transformer2\\policy.py", "rl\\policy.py"),
    ("submission_rl_transformer2\\encoding.py", "rl\\encoding.py"),
    ("submission_rl_transformer2\\search_agent.py", "rl\\search_agent.py"),
    ("submission_rl_transformer2\\card_features.py", "rl\\card_features.py"),
    ("submission_rl_transformer2\\effect_data.py", "rl\\effect_data.py"),
    ("submission_rl_transformer2\\attack_data.py", "rl\\attack_data.py"),
    ("submission_rl_transformer2\\buff_data.py", "rl\\buff_data.py"),
    ("submission_rl_transformer2\\enc_constants.py", "rl\\enc_constants.py"),
    ("submission_rl_transformer2\\decks.py", "rl\\decks.py"),
]
for a, b in pairs:
    pa, pb = os.path.join(ROOT, a), os.path.join(ROOT, b)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print("MISSING", a if not os.path.exists(pa) else b); continue
    ha, hb = sha(pa), sha(pb)
    print(("SAME " if ha == hb else "DIFF ") + "%-46s %s vs %s" % ("", a, b), ha, hb)

print("\n== submission sdk_cg contents / model.pt meta ==")
for root_, dirs, files in os.walk(os.path.join(ROOT, "submission_rl_transformer2", "sdk_cg")):
    for fn in files:
        print(" ", os.path.relpath(os.path.join(root_, fn), ROOT))
ck = torch.load(os.path.join(ROOT, "submission_rl_transformer2", "model.pt"), map_location="cpu")
print("model.pt global_step=", ck.get("global_step"), "net_config=", ck.get("net_config"))

print("\n== deck.csv: submission vs agent ==")
for p in (os.path.join(ROOT, "submission_rl_transformer2", "deck.csv"),
          os.path.join(ROOT, "agent", "deck.csv")):
    s = open(p).read().split()
    print(os.path.relpath(p, ROOT), "len=", len(s), "sha=", sha(p))

print("\n== kaggle_environments install probe ==")
import importlib
try:
    m = importlib.import_module("kaggle_environments")
    print("real pkg at", os.path.dirname(m.__file__))
except Exception as e:
    print("import fail:", type(e).__name__, e)
