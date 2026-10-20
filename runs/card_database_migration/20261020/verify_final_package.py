# -*- coding: utf-8 -*-
"""第七阶段：最终提交包验证。

1) tar 可读性与清单、大小、SHA-256
2) model.pt 是否对应指定 checkpoint（张量逐位比对 + net_config + global_step）
3) deck.csv 是否与指定 best_deck.csv 一致；EN_Card_Data.csv 是否为迁移后权威库
4) 是否混入不可分发的专有引擎二进制
5) 干净目录解包后：import main、首个决策、连续决策、60 张开局、真实对局（双座位）、
   would_ko 真实 Search API 路径
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile

ROOT = r"D:\pkmn-kaggle"
MIG = os.path.join(ROOT, "runs", "card_database_migration", "20261020")
TAR = os.path.join(ROOT, "submission_rl_transformer2_db_migrated.tar.gz")
SRC_CKPT = os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_999424.pt")
SRC_DECK = os.path.join(ROOT, "runs", "evaluation", "card_database_migration", "best_deck.csv")
SRC_LIB = os.path.join(ROOT, "EN_Card_Data.csv")
WORK = os.path.join(MIG, "final_pkg_verify")
PY = r"D:\ProgramFiles\deeplearning\python.exe"
SHIM = os.path.join(ROOT, ".engine-shim")

CHILD = r'''
import ctypes, json, os, random, sys
PKG = sys.argv[1]
LOADS, INITS = [], []
_orig = ctypes.cdll.LoadLibrary
def _load(name, *a, **k):
    lib = _orig(name, *a, **k)
    LOADS.append(os.path.basename(name))
    try:
        real = lib.GameInitialize
        lib.__dict__["GameInitialize"] = lambda *aa, **kk: (INITS.append(1), real(*aa, **kk))[1]
    except Exception:
        pass
    return lib
ctypes.cdll.LoadLibrary = _load
sys.path.insert(0, PKG)
import importlib.util
res = {"pkg": PKG}
if sys.argv[2] == "import_main":
    import sdk_cg.sim  # noqa: F401  # 引擎封装可由包内路径正常加载
    spec = importlib.util.spec_from_file_location("staged_main", os.path.join(PKG, "main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    import search_agent as sa
    WK = {"calls": 0, "nonempty": 0, "positive_ko": 0}
    orig = sa.would_ko_flags
    def w(*a, **k):
        r = orig(*a, **k)
        WK["calls"] += 1
        if r:
            WK["nonempty"] += 1
        for _i, v in (r or {}).items():
            if list(v)[0] > 0:
                WK["positive_ko"] += 1
        return r
    sa.would_ko_flags = w
    sa.annotate_would_ko.__globals__["would_ko_flags"] = w
    from sdk_cg import game
    deck = list(m.DECK)
    opp_deck = list(__import__("decks").DECKS[list(__import__("decks").DECKS)[1]])
    out = {"deck_len": len(deck), "games": [], "first_decision": None, "consecutive": None}
    rng = random.Random(20261020)
    for seat in (0, 1):
        obs, sd = game.battle_start(deck if seat == 0 else opp_deck,
                                   opp_deck if seat == 0 else deck)
        if obs is None:
            out["games"].append({"seat": seat, "accepted": False, "errorType": sd.errorType})
            continue
        steps = illegal = 0
        exc = []
        first = None
        chain = []
        while obs.get("select") is not None and steps < 250:
            acting = (obs.get("current") or {}).get("yourIndex")
            sel = obs["select"]
            n_opts = len(sel.get("option") or [])
            try:
                if acting == seat:
                    pick = m.agent(obs)
                    if first is None:
                        first = list(pick)
                    elif len(chain) < 5:
                        chain.append(list(pick))
                    if not isinstance(pick, list) or any((not isinstance(p, int)) or p < 0 or p >= n_opts for p in pick):
                        illegal += 1
                        pick = None
                else:
                    pick = None
                if pick is None:
                    opts = sel.get("option") or []
                    if not opts:
                        break
                    k = max(sel.get("minCount", 1) or 1, 1)
                    hi = max(k, min(sel.get("maxCount", 1) or 1, len(opts)))
                    pick = sorted(rng.sample(range(len(opts)), min(hi, len(opts))))
                obs = game.battle_select(list(pick))
            except Exception as e:
                exc.append(f"{type(e).__name__}: {e}")
                break
            steps += 1
        try:
            result = (obs.get("current") or {}).get("result")
        except Exception:
            result = None
        try:
            game.battle_finish()
        except Exception:
            pass
        out["games"].append({"seat": seat, "accepted": True, "steps": steps, "result": result,
                             "illegal_shape": illegal, "exceptions": exc[:3]})
        out["first_decision"] = out["first_decision"] or first
        out["consecutive"] = out["consecutive"] or chain
    out["would_ko"] = WK
    out["loads"] = LOADS
    out["game_initialize_calls"] = len(INITS)
    res.update(out)
print(json.dumps(res, ensure_ascii=False, default=str))
'''


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    import torch
    import numpy as np
    rep = {"tar": {"path": TAR, "exists": os.path.exists(TAR), "size_bytes": os.path.getsize(TAR),
                   "sha256": sha(TAR)}}
    with tarfile.open(TAR, "r:gz") as t:
        members = {m.name: m.size for m in t.getmembers()}
    rep["tar"]["members"] = members
    rep["tar"]["n_members"] = len(members)
    rep["tar"]["proprietary_binaries"] = [n for n in members
                                          if n.endswith((".dll", ".so", ".dylib"))]

    # 内容一致性（从 tar 内直接读取比对）
    with tarfile.open(TAR, "r:gz") as t:
        def read(name):
            return t.extractfile(name).read()
        rep["checks"] = {
            "deck_matches_best_deck": read("deck.csv") == open(SRC_DECK, "rb").read(),
            "library_matches_migrated_authority": read("EN_Card_Data.csv") == open(SRC_LIB, "rb").read(),
            "deck_lines": len([x for x in read("deck.csv").decode().splitlines() if x.strip()]),
            "sdk_cg_py_only": all(n.endswith(".py") or t.getmember(n).isdir()
                                  for n in members if n.startswith("sdk_cg/")),
        }
        open(os.path.join(MIG, "_tmp_model.pt"), "wb").write(read("model.pt"))
    # 两处加载对象都是本仓库自产产物（训练 checkpoint 与本次导出的 model.pt），
    # 需要读 net_config/global_step，故用 weights_only=False
    src = torch.load(SRC_CKPT, map_location="cpu", weights_only=False)
    pkg = torch.load(os.path.join(MIG, "_tmp_model.pt"), map_location="cpu", weights_only=False)
    rep["checks"]["model_pt_keys"] = sorted(pkg.keys())
    rep["checks"]["global_step_matches"] = pkg.get("global_step") == src.get("global_step")
    rep["checks"]["net_config_matches"] = (pkg.get("net_config") or {}) == (src.get("net_config") or {})
    ks = list(pkg["net"].keys())
    rep["checks"]["n_tensors"] = len(ks)
    rep["checks"]["tensors_bitwise_equal"] = all(
        bool(np.array_equal(pkg["net"][k].numpy(), src["net"][k].numpy())) for k in ks)
    rep["checks"]["first_tensor_shapes_equal"] = all(
        tuple(pkg["net"][k].shape) == tuple(src["net"][k].shape) for k in ks)
    os.remove(os.path.join(MIG, "_tmp_model.pt"))

    # 干净目录解包 + 运行
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    with tarfile.open(TAR, "r:gz") as t:
        t.extractall(WORK)
    for d in (os.path.join(WORK, "sdk_cg", "__pycache__"), os.path.join(WORK, "__pycache__")):
        shutil.rmtree(d, ignore_errors=True)
    child = os.path.join(MIG, "_pkg_child.py")
    open(child, "w", encoding="utf-8").write(CHILD)
    env = dict(os.environ, PYTHONPATH=SHIM, PYTHONUNBUFFERED="1")
    p = subprocess.run([PY, child, WORK, "import_main"], capture_output=True, text=True,
                       timeout=1800, env=env)
    rep["runtime"] = {"exit": p.returncode,
                      "stdout_json": (p.stdout or "").strip().splitlines()[-1:],
                      "stderr_tail": (p.stderr or "").strip().splitlines()[-10:]}
    if rep["runtime"]["stdout_json"]:
        try:
            rep["runtime"]["detail"] = json.loads(rep["runtime"]["stdout_json"][0])
        except Exception as e:                              # noqa: BLE001
            rep["runtime"]["parse_error"] = str(e)

    rep["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                       "numpy": np.__version__}
    with open(os.path.join(MIG, "final_package_verification.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "tar"}, ensure_ascii=False, indent=2)[:4000])
    print("TAR:", json.dumps(rep["tar"], ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
