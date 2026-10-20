# -*- coding: utf-8 -*-
"""引擎初始化机制复现（Windows / 官方 cg.dll，1.33.0）。

old = 修复前的 _load_lib（只在官方模块已导入时复用句柄）
new = 工作区 _load_lib（优先让官方模块完成唯一一次初始化并复用句柄）
子进程顺序：import 变体 sim → import 官方 kaggle_environments...cg.sim → 统计初始化 → 真实开局。
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = r"D:\pkmn-kaggle"
MIG = os.path.join(ROOT, "runs", "card_database_migration", "20261020")
WORK = os.path.join(MIG, "init_repro_work")
PY = r"D:\ProgramFiles\deeplearning\python.exe"
SHIM = os.path.join(ROOT, ".engine-shim")

CHILD = r'''
import ctypes, json, os, sys
PKG = sys.argv[1]
LOADS, INITS = [], []
_orig = ctypes.cdll.LoadLibrary
def _load(name, *a, **k):
    lib = _orig(name, *a, **k)
    LOADS.append(os.path.basename(name))
    try:
        real = lib.GameInitialize
        lib.__dict__["GameInitialize"] = lambda *aa, **kk: (INITS.append(os.path.basename(name)), real(*aa, **kk))[1]
    except Exception:
        pass
    return lib
ctypes.cdll.LoadLibrary = _load
sys.path.insert(0, PKG)
import sdk_cg.sim as ours
from kaggle_environments.envs.cabt.cg import sim as official
import sdk_cg.game as g
deck = [int(x) for x in open(os.path.join(PKG, "deck.csv")) if x.strip()]
obs, sd = g.battle_start(deck, deck)
res = {"loads": LOADS, "game_initialize_calls": len(INITS),
       "same_handle": ours.lib is official.lib,
       "battle_accepted": obs is not None, "errorType": sd.errorType}
if obs is not None:
    g.battle_finish()
print(json.dumps(res))
'''


def old_sim_source():
    cur = open(os.path.join(ROOT, "sdk_cg", "sim.py"), encoding="utf-8").read()
    start = cur.index("def _load_lib():")
    end = cur.index("lib, _loaded_here = _load_lib()")
    block = '''def _load_lib():
    # 修复前：仅当官方模块已被导入时才复用句柄；自己先加载则不管官方模块稍后的二次初始化。
    pre = sys.modules.get("kaggle_environments.envs.cabt.cg.sim")
    if pre is not None and getattr(pre, "lib", None) is not None:
        return pre.lib, False
    for p in _engine_candidates():
        if os.path.exists(p):
            return ctypes.cdll.LoadLibrary(p), True
    raise FileNotFoundError("cabt engine binary not found")


'''
    return cur[:start] + block + cur[end:]


def build(variant, pkgdir):
    if os.path.isdir(pkgdir):
        shutil.rmtree(pkgdir)
    os.makedirs(os.path.join(pkgdir, "sdk_cg"))
    for f in ("__init__.py", "api.py", "game.py", "utils.py"):
        shutil.copy(os.path.join(ROOT, "sdk_cg", f), os.path.join(pkgdir, "sdk_cg", f))
    sim = old_sim_source() if variant == "old" else open(
        os.path.join(ROOT, "sdk_cg", "sim.py"), encoding="utf-8").read()
    with open(os.path.join(pkgdir, "sdk_cg", "sim.py"), "w", encoding="utf-8") as f:
        f.write(sim)
    shutil.copy(os.path.join(ROOT, "runs", "evaluation", "card_database_migration", "best_deck.csv"),
                os.path.join(pkgdir, "deck.csv"))
    # 官方包路径靠 PYTHONPATH 指向 .engine-shim；子进程继承父环境
    return sim


def main():
    child = os.path.join(MIG, "_init_child.py")
    with open(child, "w", encoding="utf-8") as f:
        f.write(CHILD)
    results = []
    for variant in ("old", "new"):
        pkg = os.path.join(WORK, variant)
        src = build(variant, pkg)
        open(os.path.join(pkg, "sdk_cg", "sim_variant.py"), "w", encoding="utf-8").write(src)
        env = dict(os.environ, PYTHONPATH=SHIM + ";" + pkg)
        p = subprocess.run([PY, child, pkg], capture_output=True, text=True, timeout=600, env=env)
        r = {"variant": variant, "exit": p.returncode, "aborted": p.returncode < 0,
             "stdout": (p.stdout or "").strip()[-400:],
             "stderr_tail": (p.stderr or "").strip().splitlines()[-6:]}
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    with open(os.path.join(MIG, "init_repro_results_windows.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
