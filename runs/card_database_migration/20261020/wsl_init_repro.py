#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""引擎初始化差异复现（WSL2，官方 libcg.so 1.33.0）。

两个变体各自独立子进程：
  old = 用 HEAD 版 sdk_cg/sim.py（无条件 GameInitialize）
  new = 用工作区 sdk_cg/sim.py（优先复用官方模块的句柄与唯一一次初始化）
子进程顺序：import 包内 sdk_cg.sim → 再 import 官方 kaggle_environments...cg.sim
→ 统计 LoadLibrary/GameInitialize → 跑一次真实 60 张开局。
"""
import json
import os
import shutil
import subprocess
import sys

MIG = "/mnt/d/pkmn-kaggle/runs/card_database_migration/20261020"
PYLIBS = "/home/swcmb/pkmn-linux-verify-20261020/pylibs"
BASE = "/home/swcmb/pkmn-migration-verify-20261020"
PKG_NEW = os.path.join(BASE, "pkg")
PKG_OLD = os.path.join(BASE, "pkg_old")

CHILD = '''
import ctypes, json, os, sys
PKG, PYLIBS = sys.argv[1], sys.argv[2]
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
sys.path.insert(0, PYLIBS)
sys.path.insert(0, PKG)
import sdk_cg.sim as ours                      # 提交包内的引擎封装（变体决定其行为）
from kaggle_environments.envs.cabt.cg import sim as official   # harness 侧官方模块
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


def build_old_sim():
    """把工作区 sim.py 的 _load_lib 换回修复前版本（只认 sys.modules 的单向保护），
    其余声明保持一致，确保差异只来自句柄复用逻辑。"""
    cur = open("/mnt/d/pkmn-kaggle/sdk_cg/sim.py", encoding="utf-8").read()
    start = cur.index("def _load_lib():")
    end = cur.index("lib, _loaded_here = _load_lib()")
    old_block = '''def _load_lib():
    # 修复前：只在官方模块已被导入时复用句柄；自己先加载则不管官方模块稍后是否二次初始化。
    pre = sys.modules.get("kaggle_environments.envs.cabt.cg.sim")
    if pre is not None and getattr(pre, "lib", None) is not None:
        return pre.lib, False
    for p in _engine_candidates():
        if os.path.exists(p):
            return ctypes.cdll.LoadLibrary(p), True
    raise FileNotFoundError("cabt engine binary not found")


'''
    return cur[:start] + old_block + cur[end:]


def main():
    child_path = "/tmp/pkmn_init_child.py"
    with open(child_path, "w", encoding="utf-8") as f:
        f.write(CHILD)
    if os.path.isdir(PKG_OLD):
        shutil.rmtree(PKG_OLD)
    shutil.copytree(PKG_NEW, PKG_OLD)
    with open(os.path.join(PKG_OLD, "sdk_cg", "sim.py"), "w", encoding="utf-8") as f:
        f.write(build_old_sim())
    shutil.rmtree(os.path.join(PKG_OLD, "sdk_cg", "__pycache__"), ignore_errors=True)
    shutil.rmtree(os.path.join(PKG_OLD, "__pycache__"), ignore_errors=True)
    env = dict(os.environ, PYTHONPATH=PYLIBS)
    results = []
    for variant, pkg in (("old", PKG_OLD), ("new", PKG_NEW)):
        p = subprocess.run([sys.executable, child_path, pkg, PYLIBS],
                           capture_output=True, text=True, timeout=600, env=env)
        r = {"variant": variant, "pkg": pkg, "exit": p.returncode,
             "abort": p.returncode < 0,
             "stdout": (p.stdout or "").strip()[-500:],
             "stderr_tail": (p.stderr or "").strip().splitlines()[-6:]}
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    with open(os.path.join(MIG, "init_repro_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
