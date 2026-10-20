#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4 编排：在 WSL2 内准备干净解包目录、叠加迁移内容、逐用例独立进程运行并汇总。

由 Windows 侧调用：wsl -d Ubuntu-24.04 -- python3 //mnt/d/pkmn-kaggle/runs/card_database_migration/20261020/wsl_stage4_run.py
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile

SRC = "/mnt/d/pkmn-kaggle"
MIG = os.path.join(SRC, "runs", "card_database_migration", "20261020")
BASE = os.path.join(os.path.expanduser("~"), "pkmn-migration-verify-20261020")
PKG = os.path.join(BASE, "pkg")
PYLIBS = os.path.join(os.path.expanduser("~"), "pkmn-linux-verify-20261020", "pylibs")
TAR = os.path.join(SRC, "submission_rl_transformer2_dragapult_nodist.tar.gz")
CASES = ["import_clean", "env_first", "by_path", "deck_request", "games", "would_ko"]


def sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def stage():
    if os.path.isdir(PKG):
        shutil.rmtree(PKG)
    os.makedirs(PKG)
    with tarfile.open(TAR, "r:gz") as t:
        t.extractall(PKG)
    shutil.copy(os.path.join(SRC, "EN_Card_Data.csv"), os.path.join(PKG, "EN_Card_Data.csv"))
    for f in os.listdir(os.path.join(SRC, "sdk_cg")):
        if f.endswith(".py"):
            shutil.copy(os.path.join(SRC, "sdk_cg", f), os.path.join(PKG, "sdk_cg", f))
    shutil.copy(os.path.join(SRC, "runs", "evaluation", "deck_search", "dragapult_deck.csv"),
                os.path.join(PKG, "deck.csv"))
    for d in (os.path.join(PKG, "sdk_cg", "__pycache__"), os.path.join(PKG, "__pycache__")):
        if os.path.isdir(d):
            shutil.rmtree(d)
    return {"tar": TAR, "tar_sha256": sha(TAR),
            "staged": {n: sha(os.path.join(PKG, n))
                       for n in ("EN_Card_Data.csv", "deck.csv", "sdk_cg/sim.py", "main.py", "model.pt")},
            "migrated_library_sha256": sha(os.path.join(SRC, "EN_Card_Data.csv"))}


def run_case(case):
    env = dict(os.environ, PYTHONPATH=PYLIBS, PKMN_TURN_CAP="300")
    p = subprocess.run([sys.executable, os.path.join(MIG, "wsl_stage4_cases.py"), PKG, PYLIBS, case],
                       cwd=PKG, env=env, capture_output=True, text=True, timeout=1800)
    tail = (p.stdout or "").strip().splitlines()
    return {"case": case, "exit": p.returncode, "signal": (-p.returncode if p.returncode < 0 else None),
            "stdout_last_json": tail[-1] if tail else "",
            "stderr_tail": "\n".join((p.stderr or "").strip().splitlines()[-12:])}


def main():
    info = {"python": sys.version.split()[0], "staging": stage()}
    res = []
    for c in CASES:
        r = run_case(c)
        res.append(r)
        print(f"[case] {c}: exit={r['exit']}", flush=True)
        print("   " + (r["stdout_last_json"][:900] or r["stderr_tail"][:900]), flush=True)
    info["cases"] = res
    out = os.path.join(MIG, "wsl_stage4_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("[saved]", out, flush=True)


if __name__ == "__main__":
    main()
