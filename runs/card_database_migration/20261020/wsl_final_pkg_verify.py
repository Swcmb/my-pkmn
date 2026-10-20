#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终提交包的 WSL2 验证：干净解包 submission_rl_transformer2_db_migrated.tar.gz，
不做任何叠加（包内容即提交内容），跑 wsl_stage4_cases.py 的全部用例。
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

SRC = "/mnt/d/pkmn-kaggle"
MIG = os.path.join(SRC, "runs", "card_database_migration", "20261020")
BASE = os.path.join(os.path.expanduser("~"), "pkmn-finalpkg-verify-20261020")
PKG = os.path.join(BASE, "pkg")
PYLIBS = os.environ.get("PYLIBS", "/home/ubuntu/pkmn-venv/lib/python3.12/site-packages")
TAR = os.path.join(SRC, "submission_rl_transformer2_db_migrated.tar.gz")
CASES = ["import_clean", "env_first", "by_path", "deck_request", "games", "would_ko"]


def main():
    if os.path.isdir(PKG):
        shutil.rmtree(PKG)
    os.makedirs(PKG)
    with tarfile.open(TAR, "r:gz") as t:
        t.extractall(PKG)
    shutil.rmtree(os.path.join(PKG, "sdk_cg", "__pycache__"), ignore_errors=True)
    shutil.rmtree(os.path.join(PKG, "__pycache__"), ignore_errors=True)
    info = {"tar": TAR,
            "tar_sha256": hashlib.sha256(open(TAR, "rb").read()).hexdigest(),
            "staged": {n: hashlib.sha256(open(os.path.join(PKG, n), "rb").read()).hexdigest()
                       for n in ("EN_Card_Data.csv", "deck.csv", "model.pt", "main.py", "sdk_cg/sim.py")},
            "no_binaries": [n for n in os.listdir(PKG) if n.endswith((".dll", ".so", ".dylib"))]}
    results = []
    for c in CASES:
        env = dict(os.environ, PYTHONPATH=PYLIBS, PKMN_TURN_CAP="300")
        p = subprocess.run([sys.executable, os.path.join(MIG, "wsl_stage4_cases.py"), PKG, PYLIBS, c],
                           cwd=PKG, env=env, capture_output=True, text=True, timeout=1800)
        last = (p.stdout or "").strip().splitlines()
        r = {"case": c, "exit": p.returncode, "json": last[-1] if last else "",
             "stderr": (p.stderr or "").strip().splitlines()[-6:]}
        results.append(r)
        print(f"[case] {c} exit={r['exit']}", flush=True)
        print("   " + (r["json"][:900] or "\n".join(r["stderr"])[:900]), flush=True)
    info["cases"] = results
    with open(os.path.join(MIG, "final_pkg_wsl_results.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("[saved] final_pkg_wsl_results.json", flush=True)


if __name__ == "__main__":
    main()
