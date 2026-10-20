"""干净目录验证候选提交包：解包 -> import main -> 牌组请求 -> 多局决策 -> would_ko 写入。

候选包不内嵌引擎二进制；sdk_cg/sim.py 会解析官方 kaggle_environments 安装。
本脚本用 PYTHONPATH 注入本机官方引擎 shim（Windows cg.dll），全程真实引擎驱动。

  PYTHONPATH=.;<shim> python runs/evaluation/package_check.py \
      --tar submission_rl_transformer2_linux_nodist.tar.gz
"""
import argparse
import json
import os
import random
import shutil
import sys
import tarfile
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=os.path.join(
        ROOT, "submission_rl_transformer2_linux_nodist.tar.gz"))
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "evaluation",
                                                  "package_audit_run.json"))
    a = ap.parse_args()
    out_path = os.path.abspath(a.out)          # chdir 到解包目录后仍写回原路径

    work = tempfile.mkdtemp(prefix="pkmn_pkgcheck_")
    pkg = os.path.join(work, "agent")
    os.makedirs(pkg)
    with tarfile.open(a.tar) as tf:
        members = tf.getnames()
        tf.extractall(pkg)
    binaries = [n for n in members if n.endswith((".so", ".dll", ".dylib"))]

    result = {"tar": os.path.abspath(a.tar), "members": len(members),
              "binaries": binaries, "extract_dir": pkg, "games": [],
              "errors": []}
    sys.path.insert(0, pkg)
    os.chdir(pkg)

    import main as agent_main                                        # noqa: E402
    result["import_main"] = True
    result["wk_on"] = bool(agent_main._WK_ON)
    deck = agent_main.agent({"select": None})
    result["deck_request"] = {"n": len(deck), "all_int": all(isinstance(x, int) for x in deck)}

    import sdk_cg.game as game                                       # noqa: E402
    from sdk_cg.sim import Battle                                    # noqa: E402

    rng = random.Random(20261019)
    for gi in range(a.games):
        obs, sd = game.battle_start(deck, deck)
        if obs is None:
            result["errors"].append(f"game {gi}: battle_start failed p={sd.errorPlayer} "
                                    f"t={sd.errorType}")
            Battle.battle_ptr = None
            continue
        agent_main.agent({"select": None})                           # reset per-game tracker
        main_seat = gi % 2                                           # 两个座位都覆盖
        g = {"main_seat": main_seat, "decisions_main": 0, "decisions_opp": 0,
             "wk_decisions": 0, "wk_cells": 0, "bad_actions": 0,
             "attack_opts": 0, "select_types": {},
             "first_decision_ok": False, "terminal": None, "won": None, "error": None}
        try:
            for _ in range(4000):
                sel = obs.get("select")
                if sel is None:
                    break
                opts = sel.get("option") or []
                me = int((obs.get("current") or {}).get("yourIndex", 0))
                if me == main_seat:
                    act = agent_main.agent(obs)                      # 真实推理 + would_ko 写回
                    g["decisions_main"] += 1
                    t = str(sel.get("type"))
                    g["select_types"][t] = g["select_types"].get(t, 0) + 1
                    g["attack_opts"] += sum(1 for o in opts if o.get("attackId") is not None)
                    if g["decisions_main"] == 1:
                        g["first_decision_ok"] = isinstance(act, list)
                    if any("would_ko" in o for o in opts):
                        g["wk_decisions"] += 1
                        g["wk_cells"] += sum(1 for o in opts if "would_ko" in o)
                else:
                    k = min(int(sel.get("maxCount", 1) or 1), len(opts))
                    act = rng.sample(range(len(opts)), k)
                    g["decisions_opp"] += 1
                if (not isinstance(act, list) or any(not isinstance(x, int) for x in act)
                        or any(x < 0 or x >= len(opts) for x in act)
                        or len(set(act)) != len(act)):
                    g["bad_actions"] += 1
                    break
                obs = game.battle_select(act)
                cur = obs.get("current") or {}
                if int(cur.get("result", -1)) >= 0:
                    g["terminal"] = int(cur["result"])
                    g["won"] = int(cur["result"]) == main_seat
                    break
        except Exception as e:                                        # noqa: BLE001
            g["error"] = f"{type(e).__name__}: {e}"
            result["errors"].append(f"game {gi}: {g['error']}")
            traceback.print_exc()
        finally:
            try:
                game.battle_finish()
            except Exception:
                pass
            Battle.battle_ptr = None
        result["games"].append(g)
        print(f"[game {gi}] {g}", flush=True)

    result["ok"] = (result.get("import_main") and result["deck_request"]["n"] == 60
                    and result["deck_request"]["all_int"] and not result["errors"]
                    and all(x["first_decision_ok"] and x["bad_actions"] == 0 and x["error"] is None
                            for x in result["games"]))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[package_check] ok={result['ok']} binaries={binaries}")
    print(f"[out] {out_path}")
    shutil.rmtree(work, ignore_errors=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
