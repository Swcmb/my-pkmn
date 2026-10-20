#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4b：官方 harness 口径的真实对局 + 引擎初始化次数测量。

A. 干净解释器 import main，统计 LoadLibrary / GameInitialize 次数（要求各 1 次）
B. 官方 kaggle_environments 先 make('cabt')，再 import main，同样统计
C. 用官方 harness 跑真实对局：候选 agent 分别坐 0/1 号位，各 3 局，对手为随机合法策略

由 Windows 侧调用：
  wsl -d Ubuntu-24.04 -- python3 //mnt/d/pkmn-kaggle/runs/card_database_migration/20261020/wsl_stage4b_harness.py <case>
"""
import ctypes
import json
import os
import random
import sys

PKG = os.environ.get("VERIFY_PKG", "/home/swcmb/pkmn-migration-verify-20261020/pkg")
PYLIBS = os.environ.get("VERIFY_PYLIBS", "/home/swcmb/pkmn-linux-verify-20261020/pylibs")
CASE = sys.argv[1] if len(sys.argv) > 1 else "init_clean"
sys.path.insert(0, PYLIBS)
sys.path.insert(0, PKG)

LOADS, INITS = [], []
_orig_load = ctypes.cdll.LoadLibrary


def _patched_load(name, *a, **kw):
    lib = _orig_load(name, *a, **kw)
    LOADS.append(os.path.basename(name))

    def _counted(*args, **kwargs):
        INITS.append(os.path.basename(name))
        return lib.__dict__.get("_gi_real", getattr(lib, "GameInitialize"))(*args, **kwargs)
    try:
        real = lib.GameInitialize                 # 触发 ctypes 生成并缓存函数指针
        lib.__dict__["_gi_real"] = real
        lib.__dict__["GameInitialize"] = _counted
    except Exception:                              # noqa: BLE001
        pass
    return lib


ctypes.cdll.LoadLibrary = _patched_load
out = {"case": CASE, "pkg": PKG}


def read_deck():
    return [int(x) for x in open(os.path.join(PKG, "deck.csv")) if x.strip()]


def random_agent_factory(opp_deck):
    """随机合法策略，接口与 main.agent 一致：select 为 None 时交牌组。"""
    def fn(obs):
        if isinstance(obs, str):
            import json as _j
            obs = _j.loads(obs)
        sel = obs.get("select") if isinstance(obs, dict) else None
        if sel is None:
            return list(opp_deck)
        opts = sel.get("option") or []
        if not opts:
            return []
        k = max(sel.get("minCount", 1) or 1, 1)
        hi = max(k, min(sel.get("maxCount", 1) or 1, len(opts)))
        return sorted(random.sample(range(len(opts)), min(hi, len(opts))))
    return fn


def case_init(order):
    if order == "env_first":
        import kaggle_environments                       # noqa: F401
        from kaggle_environments import make
        env = make("cabt")
        out["env_created"] = type(env).__name__
        import main                                      # noqa: F401
    else:
        import main                                      # noqa: F401
    import sdk_cg.sim as s
    out["loads"] = LOADS
    out["game_initialize_calls"] = len(INITS)
    out["official_sim_imported"] = "kaggle_environments.envs.cabt.cg.sim" in sys.modules
    out["same_handle"] = s.lib.__class__.__name__
    out["deck_len"] = len(main.DECK)
    out["ok"] = True


def case_harness_games(n_each=3, turn_cap=400):
    import main
    from kaggle_environments import make
    results = []
    decks_alt = read_deck()
    for seat in (0, 1):
        for k in range(n_each):
            env = make("cabt", debug=False)
            opp = random_agent_factory(decks_alt)
            order = [main.agent, opp] if seat == 0 else [opp, main.agent]
            try:
                env.run(order)
                err = None
            except Exception as e:                        # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            try:
                rewards = [d.get("reward") for d in env.done] if env.done else None
            except Exception:                             # noqa: BLE001
                rewards = None
            n_steps = len(env.steps) if getattr(env, "steps", None) else 0
            info = {}
            try:
                last = env.steps[-1] if env.steps else {}
                info = {a: last.get(a, {}).get("status") for a in ("0", "1")}
            except Exception:                             # noqa: BLE001
                pass
            results.append({"agent_seat": seat, "game": k, "error": err,
                            "rewards": rewards, "steps": n_steps, "status": info})
            print(f"[harness] seat={seat} g={k} err={err} rewards={rewards} steps={n_steps}", flush=True)
    out["games"] = results
    out["loads"] = LOADS
    out["game_initialize_calls"] = len(INITS)
    out["errors"] = [r["error"] for r in results if r["error"]]


try:
    if CASE == "init_clean":
        case_init("clean")
    elif CASE == "init_env_first":
        case_init("env_first")
    elif CASE == "harness_games":
        case_harness_games()
    else:
        raise SystemExit(f"unknown case {CASE}")
except Exception as e:                                    # noqa: BLE001
    out["fatal"] = f"{type(e).__name__}: {e}"
    import traceback
    out["traceback"] = traceback.format_exc()[-1500:]
out["loads"] = out.get("loads", LOADS)
out["game_initialize_calls"] = out.get("game_initialize_calls", len(INITS))
print(json.dumps(out, ensure_ascii=False, default=str))
