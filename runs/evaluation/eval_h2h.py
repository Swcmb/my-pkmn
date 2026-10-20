# SPDX-License-Identifier: GPL-3.0-or-later
"""ckpt_999424 的独立对战评估驱动（任务书 Prompt/a.md）。

为什么不用 scripts/h2h_ckpt.py 直接跑：
  rl/gates.py:235 调用 `_field_candidates(...)`，该符号全仓库无定义（仓库为单一 squash 提交
  6e7829e，无历史可寻），因此 field_winrate_fast 在任何分支下都会 NameError。本脚本照搬
  field_winrate_fast 的双组批量前向骨架，但自带 candidate 序列（stride 7 在池上轮转），
  并额外统计 field_winrate_fast 不提供的：非法动作率、平均对局长度、95% CI。

与对拍口径的差异（报告须标注）：本脚本把 step/spawn 异常记入 errs 单列，不计入 draws；
field_winrate_fast 的 errs 同时计入 draws。wins/decided 两者一致。

引擎来源（本机仓库缺陷的绕行，报告须标注）：仓库 sdk_cg/ 只带 5 个 .py、缺随附的
cg.dll，sdk_cg/sim.py 在 import 期 LoadLibrary 即抛 FileNotFoundError，且失败不被
sys.modules 缓存 → 每个走到 `from sdk_cg import api`（would_ko 标注路径）的决策都重抛。
main() 起始调用 _install_sdk_cg_shim_alias()，把平台真引擎模块注入为 sdk_cg.sim，
使 `from .sim import lib` 复用同一 dll 句柄（同进程绝不二次 LoadLibrary）。

用法：
  PYTHONPATH=. PKMN_TURN_CAP=200 python runs/evaluation/eval_h2h.py --opponent all --games 100
"""

import argparse
import ctypes
import json
import math
import os
import sys
import time

# PKMN_TURN_CAP 必须在 import rl 之前设置：rl/env_selfplay.py 在模块级读入 _TURN_CAP。
_TURN_CAP = int(os.environ.get("PKMN_TURN_CAP", "0") or "0")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

torch.set_grad_enabled(False)

from rl.card_features import get_card_table                          # noqa: E402
from rl.env_selfplay import TwoSidedSelfPlayEnv                      # noqa: E402
from rl.encoding import TokenEncoder                                  # noqa: E402
from rl.policy import bucketed_opt_len_np, build_token_net            # noqa: E402

CAND_CKPT = os.path.join("runs", "train-20m-e16-1m", "ckpt_999424.pt")
BASE_CKPT = os.path.join("runs", "train-20m-e16-1m", "ckpt_500736.pt")
POOL_JSON = os.path.join("runs", "evaluation", "pool_top15.json")
OUT_DIR = os.path.join("runs", "evaluation")


class _Counted(TwoSidedSelfPlayEnv):
    """固定一副有序牌组（seat0=self.decks[0], seat1=self.decks[1]），并统计真实 _apply 次数。

    step() 中 _apply 返回 True 时会跳过 `_steps += 1` 与 turn-cap 检查，因此外层观测不到
    「这一回合真正提交了几次」；唯一准确的口径是在 _apply 内部计数。
    native_encode 由构造参数传入（本机 cg.dll 缺 GetBinaryObs 导出，统一 False）。
    """

    def __init__(self, *a, **kw):
        self._apply_calls = 0
        super().__init__(*a, **kw)

    def _sample_decks(self):
        self._deck_idx = (0, 1)
        return self.decks[0], self.decks[1]

    def _apply(self, seat, indices):
        self._apply_calls += 1
        return super()._apply(seat, indices)

    def reset(self, seed=None):
        self._apply_calls = 0
        return super().reset(seed)


def wilson(k, n, z=1.959963984540054):
    """Wilson score 区间（胜率二项比例的 95% CI，小样本下比正态近似稳健）。"""
    if n <= 0:
        return [None, None]
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    s = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return [max(0.0, (c - s) / d), min(1.0, (c + s) / d)]


def load_net(path, ct, dev):
    ck = torch.load(path, map_location="cpu")
    net = build_token_net(ct, ck["net_config"])
    net.load_state_dict(ck["net"])
    net.eval().to(dev)
    return net, int(ck.get("global_step", -1)), ck["net_config"]


def build_opponent(kind, ct, dev, seed, cand_path):
    """返回 (net_or_None, label, meta)。net=None 表示随机合法策略。"""
    if kind == "random":
        return None, "random_legal", {}
    if kind == "untrained":
        _net, step, nc = load_net(cand_path, ct, dev)
        torch.manual_seed(seed)                    # 同 net_config 的随机初始化（真 untrained）
        net = build_token_net(ct, nc)
        net.eval().to(dev)
        return net, "untrained_same_config", {
            "net_config": nc, "init_seed": seed,
            "shape_from": cand_path, "shape_step": step}
    if kind == "baseline":
        net, step, nc = load_net(BASE_CKPT, ct, dev)
        return net, "baseline_ckpt_500736", {"ckpt": BASE_CKPT, "step": step}
    if kind == "selfmirror":
        net, step, nc = load_net(cand_path, ct, dev)
        return net, "selfmirror_ckpt_999424", {"ckpt": cand_path, "step": step}
    raise SystemExit(f"unknown opponent kind: {kind}")


def run_one(kind, games, seed, pool, net_pool, dev, ct, enc, live_n, mode, cand_net,
            cand_path):
    """跑一个对手 games 局，返回统计字典。

    mode=mirror: 两座位同一副牌（deck strength 逐局抵消，wr 纯粹是 net 对 net）。
    mode=cross : 候选座位用 agent/deck.csv，对手座位用池轮转。
    """
    opp_net, label, meta = build_opponent(kind, ct, dev, seed, cand_path)
    rng = np.random.default_rng(seed)
    cnet = cand_net

    # 候选牌组序列：stride 7 在池上轮转（任务书 Decisions 项）。
    npool = len(pool)
    cand_seq = [pool[(i * 7) % npool] for i in range(games)]

    st = {"wins": 0, "losses": 0, "decided": 0, "draws": 0, "errs": 0,
          "trunc_turncap": 0, "trunc_maxsteps": 0, "true_draw": 0,
          "wins_seat0": 0, "wins_seat1": 0, "games_seat0": 0, "games_seat1": 0,
          "illegal_clamp": 0, "illegal_forfeit": 0,
          "illegal_clamp_net": 0, "illegal_clamp_opp": 0,
          "illegal_forfeit_net": 0, "illegal_forfeit_opp": 0,
          "applies": [], "turns": []}
    err_msgs = []

    def _spawn(gi):
        cs = gi % 2                      # 严格均衡先后手
        cd = cand_seq[gi]
        od = cd if mode == "mirror" else net_pool[(gi * 7) % len(net_pool)]
        d0, d1 = (od, cd) if cs else (cd, od)
        env = _Counted(decks=[d0, d1], encoder=enc, would_ko=True,
                       seed=int(seed + gi), native_encode=False)
        obs, seat, _info = env.reset()
        return {"env": env, "obs": obs, "seat": int(seat), "cs": cs}

    live, nxt, t0 = [], 0, time.time()
    while nxt < games or live:
        while len(live) < live_n and nxt < games:
            try:
                live.append(_spawn(nxt))
            except Exception as e:
                if nxt < 5:
                    raise
                st["errs"] += 1
                err_msgs.append("spawn@%d: %s: %s" % (nxt, type(e).__name__, e))
            nxt += 1
        if not live:
            break

        # 路由组：候选净 / 对手净（net 对手）/ 随机合法（random 对手）
        groups = []
        for tag in ("net", "opp", "rnd"):
            if tag == "net":
                grp = [b for b in live if b["seat"] == b["cs"]]
            elif tag == "opp":
                grp = [] if opp_net is None else [b for b in live if b["seat"] != b["cs"]]
            else:
                grp = [] if opp_net is not None else [b for b in live if b["seat"] != b["cs"]]
            if grp:
                groups.append((tag, grp))

        for tag, grp in groups:
            np_obs = {k: np.stack([np.asarray(b["obs"][k]) for b in grp])
                      for k in grp[0]["obs"]}
            if tag == "rnd":
                mask = np_obs["action_mask"] > 0.5
                for b, m in zip(grp, mask):
                    leg = np.flatnonzero(m)
                    b["act"] = int(rng.choice(leg)) if leg.size else 0
                continue
            batch = {k: torch.as_tensor(
                v, dtype=(torch.long if k in enc.int_keys else torch.float32), device=dev)
                for k, v in np_obs.items()}
            gnet = cnet if tag == "net" else opp_net
            acts = gnet.logits_value(
                batch, opt_len=bucketed_opt_len_np(np_obs["action_mask"], np_obs["opt_attr"])
            )[0].argmax(-1).cpu().numpy()
            for b, a in zip(grp, acts):
                b["act"] = int(a)

        finished = []
        for b in live:
            acting = b["seat"]                  # step 前记录 acting seat（非法动作归属）
            try:
                obs, seat, _r, done, info = b["env"].step(b["act"])
            except Exception as e:
                st["errs"] += 1
                err_msgs.append("step@%d: %s: %s" % (acting, type(e).__name__, e))
                finished.append(b)
                continue
            if info.get("illegal_action"):
                st["illegal_clamp"] += 1
                if acting == b["cs"]:
                    st["illegal_clamp_net"] += 1
                else:
                    st["illegal_clamp_opp"] += 1
            if not done:
                b["obs"], b["seat"] = obs, int(seat)
                continue
            cs = b["cs"]
            st["applies"].append(b["env"]._apply_calls)
            try:
                st["turns"].append(int((b["env"]._state() or {}).get("turn") or -1))
            except Exception:
                st["turns"].append(-1)
            st["games_seat0" if cs == 0 else "games_seat1"] += 1
            if b["env"]._forfeit is not None:
                st["illegal_forfeit"] += 1
                if b["env"]._forfeit == cs:
                    st["illegal_forfeit_net"] += 1
                else:
                    st["illegal_forfeit_opp"] += 1
            t = info.get("terminal")
            if t is None:
                if info.get("trunc_turncap"):
                    st["trunc_turncap"] += 1
                elif info.get("truncated"):
                    st["trunc_maxsteps"] += 1
                st["draws"] += 1
            elif t[cs] == t[1 - cs]:
                st["true_draw"] += 1
                st["draws"] += 1
            elif t[cs] > t[1 - cs]:
                st["wins"] += 1
                st["decided"] += 1
                st["wins_seat0" if cs == 0 else "wins_seat1"] += 1
            else:
                st["losses"] += 1
                st["decided"] += 1
            finished.append(b)

        for b in finished:
            try:
                b["env"].close()
            except Exception:
                pass
            live.remove(b)

    for b in live:
        try:
            b["env"].close()
        except Exception:
            pass

    el = time.time() - t0
    n = games
    dec = st["decided"]
    lo, hi = wilson(st["wins"], dec)
    total_illegal = st["illegal_clamp"] + st["illegal_forfeit"]
    applies = st["applies"] or [0]
    turns = [t for t in st["turns"] if t >= 0] or [0]
    return {
        "opponent": kind, "label": label, "games": n, "mode": mode, "seed": seed,
        "meta": meta,
        "wins": st["wins"], "losses": st["losses"], "decided": dec, "draws": st["draws"],
        "errs": st["errs"], "err_samples": err_msgs[:10],
        "draw_breakdown": {"turn_cap": st["trunc_turncap"], "max_steps": st["trunc_maxsteps"],
                           "true_draw": st["true_draw"]},
        "winrate": (st["wins"] / dec) if dec else None,
        "winrate_ci95": [lo, hi],
        "winrate_all_games": (st["wins"] / n) if n else None,
        "seat_balance": {"games_seat0": st["games_seat0"], "games_seat1": st["games_seat1"],
                         "wins_seat0": st["wins_seat0"], "wins_seat1": st["wins_seat1"],
                         "wr_seat0": (st["wins_seat0"] / st["games_seat0"])
                         if st["games_seat0"] else None,
                         "wr_seat1": (st["wins_seat1"] / st["games_seat1"])
                         if st["games_seat1"] else None},
        "illegal": {"clamp": st["illegal_clamp"], "forfeit": st["illegal_forfeit"],
                    "total": total_illegal,
                    "rate_per_decision": total_illegal / max(sum(applies), 1),
                    "clamp_candidate": st["illegal_clamp_net"],
                    "clamp_opponent": st["illegal_clamp_opp"],
                    "forfeit_candidate": st["illegal_forfeit_net"],
                    "forfeit_opponent": st["illegal_forfeit_opp"]},
        "length": {"mean_applies": float(np.mean(applies)),
                   "median_applies": float(np.median(applies)),
                   "p95_applies": float(np.percentile(applies, 95)),
                   "max_applies": int(np.max(applies)),
                   "mean_turn": float(np.mean(turns)),
                   "median_turn": float(np.median(turns)),
                   "p95_turn": float(np.percentile(turns, 95)),
                   "max_turn": int(np.max(turns))},
        "elapsed_s": el, "games_per_s": n / max(el, 1e-9),
    }


# --- sdk_cg 引擎别名注入（本机仓库缺 cg.dll 的绕行） --------------------------
# 仓库 sdk_cg/sim.py L22/L29：lib_path 指向 sdk_cg/cg.dll → LoadLibrary 抛
# FileNotFoundError（失败不被 sys.modules 缓存，每次导入都重抛）。平台真引擎在
# kaggle_environments.envs.cabt.cg.sim（普通对局的 JSON 路径本就跑它）。做法：把该
# 已加载的 lib 以 `sdk_cg.sim` 名字注入 sys.modules，让 api.py 的 `from .sim import lib`
# 拿到同一 dll 句柄——同进程绝不二次 LoadLibrary（引擎头表 STB_GNU_UNIQUE 使 CardTable
# 跨句柄共享，双加载＝双填 init，触发 assert/状态损坏）。shim 只声明战斗函数，缺 api.py
# 需要的 7 个搜索/枚举声明，按 repo sdk_cg/sim.py L36-71 原文补齐（逐函数 hasattr 守卫；
# 未导出符号仅告警，调用时由 ctypes 自然报错）。

_SDK_CG_DECLS = ("AgentStart", "SearchBegin", "SearchStep", "SearchEnd",
                 "SearchRelease", "AllCard", "AllAttack")


def _install_sdk_cg_shim_alias():
    """见上方注释。必须在任何 `from sdk_cg import api` 之前调用（main() 起始）。"""
    if "sdk_cg.sim" in sys.modules:
        return
    if os.path.exists(os.path.join(_ROOT, "sdk_cg", "cg.dll")):
        print("[shim] 仓库自带 sdk_cg/cg.dll，保留原生 sdk_cg.sim，不注入别名", flush=True)
        return
    try:
        from kaggle_environments.envs.cabt.cg import sim as shim_sim
        import sdk_cg
    except Exception as e:                                    # noqa: BLE001
        print("[shim] 别名注入失败，sdk_cg 将按原样报错：%s: %s"
              % (type(e).__name__, e), flush=True)
        return
    lib = getattr(shim_sim, "lib", None)
    if lib is None:
        print("[shim] 平台 shim 无 lib 句柄，别名注入跳过", flush=True)
        return

    sys.modules["sdk_cg.sim"] = shim_sim
    setattr(sdk_cg, "sim", shim_sim)

    missing = [n for n in _SDK_CG_DECLS if not hasattr(lib, n)]
    if hasattr(lib, "AgentStart"):
        lib.AgentStart.restype = ctypes.c_void_p
    if hasattr(lib, "SearchBegin"):
        lib.SearchBegin.restype = ctypes.c_char_p
        lib.SearchBegin.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.c_int]
    if hasattr(lib, "SearchStep"):
        lib.SearchStep.restype = ctypes.c_char_p
        lib.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_int64,
                                   ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    if hasattr(lib, "SearchEnd"):
        lib.SearchEnd.argtypes = [ctypes.c_void_p]
    if hasattr(lib, "SearchRelease"):
        lib.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    if hasattr(lib, "AllCard"):
        lib.AllCard.restype = ctypes.c_char_p
    if hasattr(lib, "AllAttack"):
        lib.AllAttack.restype = ctypes.c_char_p

    print("[shim] sdk_cg.sim -> 平台引擎别名已注入（补 %d 声明%s）"
          % (len(_SDK_CG_DECLS) - len(missing),
             "" if not missing else "；缺导出 " + ",".join(missing)), flush=True)


def main():
    _install_sdk_cg_shim_alias()
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="all",
                    choices=["all", "random", "untrained", "baseline", "selfmirror"])
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--pool-json", default=POOL_JSON)
    ap.add_argument("--net-pool-json", default=POOL_JSON)
    ap.add_argument("--ckpt", default=CAND_CKPT)
    ap.add_argument("--live", type=int, default=1)
    ap.add_argument("--mode", default="mirror", choices=["mirror", "cross"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # JSON 引擎 Battle.battle_ptr 是进程级类属性（单例）：多个 env 并存时，后 spawn 的
    # env 在 reset() 里 battle_finish 掉前一个 env 的战斗并清空全局指针，导致串局与首步
    # forfeit。live=1（串行）是唯一正确解，故这里强制 clamp。
    if args.live != 1:
        print("[warn] --live=%d 请求被忽略：JSON 引擎单例，强制串行 live=1"
              % args.live, flush=True)
        args.live = 1

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[env] device=%s turn_cap=%s torch=%s"
          % (dev.type, os.environ.get("PKMN_TURN_CAP"), torch.__version__), flush=True)
    if dev.type != "cuda":
        print("WARNING: no CUDA -- CPU only, expect ~50x slower", flush=True)

    ct = get_card_table()
    enc = TokenEncoder(ct)
    pj = json.load(open(args.pool_json, encoding="utf-8"))
    pool = [list(d) for d in pj["decks"]]
    npj = json.load(open(args.net_pool_json, encoding="utf-8"))
    net_pool = [list(d) for d in npj["decks"]]
    print("[pool] cand_pool=%d decks (%s); net_pool=%d decks"
          % (len(pool), args.pool_json, len(net_pool)), flush=True)

    cand_net, cand_step, _nc = load_net(args.ckpt, ct, dev)
    print("[cand] %s step=%d" % (args.ckpt, cand_step), flush=True)

    kinds = (["random", "untrained", "baseline", "selfmirror"]
             if args.opponent == "all" else [args.opponent])
    results = []
    for kind in kinds:
        print("\n=== opponent=%s games=%d seed=%d mode=%s ==="
              % (kind, args.games, args.seed, args.mode), flush=True)
        r = run_one(kind, args.games, args.seed, pool, net_pool, dev, ct, enc,
                    args.live, args.mode, cand_net, args.ckpt)
        print("[done] %s wr=%s ci95=%s draws=%d errs=%d illegal=%d "
              "len_mean=%.1f %.2f g/s"
              % (r["label"], r["winrate"], r["winrate_ci95"], r["draws"], r["errs"],
                 r["illegal"]["total"], r["length"]["mean_applies"], r["games_per_s"]),
              flush=True)
        results.append(r)

    out = args.out or os.path.join(
        OUT_DIR, "eval_%s_%s_%dg_seed%d.json"
        % (args.mode, args.opponent, args.games, args.seed))
    rec = {"cand_ckpt": args.ckpt, "cand_step": cand_step, "mode": args.mode,
           "games": args.games, "seed": args.seed, "live": args.live,
           "turn_cap": _TURN_CAP, "pool_json": args.pool_json,
           "device": dev.type, "torch": torch.__version__, "results": results}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print("\n[out] %s" % out, flush=True)


if __name__ == "__main__":
    main()