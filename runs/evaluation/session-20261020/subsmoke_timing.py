"""提交包官方等价 smoke test（阶段 1 沙箱约束验证）。

以平台调用方式导入包内 main.py，驱动真实引擎对局，逐项验证官方限制：
单步 100ms / 单局 10 分钟 / 内存 2GB / 返回值长度与下标合法 / 无异常无崩溃。
只读验证，不启动任何训练。
"""
import ctypes
import importlib.util
import json
import os
import random
import sys
import time

PKG = os.path.abspath(sys.argv[1])
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 20261020

sys.path.insert(0, PKG)
os.environ.setdefault("PYTHONPATH", PKG)

# 官方沙箱 CPU 限制 2 核：用 THREADS 环境变量约束 torch 线程，复现平台推理预算
_THREADS = int(os.environ.get("SMOKE_THREADS", "0"))
if _THREADS:
    import torch
    torch.set_num_threads(_THREADS)

import sdk_cg.game as game                                 # noqa: E402
from encoding import build_mask, SUBMIT_ACTION             # noqa: E402

spec = importlib.util.spec_from_file_location("submain", os.path.join(PKG, "main.py"))
main = importlib.util.module_from_spec(spec)
sys.modules["submain"] = main
_t_import = time.perf_counter()
spec.loader.exec_module(main)
IMPORT_S = time.perf_counter() - _t_import

DECK = [int(x) for x in open(os.path.join(PKG, "deck.csv")).read().split()]


def peak_rss_mb():
    """Windows PSAPI 工作集峰值（MB）；非 Windows 回落到 resource。"""
    if os.name == "nt":
        class CTR(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = CTR(); c.cb = ctypes.sizeof(c)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        # K32 前缀在 kernel32，旧版 GetProcessMemoryInfo 在 psapi，两个位置都试一次
        for mod, fn in ((ctypes.windll.kernel32, "K32GetProcessMemoryInfo"),
                        (ctypes.windll.psapi, "GetProcessMemoryInfo")):
            try:
                getattr(mod, fn)(ctypes.c_void_p(h), ctypes.byref(c), c.cb)
                break
            except AttributeError:
                continue
        return c.PeakWorkingSetSize / 1048576.0
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def legal_random(sel, picked, rng):
    mask = build_mask(sel, set(picked))
    legal = [i for i, m in enumerate(mask) if m]
    return rng.choice(legal) if legal else None


def scripted(sel, rng):
    picked = []
    for _ in range(sel.get("maxCount", 1) + 1):
        a = legal_random(sel, picked, rng)
        if a is None or a == SUBMIT_ACTION:
            break
        picked.append(a)
        if len(picked) >= sel.get("maxCount", 1):
            break
    return sorted(set(picked))


rng = random.Random(SEED)
st = {"games": 0, "wins": 0, "losses": 0, "draws": 0, "errors": 0,
      "illegal_len": 0, "illegal_index": 0, "duplicate_index": 0,
      "decisions": 0, "step_ms_max": 0.0, "step_ms_sum": 0.0, "game_s_max": 0.0,
      "over_100ms": 0, "over_10min": 0}
lat = []

for gi in range(GAMES):
    seat = gi % 2
    main.agent({"select": None})                       # 平台开局重置路径
    obs, _sd = game.battle_start(list(DECK), list(DECK))
    g0 = time.perf_counter()
    n = 0
    while True:
        cur = obs["current"]
        if cur["result"] >= 0:
            r = cur["result"]
            st["draws" if r == 2 else ("wins" if r == seat else "losses")] += 1
            break
        if n >= 3000:
            st["errors"] += 1
            break
        sel = obs["select"]
        who_is_agent = (cur["yourIndex"] == seat)
        if who_is_agent:
            t = time.perf_counter()
            try:
                picks = main.agent(obs)
            except Exception as e:
                st["errors"] += 1
                print(f"[err] agent {type(e).__name__}: {e}", flush=True)
                break
            ms = (time.perf_counter() - t) * 1000.0
            lat.append(ms)
            st["decisions"] += 1
            st["step_ms_sum"] += ms
            st["step_ms_max"] = max(st["step_ms_max"], ms)
            if ms > 100.0:
                st["over_100ms"] += 1
        else:
            picks = scripted(sel, rng)
        opts = (sel.get("option") or [])
        n_pick = len(picks or [])
        if not (sel.get("minCount", 1) <= n_pick <= sel.get("maxCount", 1)):
            st["illegal_len"] += 1
        if len(set(picks or [])) != len(picks or []):
            st["duplicate_index"] += 1
        for p in picks or []:
            if not (isinstance(p, int) and 0 <= p < len(opts)):
                st["illegal_index"] += 1
        try:
            obs = game.battle_select([int(x) for x in picks])
        except Exception as e:
            st["errors"] += 1
            print(f"[err] select {type(e).__name__}: {e}", flush=True)
            break
        n += 1
    wall = time.perf_counter() - g0
    st["game_s_max"] = max(st["game_s_max"], wall)
    if wall > 600.0:
        st["over_10min"] += 1
    st["games"] += 1

srt = sorted(lat)
n_dec = len(srt)
WARMUP = 10                                        # 前 10 次决策含 torch/引擎预热，单独口径
tail = lat[WARMUP:] if n_dec > WARMUP else lat
st["step_ms_p50"] = round(srt[n_dec // 2 - 1], 2) if n_dec else 0.0
st["step_ms_p99"] = srt[max(int(n_dec * 0.99) - 1, 0)] if n_dec else 0.0
st["step_ms_top5"] = [round(x, 2) for x in srt[-5:]]
st["warmup_first10_ms"] = [round(x, 2) for x in lat[:WARMUP]]
st["over_100ms_after_warmup"] = sum(1 for x in tail if x > 100.0)
st["step_ms_mean"] = st["step_ms_sum"] / max(st["decisions"], 1)
st["threads"] = _THREADS or "default"
st["import_s"] = round(IMPORT_S, 2)
st["peak_rss_mb"] = round(peak_rss_mb(), 1)
st["verdict"] = {
    "单步100ms(全量)": st["over_100ms"] == 0,
    "单步100ms(预热后)": st["over_100ms_after_warmup"] == 0,
    "单局10分钟": st["over_10min"] == 0,
    "内存2GB": st["peak_rss_mb"] < 2048.0,
    "返回长度合法": st["illegal_len"] == 0,
    "返回下标合法": st["illegal_index"] == 0,
    "无重复下标": st["duplicate_index"] == 0,
    "无异常": st["errors"] == 0,
}
st["all_pass"] = all(st["verdict"].values())
print(json.dumps(st, ensure_ascii=False, indent=2))
