"""复用 scripts/h2h_ckpt.py，额外统计非法动作。

h2h_ckpt.py -> rl.gates.field_winrate_fast 不暴露 illegal 计数；这里包一层
TwoSidedSelfPlayEnv.step，其余参数、种子、牌组池与主 200 局完全一致。
"""
import json
import os
import runpy
import sys

import rl.gates  # noqa: F401  (先导入，确保 field_winrate_fast 使用同一个类)
from rl.env_selfplay import TwoSidedSelfPlayEnv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
counts = {"step_calls": 0, "illegal_clamp": 0, "illegal_forfeit": 0,
          "finished_games": 0, "errs": 0}

_orig_step = TwoSidedSelfPlayEnv.step


def _step(self, *args, **kwargs):
    obs, seat, reward, done, info = _orig_step(self, *args, **kwargs)
    counts["step_calls"] += 1
    if info.get("illegal_action"):
        counts["illegal_clamp"] += 1
    if done:
        counts["finished_games"] += 1
        if getattr(self, "_forfeit", None) is not None:
            counts["illegal_forfeit"] += 1
    return obs, seat, reward, done, info


TwoSidedSelfPlayEnv.step = _step

sys.argv = [
    "scripts/h2h_ckpt.py",
    os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_999424.pt"),
    os.path.join(ROOT, "runs", "train-20m-e16-1m", "ckpt_500736.pt"),
    "200",
    "--pool-json", os.path.join(ROOT, "runs", "evaluation", "final_gate", "pool_control_plus_dragapult.json"),
    "--pool-weights", os.path.join(ROOT, "runs", "evaluation", "final_gate", "pool_control_plus_dragapult_w.npy"),
    "--sample", "2",
    "--seed", "777",
    "--live", "1",
    "--progress-every", "50",
    "--out", os.path.join(ROOT, "runs", "evaluation", "final_gate"),
    "--tag", "h2h_cand_vs_base_200_instr",
]

try:
    runpy.run_path(os.path.join(ROOT, "scripts", "h2h_ckpt.py"), run_name="__main__")
except SystemExit as e:
    counts["errs"] = 1
    print("[instr] SystemExit:", e, flush=True)
finally:
    out = os.path.join(ROOT, "runs", "evaluation", "final_gate", "illegal_counts_200.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)
    print("[instr]", json.dumps(counts, ensure_ascii=False), flush=True)
