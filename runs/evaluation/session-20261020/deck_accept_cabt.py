# -*- coding: utf-8 -*-
"""A5.5 §8.3 第 4 步：用真实 CABT 引擎做还原牌组的**开局接受校验**。

只做引擎侧合法性，不做强度评测（强度评测属 §8.4，且本次不跑：用户训练仍在占用 GPU/CPU，
胜率评测本身数值不受争用影响，但会抢 CPU 拖慢训练，故推迟到训练结束后）。

校验内容（每副牌 N 局）
  1. `env.reset()` 内部调用 `game.battle_start(d0, d1)`，obs is None 时抛
     `battle_start failed: errorPlayer=…` —— 这正是引擎拒牌的入口，能过即"开局被接受"。
  2.  reset 后连续走 M 个随机合法动作（build_mask 过滤），确认引擎不报错、不越界。
  3.  记录 `illegal_action` 事件数（本探针只喂合法动作，预期 0）。

用法：python runs/evaluation/session-20261020/deck_accept_cabt.py [--games 4] [--steps 60]
"""
import argparse
import glob
import io
import json
import os
import random
import sys

os.environ.setdefault("PKMN_TURN_CAP", "200")
SESS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SESS)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "runs", "evaluation"))
import eval_h2h as H  # noqa: E402

from rl.encoding import SUBMIT_ACTION, build_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--games", type=int, default=4)
ap.add_argument("--steps", type=int, default=60)
a = ap.parse_args()

H._install_sdk_cg_shim_alias()
ct = H.get_card_table()
enc = H.TokenEncoder(ct)

paths = sorted(glob.glob(os.path.join(SESS, "strong_decks", "*_reconstructed.csv")))
if not paths:
    raise SystemExit("没有还原牌组 CSV，先跑 reconstruct_decks.py")

report = []
for p in paths:
    ids = [int(x) for line in io.open(p, encoding="utf-8") for x in line.split() if x.strip()]
    deck = H._Counted(decks=[ids, list(ids)], encoder=enc, would_ko=True,
                      seed=20261020, native_encode=False)
    ok_reset, steps_seen, illegal, errors = 0, 0, 0, []
    for g in range(a.games):
        try:
            _e, _s, _i = deck.reset(seed=20261020 + g)
            ok_reset += 1
        except Exception as ex:
            errors.append(f"reset: {type(ex).__name__}: {ex}")
            continue
        rng = random.Random(20261020 + g)
        for _st in range(a.steps):
            sel = deck._obs["select"]
            mask = build_mask(sel, set(deck._picked))
            legal = [i for i, m in enumerate(mask) if m]
            if not legal:
                errors.append(f"no legal action at step {_st}")
                break
            pick = rng.choice(legal)
            if pick == SUBMIT_ACTION and len(legal) > 1 and rng.random() < 0.3:
                pick = legal[0] if legal[0] != SUBMIT_ACTION else legal[-1]
            try:
                # TwoSidedSelfPlayEnv.step 返回 (enc, seat, reward, done, info)
                _e, _seat, _r, done, info = deck.step(pick)
            except Exception as ex:
                errors.append(f"step: {type(ex).__name__}: {ex}")
                break
            steps_seen += 1
            if info.get("illegal_action"):
                illegal += 1
            if done:
                break
    name = os.path.basename(p).replace("_reconstructed.csv", "")
    row = {"deck": name, "n_cards": len(ids), "games": a.games, "resets_ok": ok_reset,
           "random_steps_executed": steps_seen, "illegal_action_events": illegal,
           "errors": errors[:5], "accepted": ok_reset == a.games and not errors}
    report.append(row)
    print(f"[{'ACCEPT' if row['accepted'] else 'REJECT '}] {name:<28} "
          f"reset {ok_reset}/{a.games}  随机合法步 {steps_seen}  "
          f"非法动作事件 {illegal}" + (f"  错误 {errors[:2]}" if errors else ""))

out_path = os.path.join(SESS, "strong_decks", "cabt_accept.json")
json.dump(report, io.open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n[saved]", out_path)
print("全部接受" if all(r["accepted"] for r in report) else "存在被引擎拒绝的牌组")
