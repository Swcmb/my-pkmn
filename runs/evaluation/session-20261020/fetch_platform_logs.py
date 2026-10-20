# -*- coding: utf-8 -*-
"""A5.5 只读支线 · 第一步：采集头部强队的真实对局日志并落盘。

只读：GET /api/matches 与 GET /api/matches/{id}/log，不写权重、不训练、不上传、不提交。
说明：`ptcg_arena/api.py:137` 的 `match_log()` 未传 `auth=True`，故 CLI 的
`agent results --log-id` 必然 401。本脚本在**临时脚本内**用包内
`api._request(..., auth=True)` 等价绕过，**不修改 CLI 源文件**（报告中标注"绕过而非修复"）。

排序依据：积分榜**累计总积分**（不是单轮胜率），与任务书 §8.2 一致。
每队取最近 N 局，跨队去重后下载，日志连同 seed/胜者/队伍一并落盘，文件名带 match_id，不覆盖。
"""
import json
import os
import sys
import time

sys.path.insert(0, r"D:\ptcg-arena-cli")
from ptcg_arena import api                                   # noqa: E402

ROOT = r"D:\pkmn-kaggle"
OUT = os.path.join(ROOT, "runs", "evaluation", "session-20261020", "platform_logs")
GAMES_PER_TEAM = int(os.environ.get("LOG_GAMES", "44"))
os.makedirs(OUT, exist_ok=True)
LOGS = os.path.join(OUT, "logs")
os.makedirs(LOGS, exist_ok=True)

lb = json.loads(api._request("GET", "/api/leaderboard/a", auth=True)[1])
teams = sorted(lb["rows"], key=lambda r: -r["total_points"])
top = teams[:5]
print("累计积分前 5 队（A 榜，已发布 %d 轮）：" % len(lb["published_rounds"]))
for i, t in enumerate(top, 1):
    rs = t["rounds"]
    w = sum(r["wins"] for r in rs); l = sum(r["losses"] for r in rs); d = sum(r["draws"] for r in rs)
    n = w + l + d
    rate = w / n if n else 0.0
    print(f"  {i}. team_id={t['team_id']:>3} {t['team_name']:<14} 积分 {t['total_points']:>5} "
          f"累计 {w}胜{l}负{d}平 = {rate:.3f} 参与 {len(rs)} 轮")

ms = json.loads(api._request("GET", "/api/matches", auth=True)[1])
print(f"\n平台对局总数 {len(ms)}，最大 id {max(m['id'] for m in ms)}")

by_team = {}
for m in ms:
    for tid in (m["player_a"], m["player_b"]):
        by_team.setdefault(tid, []).append(m)
for v in by_team.values():
    v.sort(key=lambda m: m["id"])

plan, chosen = [], {}
for t in top:
    tid = t["team_id"]
    picks = by_team.get(tid, [])[-GAMES_PER_TEAM:]
    chosen[tid] = [p["id"] for p in picks]
    plan.append((tid, t["team_name"], len(by_team.get(tid, [])), len(picks)))
    print(f"  team {t['team_name']:<14} 参与 {len(by_team.get(tid, [])):>4} 局，取最近 {len(picks)} 局")

ids = sorted({i for v in chosen.values() for i in v})
print(f"\n去重后需下载 {len(ids)} 局日志")

index, got, failed, t0 = {}, 0, [], time.time()
for mid in ids:
    path = os.path.join(LOGS, f"match_{mid}.json")
    if os.path.exists(path):
        index[str(mid)] = {"cached": True}
        continue
    try:
        d = json.loads(api._request("GET", f"/api/matches/{mid}/log", auth=True, timeout=60)[1])
    except Exception as e:                                    # noqa: BLE001
        failed.append((mid, f"{type(e).__name__}: {str(e)[:120]}"))
        continue
    meta = next(m for m in ms if m["id"] == mid)
    rec = {"match": meta, "teams": {str(meta["player_a"]): "A", str(meta["player_b"]): "B"},
           "log_chars": len(d.get("log") or ""), "log": d.get("log")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    index[str(mid)] = {"chars": rec["log_chars"]}
    got += 1
    if got % 20 == 0:
        print(f"  已下载 {got}/{len(ids)}  用时 {time.time()-t0:.0f}s", flush=True)

with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
    json.dump({"board": "A", "published_rounds": len(lb["published_rounds"]),
               "top_teams": [{"team_id": t["team_id"], "team_name": t["team_name"],
                              "total_points": t["total_points"],
                              "selected_match_ids": chosen[t["team_id"]]} for t in top],
               "games_per_team": GAMES_PER_TEAM, "downloaded_new": got,
               "failed": failed, "source": "GET /api/matches, GET /api/matches/{id}/log",
               "note": "CLI match_log() 缺 auth=True，本脚本用包内 _request(auth=True) 绕过，未修改 CLI"},
              f, ensure_ascii=False, indent=2)

tot_chars = sum(os.path.getsize(os.path.join(LOGS, x)) for x in os.listdir(LOGS))
print(f"\n新增下载 {got} 局（失败 {len(failed)}），日志库总计 {len(os.listdir(LOGS))} 文件 / {tot_chars/1e6:.1f} MB")
if failed:
    print("失败样本:", failed[:3])
