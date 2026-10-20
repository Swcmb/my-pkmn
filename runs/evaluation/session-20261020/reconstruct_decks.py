# -*- coding: utf-8 -*-
"""A5.5 只读支线 · 第二步：从中文对局日志还原头部强队的 60 张牌组。

性质：**纯文本统计推断**，不依赖"同 seed 本地可复现平台对局"这一未验证假设。
还原结果不是官方 deck，必须连同置信度一起读。

解析口径（全部实测自 platform_logs/logs，不是猜的）
  1. 侧别按**行首的 `([AB]) 方`** 归属。绝不按 `▶` 块头继承 —— 按块头继承会把对手
     在同一块里发生的抽卡/打出算到本队头上（实测污染：bug 队混进 Abra、超能队混进
     Bug Catching Set）。
  2. 同一个动作在日志里出现两次：`▶ A 方：打出「X」`（决策提示）与 `  A 方 打出「X」`
     （结算行）。计数只认结算行，否则所有次数翻倍。
  3. `将「C」附着到` 只统计 C 为能量（cardType 5/6）的行 —— 工具卡（Hop’s Choice Band、
     Hero’s Cape）也走同一个句式，不过滤会把工具当成附能次数。
  4. `场面` 块内的宝可梦名不带「」，单独解析；块内 `能量×n` 给出场上能量数。

张数证据（按强度排序）
  board_lb ：同一局 `X 方 场面` 出战+备战里该宝可梦名同时出现的最大个数 —— 硬下界。
  play_lb  ：单局内结算行 `X 方 打出「C」` 次数的最大值 —— 每次打出消耗一张实体卡，近似硬下界。
  draw_lb  ：单局内 `抽到「C」` 次数的最大值 —— 会被回收再抽，仅作展示，不参与定张数。
  非能量初始张数 = clip(max(board_lb, play_lb, 1), 1, 4)；ACE SPEC 强制 1。
  能量张数不可辨识（日志看不出牌库里还剩多少能量），只能用先验：
  E_target = clip(round(1.5 × 每局附能总次数的 p75), 10, 20)，缺数据时取 12。
  先验系数 1.5 是"场上用掉的能量之外牌库还得留备料"的经验值，**属于推断，不是证据**。

输出（§8.3）
  strong_decks/reconstructed.json        全量结构化结果
  strong_decks/<team>_reconstructed.csv  每行一个 cardId，共 60 行，可直接喂校验器
  strong_decks/<team>_evidence.md        逐卡证据表 + 置信度分级 + 方法与局限声明
"""
import csv
import io
import json
import os
import re
import statistics
from collections import Counter, defaultdict

ROOT = r"D:\pkmn-kaggle"
SESS = os.path.join(ROOT, "runs", "evaluation", "session-20261020")
LOGS = os.path.join(SESS, "platform_logs", "logs")
IDX = os.path.join(SESS, "platform_logs", "index.json")
CARDS = os.path.join(SESS, "platform_cards.csv")
OUTDIR = os.path.join(SESS, "strong_decks")
os.makedirs(OUTDIR, exist_ok=True)

CARD_RE = re.compile(r"「(.+?)」")
SIDE_RE = re.compile(r"^▶?\s*([AB])\s*方")
DRAW_RE = re.compile(r"抽到「(.+?)」")
PLAY_RE = re.compile(r"打出「(.+?)」")
ATTACH_RE = re.compile(r"^([AB])\s*方\s*将「(.+?)」附着到")
BOARD_HDR_RE = re.compile(r"^([AB])\s*方\s*场面")
ENERGY_ON_BOARD_RE = re.compile(r"能量×(\d+)")
P_MIN = 0.30
DECK_SIZE = 60
E_FLOOR, E_CEIL, E_FALLBACK = 10, 20, 12

TYPE_CN = {0: "宝可梦", 1: "道具", 2: "工具", 3: "支援者", 4: "竞技场",
           5: "基础能量", 6: "特殊能量"}
ELEMENT_TO_BASIC = {"GRASS": "Basic {G} Energy", "FIRE": "Basic {R} Energy",
                    "WATER": "Basic {W} Energy", "LIGHTNING": "Basic {L} Energy",
                    "PSYCHIC": "Basic {P} Energy", "FIGHTING": "Basic {F} Energy",
                    "DARKNESS": "Basic {D} Energy", "METAL": "Basic {M} Energy",
                    "DRAGON": "Basic {D} Energy"}

meta = {}
for r in csv.DictReader(io.open(CARDS, encoding="utf-8-sig")):
    cid = int(r["cardId"])
    meta[cid] = {"name": r["name"].strip(), "cardType": int(r["cardType"]),
                 "basic": int(r["basic"]), "aceSpec": int(r["aceSpec"]),
                 "energyType": r["energyType"].strip()}
name2id = {}
for cid, m in sorted(meta.items()):
    name2id.setdefault(m["name"], cid)   # 同名多印刷取最小 ID；同名 ≤4 本就按名称统计
is_energy = lambda cid: meta[cid]["cardType"] in (5, 6)

idx = json.load(open(IDX, encoding="utf-8"))
top = idx["top_teams"]
watch = {t["team_id"]: t["team_name"] for t in top}
points = {t["team_id"]: t["total_points"] for t in top}

freq = defaultdict(Counter)        # team -> name -> 出现总次数（所有本队行）
appear = defaultdict(Counter)      # team -> name -> 出现过的局数
board_lb = defaultdict(Counter)    # team -> name -> 场面同局最大同时张数
play_lb = defaultdict(Counter)     # team -> name -> 单局打出最大次数
draw_lb = defaultdict(Counter)     # team -> name -> 单局抽到最大次数
attach = defaultdict(Counter)      # team -> energy name -> 附能总次数
attach_per_game = defaultdict(list)     # team -> [每局能量附能总次数]
ngames = defaultdict(int)


def board_counts(lines, i):
    """解析 `X 方 场面` 块，返回 (场上宝可梦名计数, 场上能量总数)。"""
    cnt, esum = Counter(), 0
    for j in range(i + 1, min(i + 6, len(lines))):
        s = lines[j].strip()
        if not (s.startswith("出战：") or s.startswith("备战：")):
            break
        seg = s.split("：", 1)[1]
        if "空" in seg and "｜" not in seg:
            continue
        esum += sum(int(v) for v in ENERGY_ON_BOARD_RE.findall(s))
        for part in seg.split("｜"):
            part = part.strip()
            if " HP " in part:
                nm = part.split(" HP ")[0].strip()
                if nm in name2id:
                    cnt[nm] += 1
    return cnt, esum


for fn in sorted(os.listdir(LOGS)):
    rec = json.load(open(os.path.join(LOGS, fn), encoding="utf-8"))
    m = rec["match"]
    side_of = {m["player_a"]: "A", m["player_b"]: "B"}
    lines = (rec.get("log") or "").split("\n")
    per_draw, per_play = defaultdict(Counter), defaultdict(Counter)
    per_any, per_board, per_attach = (defaultdict(Counter) for _ in range(3))
    per_energy_peak = Counter()
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        bh = BOARD_HDR_RE.match(s)
        if bh:
            sd, esum = board_counts(lines, i)
            bs = bh.group(1)
            for nm, v in sd.items():
                if v > per_board[bs][nm]:
                    per_board[bs][nm] = v
            if esum > per_energy_peak[bs]:
                per_energy_peak[bs] = esum
            continue
        mm = SIDE_RE.match(s)
        if mm is None:
            continue
        side = mm.group(1)
        for nm in CARD_RE.findall(s):
            if nm in name2id:
                per_any[side][nm] += 1
        if s.startswith("▶"):            # 决策提示行：与结算行重复，跳过
            continue
        if "抽到" in s:
            for nm in DRAW_RE.findall(s):
                if nm in name2id:
                    per_draw[side][nm] += 1
        if "打出" in s:
            for nm in PLAY_RE.findall(s):
                if nm in name2id:
                    per_play[side][nm] += 1
        am = ATTACH_RE.match(s)
        if am and am.group(2) in name2id and is_energy(name2id[am.group(2)]):
            per_attach[am.group(1)][am.group(2)] += 1
    for tid, sd in side_of.items():
        if tid not in watch or not per_any[sd]:
            continue
        ngames[tid] += 1
        freq[tid] += per_any[sd]
        for nm, v in per_any[sd].items():
            appear[tid][nm] += 1
        for src, dst in ((per_draw, draw_lb), (per_play, play_lb), (per_board, board_lb)):
            for nm, v in src[sd].items():
                if v > dst[tid][nm]:
                    dst[tid][nm] = v
        for nm, v in per_attach[sd].items():
            attach[tid][nm] += v
        attach_per_game[tid].append(sum(per_attach[sd].values()))


def confidence(p):
    if p >= 0.80:
        return "高"
    if p >= 0.50:
        return "中"
    return "低"


def energy_budget(tid):
    apg = attach_per_game[tid]
    if not apg:
        return E_FALLBACK, "无附能记录，取默认 12"
    s = sorted(apg)
    p75 = s[int(0.75 * (len(s) - 1))]
    return max(E_FLOOR, min(E_CEIL, round(1.5 * p75))), \
        f"每局附能 p75={p75}，×1.5 先验后 clip[{E_FLOOR},{E_CEIL}]"


def new_card(cid, n, p, appear_n, ng, source):
    m = meta[cid]
    return {"cid": cid, "name": m["name"], "cardType": m["cardType"], "n": n, "p": p,
            "appear": appear_n, "fpg": 0.0, "board_lb": 0, "play_lb": 0, "draw_lb": 0,
            "attach": 0, "is_ace": m["aceSpec"] == 1,
            "is_special_energy": m["cardType"] == 6,
            "is_basic_poke": m["cardType"] == 0 and m["basic"] == 1,
            "conf": confidence(p), "source": source}


def reconstruct(tid):
    ng = ngames[tid] or 1
    f, ap = freq[tid], appear[tid]
    e_target, e_note = energy_budget(tid)
    budget = DECK_SIZE - e_target

    rows, erows = [], []
    for nm, c in f.items():
        cid = name2id[nm]
        r = new_card(cid, 0, ap[nm] / ng, ap[nm], ng, "p 推断")
        r.update(fpg=c / ng, board_lb=board_lb[tid][nm], play_lb=play_lb[tid][nm],
                 draw_lb=draw_lb[tid][nm], attach=attach[tid][nm])
        r["lb"] = max(r["board_lb"], r["play_lb"], 1)
        (erows if is_energy(cid) else rows).append(r)
    rows.sort(key=lambda x: (-x["p"], -x["fpg"]))
    erows.sort(key=lambda x: -x["attach"])

    sel, n_ace = {}, 0
    for r in rows:
        if r["p"] < P_MIN:
            continue
        if r["is_ace"]:
            if n_ace:
                continue
            n_ace = 1
        v = dict(r)
        v["n"] = 1 if r["is_ace"] else min(4, r["lb"])
        v["source"] = ("场面同局上限" if r["board_lb"] >= 2 else
                       ("单局打出次数" if r["play_lb"] >= 2 else "p 推断"))
        sel[r["cid"]] = v
    order = sorted(sel, key=lambda c: (-sel[c]["p"], c))

    def total():
        return sum(v["n"] for v in sel.values())

    for cid in reversed(order):                       # 超预算：从 p 最低端整种删
        if total() <= budget:
            break
        keep_basic = (sel[cid]["is_basic_poke"] and
                      not any(v["is_basic_poke"] for k, v in sel.items() if k != cid))
        if not keep_basic:
            del sel[cid]
    while total() < budget:                           # 不足：按 p 降序补到 4
        grew = False
        for cid in order:
            if cid in sel and sel[cid]["n"] < (1 if sel[cid]["is_ace"] else 4):
                sel[cid]["n"] += 1
                grew = True
                if total() >= budget:
                    break
        if not grew:
            break

    # 能量名额：先按附能占比摊派，特殊能量 ≤4、ACE ≤1、基础能量豁免
    pool = [(r["cid"], r["attach"], r["p"]) for r in erows if r["attach"] > 0]
    fallback_cids = set()
    if not any(meta[c]["cardType"] == 5 for c, _, _ in pool):
        # 日志里一种基础能量都没出现 -> 按牌组内宝可梦的属性补一张基础能量兜底
        need = Counter(meta[c]["energyType"] for c in sel
                       if meta[c]["cardType"] == 0
                       and meta[c]["energyType"] in ELEMENT_TO_BASIC)
        for el, _ in need.most_common(2):
            nm = ELEMENT_TO_BASIC[el]
            if nm in name2id and not any(c == name2id[nm] for c, _, _ in pool):
                pool.append((name2id[nm], 0, 0.0))
                fallback_cids.add(name2id[nm])
    room = DECK_SIZE - total()
    tot_w = sum(a for _, a, _ in pool) or len(pool)
    for cid, a, ep in sorted(pool, key=lambda x: -x[1]):
        if room <= 0:
            break
        cap = DECK_SIZE if meta[cid]["cardType"] == 5 else 1 if meta[cid]["aceSpec"] else 4
        if meta[cid]["aceSpec"] and any(v["is_ace"] for v in sel.values()):
            continue
        if cid not in sel:
            sel[cid] = new_card(cid, 0, ep, int(round(ep * ng)), ng,
                                "属性兜底（日志未出现，先验）" if cid in fallback_cids
                                else "附能摊派")
            sel[cid]["attach"] = a
            sel[cid]["is_energy"] = True
        sel[cid]["is_energy"] = True
        sel[cid]["n"] = min(cap, sel[cid]["n"] + max(0, min(room, round(room * (a or 1) / tot_w))))
        room = DECK_SIZE - total()

    # 仍有余额：只允许落在基础能量（同名 ≤4 豁免）上，否则退回给 p 最高的非能量卡
    basic_pool = [c for c, _, _ in pool if meta[c]["cardType"] == 5]
    while (room := DECK_SIZE - total()) > 0:
        if basic_pool:
            sel[basic_pool[0]]["n"] += room
            break
        grew = False
        for cid in order:
            if cid in sel and sel[cid]["n"] < (1 if sel[cid]["is_ace"] else 4):
                sel[cid]["n"] += 1
                grew = True
                break
        if not grew:
            break

    deck = []
    for cid in sorted(sel, key=lambda k: (-sel[k]["n"], k)):
        deck += [cid] * sel[cid]["n"]
    if len(deck) != DECK_SIZE:
        raise SystemExit(f"ERROR: {watch[tid]} 还原出 {len(deck)} 张，应为 60")
    return deck, sel, ng, rows, e_target, e_note


def check(deck):
    d = Counter(deck)
    nm = Counter(meta[c]["name"] for c in d)
    over = {k: v for k, v in nm.items() if v > 4 and meta[name2id[k]]["cardType"] != 5}
    basics = {c for c in d if meta[c]["cardType"] == 0 and meta[c]["basic"] == 1}
    ace = sum(v for c, v in d.items() if meta[c]["aceSpec"] == 1)
    mix = Counter()
    for c, v in d.items():
        mix[meta[c]["cardType"]] += v
    return {"n": len(deck), "in_pool": all(c in meta for c in deck),
            "name_over4": over, "n_basic_species": len(basics),
            "n_distinct": len(d), "ace_spec_total": ace,
            "n_energy_total": mix[5] + mix[6],
            "card_type_mix": {str(k): v for k, v in sorted(mix.items())},
            "unknown_ids": sorted({c for c in deck if c not in meta})}


def slug(name):
    """保留中文队名，只替换 Windows 文件名非法字符。"""
    return re.sub(r'[<>:"/\\|?*]+', "_", name).strip(" ._")


out = {}
for tid in sorted(watch, key=lambda t: -points[t]):
    deck, sel, ng, rows, e_target, e_note = reconstruct(tid)
    chk = check(deck)
    name = watch[tid]
    base = f"team{tid}" + (f"_{slug(name)}" if slug(name) else "")
    cards = sorted(sel.items(), key=lambda kv: (-kv[1]["p"], -kv[1]["n"]))
    out[str(tid)] = {"team_id": tid, "team_name": name, "total_points": points[tid],
                    "games_observed": ng, "energy_budget": e_target,
                    "energy_budget_note": e_note, "deck": deck, "check": chk,
                    "cards": [{"cardId": cid, "name": v["name"], "copies": v["n"],
                               "appear_games": v["appear"], "games": ng,
                               "p": round(v["p"], 4), "freq_per_game": round(v["fpg"], 3),
                               "board_lb": v["board_lb"], "play_lb": v["play_lb"],
                               "draw_lb": v["draw_lb"], "attach_total": v["attach"],
                               "cardType": meta[cid]["cardType"], "is_ace_spec": v["is_ace"],
                               "is_basic_energy": bool(v.get("is_energy")),
                               "evidence": v["source"], "confidence": v["conf"]}
                              for cid, v in cards]}
    print(f"\n=== {name}（team {tid}，积分 {points[tid]}，观测 {ng} 局，"
          f"能量预算 {e_target}：{e_note}）"
          f" -> {chk['n']} 张 / {chk['n_distinct']} 种"
          f" | 基础宝可梦 {chk['n_basic_species']} 种 | ACE {chk['ace_spec_total']}"
          f" | 能量合计 {chk['n_energy_total']} | 名>4 {chk['name_over4'] or '无'}"
          f" | 类型 {chk['card_type_mix']}")
    for cid, v in cards:
        tag = "能量" if v.get("is_energy") else ("ACE" if v["is_ace"] else "")
        print(f"    {v['n']}× {v['name']:<32} p={v['p']:.2f}({v['appear']}/{ng}) "
              f"场面{v['board_lb']} 打出{v['play_lb']} 抽{v['draw_lb']} "
              f"附能{v['attach']} {v['source']} {v['conf']} {tag}")
    if ng == 0:
        continue

    csv_path = os.path.join(OUTDIR, f"{base}_reconstructed.csv")
    with io.open(csv_path, "w", encoding="utf-8", newline="") as fh:
        for cid in deck:
            fh.write(f"{cid}\n")

    md = [f"# 还原牌组 · {name}（team_id {tid}）", "",
          f"> **性质声明（§8.3）**：本牌组是从平台公开中文对局日志**推断**出来的，"
          f"**不是**平台登记的官方 deck 列表；能量张数部分是**先验**而非证据。"
          f"逐卡置信度必须一起读。", "",
          f"- 累计积分：{points[tid]}（A 榜）",
          f"- 观测局数：{ng} 局",
          f"- 能量预算：{e_target} 张（{e_note}）；非能量卡预算 {DECK_SIZE - e_target} 张",
          f"- 自检：{chk['n']} 张 / {chk['n_distinct']} 种卡名 / ID 全部命中平台卡池 = {chk['in_pool']}"
          f" / 基础宝可梦 {chk['n_basic_species']} 种 / ACE SPEC 合计 {chk['ace_spec_total']} 张"
          f" / 同名>4 {chk['name_over4'] or '无'}",
          f"- 卡类型分布：{ {TYPE_CN.get(int(k), k): v for k, v in chk['card_type_mix'].items()} }",
          "", "## 方法", "",
          "1. **侧别按行归属**：每条事件行自带 `A 方`/`B 方`；按 `▶` 块头继承会把对手的卡漏进来（实测污染，已修）。",
          "2. **动作只数结算行**：同一动作有 `▶ A 方：打出「X」` 与 `A 方 打出「X」` 两行，只认后者，否则次数翻倍。",
          "3. **附能只认能量卡**：`将「C」附着到` 里 C 也可能是工具（Hop’s Choice Band、Hero’s Cape），必须按 cardType∈{5,6} 过滤。",
          "4. 卡名来自 `「…」`；一名多印刷按名称合并取最小 cardId。",
          f"5. 入选门槛 p ≥ {P_MIN}，p = 该卡至少出现一次的局数 / 观测局数。",
          "6. 非能量初始张数 = `max(场面同局最大同时张数, 单局打出最大次数, 1)`，上限 4；ACE SPEC 固定 1。",
          f"7. 超出非能量预算 {DECK_SIZE - e_target} 则从 p 最低端整种删除（保底 1 种基础宝可梦）；"
          "不足则按 p 降序逐个 +1（上限 4）。",
          f"8. 能量 {e_target} 张按附能总次数占比摊派：基础能量豁免同名 ≤4，特殊能量封顶 4，ACE SPEC 全局 ≤1。"
          "若日志中一种基础能量都没出现，则按牌组内宝可梦的 `energyType` 补对应基础能量。",
          "", "## 局限", "",
          "| 局限 | 影响 |", "| :--- | :--- |",
          "| 只统计日志中被提及的卡 | 起手卡、整局未用的卡会漏，卡种数偏少 |",
          "| 场面解析只覆盖宝可梦 | 道具/支援者张数靠单局打出次数与 p，弱一档 |",
          "| 假设该队在这些局里牌组不变 | 跨轮改牌组会把两套卡的证据混在一起 |",
          "| **能量张数是先验不是证据** | p75 附能 ×1.5 只是经验系数，能量构成最不可靠 |",
          "| 基础能量兜底靠属性推断 | 日志没出现的属性可能被补错 |",
          "| 未与平台 deck 接口比对 | 平台不公开 deck 列表，无法证伪 |",
          "", "## 逐卡证据", "",
          "| cardId | 卡名 | 推断张数 | 出现局数 | p | 次数/局 | 场面上限 | 单局打出上限 | "
          "单局抽到上限 | 附能总次数 | 类型 | 张数依据 | 置信度 |",
          "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |"]
    for cid, v in cards:
        flag = " 🅑" if v.get("is_energy") else (" 🅐" if v["is_ace"] else "")
        md.append(f"| {cid} | {v['name']}{flag} | {v['n']} | {v['appear']}/{ng} | "
                  f"{v['p']:.2f} | {v['fpg']:.2f} | {v['board_lb']} | {v['play_lb']} | "
                  f"{v['draw_lb']} | {v['attach']} | {TYPE_CN.get(meta[cid]['cardType'])} "
                  f"| {v['source']} | {v['conf']} |")
    md += ["", "## 置信度分级口径", "",
           "| 级别 | 判据 | 含义 |", "| :--- | :--- | :--- |",
           "| 高 | p ≥ 0.80 | 几乎每局都出现，卡种判定可靠；张数仍可能因漏统计而偏低 |",
           "| 中 | 0.50 ≤ p < 0.80 | 多数局出现，卡种判定较可靠，是否满 4 张不确定 |",
           f"| 低 | {P_MIN} ≤ p < 0.50 | 仅少数局出现，可能是关键单卡，也可能是回收/换牌噪声 |",
           f"| 未列出 | p < {P_MIN} | 出现但未过门槛，证据不足，未纳入 |",
           "", f"## 门槛之下出现过的卡（p < {P_MIN}，仅参考，未纳入）", ""]
    rest = [r for r in rows if r["p"] < P_MIN]
    if rest:
        md += ["| 卡名 | cardId | 出现局数 | p | 场面上限 | 单局打出上限 |",
               "| :--- | ---: | ---: | ---: | ---: | ---: |"]
        md += [f"| {r['name']} | {r['cid']} | {r['appear']}/{ng} | {r['p']:.2f} | "
               f"{r['board_lb']} | {r['play_lb']} |"
               for r in sorted(rest, key=lambda x: (-x["p"], -x["fpg"]))[:40]]
    else:
        md.append("（无）")
    io.open(os.path.join(OUTDIR, f"{base}_evidence.md"), "w", encoding="utf-8").write(
        "\n".join(md) + "\n")
    print(f"    -> {os.path.relpath(csv_path, ROOT)} / "
          f"{os.path.relpath(os.path.join(OUTDIR, base + '_evidence.md'), ROOT)}")

with open(os.path.join(OUTDIR, "reconstructed.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n[saved]", os.path.join(OUTDIR, "reconstructed.json"))
