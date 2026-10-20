"""Teacher rollout dumper -> BC-consumable .npz (route A, stage A3).

Writes the SAME flat layout `scripts/bc_train.py` already reads: one array per TokenEncoder obs key
(int keys -> int32, others -> float32) plus `__labels__` (int64) = the teacher's argmax action.
Rows stay grouped by episode so bc_train's contiguous-tail game-level validation split is leak-free.
bc_train treats EVERY npz key except `__labels__` / `__is_attack__` / `__group__` as an obs feature,
so the extra per-row columns (teacher value, acting seat, game id, terminal outcome of that seat) go
into a sidecar `<out>.aux.npz` aligned 1:1 by row index, never into the training file.
`__is_attack__` / `__group__` are NOT produced: the collector path has no attack signal, and a
canonical-option group map would need a dedup builder this repo does not contain, so
`bc_train --dedup` cannot be used on files from here.

Reads weights, writes data only: no optimizer, no checkpoint, no in-place tensor update.

Engine note: this machine has no `sdk_cg/cg.dll`, so `from sdk_cg import api` (needed by the would_ko
annotation path) raises on every decision. The proven workaround is
`runs/evaluation/eval_h2h.py::_install_sdk_cg_shim_alias`, imported here instead of duplicated so the
two paths cannot drift; it no-ops when the real dll is present.

  PYTHONPATH=D:\\pkmn-kaggle\\.engine-shim;D:\\pkmn-kaggle  PKMN_TURN_CAP=200
  python scripts/export_distill_data.py --ckpt <teacher.pt> --games 2000 --seed 20261020
      --deck runs/evaluation/card_database_migration/best_deck.csv
      --ladder-json runs/evaluation/session-20261020/opp_pool_train30.json
      --out runs/evaluation/session-20261020/distill/teacher_data.npz

Rails: `--max-ram-gb` refuses up front when the projected dataset cannot fit in RAM,
`--limit-decisions` caps rows, `--shard k/n` splits games across runs, `--max-bytes` warns on output
size, and the end-of-run self-check requires masked_rate == 0 (every stored label legal under its own
action_mask).
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("PKMN_TURN_CAP", "200")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "runs", "evaluation"))

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402

from rl.card_features import get_card_table                   # noqa: E402
from rl.encoding import TokenEncoder, build_mask              # noqa: E402
from rl.env_selfplay import TwoSidedSelfPlayEnv               # noqa: E402
from rl.policy import bucketed_opt_len_np, build_token_net    # noqa: E402


def log(msg):
    print(msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_csv_deck(path):
    ids = [int(x) for row in csv.reader(open(path, encoding="utf-8-sig")) for x in row if x.strip()]
    if len(ids) != 60:
        raise SystemExit(f"ERROR: {path} 有 {len(ids)} 张，应为 60")
    return ids


def load_net(path, ct, dev):
    """weights_only=True：ckpt 里只有 net 张量 / net_config 标量字典 / global_step。"""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = build_token_net(ct, ck["net_config"])
    net.load_state_dict(ck["net"])
    net.eval().to(dev)
    return net, int(ck.get("global_step", -1)), ck["net_config"]


class FixedDeckEnv(TwoSidedSelfPlayEnv):
    """固定两副牌（seat0 = decks[0]），不做牌池随机采样。"""

    def _sample_decks(self):
        self._deck_idx = (0, 1)
        return self.decks[0], self.decks[1]


def make_env(cand_deck, opp_deck, cand_seat, enc, seed):
    d0, d1 = (cand_deck, opp_deck) if cand_seat == 0 else (opp_deck, cand_deck)
    return FixedDeckEnv(decks=[list(d0), list(d1)], encoder=enc, would_ko=True,
                        seed=int(seed), native_encode=False)


def teacher_pick(net, int_keys, dev, obs):
    """返回 (argmax 动作下标, 状态 value)。obs 是单行 dict。

    opt_len 走 numpy 口径（bucketed_opt_len_np 是 numpy 实现），网络输入走 torch 张量，
    与 runs/evaluation/eval_h2h.py 的批处理路径保持一致：int 键 long、其余 float32。
    """
    np_batch = {k: np.asarray(v)[None, ...] for k, v in obs.items()}
    opt_len = bucketed_opt_len_np(np_batch["action_mask"], np_batch["opt_attr"])
    batch = {k: torch.as_tensor(v, dtype=(torch.long if k in int_keys else torch.float32),
                               device=dev) for k, v in np_batch.items()}
    with torch.no_grad():
        logits, value = net.logits_value(batch, opt_len=opt_len)
    return int(logits[0].argmax(-1).item()), float(value.reshape(-1)[0].item())


def close(env):
    try:
        env.close()
    except Exception:
        pass


def collect(net, enc, int_keys, dev, cand_deck, opp_pool, games, seed, selfplay,
            limit_decisions, game_filter):
    """逐局跑教师（selfplay=True 时两侧都由同一教师驾驶），按行落数据。"""
    rng = np.random.default_rng(seed)
    rows, labels, values, seats, gids = [], [], [], [], []
    metas, rets = [], []
    games_done = reset_fail = illegal_pick = rows_dropped = skipped_shard = 0
    t0 = last = time.time()
    for gi in range(games):
        if not game_filter(gi):
            skipped_shard += 1
            continue
        if limit_decisions and len(rows) >= limit_decisions:
            log(f"[dump] 命中 --limit-decisions={limit_decisions}，停在第 {gi} 局")
            break
        cand_seat = gi % 2                      # 严格均衡候选牌组所在座位
        opp_deck = opp_pool[int(rng.integers(len(opp_pool)))]
        env = make_env(cand_deck, opp_deck, cand_seat, enc, seed + gi)
        try:
            obs, seat, _info = env.reset()
        except Exception as e:
            reset_fail += 1
            log(f"[dump] reset 失败 game {gi}: {type(e).__name__}: {e}")
            close(env)
            continue
        n0 = len(rows)
        trunc = True                            # 未拿到终局一律按截断处理
        terminal = None
        steps = 0
        while True:
            steps += 1
            acting = int(seat)
            piloting = selfplay or acting == cand_seat
            if limit_decisions and len(rows) >= limit_decisions:
                break
            act, val = teacher_pick(net, int_keys, dev, obs)
            mask = build_mask(env._obs["select"], set(env._picked))
            if act >= len(mask) or mask[act] == 0:
                illegal_pick += 1               # 非法下标：不入库，让 env 自行 clamp 继续跑
            elif piloting:
                rows.append({k: np.asarray(v) for k, v in obs.items()})
                labels.append(act)
                values.append(val)
                seats.append(acting)
                gids.append(games_done)
            obs, seat, _r, done, info = env.step(act)
            if done:
                trunc = bool(info.get("trunc_turncap") or info.get("truncated"))
                terminal = info.get("terminal")
                break
        games_done += 1
        if trunc or terminal is None:
            rows_dropped += len(rows) - n0      # 截断/无终局：整局行丢弃
            del rows[n0:], labels[n0:], values[n0:], seats[n0:], gids[n0:]
            ret = None
        else:
            ret = [int(terminal[0]), int(terminal[1])]
            rets.extend(ret[seats[i]] for i in range(n0, len(rows)))
        metas.append({"game": gi, "cand_seat": cand_seat, "steps": steps,
                      "stored_rows": len(rows) - n0, "truncated": bool(trunc),
                      "terminal": ret})
        close(env)
        now = time.time()
        if now - last >= 20:
            last = now
            log(f"[dump] {games_done} games -> {len(rows)} rows "
                f"({len(rows) / max(1e-9, now - t0):.1f} rows/s，"
                f"截断丢弃 {rows_dropped}，reset 失败 {reset_fail})")
    if len(rets) != len(rows):
        raise SystemExit(f"ERROR: __ret__ 行数 {len(rets)} != 数据行 {len(rows)}")
    return dict(rows=rows, labels=labels, values=values, seats=seats, gids=gids, rets=rets,
                metas=metas, games=games_done, reset_fail=reset_fail,
                illegal_pick=illegal_pick, rows_dropped=rows_dropped,
                skipped_shard=skipped_shard, seconds=time.time() - t0)


def stack_dataset(out, int_keys):
    """按 bc_train 的 dtype 约定堆成 (训练集 dict, 辅助列 dict)。

    bc_train 把 npz 里除 `__labels__` / `__is_attack__` / `__group__` 之外的**每一个键**都当作
    观测特征喂给网络，所以 value/seat/game/ret 这些附加列绝不能混在同一个 npz 里，
    它们进 sidecar（行数与训练集严格 1:1 对齐）。
    """
    keys = list(out["rows"][0].keys())
    ds = {}
    for k in keys:
        dt = np.int32 if k in int_keys else np.float32
        ds[k] = np.stack([r[k] for r in out["rows"]]).astype(dt)
    ds["__labels__"] = np.asarray(out["labels"], dtype=np.int64)
    aux = {"value": np.asarray(out["values"], dtype=np.float32),
           "seat": np.asarray(out["seats"], dtype=np.int8),
           "game": np.asarray(out["gids"], dtype=np.int64),
           "ret": np.asarray(out["rets"], dtype=np.int8),
           "row_index": np.arange(len(ds["__labels__"]), dtype=np.int64)}
    return ds, aux, keys


def selfcheck(ds):
    """每条 label 在自己的 action_mask 下必须合法（masked_rate 必须为 0）。"""
    am, lab = ds["action_mask"], ds["__labels__"]
    masked = float((am[np.arange(len(lab)), lab] < 0.5).mean())
    dist = np.bincount(lab, minlength=am.shape[1])
    top = np.argsort(-dist)[:3]
    return masked, [(int(i), int(dist[i]), round(float(dist[i]) / len(lab), 4)) for i in top]


def main():
    p = argparse.ArgumentParser(description="教师 rollout -> BC .npz（只读权重，不写 ckpt）")
    p.add_argument("--ckpt", required=True, help="教师 ckpt（transformer2）")
    p.add_argument("--deck", default=os.path.join(_ROOT, "runs", "evaluation",
                                                  "card_database_migration", "best_deck.csv"))
    p.add_argument("--ladder-json", required=True, help="对手牌组池 JSON（load_ladder_decks 格式）")
    p.add_argument("--out", required=True, help="输出 .npz（新文件名，存在即拒绝覆盖）")
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-decisions", type=int, default=0,
                   help="0 = 不限；命中上限时未打完的最后半局会被整局丢弃（可能少于上限）")
    p.add_argument("--shard", type=str, default="", help="k/n：只跑 game 索引除以 n 余 k 的局")
    p.add_argument("--opp-policy", choices=["teacher", "random"], default="teacher",
                   help="teacher=自博弈，两侧都入库；random=对手随机合法，只入库教师侧")
    p.add_argument("--max-ram-gb", type=float, default=24.0, help="预计数据量超此值则直接拒绝启动")
    p.add_argument("--max-bytes", type=int, default=0, help="0 = 不检查（这是训练数据，非提交包）")
    p.add_argument("--row-bytes", type=int, default=40960,
                   help="单行字节数估计，用于 RAM 预测（实测 38.2 KB/行，取整偏保守）")
    p.add_argument("--rows-per-game", type=int, default=98,
                   help="每局决策数估计，用于 RAM 预测（6 局实测 97.7，取 98 偏保守）")
    a = p.parse_args()

    if os.path.exists(a.out):
        raise SystemExit(f"ERROR: --out 已存在，拒绝覆盖：{a.out}")
    if not a.out.endswith(".npz"):
        raise SystemExit("ERROR: --out 必须以 .npz 结尾")

    from eval_h2h import _install_sdk_cg_shim_alias
    _install_sdk_cg_shim_alias()
    from rl.decks_ladder import load_ladder_decks
    opp_pool, _ = load_ladder_decks(a.ladder_json)
    cand_deck = load_csv_deck(a.deck)

    ct = get_card_table()
    enc = TokenEncoder(ct)
    int_keys = set(enc.int_keys)
    dev = torch.device(a.device)
    net, step, nc = load_net(a.ckpt, ct, dev)
    if nc.get("arch") != "transformer2":
        raise SystemExit(f"ERROR: 教师 arch={nc.get('arch')}，本脚本只支持 transformer2")
    log(f"[dump] 教师 {a.ckpt} step={step} config={nc} params="
        f"{sum(v.numel() for v in net.parameters()):,}")

    proj_rows = a.games * a.rows_per_game
    proj_gb = proj_rows * a.row_bytes / 1e9
    log(f"[dump] 预计 {a.games} 局 ≈ {proj_rows:,} 行 ≈ {proj_gb:.1f} GB（"
        f"按 {a.row_bytes:,} B/行；上限 --max-ram-gb={a.max_ram_gb}）")
    if proj_gb > a.max_ram_gb:
        raise SystemExit(f"ERROR: 预计 {proj_gb:.1f} GB 超过 --max-ram-gb={a.max_ram_gb}；"
                         f"减小 --games 或用 --shard 分次跑")

    shard_k = shard_n = None
    if a.shard:
        k, n = a.shard.split("/")
        shard_k, shard_n = int(k), int(n)
        if not 0 <= shard_k < shard_n:
            raise SystemExit(f"ERROR: --shard 需要 0<=k<n，收到 {a.shard}")
        log(f"[dump] 分片 {shard_k}/{shard_n}：只跑 game 索引 % {shard_n} == {shard_k} 的局")
    game_filter = (lambda gi: True) if shard_n is None else (lambda gi: gi % shard_n == shard_k)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = collect(net=net, enc=enc, int_keys=int_keys, dev=dev, cand_deck=cand_deck,
                  opp_pool=opp_pool, games=a.games,
                  seed=a.seed + (shard_k or 0) * 1000003,
                  selfplay=(a.opp_policy == "teacher"),
                  limit_decisions=a.limit_decisions, game_filter=game_filter)
    if not out["rows"]:
        raise SystemExit("ERROR: 一条数据都没采到，检查上面 reset 失败原因")

    ds, aux, keys = stack_dataset(out, int_keys)
    masked, top3 = selfcheck(ds)
    if masked != 0.0:
        raise SystemExit(f"ERROR: masked_rate={masked} != 0，标签与掩码不一致，拒绝落盘")
    if len(aux["row_index"]) != len(ds["__labels__"]):
        raise SystemExit("ERROR: sidecar 行数与训练集不一致")
    meta = {"teacher_ckpt": os.path.abspath(a.ckpt), "teacher_sha256": sha256(a.ckpt),
            "teacher_global_step": step, "net_config": nc,
            "deck": os.path.abspath(a.deck), "deck_sha256": sha256(a.deck),
            "ladder_json": os.path.abspath(a.ladder_json),
            "ladder_sha256": sha256(a.ladder_json), "opp_pool_size": len(opp_pool),
            "card_table_sha256": sha256(os.path.join(_ROOT, "EN_Card_Data.csv")),
            "seed": a.seed, "shard": a.shard or "none", "games_requested": a.games,
            "games_done": out["games"], "reset_fail": out["reset_fail"],
            "skipped_by_shard": out["skipped_shard"],
            "decisions": int(len(ds["__labels__"])), "rows_dropped_truncated": out["rows_dropped"],
            "illegal_teacher_picks": out["illegal_pick"], "masked_rate": masked,
            "label_top3_action": top3, "opp_policy": a.opp_policy,
            "turn_cap": os.environ.get("PKMN_TURN_CAP"), "native_encode": False,
            "would_ko": True, "obs_keys": keys, "columns": sorted(ds.keys()),
            "sidecar": os.path.abspath(a.out) + ".aux.npz", "sidecar_columns": sorted(aux.keys()),
            "seconds": round(out["seconds"], 1), "produced_by": "scripts/export_distill_data.py",
            "per_game": out["metas"]}
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, **ds)
    size = os.path.getsize(tmp)
    if a.max_bytes and size > a.max_bytes:
        os.replace(tmp, a.out + ".oversized")
        raise SystemExit(f"ERROR: 输出 {size:,} B 超过 --max-bytes={a.max_bytes:,}；"
                         f"残件保留为 {a.out}.oversized 供检查")
    os.replace(tmp, a.out)
    np.savez(a.out + ".aux.npz", **aux)
    with open(a.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    log(f"[dump] wrote {a.out}: {meta['decisions']} rows / {out['games']} games  "
        f"{size / 1e9:.2f} GB  masked_rate={masked:.6f}  "
        f"截断丢弃 {out['rows_dropped']} 非法下标 {out['illegal_pick']} "
        f"reset失败 {out['reset_fail']}")
    log(f"[dump] sidecar {a.out}.aux.npz（{sorted(aux.keys())}，与训练集行号 1:1 对齐，"
        f"bc_train 不读它）")
    log(f"[dump] 动作分布 top3 {top3}")


if __name__ == "__main__":
    main()
