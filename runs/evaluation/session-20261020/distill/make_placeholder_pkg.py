# -*- coding: utf-8 -*-
"""A2 · 极小占位包：<1 MiB 的合法提交包，用来确认两件从未验证过的事
   ① 平台是否接受 main.py/deck.csv 之外的多文件结构；② 沙箱是否预装 torch。

做法：随机初始化一个 64/4/3/128 小网（**不训练**，只构造 + 存权张量），
然后复用正式导出脚本 `scripts/export_rl_submission.py` 打包 ——
这样体积门禁（`--max-bytes`）与 `torch.set_num_threads(2)` 补丁都走同一条已测过的路径，
且**不修改正式模板**（§4.1 明确要求）。

产物全部落在 distill/placeholder/ 下的新名字，`--out`/`--builddir` 存在即拒绝，不覆盖任何东西。

  PKMN_TURN_CAP=200 PYTHONPATH=D:\\pkmn-kaggle\\.engine-shim;D:\\pkmn-kaggle \\
  python runs/evaluation/session-20261020/distill/make_placeholder_pkg.py [--d-model 64 ...]

本脚本不产生任何训练效果：权重来自 `build_token_net` 的随机初始化，没有优化器、
没有梯度、没有对任何已有 ckpt 的写入。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = r"D:\pkmn-kaggle"
sys.path.insert(0, ROOT)
SESS = os.path.join(ROOT, "runs", "evaluation", "session-20261020")
OUTDIR = os.path.join(SESS, "distill", "placeholder")
SHIP_DECK = os.path.join(ROOT, "runs", "evaluation", "card_database_migration", "best_deck.csv")
MAX_BYTES = 1_047_000


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=3)
    ap.add_argument("--ff", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--tag", default="placeholder", help="产物文件名前缀（须为新名字）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行的导出命令，不落盘")
    a = ap.parse_args()

    cfg = {"arch": "transformer2", "d_model": a.d_model, "nhead": a.nhead,
           "nlayers": a.nlayers, "ff": a.ff, "static": False, "would_ko": True,
           "split_heads": False, "value_categorical": False,
           "value_atoms": 51, "value_vmax": 1.0}
    out_tar = os.path.join(OUTDIR, f"{a.tag}.tar.gz")
    build = os.path.join(OUTDIR, a.tag)
    ckpt = os.path.join(OUTDIR, f"{a.tag}_untrained.pt")
    for p in (out_tar, build):
        if os.path.exists(p):
            raise SystemExit(f"ERROR: 目标已存在，拒绝覆盖：{p}（换 --tag）")

    cmd = [sys.executable, os.path.join(ROOT, "scripts", "export_rl_submission.py"),
           "--ckpt", ckpt, "--backend", "transformer2", "--deck", SHIP_DECK,
           "--out", out_tar, "--builddir", build, "--max-bytes", str(MAX_BYTES)]
    if a.dry_run:
        print("[dry-run] 1) 先写随机初始化小网 ckpt：", ckpt, cfg)
        print("[dry-run] 2) 再调用正式导出脚本：")
        print("    " + " ".join(cmd))
        return

    import torch
    from rl.card_features import get_card_table
    from rl.policy import build_token_net
    torch.set_grad_enabled(False)
    torch.manual_seed(a.seed)
    os.makedirs(OUTDIR, exist_ok=True)
    ct = get_card_table(os.path.join(ROOT, "EN_Card_Data.csv"))
    net = build_token_net(ct, cfg)
    n_params = sum(p.numel() for p in net.parameters())
    torch.save({"net": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "net_config": cfg, "global_step": 0}, ckpt)
    print(f"[placeholder] 随机初始化 {a.d_model}/{a.nhead}/{a.nlayers}/{a.ff} "
          f"= {n_params:,} 参数 -> {ckpt}（{os.path.getsize(ckpt):,} B）")
    fp16_bytes = n_params * 2
    print(f"[placeholder] 权重预算核对：fp32 存 {n_params*4:,} B / 若 fp16 {fp16_bytes:,} B；"
          f"1 MiB 包内权重上限约 926,000 B")

    env = dict(os.environ, PYTHONPATH=r"D:\pkmn-kaggle\.engine-shim;D:\pkmn-kaggle",
               PKMN_TURN_CAP="200")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    print(p.stdout[-2500:])
    if p.returncode != 0:
        print(p.stderr[-2500:])
        raise SystemExit(f"ERROR: 导出失败 rc={p.returncode}")

    size = os.path.getsize(out_tar)
    members = sorted(os.listdir(build))
    report = {"kind": "untrained placeholder (no training, random init)",
              "net_config": cfg, "params": n_params, "init_seed": a.seed,
              "ckpt": os.path.abspath(ckpt), "ckpt_sha256": sha(ckpt),
              "tar": os.path.abspath(out_tar), "tar_sha256": sha(out_tar), "tar_bytes": size,
              "max_bytes": MAX_BYTES, "fits": size < MAX_BYTES,
              "deck": os.path.abspath(SHIP_DECK), "deck_sha256": sha(SHIP_DECK),
              "builddir": os.path.abspath(build), "top_level_members": members}
    with open(os.path.join(OUTDIR, f"{a.tag}_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n[placeholder] {out_tar}  {size:,} B  ({size/MAX_BYTES:.1%} of {MAX_BYTES:,})  "
          f"fits={size < MAX_BYTES}")
    print("[placeholder] 顶层成员:", members)
    print("[placeholder] report ->", os.path.join(OUTDIR, f"{a.tag}_report.json"))
    if size >= MAX_BYTES:
        raise SystemExit(f"ERROR: 占位包 {size:,} B 仍超预算，需要降级架构（--nlayers 2）")


if __name__ == "__main__":
    main()
