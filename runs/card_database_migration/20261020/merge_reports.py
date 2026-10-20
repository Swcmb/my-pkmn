# -*- coding: utf-8 -*-
"""把本次迁移的六份报告合并成单文件，原文逐段保留，只做标题层级下沉与目录生成。

标题下沉对 fenced code block 内的行不生效（避免改动命令与日志原文）。
不删除、不修改任何来源文件。
"""
import datetime
import os
import re

ROOT = r"D:\pkmn-kaggle"
MIG = os.path.join(ROOT, "runs", "card_database_migration", "20261020")
EVAL = os.path.join(ROOT, "runs", "evaluation", "card_database_migration")
OUT = os.path.join(EVAL, "CARD_DATABASE_MIGRATION_ALL_REPORTS.md")

SOURCES = [
    ("阶段 1 · 备份 / 校验 / schema 与 ID 对比（含方案 A 落地结果）",
     os.path.join(MIG, "stage1_migration_report.md")),
    ("牌组验证与双向交叉评估", os.path.join(EVAL, "deck_validation_report.md")),
    ("Linux（WSL2）真实运行时验证（含最终包复验）", os.path.join(EVAL, "linux_runtime_report.md")),
    ("评估门禁重验（迁移后新库）", os.path.join(EVAL, "final_gate_report.md")),
    ("最终导出前门禁核对表", os.path.join(EVAL, "pre_export_checklist.md")),
    ("最终综合报告", os.path.join(EVAL, "final_report.md")),
]

HEAD_RE = re.compile(r"^(#{1,6})(\s.*)?$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def demote(text):
    """所有 ATX 标题降一级（#→##），围栏代码块内保持原样。"""
    out, in_fence = [], False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence:
            m = HEAD_RE.match(line)
            if m:
                hashes = m.group(1)
                if len(hashes) < 6:
                    line = "#" + line
        out.append(line)
    return "\n".join(out)


PENDING = """# 待用户决定的两项与微调状态

## 待决定 1 · 是否先把卡牌库扩到引擎真值 1431 卡

引擎 1.33.0 实际卡池为 1431 张，而两份 CSV 都只覆盖 ID 1–1267；ID 1268–1431 的 164 张新卡
（140 宝可梦 / 8 支援者 / 6 物品 / 5 特殊能量 / 3 竞技场 / 2 道具）在编码器里被
`np.clip(..., 0, UNK)` 夹到 UNK 行（`rl/encoding.py:1416`）。
先扩库再微调，策略才能真正“看见”这些牌；直接微调会继续学习一张看不见的牌。
代价：`card_emb.weight` 需从 `(1269, 544)` 扩到 `(1432, 544)`，新增 163 行的初始化方式
（复制 UNK 行 / 零初始化）与是否需要重训补偿，属需评估的高影响改动。

## 待决定 2 · `best_deck.csv` 是否加入 `rl/decks_train.py` 的 `TRAIN_TOP50` 池

`--finetune-deck` 只接受池内**牌组名**（`rl/train_selfplay.py:312`），
`--gate-decks-file` 接受的是池下标 JSON 列表（同文件 L294），两者都无法直接指向牌组文件。
实测 `TRAIN_TOP50` 共 74 张池内牌组，`best_deck.csv` 的 60 张序列**无同序列匹配**，
最接近的是 `k06_dragapult_ex_fezandipiti_ex_latias_ex_me`（与提交牌组不同）。
选项：(a) 把 `best_deck.csv` 作为新命名条目入池（一处数据改动，需批准）；
(b) 用池内 `k06_dragapult…` 训练并接受差异；(c) 不入池，微调只做通用自博弈。
本次未擅自修改池文件。

## 0.5M 微调状态

配置、预计耗时（约 1.6 h 纯训练 / 规划 2 h 机器时间，按实测 88.095 env-step/s，
来源 `runs/bench-20m/report.md` E16 推荐配置）与六项验收阈值见合订本第 6 部分 §9。
**训练未启动**，等待用户明确决定。

## 本次会话用量

约 56 轮、3376 秒。
"""


def main():
    now = datetime.date.today().isoformat()
    parts = [f"""# 卡牌库迁移 · 全部报告合订

生成日期：{now}　项目：`D:\\pkmn-kaggle`
最终提交包：`submission_rl_transformer2_db_migrated.tar.gz`
SHA-256 `8d4ec3ccdebd563c259efa12bfe1061493dcaff7c1f0154fcfff6f128d58ae71`（75,126,979 B）

本文件按执行顺序合订本次迁移的六份报告，正文逐段保留原文（仅标题下沉一级，代码块与日志原文不动）。
来源文件仍然存在且未被修改：

| 部分 | 来源文件 |
|:---|:---|
"""]
    for i, (title, path) in enumerate(SOURCES, start=1):
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        parts[0] += f"| 第 {i} 部分 · {title} | `{rel}` |\n"
    parts.append("\n" + PENDING + "\n")

    toc = ["## 目录\n"]
    for i, (title, _p) in enumerate(SOURCES, start=1):
        toc.append(f"{i}. 第 {i} 部分 · {title}")
    toc.append(f"{len(SOURCES) + 1}. 第 0 部分 · 待用户决定的两项与微调状态（已前置）\n")
    parts.insert(1, "\n".join(toc) + "\n---\n")

    for i, (title, path) in enumerate(SOURCES, start=1):
        text = open(path, encoding="utf-8").read().rstrip()
        parts.append(f"\n---\n\n# 第 {i} 部分 · {title}\n\n" + demote(text) + "\n")

    body = "".join(parts)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    print("写出:", OUT, len(body), "字符")
    print("一级标题:")
    for line in body.splitlines():
        if re.match(r"^# ", line):
            print("  ", line[:90])


if __name__ == "__main__":
    main()
