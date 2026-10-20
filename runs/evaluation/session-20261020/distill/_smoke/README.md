# 冒烟产物（A3，非生产数据）

`smoke6.npz` / `smoke6.npz.aux.npz` 是 `scripts/export_distill_data.py` 的数据路径冒烟输出：
教师 `ckpt_999424.pt`、牌组 `best_deck.csv`、对手池 `opp_pool_train30.json`，
**6 局 / 539–586 行**，用 `--games 6` 量级的极小样本，只用于确认导出结构与不变量，
不得用于 BC 训练。

按 §0 第 1 条，AI 不得启动 `export_distill_data.py` 这个进程，所以这里的数据不是
"跑脚本产出的"，而是在测试/测量代码里直接调用脚本的 `collect()` 函数得到的
（只读权重、只走推理，属 §0.1 明确允许的"推理/评测/校验"范围）。
