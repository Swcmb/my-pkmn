# 追加一条证据完整性说明
import io
P = r"D:\pkmn-kaggle\runs\evaluation\audit_report.md"
s = io.open(P, encoding="utf-8").read()
old = """| 证据目录 | `runs/evaluation/audit_artifacts/`（新增日志与汇总表）、`runs/evaluation/*.json` |"""
new = """| 证据目录 | `runs/evaluation/audit_artifacts/`（新增日志与汇总表）、`runs/evaluation/*.json` |
| 版本一致性 | `eval_h2h.py` 的 mtime（今日 12:21）早于全部 `eval_*_100.json`（12:25-12:39），即已有 100 局数据产生后评估代码未被改动；本轮新增 JSON 均由同一版本产生 |"""
assert s.count(old) == 1
io.open(P, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("OK")
