# 三处交叉引用修正
import io, sys
P = r"D:\pkmn-kaggle\runs\evaluation\audit_report.md"
s = io.open(P, encoding="utf-8").read()
R = [
 ("   决策就会崩。这是当前提交包的首要阻塞，修复方式见 §5.7。",
  "   决策就会崩。这是当前提交包的首要阻塞，修复方式见 §5.6。"),
 ("| 引擎 | 官方 `kaggle_environments` wheel 的 CABT 二进制（`cg.dll` 1,659,392 B；来源见 §5.3） |",
  "| 引擎 | 官方 `kaggle_environments` wheel 的 CABT 二进制（`cg.dll` 1,659,392 B；来源见 §5.2） |"),
 ("- 真实 ckpt 路径与步数已核对：所有 `eval_*.json`/`ctl*.json` 顶层 `cand_ckpt/cand_step` 与\n  meta 中的对手路径/步数均指向实际文件（`audit_artifacts/audit_subs.log` 同批验证）。",
  "- 真实 ckpt 路径与步数已核对：所有 `eval_*.json`/`ctl*.json` 顶层 `cand_ckpt/cand_step` 与\n  meta 中的对手路径/步数均指向实际文件（`audit_artifacts/audit_meta.log` 逐行记录）。"),
]
for i, (o, n) in enumerate(R, 1):
    c = s.count(o)
    if c != 1:
        print("REPL %d matched %d -> abort" % (i, c)); sys.exit(1)
    s = s.replace(o, n)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("OK cross-refs fixed")
