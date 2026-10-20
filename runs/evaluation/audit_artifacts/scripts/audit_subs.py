# 审计脚本 2：提交包差异、would_ko 调用链符号依赖、bench-shim 引擎溯源
import difflib, os, re, ctypes, hashlib

ROOT = r"D:\pkmn-kaggle"
SUB = os.path.join(ROOT, "submission_rl_transformer2")

def diff_summary(rel, n=200):
    a = open(os.path.join(SUB, rel), encoding="utf-8").readlines()
    b = open(os.path.join(ROOT, "rl", rel), encoding="utf-8").readlines()
    d = list(difflib.unified_diff(b, a, "rl/" + rel, "sub/" + rel, n=1))
    print("==== %s : %d diff lines ====" % (rel, len(d)))
    print("".join(d[:n]))

diff_summary("encoding.py")
diff_summary("search_agent.py")

print("\n==== search_agent: sdk 调用与 WouldKO 类符号使用 ====")
src = open(os.path.join(SUB, "search_agent.py"), encoding="utf-8").read()
for ln in re.findall(r"^(?:.*(?:from sdk_cg|import api|api\.\w+|lib\.\w+|WouldKO|GetSerialObs|GetBinaryObs|AdvanceLog).*)$", src, re.M):
    print("  " + ln.strip())

print("\n==== 全仓库 WouldKO/GetSerialObs/GetBinaryObs/AdvanceLog 引用 ====")
for dirpath, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
    for fn in files:
        if not fn.endswith((".py", ".pyx", ".h", ".md", ".sh")):
            continue
        p = os.path.join(dirpath, fn)
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        hits = [s for s in ("WouldKO", "GetSerialObs", "GetBinaryObs", "AdvanceLog") if s in t]
        if hits:
            print("  %-55s %s" % (os.path.relpath(p, ROOT), hits))

print("\n==== bench-shim 引擎溯源 ====")
SHIM = r"C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim"
for root_, dirs, files in os.walk(SHIM):
    for fn in files:
        p = os.path.join(root_, fn)
        st = os.stat(p)
        print("  %-70s %10d B  mtime=%s" % (os.path.relpath(p, SHIM), st.st_size,
              __import__("datetime").datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")))
si = os.path.join(SHIM, "kaggle_environments", "envs", "cabt", "cg", "sim.py")
if os.path.exists(si):
    print("---- shim sim.py (全文) ----")
    print(open(si, encoding="utf-8", errors="ignore").read())

print("\n==== shim cg.dll 导出 ====")
try:
    lib = ctypes.cdll.LoadLibrary(os.path.join(SHIM, "kaggle_environments", "envs", "cabt", "cg", "cg.dll"))
    for s in ("GameInitialize", "BattleStart", "BattleFinish", "GetBattleData", "Select",
              "VisualizeData", "AgentStart", "SearchBegin", "SearchStep", "SearchEnd",
              "SearchRelease", "AllCard", "AllAttack", "WouldKO", "GetSerialObs",
              "GetBinaryObs", "AdvanceLog"):
        try:
            getattr(lib, s); print("  EXPORT", s)
        except AttributeError:
            print("  missing", s)
except Exception as e:
    print("  LoadLibrary fail:", type(e).__name__, e)
