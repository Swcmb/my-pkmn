# 提交包"按原样"验证：tar 内容 + 不含引擎二进制时的真实失败点
import importlib.util, json, os, shutil, sys, tarfile

ROOT = r"D:\pkmn-kaggle"
TAR = os.path.join(ROOT, "submission_rl_transformer2.tar.gz")
CLEAN = r"C:\Users\Swcmb\AppData\Local\Temp\opencode\subclean"
print("== tar 成员 ==")
with tarfile.open(TAR) as t:
    names = t.getnames()
    for n in names:
        m = t.getmember(n)
        print("  %-28s %10d B" % (n, m.size))
    print("  member_count =", len(names))
    print("  含二进制(.dll/.so/.dylib) =", [n for n in names if n.endswith((".dll", ".so", ".dylib"))])

print("\n== 按原样解包 → 干净环境 import/推理 ==")
if os.path.exists(CLEAN):
    shutil.rmtree(CLEAN)
os.makedirs(CLEAN)
with tarfile.open(TAR) as t:
    t.extractall(CLEAN)
sys.path.insert(0, CLEAN)
spec = importlib.util.spec_from_file_location("cleanmain", os.path.join(CLEAN, "main.py"))
main = importlib.util.module_from_spec(spec)
sys.modules["cleanmain"] = main
try:
    spec.loader.exec_module(main)
    print("import main: OK; _WK_ON =", main._WK_ON)
except Exception as e:
    print("import main: FAIL", type(e).__name__, e)
    raise SystemExit

deck = main.agent({"select": None})
print("deck 请求分支: OK, len =", len(deck) if isinstance(deck, list) else deck)

# 触发 would_ko 标注路径的第一步（annotate_would_ko 内部执行 `from sdk_cg import api`）
try:
    from sdk_cg import api                                     # 需要 sdk_cg/cg.dll
    print("sdk_cg import: OK")
except Exception as e:
    print("sdk_cg import: FAIL -> %s: %s" % (type(e).__name__, e))
    print("  => 首个需要 would_ko 标注的决策会在 search_agent.would_ko_flags 内部此处抛出")

