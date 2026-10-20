import ctypes
import os
import platform
import sys

class StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]

class SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int)
    ]

def _engine_candidates():
    """包内二进制优先；包内缺失时指向官方 kaggle_environments 安装，避免再分发引擎。"""
    here = os.path.dirname(os.path.abspath(__file__))
    os_name = platform.system()
    if os_name == "Windows":
        names = ["cg.dll"]
    elif os_name == "Darwin":
        names = ["libcg.dylib"]
    elif platform.machine() in ("arm64", "aarch64"):
        names = ["libcg-arm64.so"]
    else:
        names = ["libcg.so"]
    for n in names:
        yield os.path.join(here, n)
    try:
        import kaggle_environments.envs.cabt.cg as official_cg
        for n in names:
            yield os.path.join(os.path.dirname(official_cg.__file__), n)
    except Exception:
        pass


def _load_lib():
    # 平台 harness 已加载官方引擎时复用同一句柄，避免同一进程二次 LoadLibrary/初始化。
    pre = sys.modules.get("kaggle_environments.envs.cabt.cg.sim")
    if pre is not None and getattr(pre, "lib", None) is not None:
        return pre.lib, False
    for p in _engine_candidates():
        if os.path.exists(p):
            return ctypes.cdll.LoadLibrary(p), True
    raise FileNotFoundError(
        "cabt engine binary not found: bundle sdk_cg/cg.dll|libcg.so or install the official "
        "kaggle-environments package (kaggle_environments.envs.cabt.cg)")


lib, _loaded_here = _load_lib()
if _loaded_here:
    lib.GameInitialize()
