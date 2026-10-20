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


_OFFICIAL_SIM = "kaggle_environments.envs.cabt.cg.sim"


def _load_lib():
    # 修复前：仅当官方模块已被导入时才复用句柄；自己先加载则不管官方模块稍后的二次初始化。
    pre = sys.modules.get("kaggle_environments.envs.cabt.cg.sim")
    if pre is not None and getattr(pre, "lib", None) is not None:
        return pre.lib, False
    for p in _engine_candidates():
        if os.path.exists(p):
            return ctypes.cdll.LoadLibrary(p), True
    raise FileNotFoundError("cabt engine binary not found")


lib, _loaded_here = _load_lib()
if _loaded_here:
    lib.GameInitialize()

lib.BattleStart.restype = StartData
lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]

lib.AgentStart.restype = ctypes.c_void_p

lib.BattleFinish.argtypes = [ctypes.c_void_p]

lib.GetBattleData.restype = SerialData
lib.GetBattleData.argtypes = [ctypes.c_void_p]

lib.Select.restype = ctypes.c_int
lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

lib.VisualizeData.restype = ctypes.c_char_p
lib.VisualizeData.argtypes = [ctypes.c_void_p]

lib.SearchBegin.restype = ctypes.c_char_p
lib.SearchBegin.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int]

lib.SearchStep.restype = ctypes.c_char_p
lib.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

lib.SearchEnd.argtypes = [ctypes.c_void_p]

lib.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_int64]

lib.AllCard.restype = ctypes.c_char_p

lib.AllAttack.restype = ctypes.c_char_p

class Battle:
    battle_ptr = None
    obs = None
