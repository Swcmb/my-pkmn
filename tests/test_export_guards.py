"""导出侧守卫：推理线程钉 + 包体体积门禁（回归测试）。

两条都是实测出来的约束，缺失的后果分别是"平台判负"与"上传被 nginx 丢弃"，且都只体现在
生成产物的字节里，所以在此钉住。

依据：平台单步限时 100ms、CPU 2 核。空闲态 A/B 实测不钉线程 38/964 次决策越界
（max 350.3ms），钉到 2 线程后 1/571（max 112.7ms）。这条约束只体现在生成的
main.py 文本里，一旦模板被改写就静默失效，所以在此钉住。
"""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_rl_submission",
        os.path.join(ROOT, "scripts", "export_rl_submission.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_transformer2_head_pins_threads_before_model_load():
    m = _load_export_module()
    head = m.TRANSFORMER2_HEAD
    assert "torch.set_num_threads(2)" in head
    # 必须在加载模型与任何前向之前设置，否则线程池已按宿主机核数建好
    assert head.index("torch.set_num_threads(2)") < head.index("torch.load(")
    assert head.index("torch.set_num_threads(2)") < head.index("_NET.logits_value")


def test_generated_mainpy_contains_pin():
    m = _load_export_module()
    src = m._DIR_FINDER + m.TRANSFORMER2_HEAD
    compile(src, "main.py", "exec")            # 拼接后仍是合法 Python
    assert "torch.set_num_threads(2)" in src


def test_size_gate_exists_and_sits_under_platform_body_limit():
    """导出侧必须自带体积门禁：平台 nginx client_max_body_size 实测 1 MiB，超限直接 HTTP 413。"""
    src = open(os.path.join(ROOT, "scripts", "export_rl_submission.py"), encoding="utf-8").read()
    assert "--max-bytes" in src, "体积门禁开关被删掉了"
    assert "超过提交上限" in src, "包体超限时必须让导出失败，而不是留下一个上传不了的包"
    # 默认值必须严格小于 1 MiB（留 multipart 边界余量）
    import re
    m = re.search(r'"--max-bytes",\s*type=int,\s*default=(\d[\d_]*)', src)
    assert m and 0 < int(m.group(1).replace("_", "")) < 1_048_576
