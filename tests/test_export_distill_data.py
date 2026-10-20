"""Contract guards for scripts/export_distill_data.py (route A, stage A3).

The decisive one is the key-set guard: bc_train takes EVERY npz key except `__labels__` /
`__is_attack__` / `__group__` and feeds it to the network as an obs feature. A single extra column
sneaking into the training file therefore corrupts BC silently, so the test asserts the exact key
set instead of spot-checking.

One module-scoped rollout of 3 real games is shared by the dataset tests: CPU only, reads weights,
never writes a checkpoint, so it cannot produce new weights.

  Run: python -m pytest tests/test_export_distill_data.py -v
"""
import csv
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "runs", "evaluation"))

from rl.card_features import get_card_table                    # noqa: E402
from rl.encoding import TokenEncoder                           # noqa: E402
from rl.decks_ladder import load_ladder_decks                  # noqa: E402

SPEC = os.path.join(_ROOT, "scripts", "export_distill_data.py")
CKPT = os.path.join(_ROOT, "runs", "train-20m-e16-1m", "ckpt_999424.pt")
DECK = os.path.join(_ROOT, "runs", "evaluation", "card_database_migration", "best_deck.csv")
POOL = os.path.join(_ROOT, "runs", "evaluation", "session-20261020", "opp_pool_train30.json")
RESERVED = ("__labels__", "__is_attack__", "__group__")


def _install_engine_alias():
    """`runs/evaluation/eval_h2h.py:39` 在 import 期执行全局 `torch.set_grad_enabled(False)`，
    任何导入它的模块都会连带失去 autograd。这里导入后立刻恢复原状，避免污染同进程的其他测试。
    """
    before = torch.is_grad_enabled()
    from eval_h2h import _install_sdk_cg_shim_alias
    _install_sdk_cg_shim_alias()
    torch.set_grad_enabled(before)


def _load_dumper():
    spec = importlib.util.spec_from_file_location("export_distill_data", SPEC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rollout():
    """真引擎 + 真教师，跑 3 局完整对局（不限决策，避免半局被整局丢弃）。"""
    d = _load_dumper()
    _install_engine_alias()
    ct = get_card_table()
    enc = TokenEncoder(ct)
    net, _step, nc = d.load_net(CKPT, ct, torch.device("cpu"))
    assert nc.get("arch") == "transformer2", nc
    deck = [int(x) for row in csv.reader(open(DECK, encoding="utf-8-sig")) for x in row if x.strip()]
    assert len(deck) == 60, len(deck)
    pool, _c = load_ladder_decks(POOL)
    out = d.collect(net=net, enc=enc, int_keys=set(enc.int_keys), dev=torch.device("cpu"),
                    cand_deck=deck, opp_pool=pool, games=3, seed=20261020,
                    selfplay=True, limit_decisions=0, game_filter=lambda gi: True)
    assert out["rows"], f"没采到数据：reset_fail={out['reset_fail']} metas={out['metas']}"
    ds, aux, keys = d.stack_dataset(out, set(enc.int_keys))
    return d, out, ds, aux, keys


def test_npz_key_set_is_exactly_obs_plus_labels(rollout):
    """训练文件里只能有观测键 + __labels__；任何附加列都必须留在 sidecar。"""
    _d, _out, ds, aux, keys = rollout
    extra = [k for k in ds if k.startswith("__") and k not in RESERVED]
    assert not extra, f"内部键被误写入训练文件，会被 bc_train 当观测特征：{sorted(extra)}"
    assert set(keys) | {"__labels__"} == set(ds)
    for k in aux:
        assert k not in ds, f"sidecar 列混进了训练文件：{k}"


def test_dtypes_follow_bc_train_convention(rollout):
    _d, _out, ds, _aux, keys = rollout
    int_keys = set(TokenEncoder(get_card_table()).int_keys)
    for k in keys:
        if k == "__labels__":
            continue
        want = np.int32 if k in int_keys else np.float32
        assert ds[k].dtype == want, f"{k}: {ds[k].dtype} != {want}"
    assert ds["__labels__"].dtype == np.int64


def test_every_label_is_legal_under_its_own_mask(rollout):
    """masked_rate 必须为 0，否则 bc_train 会在非法下标上学动作。"""
    d, _out, ds, _aux, _keys = rollout
    masked, _top3 = d.selfcheck(ds)
    assert masked == 0.0, f"masked_rate={masked}"
    am = ds["action_mask"]
    assert am.shape[0] == ds["__labels__"].shape[0]
    assert float(am.min()) >= 0.0 and float(am.max()) <= 1.0


def test_sidecar_aligned_row_by_row(rollout):
    _d, out, ds, aux, _keys = rollout
    n = ds["__labels__"].shape[0]
    for k in ("value", "seat", "game", "ret", "row_index"):
        assert aux[k].shape[0] == n, f"{k}: {aux[k].shape[0]} != {n}"
    assert (aux["row_index"] == np.arange(n)).all()
    assert set(aux["seat"].tolist()) <= {0, 1}
    assert set(aux["ret"].tolist()) <= {-1, 0, 1}
    assert n == sum(m["stored_rows"] for m in out["metas"])


def test_rows_grouped_by_game_for_leak_free_tail_split(rollout):
    """bc_train 用连续尾部当验证集，同一局的行必须连续成块。"""
    _d, out, _ds, aux, _keys = rollout
    if len(out["metas"]) < 2:
        pytest.skip("只采到一局，无从验证分块")
    runs = []
    for v in aux["game"].tolist():
        if not runs or runs[-1][0] != v:
            runs.append([v, 0])
        runs[-1][1] += 1
    assert len({r[0] for r in runs}) == len(runs), "同一局的行被拆散，尾部切分会泄漏"


def test_selfplay_covers_both_seats(rollout):
    """自博弈两侧都入库 + 座位按局交替，牌组/座位效应才会被抵消。"""
    _d, out, _ds, aux, _keys = rollout
    assert set(aux["seat"].tolist()) == {0, 1}
    assert [m["cand_seat"] for m in out["metas"]] == [0, 1, 0]


def test_weights_only_loader_reads_the_teacher():
    """ckpt 必须能用 weights_only=True 打开（不得依赖 pickle 执行）。"""
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    assert {"net", "net_config", "global_step"} <= set(ck)
    assert ck["net_config"]["arch"] == "transformer2"


def test_importing_the_harness_leaves_autograd_enabled(rollout):
    """守卫：eval_h2h 在 import 期关掉全局 grad，本文件必须把它恢复回来，
    否则同进程的梯度类测试会在毫无征兆的情况下失败。"""
    assert torch.is_grad_enabled()
    lin = torch.nn.Linear(2, 2)
    loss = lin(torch.ones(1, 2)).sum()
    loss.backward()
    assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()


def test_deck_helper_rejects_wrong_size(tmp_path):
    d = _load_dumper()
    bad = tmp_path / "bad.csv"
    bad.write_text("1\n2\n3\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        d.load_csv_deck(str(bad))
    assert len(d.sha256(SPEC)) == 64
