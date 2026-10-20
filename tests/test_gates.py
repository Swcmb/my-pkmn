"""rl/gates schedules: small pools keep the byte-identical round-robin; pools that outgrow the
gate (ladder_all) switch to a FIXED-SEED uniform draw (unbiased + identical every gate)."""
from rl.gates import _field_candidates, _gate_schedule


def test_small_pool_round_robin_unchanged():
    # 30-deck mirror 400g = the gen5m schedule: sequential pilot-swap blocks, full coverage
    s = _gate_schedule(30, 400, "mirror")
    assert s[:4] == [(0, 0), (0, 0), (1, 1), (1, 1)]
    assert len(s) == 400
    assert {i for i, _ in s} == set(range(30))
    # finetune pool (n=1) degenerate case
    assert _gate_schedule(1, 4, "mirror") == [(0, 0)] * 4


def test_big_pool_uniform_draw_mirror():
    s1 = _gate_schedule(1107, 400, "mirror")
    assert s1 == _gate_schedule(1107, 400, "mirror")     # pure function -> identical every gate
    assert len(s1) == 400
    assert all(i == j for i, j in s1)                    # mirror: pure piloting probe
    decks = [i for i, _ in s1]
    assert len(set(decks)) == 200                        # 200 distinct swap-blocks (no replacement)
    assert max(decks) >= 400                             # NOT the head-prefix the round-robin gives
    assert all(s1[2 * b] == s1[2 * b + 1] for b in range(200))   # pilot-swap blocks intact


def test_big_pool_cross_swap_blocks():
    s = _gate_schedule(1107, 400, "cross")
    assert s == _gate_schedule(1107, 400, "cross")
    for b in range(200):
        i, j = s[2 * b]
        assert s[2 * b + 1] == (j, i) and i != j         # (i,j) then (j,i), deck strength cancels


def test_field_candidates_cycle_pure_and_complete():
    c = _field_candidates(30, 100)
    assert len(c) == 100
    assert c == _field_candidates(30, 100)               # 纯函数：每次 gate 相同
    assert all(0 <= i < 30 for i in c)
    assert set(c) == set(range(30))                      # 全覆盖，且每副 3-4 次
    assert c[:32] == list(range(30)) + [0, 1]
    # finetune 单牌组退化情形
    assert _field_candidates(1, 5) == [0] * 5


def test_field_candidates_rejects_empty_pool():
    import pytest
    with pytest.raises(ValueError):
        _field_candidates(0, 10)

