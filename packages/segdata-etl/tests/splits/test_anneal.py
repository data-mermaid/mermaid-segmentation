from __future__ import annotations

import numpy as np

from segdata_etl.splits.anneal import anneal


def _run(eligible, fixed=None, seed=0, iters=20_000):
    groups = len(eligible)
    X = np.full((groups, 1), 10.0)
    n = np.full(groups, 10.0)
    P = X.sum(0)
    total = n.sum()
    return anneal(
        X,
        n,
        P,
        np.asarray(eligible),
        (0.1 * total, 0.1 * total),
        (0.1, 0.1),
        10.0,
        np.array([True]),
        np.array([0.0]),
        1.0,
        iters,
        seed,
        fixed,
    )


def test_same_seed_gives_an_identical_assignment():
    eligible = [True] * 10
    first, first_cost = _run(eligible, seed=3, iters=500)
    second, second_cost = _run(eligible, seed=3, iters=500)
    assert first.tolist() == second.tolist()
    assert first_cost == second_cost


def test_fixed_groups_keep_their_split_and_ineligible_groups_stay_train():
    # Group 1 is eligible and pinned to test; a free group would move off it on some seed.
    eligible = [False, True, True, True, True, True, True, True, True, True]
    fixed = [-1, 2, -1, -1, -1, -1, -1, -1, -1, -1]
    for seed in range(6):
        assignment, _ = _run(eligible, fixed=fixed, seed=seed, iters=5_000)
        assert assignment[0] == 0
        assert assignment[1] == 2


def test_ten_equal_groups_put_exactly_one_in_val_and_one_in_test():
    assignment, _ = _run([True] * 10)
    assert sorted(assignment.tolist()) == [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
