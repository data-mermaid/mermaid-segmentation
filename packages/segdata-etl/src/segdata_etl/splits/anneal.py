"""Simulated annealing of source groups into train (0), val (1) and test (2)."""

from __future__ import annotations

from typing import Any

import numpy as np


def anneal(
    X: np.ndarray,
    n: np.ndarray,
    P: np.ndarray,
    eligible: np.ndarray,
    targets: tuple[float, float],
    fracs: tuple[float, float],
    w_size: float,
    cls_mask: np.ndarray,
    min_pts: np.ndarray,
    w_cov: float,
    iters: int,
    seed: int,
    fixed: Any = None,
) -> tuple[np.ndarray, float]:
    """Assign eligible groups to 0 (train), 1 (val), 2 (test); returns (assignment, cost).

    ``fixed``: per group, a split it is pinned to, or -1 (free; the default for every group).
    """
    rng = np.random.default_rng(seed)
    fixed = np.full(len(n), -1) if fixed is None else np.asarray(fixed)
    idx = np.flatnonzero(eligible & (fixed < 0))
    a = np.where(fixed >= 0, fixed, 0)
    p0 = max(sum(targets) - n[fixed > 0].sum(), 0) / n[idx].sum()
    a[idx] = np.where(rng.random(len(idx)) < p0, rng.integers(1, 3, len(idx)), 0)
    S = {s: X[a == s].sum(0) for s in (1, 2)}
    N = {s: n[a == s].sum() for s in (1, 2)}
    Pm = np.where(P > 0, P, 1)
    has_min = min_pts > 0
    m = min_pts[has_min]

    def split_cost(s: int, Ss: np.ndarray, Ns: float) -> float:
        size = ((Ns - targets[s - 1]) / targets[s - 1]) ** 2
        share = ((Ss / Pm - fracs[s - 1]) / fracs[s - 1]) ** 2
        coverage = (np.clip(1 - Ss[has_min] / m, 0, None) ** 2).sum()
        return float(w_size * size + share[cls_mask].mean() + w_cov * coverage)

    cost = {s: split_cost(s, S[s], N[s]) for s in (1, 2)}
    t0, t1 = 1.0, 1e-5
    for it in range(iters):
        temp = t0 * (t1 / t0) ** (it / iters)
        # Half the moves send one group to another split; the other half swap two groups in
        # different splits, which lets a large group trade places without breaking the sizes.
        g = idx[rng.integers(len(idx))]
        if rng.random() < 0.5:
            moves = [(g, (a[g] + rng.integers(1, 3)) % 3)]
        else:
            g2 = idx[rng.integers(len(idx))]
            if a[g2] == a[g]:
                continue
            moves = [(g, a[g2]), (g2, a[g])]
        dS = {s: np.zeros_like(P) for s in (1, 2)}
        dN = {1: 0.0, 2: 0.0}
        for grp, new in moves:
            for s, sign in ((a[grp], -1), (new, 1)):
                if s:
                    dS[s] += sign * X[grp]
                    dN[s] += sign * n[grp]
        delta, upd = 0.0, {}
        for s in (1, 2):
            if dN[s] or dS[s].any():
                Ss, Ns = S[s] + dS[s], N[s] + dN[s]
                c = split_cost(s, Ss, Ns)
                upd[s] = (Ss, Ns, c)
                delta += c - cost[s]
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            for grp, new in moves:
                a[grp] = new
            for s, (Ss, Ns, c) in upd.items():
                S[s], N[s], cost[s] = Ss, Ns, c
    return a, cost[1] + cost[2]
