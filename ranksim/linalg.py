"""Tiny dense linear algebra: Cholesky solve for the ridge normal equations.

The design matrices here are (alliance-appearances x teams) with ~30 columns, so a
plain O(n^3) Cholesky in pure Python is instant and keeps the tool dependency-free.
"""

from __future__ import annotations

import math


def cholesky(a: list[list[float]]) -> list[list[float]]:
    """Lower-triangular L with L @ L.T == a, for symmetric positive definite a."""
    n = len(a)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                d = a[i][i] - s
                if d <= 0.0:
                    # Ridge should prevent this; nudge rather than explode.
                    d = 1e-9
                lower[i][j] = math.sqrt(d)
            else:
                lower[i][j] = (a[i][j] - s) / lower[j][j]
    return lower


def chol_solve(lower: list[list[float]], b: list[float]) -> list[float]:
    """Solve (L L.T) x = b."""
    n = len(lower)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(lower[i][k] * y[k] for k in range(i))) / lower[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(lower[k][i] * x[k] for k in range(i + 1, n))) / lower[i][i]
    return x


def chol_inv_diag(lower: list[list[float]]) -> list[float]:
    """Diagonal of (L L.T)^-1, used for parameter standard errors."""
    n = len(lower)
    diag = [0.0] * n
    for col in range(n):
        e = [0.0] * n
        e[col] = 1.0
        diag[col] = chol_solve(lower, e)[col]
    return diag


def normal_equations(
    rows: list[list[int]], y: list[float], n_cols: int, ridge: float
) -> tuple[list[list[float]], list[float]]:
    """Build (X'X + ridge*I, X'y) from sparse rows of column indices.

    Every design row here is an alliance: three 1s and the rest zeros, so the
    matrix is assembled from index pairs instead of a dense multiply.
    """
    ata = [[0.0] * n_cols for _ in range(n_cols)]
    aty = [0.0] * n_cols
    for cols, target in zip(rows, y):
        for i in cols:
            aty[i] += target
            for j in cols:
                ata[i][j] += 1.0
    for i in range(n_cols):
        ata[i][i] += ridge
    return ata, aty
