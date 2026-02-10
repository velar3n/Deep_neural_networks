"""Minimalne wizualizacje FrozenLake do dydaktyki (Jupyter/Matplotlib).

Inspiracja: popularne kursy RL, które pokazują politykę jako strzałki i Q jako 4 liczby w komórce.

Użycie w notebooku:
    from frozenlake_viz import plot_policy_arrows, plot_q_triangles

    plot_policy_arrows(pi_greedy, shape=(nrow,ncol), desc=desc)
    plot_q_triangles(Q, shape=(nrow,ncol))
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt


_ARROW = {0: "←", 1: "↓", 2: "→", 3: "↑"}


def _to_str(ch):
    if isinstance(ch, (bytes, np.bytes_)):
        return ch.decode("utf-8")
    return str(ch)


def plot_policy_arrows(
    pi: Sequence[int],
    shape: Tuple[int, int],
    desc: Optional[Sequence[Sequence[object]]] = None,
    title: str = "Policy (arrows)",
    fontsize: int = 18,
):
    """Rysuje deterministyczną politykę jako strzałki na siatce.

    Jeśli podasz desc (np. z FrozenLake), to dla komórek S/H/G wypisze literę zamiast strzałki.
    """
    nrow, ncol = shape
    pi_grid = np.array(pi, dtype=int).reshape(nrow, ncol)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.zeros((nrow, ncol)), alpha=0.0)

    # grid lines
    for r in range(nrow + 1):
        ax.axhline(r - 0.5, color="black", linewidth=0.3)
    for c in range(ncol + 1):
        ax.axvline(c - 0.5, color="black", linewidth=0.3)

    for r in range(nrow):
        for c in range(ncol):
            cell = None
            if desc is not None:
                cell = _to_str(desc[r][c])
            if cell in {"S", "H", "G"}:
                ax.text(c, r, cell, ha="center", va="center", fontsize=fontsize)
            else:
                ax.text(c, r, _ARROW[int(pi_grid[r, c])], ha="center", va="center", fontsize=fontsize)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def plot_q_triangles(
    Q: np.ndarray,
    shape: Tuple[int, int],
    title: str = "Q(s,a) (triangles)",
    fmt: str = "{:.2f}",
    fontsize: int = 8,
):
    """Rysuje Q(s,a) jako 4 liczby w komórce (lewo/dół/prawo/góra).

    Zakłada standardowe kodowanie akcji FrozenLake:
        0=LEFT, 1=DOWN, 2=RIGHT, 3=UP.
    """
    nrow, ncol = shape
    Q = np.asarray(Q, dtype=float).reshape(nrow, ncol, 4)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.zeros((nrow, ncol)), alpha=0.0)

    # grid + diagonals
    for r in range(nrow + 1):
        ax.axhline(r - 0.5, color="black", linewidth=0.3)
    for c in range(ncol + 1):
        ax.axvline(c - 0.5, color="black", linewidth=0.3)

    for r in range(nrow):
        for c in range(ncol):
            # diagonals in each cell
            ax.plot([c - 0.5, c + 0.5], [r - 0.5, r + 0.5], color="black", linewidth=0.3)
            ax.plot([c - 0.5, c + 0.5], [r + 0.5, r - 0.5], color="black", linewidth=0.3)

    # write numbers
    for r in range(nrow):
        for c in range(ncol):
            q_left, q_down, q_right, q_up = Q[r, c]
            ax.text(c - 0.30, r, fmt.format(q_left), ha="center", va="center", fontsize=fontsize)   # LEFT
            ax.text(c, r + 0.30, fmt.format(q_down), ha="center", va="center", fontsize=fontsize)   # DOWN
            ax.text(c + 0.30, r, fmt.format(q_right), ha="center", va="center", fontsize=fontsize)  # RIGHT
            ax.text(c, r - 0.30, fmt.format(q_up), ha="center", va="center", fontsize=fontsize)     # UP

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
