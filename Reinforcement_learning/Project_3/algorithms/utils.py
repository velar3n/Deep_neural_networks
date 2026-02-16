"""
Utils: rollout, ε-greedy, wykresy (value heatmap), proste logowanie.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 0):
    np.random.seed(seed)


def epsilon_greedy(q_s: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Wybierz akcję ε-greedy dla wektora Q(s,·)."""
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_s)))
    # tie-break losowo
    maxv = np.max(q_s)
    best = np.flatnonzero(q_s == maxv)
    return int(rng.choice(best))


def rollout_episode(env, policy: Callable[[int], int], max_steps: int = 10_000):
    """Zwraca listę (s,a,r) dla pełnego epizodu."""
    traj = []
    s, _ = env.reset()
    for _ in range(max_steps):
        a = int(policy(s))
        sp, r, terminated, truncated, _ = env.step(a)
        traj.append((int(s), int(a), float(r)))
        s = sp
        if terminated or truncated:
            break
    return traj


def returns_from_trajectory(traj: List[Tuple[int,int,float]], gamma: float):
    """Dla trajektorii (s,a,r) zwraca listę zwrotów G_t (dla każdego kroku)."""
    G = 0.0
    rets = []
    for (_,_,r) in reversed(traj):
        G = r + gamma * G
        rets.append(G)
    rets.reverse()
    return rets


def plot_value_heatmap(V: np.ndarray, shape: Tuple[int,int], title: str = "V(s)"):
    H,W = shape
    grid = V.reshape(H,W)
    plt.figure()
    plt.imshow(grid)
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(W))
    plt.yticks(range(H))
    plt.show()


def plot_learning_curve(x: List[float], y: List[float], title: str, xlabel: str = "episode", ylabel: str = "metric"):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
