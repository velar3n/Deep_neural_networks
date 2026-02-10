"""
Dynamic Programming dla środowisk z modelem P (jak FrozenLake / nasz GridWorld).
P[s][a] = [(prob, s', reward, terminated), ...]
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np


def policy_evaluation(P, pi: np.ndarray, gamma: float = 0.99, theta: float = 1e-10, max_iter: int = 1_000_000):
    """Iteracyjna ewaluacja polityki: zwraca V."""
    nS = len(P)
    V = np.zeros(nS, dtype=np.float64)

    for _ in range(max_iter):
        delta = 0.0
        for s in range(nS):
            v = 0.0
            for a, p_a in enumerate(pi[s]):
                if p_a == 0: 
                    continue
                for prob, sp, r, done in P[s][a]:
                    v += p_a * prob * (r + (0.0 if done else gamma * V[sp]))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V


def value_iteration(P, gamma: float = 0.99, theta: float = 1e-10, max_iter: int = 1_000_000):
    """Value Iteration: zwraca V*."""
    nS = len(P)
    nA = len(P[0])
    V = np.zeros(nS, dtype=np.float64)

    for _ in range(max_iter):
        delta = 0.0
        for s in range(nS):
            q_sa = np.zeros(nA, dtype=np.float64)
            for a in range(nA):
                for prob, sp, r, done in P[s][a]:
                    q_sa[a] += prob * (r + (0.0 if done else gamma * V[sp]))
            v_new = np.max(q_sa)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V


def greedy_policy_from_V(P, V: np.ndarray, gamma: float = 0.99):
    """Wyciągnij deterministyczną politykę zachłanną względem V."""
    nS = len(P)
    nA = len(P[0])
    pi = np.zeros((nS, nA), dtype=np.float64)
    for s in range(nS):
        q_sa = np.zeros(nA, dtype=np.float64)
        for a in range(nA):
            for prob, sp, r, done in P[s][a]:
                q_sa[a] += prob * (r + (0.0 if done else gamma * V[sp]))
        best = np.flatnonzero(q_sa == np.max(q_sa))
        # rozkład równomierny na remisach
        pi[s, best] = 1.0 / len(best)
    return pi
