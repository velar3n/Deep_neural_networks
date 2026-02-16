"""
Temporal Difference (TD) — wersja STUDENT.

Zaimplementuj podstawowe algorytmy TD dla środowisk dyskretnych:
- TD(0) prediction: estymacja V(s) ≈ v_π(s) dla ustalonej polityki π
- SARSA: on-policy TD control (uczenie Q(s,a) i polityki ε-greedy)
- Q-learning: off-policy TD control (uczenie Q(s,a) z targetem max)

Wskazówki:
- Dla terminala przyjmujemy V(terminal)=0 oraz Q(terminal,·)=0.
- Użyj helpera epsilon_greedy z utils (jak w mc_student.py).
- Zwróć uwagę na typy: s, a, sp powinny być int (indeksy tablic numpy).
"""

from __future__ import annotations
from typing import Callable
import numpy as np
from .utils import epsilon_greedy


def td0_prediction(
    env,
    policy: Callable[[int], int],
    nS: int,
    gamma: float = 0.99,
    alpha: float = 0.1,
    episodes: int = 10_000,
    max_steps: int = 10_000,
):
    """TD(0) prediction: uczymy V(s) dla ustalonej polityki π.

    Update (tabular):
        V(S_t) <- V(S_t) + α [ r_{t+1} + γ V(S_{t+1}) - V(S_t) ]
    gdzie przy terminalu: V(S_{t+1}) = 0.
    """
    V = np.zeros(nS, dtype=np.float64)

    for _ in range(episodes):
        s, _ = env.reset()
        s = int(s)

        for _ in range(max_steps):
            # wybór akcji zgodnie z daną polityką π
            a = policy(s)

            sp, r, terminated, truncated, _ = env.step(int(a))
            sp = int(sp)
            done = bool(terminated or truncated)

            # TD target = r + γ V(sp), ale jeśli done -> bootstrap = 0
            target = r if done else r + gamma * V[sp]

            # update TD(0)
            V[s] += alpha * (target - V[s])
            s = sp

            if done:
                break

    return V


def sarsa(
    env,
    nS: int,
    nA: int,
    gamma: float = 0.99,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    episodes: int = 20_000,
    seed: int = 0,
    max_steps: int = 10_000,
):
    """SARSA (on-policy TD control).

    Update:
        Q(S_t, A_t) <- Q(S_t, A_t) + α [ r_{t+1} + γ Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) ]
    gdzie A_{t+1} wybieramy tą samą polityką ε-greedy (on-policy).
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((nS, nA), dtype=np.float64)

    for _ in range(episodes):
        s, _ = env.reset()
        s = int(s)

        # wybór akcji startowej A z S używając ε-greedy w Q
        a = epsilon_greedy(Q[s], epsilon, rng)

        for _ in range(max_steps):
            sp, r, terminated, truncated, _ = env.step(int(a))
            sp = int(sp)
            done = bool(terminated or truncated)

            if done:
                # dla terminala target = r
                target = r
                Q[s, a] += alpha * (target - Q[s, a])
                break

            # wybór A' w S' tą samą polityką ε-greedy (on-policy)
            ap = epsilon_greedy(Q[sp], epsilon, rng)

            # target = r + γ Q(S', A')
            target = r + gamma * Q[sp, ap]

            # update Q
            Q[s, a] += alpha * (target - Q[s, a])

            s, a = sp, ap

    pi_greedy = np.argmax(Q, axis=1)
    return Q, pi_greedy


def q_learning(
    env,
    nS: int,
    nA: int,
    gamma: float = 0.99,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    episodes: int = 20_000,
    seed: int = 0,
    max_steps: int = 10_000,
):
    """Q-learning (off-policy TD control).

    Update:
        Q(S_t, A_t) <- Q(S_t, A_t) + α [ r_{t+1} + γ max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) ]

    Uwaga: zachowanie (behavior) zwykle ε-greedy, ale target jest greedy (max).
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((nS, nA), dtype=np.float64)

    for _ in range(episodes):
        s, _ = env.reset()
        s = int(s)

        for _ in range(max_steps):
            # wybór akcji ε-greedy (behavior)
            a = epsilon_greedy(Q[s], epsilon, rng)

            sp, r, terminated, truncated, _ = env.step(int(a))
            sp = int(sp)
            done = bool(terminated or truncated)

            if done:
                # dla terminala target = r
                target = r
                Q[s, a] += alpha * (target - Q[s, a])
                break

            # target = r + γ max_a Q(S',a)
            target = r + gamma * np.max(Q[sp])

            # update Q
            Q[s, a] += alpha * (target - Q[s, a])
            
            s = sp

    pi_greedy = np.argmax(Q, axis=1)
    return Q, pi_greedy
