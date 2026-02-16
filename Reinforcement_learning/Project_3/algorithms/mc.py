"""
Monte Carlo (model-free) dla środowisk dyskretnych.
- prediction: V(s) z pełnych epizodów
- control: first-visit MC control (ε-greedy)
"""

from __future__ import annotations
from typing import Callable, Tuple, Optional
import numpy as np
from .utils import rollout_episode, returns_from_trajectory, epsilon_greedy


def mc_prediction(env, policy: Callable[[int], int], nS: int, gamma: float = 0.99, episodes: int = 10_000, max_steps: int = 10_000):
    V = np.zeros(nS, dtype=np.float64)
    N = np.zeros(nS, dtype=np.int64)

    for _ in range(episodes):
        traj = rollout_episode(env, policy, max_steps=max_steps)

        # zwroty G_t dla każdego kroku
        Gs = returns_from_trajectory(traj, gamma)

        # zbiór odwiedzonych stanów (first-visit)
        visited = set()

        for (t, (s, a, r)) in enumerate(traj):
            if s in visited:
                continue  # first-visit

            visited.add(s)  # dodanie bieżącego stanu s do visited

            G = Gs[t]  # zwrot dla kroku t (to jest target do aktualizacji V[s])

            N[s] += 1  # zwiększenie licznika wizyt stanu s (potrzebny do średniej inkrementalnej)

            # aktualizacja średniej inkrementalnej V[s] zgodnie ze wzorem: V <- V + (G - V)/N
            V[s] += (G - V[s]) / N[s] 

    return V, N

def mc_control_epsilon_greedy(env, nS: int, nA: int, gamma: float = 0.99, episodes: int = 50_000, epsilon: float = 0.1, seed: int = 0, max_steps: int = 10_000):
    rng = np.random.default_rng(seed)
    Q = np.zeros((nS, nA), dtype=np.float64)
    N = np.zeros((nS, nA), dtype=np.int64)

    def policy(s: int) -> int:
        # ε-greedy względem Q[s]
        return epsilon_greedy(Q[s], epsilon, rng)

    for _ in range(episodes):
        traj = rollout_episode(env, policy, max_steps=max_steps)

        # zwroty G_t dla każdego kroku t (używa returns_from_trajectory z utils)
        Gs = returns_from_trajectory(traj, gamma)  # Gs[t] jest zwrotem od kroku t (target do aktualizacji Q)

        visited = set()  # zbiór visited dla first-visit par (s,a) w tym epizodzie

        for (t, (s, a, r)) in enumerate(traj):
            key = (s, a)  # klucz pary (s,a), np. (s, a)

            # first-visit MC control: jeśli (s,a) już było w epizodzie, zostaje pominięte
            if key in visited:
                continue  # first-visit (aktualizujemy tylko pierwsze wystąpienie pary (s,a))

            visited.add(key)  # dodanie key do visited (oznaczamy, że para (s,a) już była)

            G = Gs[t]  # zwrot dla kroku t: G = Gs[t] (to jest target dla Q[s,a])

            N[s, a] += 1  # zwiększenie licznika wizyt pary (s,a) (potrzebny do średniej inkrementalnej)

            # aktualizacja średniej inkrementalnej Q[s,a] zgodnie ze wzorem: Q <- Q + (G - Q)/N
            Q[s, a] += (G - Q[s, a]) / N[s, a]

    pi_greedy = np.argmax(Q, axis=1)  # gotowa polityka do ewaluacji
    return Q, pi_greedy, N
