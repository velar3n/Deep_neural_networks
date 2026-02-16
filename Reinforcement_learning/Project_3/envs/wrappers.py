"""
Wrappery / helpery do środowisk dyskretnych.

Cel kursu (blok 1):
- pokazać te same idee na GridWorld i FrozenLake,
- w jednym, prostym API: reset/step + (opcjonalnie) model P[s][a].

Uwaga:
W tej wersji domyślnie NIE wymagamy Gym/Gymnasium do FrozenLake,
bo budujemy model w czystym Pythonie (build_frozenlake_P) i próbkujemy
z niego przez PModelEnv (jak w notebookach Ch04/Ch05).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict, Set, List
import numpy as np

try:
    import gymnasium as gym  # opcjonalnie, jeśli ktoś woli backend Gym
except Exception:
    try:
        import gym  # type: ignore
    except Exception:
        gym = None

from .gridworld import GridWorld, GridWorldConfig
from .frozenlake import (
    FROZENLAKE_MAPS,
    build_frozenlake_P,
    find_start_state,
    PModelEnv,
)


@dataclass
class DiscreteEnvSpec:
    nS: int
    nA: int
    shape: Tuple[int, int]
    terminals: Set[int]
    gamma: float = 0.99
    desc: Optional[np.ndarray] = None  # opcjonalnie: plansza (np. FrozenLake)


def make_gridworld(
    height=4,
    width=4,
    start=(0, 0),
    goal=(3, 3),
    holes=((1, 1), (2, 3)),
    step_cost=0.0,
    gamma=0.99,
    max_steps=200,
    seed=0,
):
    cfg = GridWorldConfig(
        height=height,
        width=width,
        start=start,
        goal=goal,
        holes=holes,
        step_cost=step_cost,
        gamma=gamma,
        max_steps=max_steps,
    )
    env = GridWorld(cfg=cfg, seed=seed)
    return env


def make_frozenlake(
    map_name: str = "4x4",
    is_slippery: bool = True,
    seed: int = 0,
    backend: str = "pmodel",
    desc: Optional[List[str]] = None,
):
    """
    Tworzy FrozenLake.

    backend:
      - "pmodel" (domyślnie): buduje P przez build_frozenlake_P i zwraca PModelEnv
      - "gym": używa gymnasium (jeśli zainstalowane) i zwraca env Gym

    desc:
      - jeśli podasz listę stringów mapy, nadpisze map_name.
    """
    if backend == "gym":
        if gym is None:
            raise ImportError("gymnasium/gym nie jest zainstalowany. Użyj backend='pmodel' albo zainstaluj gymnasium.")
        env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
        env.reset(seed=seed)
        return env

    # backend == "pmodel"
    if desc is None:
        if map_name not in FROZENLAKE_MAPS:
            raise ValueError(f"Nieznany map_name={map_name}. Dostępne: {list(FROZENLAKE_MAPS.keys())} albo podaj desc.")
        desc = FROZENLAKE_MAPS[map_name]

    P, nS, nA, nrow, ncol, desc_arr = build_frozenlake_P(desc, is_slippery=is_slippery)
    s0 = find_start_state(desc_arr)
    env = PModelEnv(P, start_state=s0, seed=seed, desc=desc_arr)
    # atrybuty pomocnicze (pod DP / viz)
    env.nrow, env.ncol = nrow, ncol
    env.desc = desc_arr
    return env


def get_spec(env, gamma: float = 0.99) -> DiscreteEnvSpec:
    """
    Wyciąga (nS,nA,shape,terminals) z:
    - naszego GridWorld,
    - PModelEnv (FrozenLake z modelu),
    - opcjonalnie: Gymnasium FrozenLake (unwrapped.P).
    """
    # 1) Nasz GridWorld
    if hasattr(env, "nS") and hasattr(env, "nA") and hasattr(env, "cfg"):
        nS, nA = int(env.nS), int(env.nA)
        shape = (env.cfg.height, env.cfg.width)
        terminals = set()
        terminals.add(env._to_s(env.cfg.goal))
        for h in env.cfg.holes:
            terminals.add(env._to_s(h))
        return DiscreteEnvSpec(nS=nS, nA=nA, shape=shape, terminals=terminals, gamma=gamma, desc=None)

    # 2) PModelEnv / ogólne env z jawym P
    if hasattr(env, "P") and hasattr(env, "nS") and hasattr(env, "nA"):
        nS, nA = int(env.nS), int(env.nA)
        desc = getattr(env, "desc", None)
        if desc is not None:
            H, W = desc.shape
            terminals = set()
            for r in range(H):
                for c in range(W):
                    ch = str(desc[r, c])
                    if ch in ("H", "G"):
                        terminals.add(r * W + c)
            shape = (int(H), int(W))
            return DiscreteEnvSpec(nS=nS, nA=nA, shape=shape, terminals=terminals, gamma=gamma, desc=desc)
        # fallback
        side = int(np.sqrt(nS))
        shape = (side, side)
        return DiscreteEnvSpec(nS=nS, nA=nA, shape=shape, terminals=set(), gamma=gamma, desc=None)

    # 3) Gymnasium FrozenLake (dla kompatybilności)
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "P"):
        P = env.unwrapped.P
        nS = len(P)
        nA = len(P[0])
        desc = getattr(env.unwrapped, "desc", None)
        if desc is not None:
            H, W = desc.shape
            terminals = set()
            for r in range(H):
                for c in range(W):
                    ch = desc[r, c].decode("utf-8")
                    s = r * W + c
                    if ch in ("H", "G"):
                        terminals.add(s)
            shape = (H, W)
        else:
            shape = (int(np.sqrt(nS)), int(np.sqrt(nS)))
            terminals = set()
        return DiscreteEnvSpec(nS=int(nS), nA=int(nA), shape=shape, terminals=terminals, gamma=gamma, desc=desc)

    raise ValueError("Nie rozpoznaję środowiska. Oczekuję GridWorld (nS,nA,cfg) lub env z atrybutem P/nS/nA, albo gymnasium FrozenLake (unwrapped.P).")


def state_to_rc(s: int, shape: Tuple[int, int]) -> Tuple[int, int]:
    H, W = shape
    return divmod(int(s), W)
