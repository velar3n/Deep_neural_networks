"""
FrozenLake jako jawny model MDP P[s][a] + minimalne środowisko do próbkowania trajektorii.

Cel dydaktyczny:
- DP (model-based): używa pełnego P[s][a] i liczy wartości przez sumowanie po outcomes.
- MC/TD (model-free): generuje próbki (epizody) z tego samego P i uczy się statystycznie.

Kompatybilny z notebookami Ch04/Ch05:
- build_frozenlake_P(desc, is_slippery=False) -> (P, nS, nA, nrow, ncol, desc_arr)
- PModelEnv: reset/step na bazie P (losuje przejście wg prawdopodobieństw).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import numpy as np


# Standardowe mapy jak w Gym/Gymnasium (do wygody).
FROZENLAKE_MAPS: Dict[str, List[str]] = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

# Konwencja akcji (jak w Gym): 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


def build_frozenlake_P(desc: List[str], is_slippery: bool = False):
    """
    Buduje model FrozenLake w formacie P[s][a] bez użycia Gym.

    desc: lista stringów opisujących mapę, np.
          ["SFFF",
           "FHFH",
           "FFFH",
           "HFFG"]

    Konwencja akcji (jak w Gym):
      0 = LEFT, 1 = DOWN, 2 = RIGHT, 3 = UP

    Nagroda:
      +1 za wejście na G, w pozostałych przypadkach 0.

    Stany terminalne:
      H (hole) i G (goal).

    Zwraca:
      P: słownik P[s][a] = [(p, s2, r, terminated), ...]
      nS, nA, nrow, ncol, desc_arr
    """

    # Zamieniamy opis mapy na tablicę znaków
    desc_arr = np.asarray([list(row) for row in desc], dtype="<U1")
    nrow, ncol = desc_arr.shape
    nS, nA = nrow * ncol, 4

    # Wektory ruchu dla każdej akcji
    moves = {
        LEFT:  (0, -1),
        DOWN:  (1,  0),
        RIGHT: (0,  1),
        UP:    (-1, 0),
    }

    # Zamiana (r, c) -> indeks stanu
    def to_s(r: int, c: int) -> int:
        return r * ncol + c

    # Jeden krok z pozycji (r, c) dla akcji a
    def step_from(r: int, c: int, a: int) -> Tuple[int, int]:
        dr, dc = moves[a]
        r2, c2 = r + dr, c + dc

        # FrozenLake: wyjście poza planszę = zostajemy w miejscu
        if (r2 < 0) or (r2 >= nrow) or (c2 < 0) or (c2 >= ncol):
            r2, c2 = r, c

        return r2, c2

    # Inicjalizacja struktury P[s][a]
    P: Dict[int, Dict[int, List[Tuple[float,int,float,bool]]]] = {s: {a: [] for a in range(nA)} for s in range(nS)}

    # Iterujemy po wszystkich polach planszy
    for r in range(nrow):
        for c in range(ncol):
            s = to_s(r, c)
            tile = desc_arr[r, c]

            # Stany terminalne (H i G):
            # są absorbujące — niezależnie od akcji zostajemy w tym samym stanie
            if tile in ("H", "G"):
                for a in range(nA):
                    P[s][a] = [(1.0, s, 0.0, True)]
                continue

            # Stany nieterminalne
            for a in range(nA):
                outcomes: List[Tuple[float,int,float,bool]] = []

                # Wersja śliska: akcja nie realizuje się dokładnie
                if is_slippery:
                    # Agent wybiera a, ale środowisko wykonuje (a-1), a, (a+1) z prawd. 1/3
                    for a_real in ((a - 1) % 4, a, (a + 1) % 4):
                        r2, c2 = step_from(r, c, a_real)
                        s2 = to_s(r2, c2)
                        tile2 = desc_arr[r2, c2]
                        terminated = tile2 in ("H", "G")
                        reward = 1.0 if tile2 == "G" else 0.0
                        outcomes.append((1.0/3.0, s2, reward, terminated))
                else:
                    r2, c2 = step_from(r, c, a)
                    s2 = to_s(r2, c2)
                    tile2 = desc_arr[r2, c2]
                    terminated = tile2 in ("H", "G")
                    reward = 1.0 if tile2 == "G" else 0.0
                    outcomes.append((1.0, s2, reward, terminated))

                # Scalanie identycznych następstw (różne a_real mogą prowadzić do tego samego s2)
                merged: Dict[Tuple[int,float,bool], float] = {}
                for p, s2, rwd, term in outcomes:
                    key = (int(s2), float(rwd), bool(term))
                    merged[key] = merged.get(key, 0.0) + float(p)

                # Finalna lista wyników dla (s, a)
                P[s][a] = [(p, s2, rwd, term) for (s2, rwd, term), p in merged.items()]

    return P, nS, nA, nrow, ncol, desc_arr


def find_start_state(desc_arr: np.ndarray) -> int:
    """Zwraca indeks stanu startowego 'S'."""
    nrow, ncol = desc_arr.shape
    for r in range(nrow):
        for c in range(ncol):
            if desc_arr[r, c] == "S":
                return r * ncol + c
    return 0


class _Discrete:
    """Minimalny odpowiednik gym.spaces.Discrete (tylko atrybut .n)."""
    def __init__(self, n: int):
        self.n = int(n)


class PModelEnv:
    """
    Minimalne środowisko typu Gym (reset/step) oparte o jawny model P[s][a].

    - P[s][a] = [(p, s2, r, terminated), ...]
    - step(a) losuje jedno przejście zgodnie z p.
    - truncated=False zawsze; limit kroków realizuj w kodzie rolloutów (max_steps).

    Dodatkowe atrybuty:
    - nS, nA
    - desc (jeśli podasz)
    - nrow, ncol (jeśli podasz)
    """
    def __init__(self, P, start_state: int = 0, seed: int = 0, desc: Optional[np.ndarray] = None):
        self.P = P
        self.nS = len(P)
        s0 = next(iter(P))
        self.nA = len(P[s0])
        self.action_space = _Discrete(self.nA)
        self.observation_space = _Discrete(self.nS)

        self.start_state = int(start_state)
        self.rng = np.random.default_rng(seed)
        self.s = int(start_state)

        self.desc = desc
        if desc is not None:
            self.nrow, self.ncol = desc.shape
        else:
            self.nrow = self.ncol = None

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.s = int(self.start_state)
        return int(self.s), {}

    def step(self, a: int):
        a = int(a)
        outcomes = self.P[self.s][a]  # [(p, s2, r, terminated), ...]
        ps = np.array([o[0] for o in outcomes], dtype=float)
        ps = ps / ps.sum()
        idx = int(self.rng.choice(len(outcomes), p=ps))
        p, s2, r, terminated = outcomes[idx]
        self.s = int(s2)
        truncated = False
        info: Dict[str, Any] = {"p": float(p)}
        return int(s2), float(r), bool(terminated), bool(truncated), info
