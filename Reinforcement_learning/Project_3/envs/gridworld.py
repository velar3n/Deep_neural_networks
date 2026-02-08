"""
Minimalny dyskretny GridWorld do zajęć.
- Dyskretne stany: (r,c) mapowane na indeks s = r*W+c
- Akcje: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
- Terminale: goal + holes
- Nagrody: +1 na celu, 0 w pozostałych krokach (domyślnie)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

Action = int
State = int

A_LEFT, A_DOWN, A_RIGHT, A_UP = 0, 1, 2, 3

@dataclass
class GridWorldConfig:
    height: int = 4
    width: int = 4
    start: Tuple[int,int] = (0,0)
    goal: Tuple[int,int] = (3,3)
    holes: Tuple[Tuple[int,int], ...] = ((1,1),(2,3))
    step_cost: float = 0.0  # np. -0.01 jeśli chcesz zachęcić do krótszych ścieżek
    gamma: float = 0.99
    max_steps: int = 200

class GridWorld:
    """
    Prosty environment w stylu Gymnasium (reset/step).
    Dodatkowo udostępnia model przejść `P` kompatybilny z FrozenLake:
        P[s][a] = [(prob, s', reward, terminated)]
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, cfg: GridWorldConfig = GridWorldConfig(), seed: Optional[int] = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.nS = cfg.height * cfg.width
        self.nA = 4
        self._state = self._to_s(cfg.start)
        self._steps = 0
        self.P = self._build_model()

    def _to_s(self, rc: Tuple[int,int]) -> int:
        r,c = rc
        return r * self.cfg.width + c

    def _to_rc(self, s: int) -> Tuple[int,int]:
        return divmod(s, self.cfg.width)

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.cfg.height and 0 <= c < self.cfg.width

    def _is_terminal_rc(self, r: int, c: int) -> bool:
        return (r,c) == self.cfg.goal or (r,c) in self.cfg.holes

    def _transition(self, s: int, a: int) -> Tuple[int, float, bool]:
        r,c = self._to_rc(s)
        if self._is_terminal_rc(r,c):
            # stan terminalny: brak ruchu
            return s, 0.0, True

        drdc = {A_LEFT:(0,-1), A_DOWN:(1,0), A_RIGHT:(0,1), A_UP:(-1,0)}[a]
        nr, nc = r + drdc[0], c + drdc[1]
        if not self._in_bounds(nr,nc):
            nr, nc = r, c

        terminated = self._is_terminal_rc(nr,nc)
        reward = self.cfg.step_cost
        if (nr,nc) == self.cfg.goal:
            reward = 1.0
        elif (nr,nc) in self.cfg.holes:
            reward = 0.0  # możesz ustawić np. -1.0 jeśli chcesz
        return self._to_s((nr,nc)), float(reward), bool(terminated)

    def _build_model(self):
        P: Dict[int, Dict[int, List[Tuple[float,int,float,bool]]]] = {s:{} for s in range(self.nS)}
        for s in range(self.nS):
            for a in range(self.nA):
                sp, r, done = self._transition(s,a)
                P[s][a] = [(1.0, sp, r, done)]
        return P

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._state = self._to_s(self.cfg.start)
        self._steps = 0
        info = {}
        return self._state, info

    def step(self, action: int):
        self._steps += 1
        sp, r, done = self._transition(self._state, int(action))
        self._state = sp
        truncated = self._steps >= self.cfg.max_steps
        info = {}
        return sp, r, done, truncated, info

    def render(self) -> str:
        H,W = self.cfg.height, self.cfg.width
        grid = [["." for _ in range(W)] for _ in range(H)]
        gr,gc = self.cfg.goal
        grid[gr][gc] = "G"
        for hr,hc in self.cfg.holes:
            grid[hr][hc] = "H"
        sr,sc = self._to_rc(self._state)
        if grid[sr][sc] == ".":
            grid[sr][sc] = "A"
        return "\n".join(" ".join(row) for row in grid)
