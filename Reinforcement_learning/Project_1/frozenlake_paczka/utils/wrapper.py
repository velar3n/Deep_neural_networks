"""Jupyter-friendly rendering helpers for the FrozenLake notebooks.

This wrapper renders the environment (RGB array) together with optional
visualizations (V, Q, policy, learned model) in a single matplotlib figure.

Why this file exists
--------------------
Gymnasium environments are often wrapped (e.g., by `TimeLimit`). Some wrappers
(do not always forward) attributes like `nrow`/`ncol`. FrozenLake's grid
metadata lives on the *base* environment, so we always read it from
`env.unwrapped` when available.
"""

from __future__ import annotations

import gymnasium as gym
from IPython import display
import matplotlib.pyplot as plt

from utils.visualize import (
    visualize_policy,
    visualize_q,
    visualize_model,
    visualize_v,
)


class JupyterRender(gym.Wrapper):
    """A small wrapper that renders FrozenLake + helper plots inside notebooks."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Keep an explicit reference for clarity
        self.env = env

    def _base_env(self):
        """Return the underlying (unwrapped) environment if available."""
        try:
            return self.env.unwrapped
        except Exception:
            return self.env

    def _grid_shape(self):
        base = self._base_env()
        nrow = getattr(base, "nrow", None)
        ncol = getattr(base, "ncol", None)
        # Fallback to the standard 4x4 FrozenLake grid if attributes are missing
        return (int(nrow) if nrow is not None else 4, int(ncol) if ncol is not None else 4)

    def render(
        self,
        title: str = "Environment",
        v=None,
        q=None,
        policy=None,
        model_r=None,
        model_ns=None,
    ):
        """Render the environment + optional overlays.

        Parameters
        ----------
        title:
            Title for the environment panel.
        v, q, policy:
            Arrays to visualize (tabular). Shapes follow the notebooks.
        model_r, model_ns:
            Learned reward / next-state model (for Dyna-Q chapter).
        """

        viz_list = {}
        if v is not None:
            viz_list["v"] = v
        if q is not None:
            viz_list["q"] = q
        if policy is not None:
            viz_list["policy"] = policy
        if model_r is not None:
            viz_list["model_r"] = model_r
        if model_ns is not None:
            viz_list["model_ns"] = model_ns

        nrow, ncol = self._grid_shape()

        fig = plt.figure(figsize=(8, 8))

        ax_list = [fig.add_subplot(2, 2, 1)]
        ax_list[0].imshow(self.env.render())
        ax_list[0].set_title(title)

        # Additional panels
        for i in range(2, 2 + len(viz_list)):
            ax_list.append(fig.add_subplot(2, 2, i))

        ax_index = 1
        for key, value in viz_list.items():
            if key == "policy":
                visualize_policy(value, ax_list[ax_index], nrow, ncol)
            elif key == "v":
                visualize_v(value, ax_list[ax_index], nrow, ncol)
            elif key == "q":
                visualize_q(value, ax_list[ax_index], nrow, ncol)
            else:
                if key == "model_r":
                    panel_title = "Reward Model"
                elif key == "model_ns":
                    panel_title = "Next State Model"
                else:
                    panel_title = "Model"
                visualize_model(value, ax_list[ax_index], nrow, ncol, panel_title)

            ax_index += 1

        for ax in ax_list:
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        display.display(plt.gcf())
        display.clear_output(wait=True)
        plt.close()


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)
    env = JupyterRender(env)
    env.reset(seed=0)
    env.render(title="FrozenLake")
