"""
OBD Simulated Bandit Environment for TF-Agents.

Wraps the reward model (logistic regression) as a BanditPyEnvironment.
At each step:
  - observation: a context vector (20d) sampled from train+val contexts
  - action: integer 0..n_actions-1
  - reward: Bernoulli sample from reward_model.predict_proba(concat(ctx, action_ctx[action]))
"""

from __future__ import annotations

import numpy as np
from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.specs import array_spec

from src.bandits.reward_model import build_features, load_reward_model


class OBDSimBanditEnv(bandit_py_environment.BanditPyEnvironment):
    """
    Simulated contextual bandit environment backed by a trained reward model.

    observation_spec: (20,) float32  — user/round context
    action_spec:      int32 discrete — item index 0..n_actions-1
    """

    def __init__(
        self,
        model_dir: str,
        contexts: np.ndarray,
        seed: int = 42,
    ) -> None:
        self._model, self._action_context = load_reward_model(model_dir)
        self._contexts = contexts.astype(np.float32)
        self._n_actions = self._action_context.shape[0]
        self._n_contexts = len(contexts)
        self._rng = np.random.RandomState(seed)
        self._current_context: np.ndarray | None = None

        observation_spec = array_spec.ArraySpec(
            shape=(contexts.shape[1],), dtype=np.float32, name="observation"
        )
        action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self._n_actions - 1, name="action"
        )
        super().__init__(observation_spec, action_spec)

    def _observe(self) -> np.ndarray:
        idx = self._rng.randint(0, self._n_contexts)
        self._current_context = self._contexts[idx]
        return self._current_context

    def _apply_action(self, action: np.ndarray) -> float:
        action_int = int(action)
        ctx = self._current_context[np.newaxis, :]
        X = build_features(ctx, self._action_context, np.array([action_int]))
        p_click = float(self._model.predict_proba(X)[0, 1])
        reward = float(self._rng.binomial(1, p_click))
        return reward
