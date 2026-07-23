# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Transition schema with a lossless audit trail and upstream SERL mapping."""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TransitionSource = Literal["policy", "demonstration", "intervention"]
SCHEMA_VERSION = 1
_VALID_SOURCES = {"policy", "demonstration", "intervention"}


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    policy_action: np.ndarray
    executed_action: np.ndarray
    next_observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    source: TransitionSource
    intervention: bool
    episode_id: int
    step_id: int
    success: bool

    def __post_init__(self) -> None:
        observation = _float_array(self.observation, "observation")
        next_observation = _float_array(self.next_observation, "next_observation")
        policy_action = _float_array(self.policy_action, "policy_action")
        executed_action = _float_array(self.executed_action, "executed_action")
        assert observation.ndim == 1, f"observation must be one-dimensional, received {observation.shape}"
        assert (
            next_observation.shape == observation.shape
        ), f"next_observation shape {next_observation.shape} does not match {observation.shape}"
        assert policy_action.ndim == 1, f"policy_action must be one-dimensional, received {policy_action.shape}"
        assert (
            executed_action.shape == policy_action.shape
        ), f"executed_action shape {executed_action.shape} does not match {policy_action.shape}"
        assert np.isfinite(self.reward), f"reward is not finite: {self.reward}"
        assert self.source in _VALID_SOURCES, f"Unknown transition source: {self.source!r}"
        assert self.episode_id >= 0, f"episode_id must be non-negative: {self.episode_id}"
        assert self.step_id >= 0, f"step_id must be non-negative: {self.step_id}"
        if self.intervention:
            assert self.source == "intervention", "Intervention transitions must use source='intervention'"
        if self.source == "intervention":
            assert self.intervention, "source='intervention' requires intervention=True"
        if not self.intervention:
            assert np.array_equal(
                policy_action, executed_action
            ), "Non-intervention transition changed the executed action"

        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "next_observation", next_observation)
        object.__setattr__(self, "policy_action", policy_action)
        object.__setattr__(self, "executed_action", executed_action)
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(self, "intervention", bool(self.intervention))
        object.__setattr__(self, "episode_id", int(self.episode_id))
        object.__setattr__(self, "step_id", int(self.step_id))
        object.__setattr__(self, "success", bool(self.success))

    @classmethod
    def from_env_step(
        cls,
        *,
        observation: Any,
        policy_action: Any,
        next_observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        source: TransitionSource | None = None,
    ) -> Transition:
        intervention = bool(info.get("intervention", "intervene_action" in info))
        if source is None:
            source = "intervention" if intervention else "policy"
        executed_action = info.get("executed_action", info.get("intervene_action", policy_action))
        return cls(
            observation=observation,
            policy_action=policy_action,
            executed_action=executed_action,
            next_observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            source=source,
            intervention=intervention,
            episode_id=info["episode_id"],
            step_id=info["step_id"],
            success=bool(info.get("success", reward > 0.0)),
        )

    def to_serl_dict(self) -> dict[str, Any]:
        """Return the exact keys consumed by upstream ``train_rlpd.py``.

        Bellman updates intentionally use ``executed_action``. Time-limit
        truncation keeps ``masks=1`` so the critic can bootstrap.
        """
        return {
            "observations": self.observation,
            "actions": self.executed_action,
            "next_observations": self.next_observation,
            "rewards": np.float32(self.reward),
            "masks": np.float32(0.0 if self.terminated else 1.0),
            "dones": np.bool_(self.terminated),
        }


def validate_episode_boundaries(transitions: Sequence[Transition]) -> None:
    """Ensure steps are contiguous and no transition crosses a reset boundary."""
    if not transitions:
        return
    assert transitions[0].step_id == 0, "The first transition of an episode must have step_id=0"
    for previous, current in zip(transitions, transitions[1:], strict=False):
        previous_done = previous.terminated or previous.truncated
        if current.episode_id == previous.episode_id:
            assert (
                not previous_done
            ), f"Episode {previous.episode_id} contains a transition after terminal step {previous.step_id}"
            assert (
                current.step_id == previous.step_id + 1
            ), f"Episode {current.episode_id} step jumped from {previous.step_id} to {current.step_id}"
        else:
            assert (
                current.episode_id > previous.episode_id
            ), f"episode_id moved backwards from {previous.episode_id} to {current.episode_id}"
            assert (
                previous_done
            ), f"Episode changed from {previous.episode_id} to {current.episode_id} without termination"
            assert current.step_id == 0, f"Episode {current.episode_id} must start at step_id=0"


def save_transition_dataset(path: str | Path, transitions: Iterable[Transition]) -> Path:
    """Save a fixed-shape state dataset without pickle."""
    path = Path(path)
    items = list(transitions)
    assert items, "Cannot save an empty transition dataset"
    validate_episode_boundaries(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        observations=np.stack([item.observation for item in items]),
        policy_actions=np.stack([item.policy_action for item in items]),
        executed_actions=np.stack([item.executed_action for item in items]),
        next_observations=np.stack([item.next_observation for item in items]),
        rewards=np.asarray([item.reward for item in items], dtype=np.float32),
        terminated=np.asarray([item.terminated for item in items], dtype=np.bool_),
        truncated=np.asarray([item.truncated for item in items], dtype=np.bool_),
        sources=np.asarray([item.source for item in items], dtype="U16"),
        interventions=np.asarray([item.intervention for item in items], dtype=np.bool_),
        episode_ids=np.asarray([item.episode_id for item in items], dtype=np.int64),
        step_ids=np.asarray([item.step_id for item in items], dtype=np.int64),
        successes=np.asarray([item.success for item in items], dtype=np.bool_),
    )
    return path


def load_transition_dataset(path: str | Path) -> list[Transition]:
    """Load and validate a dataset written by :func:`save_transition_dataset`."""
    with np.load(Path(path), allow_pickle=False) as dataset:
        version = int(dataset["schema_version"])
        assert version == SCHEMA_VERSION, f"Unsupported transition schema version: {version}"
        size = len(dataset["rewards"])
        transitions = [
            Transition(
                observation=dataset["observations"][index],
                policy_action=dataset["policy_actions"][index],
                executed_action=dataset["executed_actions"][index],
                next_observation=dataset["next_observations"][index],
                reward=float(dataset["rewards"][index]),
                terminated=bool(dataset["terminated"][index]),
                truncated=bool(dataset["truncated"][index]),
                source=str(dataset["sources"][index]),
                intervention=bool(dataset["interventions"][index]),
                episode_id=int(dataset["episode_ids"][index]),
                step_id=int(dataset["step_ids"][index]),
                success=bool(dataset["successes"][index]),
            )
            for index in range(size)
        ]
    validate_episode_boundaries(transitions)
    return transitions


def _float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    assert np.isfinite(array).all(), f"{name} contains NaN or Inf"
    return array.copy()
