# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Thin constructors around the upstream HIL-SERL learner."""

from .state_sac import make_state_sac_agent

__all__ = ["make_state_sac_agent"]
