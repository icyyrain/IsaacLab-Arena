# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

_EMBODIMENT_MODULES = (
    ".agibot.agibot",
    ".droid.droid",
    ".franka.franka",
    ".g1.g1",
    ".galbot.galbot",
    ".gr1t2.gr1t2",
    ".kuka_allegro.kuka_allegro",
)

for module_name in _EMBODIMENT_MODULES:
    try:
        module = import_module(module_name, package=__name__)
    except ModuleNotFoundError:
        continue
    public_names = getattr(module, "__all__", None)
    if public_names is None:
        public_names = tuple(name for name in vars(module) if not name.startswith("_"))
    for name in public_names:
        globals()[name] = getattr(module, name)
