# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert a NumPy 2 RL-Games checkpoint for loading with NumPy 1.x."""

from __future__ import annotations

import argparse
import numpy as np
import sys
import torch
from pathlib import Path


def install_numpy_checkpoint_compatibility() -> None:
    """Expose NumPy 2 pickle module names while reading the source checkpoint."""
    if hasattr(np, "_core"):
        return

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


def main() -> None:
    """Convert the input checkpoint and write a locally compatible copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source RL-Games checkpoint")
    parser.add_argument("output", type=Path, help="Converted checkpoint destination")
    args = parser.parse_args()

    assert args.input.is_file(), f"Checkpoint not found: {args.input}"
    install_numpy_checkpoint_compatibility()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Converted checkpoint: {args.output}", flush=True)


if __name__ == "__main__":
    main()
