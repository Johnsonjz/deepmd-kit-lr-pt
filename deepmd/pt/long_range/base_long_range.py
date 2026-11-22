# SPDX-License-Identifier: LGPL-3.0-or-later
"""Long-range module registry of the pt (legacy) backend.

The behaviour lives in :mod:`deepmd.dpmodel.long_range`; this module only
gives the backend its own registry, so a configuration builds the long-range
module native to the backend that reads it.
"""

from __future__ import annotations

import torch

from deepmd.dpmodel.long_range import (
    make_base_long_range,
)

BaseLongRange = make_base_long_range(torch.Tensor)

__all__ = ["BaseLongRange"]
