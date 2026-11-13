# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    ClassVar,
)

import torch

from deepmd.dpmodel.fitting.latent_charge import (
    LatentChargeFitting as LatentChargeFittingDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)


@torch_module
class LatentChargeFitting(LatentChargeFittingDP):
    """Per-atom latent charges predicted from the invariant descriptor (pt_expt).

    The per-type charge bias (``bias_atom_q``) is promoted from a buffer to an
    ``nn.Parameter``, so the trainer optimises it like the rest of the charge
    network.  The dpmodel layer adds it as a constant offset; keeping it
    trainable is what lets the charge scale of every type move instead of being
    pinned to its initial value for the whole run.
    """

    TRAINABLE_BIASES: ClassVar[tuple[str, ...]] = ("bias_atom_e",)
    """Per-type offsets of the output that stay trainable in this backend."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The bias follows the dtype of the network it offsets.  A float32 model
        # that carried a float64 parameter would, on resume, restore float64
        # optimizer moments that the fused float32 Adam path cannot mix with its
        # gradients.
        bias_dtype = next(
            (p.dtype for p in self.parameters() if p.is_floating_point()), None
        )
        for name in self.TRAINABLE_BIASES:
            tensor = self._buffers.get(name)
            if tensor is None:
                continue
            dtype = bias_dtype if bias_dtype is not None else tensor.dtype
            # `torch.nn.Module.__setattr__` drops the same-named buffer when
            # the value is a Parameter, so the promotion needs no cleanup.
            torch.nn.Module.__setattr__(
                self, name, torch.nn.Parameter(tensor.detach().to(dtype).clone())
            )
