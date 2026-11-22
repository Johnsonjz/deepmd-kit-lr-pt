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

    def _bias_requires_grad(self) -> bool:
        trainable = getattr(self, "trainable", None)
        if trainable is None:
            return True
        if isinstance(trainable, bool):
            return trainable
        return bool(trainable[-1]) if trainable else True

    def __setattr__(self, name: str, value: Any) -> None:
        # A deserialized snapshot may assign the bias after construction; keep
        # the output-layer trainable flag authoritative for it.
        if (
            name in self.TRAINABLE_BIASES
            and "_parameters" in self.__dict__
            and not isinstance(value, torch.nn.Parameter)
        ):
            value = torch.nn.Parameter(
                torch.as_tensor(value).detach().clone(),
                requires_grad=self._bias_requires_grad(),
            )
        elif name in self.TRAINABLE_BIASES and isinstance(value, torch.nn.Parameter):
            value.requires_grad_(self._bias_requires_grad())
        super().__setattr__(name, value)

    def train(self, mode: bool = True) -> "LatentChargeFitting":
        super().train(mode)
        param = self._parameters.get("bias_atom_e")
        if param is not None:
            param.requires_grad_(self._bias_requires_grad())
        return self

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The bias follows the dtype of the network it offsets.  A float32 model
        # that carried a float64 parameter would, on resume, restore float64
        # optimizer moments that the fused float32 Adam path cannot mix with its
        # gradients.
        bias_dtype = next(
            (p.dtype for p in self.parameters() if p.is_floating_point()), None
        )
        # Honor the fitting-level `trainable` flags: the charge bias is the
        # output-layer parameter, so it freezes with the output layer.
        bias_trainable = self._bias_requires_grad()
        for name in self.TRAINABLE_BIASES:
            existing = self._parameters.get(name)
            if existing is not None:
                existing.requires_grad_(bias_trainable)
                continue
            tensor = self._buffers.get(name)
            if tensor is None:
                continue
            dtype = bias_dtype if bias_dtype is not None else tensor.dtype
            # `torch.nn.Module.__setattr__` drops the same-named buffer when
            # the value is a Parameter, so the promotion needs no cleanup.
            torch.nn.Module.__setattr__(
                self,
                name,
                torch.nn.Parameter(
                    tensor.detach().to(dtype).clone(),
                    requires_grad=bias_trainable,
                ),
            )
