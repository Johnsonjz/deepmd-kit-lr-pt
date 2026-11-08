# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pair interaction kernels of the pt (legacy) backend.

The dpmodel kernels own the physics; this module wraps them as
``torch.nn.Module`` objects whose trainable parameters are ``nn.Parameter``,
so the pt optimizer and the checkpoint serializer see them like any other
parameter of the model.
"""

from __future__ import annotations

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.long_range.kernel import (
    LESKernel as LESKernelDP,
)
from deepmd.dpmodel.long_range.kernel import (
    SOGKernel as SOGKernelDP,
)


def _torch_kernel(kernel_dp: type) -> type:
    """Wrap a dpmodel interaction kernel as a torch module.

    The attributes listed in ``PARAMETER_NAMES`` become ``nn.Parameter`` that
    require a gradient exactly when the kernel is trainable, both at
    construction and when a deserialized value is assigned.
    """

    class TorchKernel(kernel_dp, torch.nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            torch.nn.Module.__init__(self)
            kernel_dp.__init__(self, *args, **kwargs)

        def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            # ``NativeOP`` may define ``__call__``; drive ``forward`` through
            # torch on purpose so tracing and ``to(device)`` see the module.
            return torch.nn.Module.__call__(self, *args, **kwargs)

        def forward(self, r_sq: torch.Tensor) -> torch.Tensor:
            return self.call(r_sq)

        def __setattr__(self, name: str, value: Any) -> None:
            if name in self.PARAMETER_NAMES and "_parameters" in self.__dict__:
                if not isinstance(value, torch.nn.Parameter):
                    value = torch.nn.Parameter(
                        torch.as_tensor(value, device="cpu").detach().clone(),
                        requires_grad=bool(self.trainable),
                    )
                if name in self._parameters:
                    self._parameters[name] = value
                    return
            super().__setattr__(name, value)

    TorchKernel.__name__ = TorchKernel.__qualname__ = kernel_dp.__name__
    return TorchKernel


LESKernel = _torch_kernel(LESKernelDP)
SOGKernel = _torch_kernel(SOGKernelDP)

__all__ = ["LESKernel", "SOGKernel"]
