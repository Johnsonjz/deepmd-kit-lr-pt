# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-atom latent-charge fitting of the pt (legacy) backend.

The network maps the invariant descriptor of an atom to ``dim_out`` latent
charge channels in units of the elementary charge. The charges are an
intermediate field of a long-range module rather than a model output: the
module constrains and pairs them, and the potential energy is the only
quantity they are trained through.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.pt.model.task.fitting import (
    GeneralFitting,
)
from deepmd.pt.model.task.invar_fitting import (
    InvarFitting,
)

if TYPE_CHECKING:
    import torch


@GeneralFitting.register("latent_charge")
@fitting_check_output
class LatentChargeFitting(InvarFitting):
    """Per-atom latent charges predicted from the invariant descriptor (pt)."""

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "latent_charge"
        return data

    def call_graph(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        vacuum_descriptor: torch.Tensor | None = None,
        node_gate: torch.Tensor | None = None,
        return_middle_output: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Graph-native (flat node axis) forward of the charge network.

        The pt ``InvarFitting.forward`` is the dense ``(nf, nloc, ...)``
        contract; the long-range module works on the flat ``(N,)`` node axis,
        so this reuses the dense forward with ``nf'=N`` single-atom frames
        and restores the flat axis on the way out.
        """
        if return_middle_output:
            raise NotImplementedError(
                "the latent-charge fitting does not expose a middle output"
            )
        n, nd = descriptor.shape
        d1 = descriptor.reshape(n, 1, nd)
        a1 = atype.reshape(n, 1)
        g1 = None if gr is None else gr.reshape(n, 1, gr.shape[-2], 3)
        ap1 = None if aparam is None else aparam.reshape(n, 1, aparam.shape[-1])
        gate = None if node_gate is None else node_gate.reshape(n, 1, 1)
        ret = self.forward(
            d1,
            a1,
            gr=g1,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=ap1,
            vacuum_descriptor=vacuum_descriptor,
            node_gate=gate,
        )
        return {kk: vv.reshape(n, *vv.shape[2:]) for kk, vv in ret.items()}


__all__ = ["LatentChargeFitting"]
