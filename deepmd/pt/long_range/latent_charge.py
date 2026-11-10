# SPDX-License-Identifier: LGPL-3.0-or-later
"""Latent-charge long-range module of the pt (legacy) backend.

The physics is the dpmodel implementation; this module gives the pt backend a
native registry entry, builds the charge network and the interaction kernel
from the pt fitting classes, and makes the module a ``torch.nn.Module`` so its
parameters are trained and serialized with the model.
"""

from __future__ import annotations

from typing import (
    Any,
)

import numpy as np
import torch

from deepmd.dpmodel.long_range.latent_charge import (
    CHARGE_NET_NEURON,
    CHARGE_NET_SEED_INDEX,
    SOG_DEFAULT_M,
    sog_default_ladder,
)
from deepmd.dpmodel.long_range.latent_charge import (
    LatentChargeLongRange as LatentChargeLongRangeDP,
)
from deepmd.dpmodel.long_range.reciprocal import (
    mesh_energy,
    kspace_energy,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    K_GRID,
    PME_MESH,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.task.latent_charge import (
    LatentChargeFitting,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_long_range import (
    BaseLongRange,
)
from .kernel import (
    LESKernel,
    SOGKernel,
)

_KERNEL_CLASSES = {"les": LESKernel, "sog": SOGKernel}

# The SOG library's default ladder: sigma = nlayers * rcut / RCUT_TO_SIGMA,
# b = 2, M = 12.  The pt backend keeps this rule for new configurations
# (``sog_default_ladder``); a migrated checkpoint passes its explicit widths
# and amplitudes instead.
_SOG_DEFAULT_M = SOG_DEFAULT_M


@BaseLongRange.register("latent_charge")
class LatentChargeLongRange(LatentChargeLongRangeDP, torch.nn.Module):
    """Energy of interacting latent charges, periodic or not (pt backend)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        LatentChargeLongRangeDP.__init__(self, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        # ``NativeOP`` owns ``__call__``; drive ``forward`` through torch so
        # tracing and ``to(device)`` see the module.
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        node_mask: torch.Tensor,
        n_node: torch.Tensor,
        topologies: dict[str, Any],
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        want_potential: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.call_graph(
            descriptor,
            atype,
            node_mask,
            n_node,
            topologies,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            want_potential=want_potential,
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
        descriptor: Any,
        fitting: Any,
        fitting_seed: int | list[int] | None = None,
    ) -> LatentChargeLongRange:
        """Build the module from its configuration on the pt backend."""
        charge_net_config = {
            "neuron": CHARGE_NET_NEURON,
            "activation_function": fitting.activation_function,
            "resnet_dt": fitting.resnet_dt,
            "precision": fitting.precision,
            "seed": child_seed(fitting_seed, CHARGE_NET_SEED_INDEX),
            **config.get("charge_net", {}),
        }
        bias_atom_q = charge_net_config.pop("bias_atom_q", None)
        kernel_config = {
            "type": "les",
            "precision": charge_net_config["precision"],
            **config.get("kernel", {}),
        }
        kernel_type = str(kernel_config.pop("type"))
        if kernel_type == "sog" and not any(
            kernel_config.get(key) is not None
            for key in ("width", "amplitude", "base_width", "width_ratio")
        ):
            # D12: new configurations follow the SOG library rule rather than
            # the upstream base_width/width_ratio defaults.
            widths, amplitudes = sog_default_ladder(
                descriptor,
                n_gaussian=int(kernel_config.get("n_gaussian", _SOG_DEFAULT_M)),
            )
            kernel_config["width"] = widths
            kernel_config["amplitude"] = amplitudes
        try:
            kernel_cls = _KERNEL_CLASSES[kernel_type]
        except KeyError as exc:
            raise ValueError(
                f"unknown long-range kernel {kernel_type!r}; expected one of "
                f"{tuple(_KERNEL_CLASSES)}"
            ) from exc
        charge_net = LatentChargeFitting(
            var_name="latent_charge",
            ntypes=descriptor.get_ntypes(),
            dim_descrpt=descriptor.get_dim_out(),
            dim_out=int(config.get("channels", 1)),
            numb_fparam=fitting.get_dim_fparam(),
            numb_aparam=fitting.get_dim_aparam(),
            dim_case_embd=fitting.dim_case_embd,
            default_fparam=fitting.get_default_fparam(),
            use_aparam_as_mask=fitting.use_aparam_as_mask,
            mixed_types=descriptor.mixed_types(),
            type_map=fitting.get_type_map(),
            bias_atom_e=(
                None
                if bias_atom_q is None
                else torch.as_tensor(
                    bias_atom_q,
                    device="cpu",
                    dtype=(
                        torch.float64
                        if str(charge_net_config["precision"]).lower()
                        in ("float64", "double")
                        else torch.float32
                    ),
                ).reshape(-1, int(config.get("channels", 1)))
            ),
            **charge_net_config,
        )
        return cls(
            charge_net,
            kernel_cls(**kernel_config),
            config.get("charge_constraint"),
            periodic=config.get("periodic", False),
            kspace_tolerance=config.get("kspace_tolerance", 1e-8),
            k_grid_mode=config.get("k_grid_mode"),
            k_grid_bounds=config.get("k_grid_bounds"),
            decouple_k_grid=config.get("decouple_k_grid", False),
            pme_spacing=config.get("pme_spacing"),
            pme_order=config.get("pme_order"),
        )

    @classmethod
    def deserialize(cls, data: dict) -> LatentChargeLongRange:
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        kernel_data = data["kernel"].copy()
        kernel_type = str(kernel_data.get("type", "les"))
        try:
            kernel_cls = _KERNEL_CLASSES[kernel_type]
        except KeyError as exc:
            raise ValueError(
                f"unknown long-range kernel {kernel_type!r}; expected one of "
                f"{tuple(_KERNEL_CLASSES)}"
            ) from exc
        obj = cls(
            LatentChargeFitting.deserialize(data["charge_net"]),
            kernel_cls.deserialize(kernel_data),
            data["charge_constraint"],
            periodic=data.get("periodic", False),
            kspace_tolerance=data.get("kspace_tolerance", 1e-8),
            k_grid_mode=data.get("k_grid_mode", "ragged"),
            k_grid_bounds=data.get("k_grid_bounds"),
            decouple_k_grid=data.get("decouple_k_grid", False),
            pme_spacing=data.get("pme_spacing"),
            pme_order=data.get("pme_order"),
        )
        if data.get("frozen_k_max") is not None:
            obj.frozen_k_max = float(data["frozen_k_max"])
        charge_state = data.get("charge_state")
        if charge_state is not None:
            obj.fold_charge_state(np.asarray(charge_state).reshape(-1).tolist())
        return obj


# Re-export the physics helpers for callers that evaluate a module directly.
__all__ = [
    "K_GRID",
    "PME_MESH",
    "LatentChargeLongRange",
    "kspace_energy",
    "mesh_energy",
]
