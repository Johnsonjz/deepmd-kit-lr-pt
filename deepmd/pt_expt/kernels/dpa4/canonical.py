# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compact canonical deployment of the DPA4/SeZM family.

The DPA4C compact lower fuses the descriptor and the fitting into one device
kernel (``dpa4c_canonical_compress_*``). SeZM is a message-passing stack with
no such fused kernel, so this module adapts the *compact canonical input ABI*
onto the existing graph lower instead of fusing:

* the destination CSR names the storage slot range of every node, so the
  edge endpoints are recovered with two pure tensor ops (``searchsorted`` and
  ``arange``) and the geometry (``edge_vec``) is passed through untouched;
* the descriptor, the message passing, the force assembly and the virial are
  the same operators as the graph lower, so the numerics match it exactly.

A long-range module needs its reciprocal-space topology, whose mesh dimensions
are computed by ``mesh_dims_of_cells`` after a detach to NumPy; that cannot be
traced into the AOTInductor package. A SeZM model with a long-range module
therefore evaluates the module through the fused ``latent_charge_sum``
operator, which sizes the mesh of every frame from its cells inside the
compiled program (see :func:`sezm_canonical_energy_force`); the canonical ABI
then carries the node positions, the communicator and the cells, and no
topology tensors.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

if TYPE_CHECKING:
    from deepmd.pt_expt.utils.canonical_graph import (
        CanonicalGraph,
    )


def _source_indices(source: torch.Tensor) -> torch.Tensor:
    """Recover node indices from the source carrier, whatever its word width.

    The compact ABI exports ``source`` as little-endian bytes with a
    *dynamic* word width (four or eight bytes), so ``view(int32/int64)``,
    which needs a static width, does not trace. Reading the bytes as a
    base-256 little-endian integer is width-agnostic and traces: every
    deployed node index is far below ``2**31``, so the high words are zero
    and no partial product overflows int64.
    """
    if source.dtype != torch.uint8:
        return source.to(torch.int64)
    words = source.to(torch.int64)
    width = words.shape[-1]
    weights = torch.pow(
        torch.tensor(256, dtype=torch.int64, device=words.device),
        torch.arange(width, dtype=torch.int64, device=words.device),
    )
    return (words * weights).sum(dim=-1)


def canonical_model_eligible(model: Any) -> bool:
    """Return whether a DPA4/SeZM model can use the compact canonical lower.

    The model must be a plain (unbridged) energy model whose descriptor is
    :class:`~deepmd.pt_expt.descriptor.dpa4.DescrptDPA4`, with an exportable
    graph lower and no atom/pair exclusion that the graph lower could not
    apply on the compact node axis.
    """
    atomic_model = getattr(model, "atomic_model", None)
    if atomic_model is None:
        return False
    from deepmd.pt_expt.kernels.utils import (
        backend_device_type,
    )

    if backend_device_type() != "cuda":
        return False
    parts = atomic_model.fused_decomposition()
    if parts is None:
        return False
    learned = parts[0]
    if learned is None or parts[1] is not None:
        return False
    from deepmd.pt_expt.descriptor.dpa4 import (
        DescrptDPA4,
    )

    if not isinstance(learned.descriptor, DescrptDPA4):
        return False
    if getattr(atomic_model, "pair_excl", None) is not None:
        return False
    if getattr(atomic_model, "atom_excl", None) is not None:
        return False
    from deepmd.pt_expt.model.graph_lower import (
        model_uses_graph_lower,
    )

    return model_uses_graph_lower(model)


def canonical_graph_edge_inputs(
    graph: CanonicalGraph,
    node_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover the graph-lower edge tensors from the compact canonical graph.

    The destination CSR row ``i`` owns the storage slots
    ``[destination_row_ptr[i], destination_row_ptr[i + 1])``. The destination
    of every slot is therefore the last row whose start is not above it, which
    ``searchsorted`` returns in one op; guard slots past the physical edge
    count land on the last node and are excluded by ``edge_mask``.

    Parameters
    ----------
    graph
        The compact canonical graph.
    node_count
        The flat node count ``N`` (local plus halo).

    Returns
    -------
    tuple
        ``edge_index`` ``(2, S)`` int64, ``edge_vec`` ``(S, 3)`` float32,
        ``edge_mask`` ``(S,)`` bool and the identity ``destination_order``
        ``(S,)`` int64.
    """
    device = graph.edge_vec.device
    source = _source_indices(graph.source)
    if source.dtype != torch.int64:
        source = source.to(torch.int64)
    row_ptr = graph.destination_row_ptr.to(torch.int64)
    slots = torch.arange(source.shape[0], dtype=torch.int64, device=device)
    # searchsorted(row_ptr, slot, right=True) - 1 is the row index whose
    # half-open [start, end) contains the slot; guard slots clamp to the last
    # row and are masked out by edge_mask.
    destination = (torch.searchsorted(row_ptr, slots, right=True) - 1).clamp(
        0, node_count - 1
    )
    edge_index = torch.stack([source, destination], dim=0)
    edge_vec = graph.edge_vec
    edge_mask = slots < row_ptr[-1]
    return edge_index, edge_vec, edge_mask, slots


def canonical_long_range_op_eligible(model: Any) -> bool:
    """Whether the long-range module of a SeZM model can run in the canonical lower.

    The compact canonical ABI carries no reciprocal-space topology, so the
    module must evaluate its sum inside the program through the fused
    ``latent_charge_sum`` operator, which sizes the mesh of every frame from
    its cells at run time. A module the operator does not serve -- another
    charge network width, frame parameters, a charge/spin condition -- needs
    the topology tensors the graph lower takes instead.
    """
    atomic_model = getattr(model, "atomic_model", None)
    if atomic_model is None or not atomic_model.has_long_range():
        return False
    eligible = getattr(atomic_model.long_range, "operator_eligible", None)
    return bool(callable(eligible) and eligible())


def sezm_canonical_energy_force(
    model: Any,
    graph: CanonicalGraph,
    atype: torch.Tensor,
    *,
    do_atomic_virial: bool,
    topologies: dict[str, Any] | None = None,
    spin: torch.Tensor | None = None,
    coord: torch.Tensor | None = None,
    communicator: torch.Tensor | None = None,
    box: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate a SeZM energy model on the compact canonical graph.

    Reconstructs the graph-lower edge tensors and delegates to the model's
    ordinary graph evaluation, then renames the outputs to the public
    canonical keys and adds the owned-node ``mask`` the deployment expects.

    A model with a long-range module passes the node positions, the
    communicator and the cells through to that evaluation, where the fused
    operator of the module evaluates the reciprocal-space sum; the canonical
    ABI then needs no topology tensors, whose mesh sizing detaches to NumPy
    and does not trace.
    """
    atomic_model = model.atomic_model
    has_long_range = atomic_model.has_long_range()
    if has_long_range and not canonical_long_range_op_eligible(model):
        raise NotImplementedError(
            "the compact canonical lower of a DPA4/SeZM model with a "
            "long-range module evaluates the module through the fused "
            "`latent_charge_sum` operator, which does not serve this module "
            "(see `operator_eligible`). Use the graph lower for this model."
        )
    n_node = graph.n_node
    n_local = graph.n_local
    edge_index, edge_vec, edge_mask, destination_order = canonical_graph_edge_inputs(
        graph, atype.shape[0]
    )
    long_range_op = None
    if has_long_range:
        if coord is None or box is None:
            raise ValueError(
                "the compact canonical lower of a model with a long-range "
                "module takes the node positions and the cells together"
            )
        long_range_op = {"coord": coord, "communicator": communicator, "box": box}
        # The operator path carries the long range itself; a topology would
        # only add autograd leaves that the energy does not read, which
        # `torch.autograd.grad` rejects as unused.
        topologies = None
    ret = model.forward_common_lower_graph(
        atype,
        n_node,
        n_local,
        edge_index,
        edge_vec,
        edge_mask,
        destination_order=destination_order,
        destination_row_ptr=graph.destination_row_ptr,
        source_order=graph.source_order,
        source_row_ptr=graph.source_row_ptr,
        destination_sorted=True,
        do_atomic_virial=do_atomic_virial,
        topologies=topologies,
        spin=spin,
        long_range_op=long_range_op,
    )
    public = model._to_public_keys(ret, do_atomic_virial)
    _, output_mask = atomic_model._prepare_graph_nodes(
        n_node,
        n_local,
        atype,
        edge_vec,
    )
    result = {
        "atom_energy": public["atom_energy"],
        "energy": public["energy"],
        "force": public["force"],
        "virial": public["virial"],
        "mask": output_mask.to(torch.int32),
    }
    if do_atomic_virial and "atom_virial" in public:
        result["atom_virial"] = public["atom_virial"]
    return result
