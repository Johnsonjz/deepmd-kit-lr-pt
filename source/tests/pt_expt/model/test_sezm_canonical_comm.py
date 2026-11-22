# SPDX-License-Identifier: LGPL-3.0-or-later
"""The with-comm gate of the compact canonical SeZM lower.

The gate itself is CPU-only: it decides from the atomic model's capabilities
whether a compact canonical artifact needs the eight-tensor border exchange.
The export-tracing tests that need the CUDA exporter are skipped without CUDA.
The canonical ghost fold is exercised on one rank by
``source/tests/pt_expt/model/test_dpa4c_parallel.py`` and end to end under
LAMMPS/MPI by ``source/tests/pt_expt/model/test_dpa4_canonical_lammps.py``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils.serialization import (
    _needs_canonical_with_comm,
    _needs_with_comm_artifact,
    _trace_and_export,
)


class _AtomicModel:
    def __init__(self, message_passing: bool, edge_parallel: bool = True) -> None:
        self._message_passing = message_passing
        self._edge_parallel = edge_parallel

    def has_message_passing_across_ranks(self) -> bool:
        return self._message_passing

    def supports_edge_parallel(self) -> bool:
        return self._edge_parallel


class _Model:
    def __init__(self, message_passing: bool, edge_parallel: bool = True) -> None:
        self.atomic_model = _AtomicModel(message_passing, edge_parallel)


@pytest.mark.parametrize("lower_kind", ["dpa4c_canonical", "dpa1_canonical"])
def test_a_message_passing_model_without_a_sezm_descriptor_keeps_the_plain_abi(
    lower_kind: str,
) -> None:
    """Only the SeZM canonical lower freezes a sibling.

    The fused DPA4C lower carries its halo exchange in the main ABI and the
    dense DPA1 lower has no sibling, so a model that does not decompose into a
    learned SeZM descriptor keeps the plain canonical archive even when its
    descriptor reports cross-rank message passing.
    """
    model = _Model(message_passing=True)
    assert _needs_canonical_with_comm(model, lower_kind) is False
    assert _needs_with_comm_artifact(model, lower_kind) is False


def test_a_single_block_canonical_model_needs_no_sibling() -> None:
    """A model without cross-rank message passing keeps the plain ABI."""
    model = _Model(message_passing=False)
    assert _needs_canonical_with_comm(model, "dpa4c_canonical") is False
    assert _needs_with_comm_artifact(model, "dpa4c_canonical") is False


def test_a_model_without_edge_parallel_support_needs_no_sibling() -> None:
    """A module that cannot follow a decomposed frame keeps single rank."""
    model = _Model(message_passing=True, edge_parallel=False)
    assert _needs_canonical_with_comm(model, "dpa4c_canonical") is False
    assert _needs_with_comm_artifact(model, "dpa4c_canonical") is False


def test_a_real_two_block_sezm_model_needs_the_canonical_sibling() -> None:
    """The gate sees the real descriptor capability, not only a fake model."""
    config = {
        "type": "dpa4",
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa4",
            "sel": 20,
            "rcut": 4.0,
            "channels": 16,
            "n_radial": 8,
            "lmax": 2,
            "mmax": 1,
            "n_blocks": 2,
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [16],
            "precision": "float64",
            "seed": 1,
        },
    }
    model = get_model(config)
    assert model.atomic_model.has_message_passing_across_ranks() is True
    assert _needs_canonical_with_comm(model, "dpa4c_canonical") is True
    assert _needs_with_comm_artifact(model, "dpa4c_canonical") is True


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="canonical export is CUDA-only"
)
def test_a_real_one_block_sezm_model_needs_no_canonical_sibling() -> None:
    """A one-block DPA4 reads no ghost feature beyond its own block."""
    config = {
        "type": "dpa4",
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa4",
            "sel": 20,
            "rcut": 4.0,
            "channels": 16,
            "n_radial": 8,
            "lmax": 2,
            "mmax": 1,
            "n_blocks": 1,
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [16],
            "precision": "float64",
            "seed": 1,
        },
    }
    model = get_model(config)
    assert model.atomic_model.has_message_passing_across_ranks() is False
    assert _needs_canonical_with_comm(model, "dpa4c_canonical") is False
    assert _needs_with_comm_artifact(model, "dpa4c_canonical") is False


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="canonical export is CUDA-only"
)
def test_two_block_canonical_export_traces_the_border_exchange() -> None:
    """The 2L DPA4 canonical with-comm program traces the border exchange."""
    config = {
        "type": "dpa4",
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa4",
            "sel": 20,
            "rcut": 4.0,
            "channels": 16,
            "n_radial": 8,
            "lmax": 2,
            "mmax": 1,
            "n_blocks": 2,
            "precision": "float32",
            "seed": 1,
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [16],
            "precision": "float32",
            "seed": 1,
        },
    }
    model = get_model(config).to("cuda").eval()
    model.atomic_model.enable_compression(0.5)
    exported, _metadata, _model_json, _keys = _trace_and_export(
        {"model": model.serialize()},
        lower_kind="dpa4c_canonical",
        with_comm_dict=True,
    )
    code = str(exported.module().code)
    assert "deepmd_export.border_op" in code
