# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test the Kokkos pair style with a two-block DPA4 long-range canonical model.

``pair_style deepmd/kk`` evaluates the archive through its compact canonical
lower.  A two-block SeZM descriptor also freezes the cross-rank sibling
(``has_comm_artifact``), and that sibling consumes the **full source CSR** of
the graph even when ``canonical_requires_source_csr`` is false: passing the
two-slot placeholder instead makes AOTI fail on the first call, which is the
deployment path this module guards.  The sibling also needs the halo plan, so
the two ranks of the parallel run exchange their node features.

Run ``source/install/test_lammps_kokkos.sh`` after
``source/install/test_cc_local.sh`` on a CUDA host: that job generates the
fixture with ``source/tests/infer/gen_dpa4_2l_canonical.py`` and sets
``DPA4_2L_KOKKOS_REQUIRED=1`` so that no missing prerequisite skips the test
silently.  The model is untrained: only the agreement between the one-rank and
two-rank evaluations of the same archive is asserted.
"""

from __future__ import (
    annotations,
)

import importlib.util
import json
import os
import shutil
import zipfile
from pathlib import (
    Path,
)

import numpy as np
import pytest
from expected_ref import (
    read_expected_ref,
)
from lammps_test_utils import (
    run_mpi_pair_runner,
)
from write_lmp_data import (
    write_lmp_data,
)

_INFER = Path(__file__).resolve().parents[2] / "tests" / "infer"
_RUNNER = Path(__file__).with_name("run_mpi_pair_deepmd_dpa3_pt2.py")


@pytest.fixture(scope="module")
def kokkos_job() -> None:
    # The ordinary LAMMPS suite also runs on CUDA, but uses a non-Kokkos wheel.
    # Only the dedicated matching library/plugin job opts into this regression.
    if os.environ.get("DPA4_2L_KOKKOS_REQUIRED") != "1":
        pytest.skip("Enable the dedicated CUDA Kokkos plugin regression explicitly")
    for module in ("lammps", "mpi4py"):
        if importlib.util.find_spec(module) is None:
            pytest.fail(f"Required Python package missing: {module}")
    if shutil.which("mpirun") is None:
        pytest.fail("mpirun required for the Kokkos regression")


def test_two_block_with_comm_agrees_across_ranks(
    kokkos_job: None, tmp_path: Path
) -> None:
    """The with-comm canonical sibling yields the one-rank result on two ranks."""
    model = _INFER / "deeppot_dpa4_2l_canonical.pt2"
    assert model.is_file(), f"Generate the fixture first: {model}"
    with zipfile.ZipFile(model) as archive:
        metadata = json.loads(archive.read("model/extra/metadata.json"))
    assert metadata["lower_input_kind"] == "dpa4c_canonical"
    assert metadata["has_message_passing"]
    assert metadata["has_comm_artifact"]
    system = read_expected_ref(model.with_suffix(".expected"))["system"]
    cell = system["box"].reshape(3, 3)
    # xlo xhi ylo yhi zlo zhi xy xz yz, with the lower corner at the origin.
    box = np.array([0, cell[0, 0], 0, cell[1, 1], 0, cell[2, 2], 0, 0, 0], dtype=float)
    data = tmp_path / "lattice.lmp"
    write_lmp_data(
        box,
        system["coord"].reshape(-1, 3),
        system["atype"].astype(int) + 1,
        data,
    )
    single, parallel = (
        run_mpi_pair_runner(
            _RUNNER,
            data,
            model,
            nprocs=ranks,
            processors="1 1 1" if ranks == 1 else f"{ranks} 1 1",
            runner_args=["--kokkos"],
            timeout=600,
        )
        for ranks in (1, 2)
    )
    assert parallel["pe"] == pytest.approx(single["pe"], rel=1e-5, abs=1e-5)
    np.testing.assert_allclose(
        parallel["forces"], single["forces"], rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        parallel["virials"], single["virials"], rtol=1e-4, atol=1e-5
    )
