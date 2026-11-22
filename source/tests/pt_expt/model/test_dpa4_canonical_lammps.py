# SPDX-License-Identifier: LGPL-3.0-or-later
"""Optional LAMMPS smoke for a two-block DPA4 canonical artifact.

The model archive is not committed (it is a trained checkpoint), so this
test runs only when the caller provides the same environment as the
``examples/lammps/water_scan0_lr`` harness:

``WATER_SCAN0_LMP_DATA``, ``WATER_SCAN0_MODEL``, ``WATER_SCAN0_PLUGIN``,
``LMP`` and ``mpirun`` on PATH drive the host pair style at np=1/2/4.  The
Kokkos leg is a second test that additionally needs ``WATER_SCAN0_PLUGIN_KK``,
``WATER_SCAN0_BACKEND_PLUGIN_KK`` and ``LMP_KK``; it runs the same fixture on
the 192-atom two-block canonical with-comm sibling.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import (
    Path,
)

import pytest

pytestmark = pytest.mark.skipif(
    not all(
        os.environ.get(name)
        for name in (
            "WATER_SCAN0_LMP_DATA",
            "WATER_SCAN0_MODEL",
            "WATER_SCAN0_PLUGIN",
            "LMP",
        )
    )
    or shutil.which("mpirun") is None,
    reason="the optional LAMMPS/model fixture is not configured",
)


def test_two_block_canonical_runs_multi_rank_in_lammps() -> None:
    """The host pair style runs np=1/2/4 with matching E/P."""
    here = Path(__file__).resolve().parents[4]
    script = here / "examples" / "lammps" / "water_scan0_lr" / "run_validation.sh"
    env = dict(os.environ)
    env["WATER_SCAN0_RUN_KK"] = "0"
    subprocess.run([str(script)], cwd=script.parent, env=env, check=True)


@pytest.mark.skipif(
    not all(
        os.environ.get(name)
        for name in (
            "WATER_SCAN0_LMP_DATA",
            "WATER_SCAN0_MODEL",
            "WATER_SCAN0_PLUGIN_KK",
            "WATER_SCAN0_BACKEND_PLUGIN_KK",
            "LMP_KK",
        )
    )
    or shutil.which("mpirun") is None,
    reason="the optional Kokkos LAMMPS fixture is not configured",
)
def test_two_block_canonical_runs_kokkos_multi_rank() -> None:
    """Kokkos exercises the with-comm sibling at np=1/2/4.

    The canonical with-comm sibling consumes the full source CSR even when
    ``canonical_requires_source_csr`` is false; passing the placeholder makes
    AOTI fail on the first Kokkos call.  This is the regression guard for that
    path.
    """
    here = Path(__file__).resolve().parents[4]
    input_file = here / "examples" / "lammps" / "water_scan0_lr" / "in.water.lammps"
    env = dict(os.environ)
    env["DP_BACKEND_PLUGIN_PATH"] = env["WATER_SCAN0_BACKEND_PLUGIN_KK"]
    host_lib = os.path.dirname(os.path.abspath(env["WATER_SCAN0_PLUGIN"]))
    other = [
        p
        for p in env.get("LD_LIBRARY_PATH", "").split(":")
        if p and os.path.abspath(p) != host_lib
    ]
    env["LD_LIBRARY_PATH"] = ":".join([env["WATER_SCAN0_BACKEND_PLUGIN_KK"], *other])
    results = []
    for np in (1, 2, 4):
        cmd = [
            shutil.which("mpirun"),
            "-np",
            str(np),
            "bash",
            "-c",
            'export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK:-0}; exec "$@"',
            "bash",
            env["LMP_KK"],
            "-k",
            "on",
            "g",
            "1",
            "-sf",
            "kk",
            "-var",
            "DATA",
            env["WATER_SCAN0_LMP_DATA"],
            "-var",
            "MODEL",
            env["WATER_SCAN0_MODEL"],
            "-var",
            "PLUGIN",
            env["WATER_SCAN0_PLUGIN_KK"],
            "-log",
            "none",
            "-in",
            str(input_file),
        ]
        proc = subprocess.run(
            cmd,
            cwd=input_file.parent,
            env=env,
            text=True,
            capture_output=True,
        )
        assert proc.returncode == 0, proc.stderr
        out = proc.stdout
        match = re.search(r"^\s*0\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", out, re.M)
        assert match is not None, out
        results.append((float(match.group(1)), float(match.group(2))))
    energies = [row[0] for row in results]
    pressures = [row[1] for row in results]
    assert max(energies) - min(energies) < 1e-4, results
    assert max(pressures) - min(pressures) < 0.01, results
