#!/usr/bin/env python3
"""Tier-3 parity: the same frame on np = 1, 2 and 4, replicated and z-slab.

The fixed-point (2^40) charge spread makes the accumulation independent of the
order of the atomic additions, so with the shipped engine the energy per atom
and the per-atom forces are *bitwise* identical across rank counts and across
the two mesh modes.  Anything else is a distributed bug (a halo plane, a
reduction, an ownership mask), and this is the cheapest way to see one.

usage: mp_parity.py [N] [model.pt2 ...]
"""
import math
import os
import subprocess
import sys

BENCH = "/data/home/public/jiangzhen/dp/sog_work/bench"
FR = "/data/home/public/jiangzhen/dp/sog_work/mdsuite/frozen/dpa4_campaign"
LMP = "/data/home/public/jiangzhen/dp/lmp_kokkos_install/bin/lmp"
OMPI = "/data/app/openmpi-4.1.4/install/openmpi/4.1.4"
MPIRUN = f"{OMPI}/bin/mpirun"
CONDA = "/data/home/jiangzhen/.conda/envs/lr212"
TORCH = f"{CONDA}/lib/python3.11/site-packages/torch"
NVD = f"{CONDA}/lib/python3.11/site-packages/nvidia"
INSTALL = "/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib"
KK_PLUGIN = "/tmp/kk_lrpt_plugin"
RANK_DEV = "/tmp/lmp_rank_dev.sh"   # maps OMPI_COMM_WORLD_LOCAL_RANK to a GPU
OUT = "/tmp/mp_parity"

ENV = {
    "PATH": "/usr/local/cuda-12.4/bin:" + OMPI + "/bin:" + os.environ.get("PATH", ""),
    "LD_LIBRARY_PATH": ":".join([
        OMPI + "/lib", f"{CONDA}/lib", f"{TORCH}/lib",
        f"{NVD}/cuda_runtime/lib", f"{NVD}/cuda_cupti/lib", f"{NVD}/nvjitlink/lib",
        f"{NVD}/cudnn/lib", f"{NVD}/cublas/lib", f"{NVD}/cusparse/lib",
        f"{NVD}/cusparselt/lib", f"{NVD}/curand/lib", f"{NVD}/cufft/lib",
        f"{NVD}/cufile/lib", f"{NVD}/nccl/lib", f"{NVD}/nvshmem/lib",
        "/data/home/public/jiangzhen/dp/lmp_kokkos_install/lib", INSTALL,
    ]),
    "DP_BACKEND_PLUGIN_PATH": INSTALL,
    "LAMMPS_PLUGIN_PATH": KK_PLUGIN,
    "OMP_NUM_THREADS": "1",
}

MODELS = sys.argv[2:] or [
    "dpa4c-sog-bs6.pt2", "dpa4c-sog-mid6ad.pt2", "dpa4c-sog-mid6ik.pt2",
    "dpa4c-sog-mid6ad-s102.pt2", "dpa4c-sog-qs6ad-s102.pt2",
]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12288
MODES = [("rep", {}), ("slab", {"SOG_MIDTOWN_SLAB": "1", "SOG_SLAB_IK": "1"})]


def deck(model, np_ranks, tag):
    lines = ["units metal", "boundary p p p", "atom_style charge/kk",
             "atom_modify map yes"]
    if np_ranks > 1:
        lines.append(f"processors      1 1 {np_ranks}")
    lines += [
        "neighbor        2.0 bin", "neigh_modify every 20 delay 0 check no",
        f"read_data       {BENCH}/data_{N}.lmp",
        "mass            1 16", "mass            2 2",
        f"pair_style      deepmd/kk {FR}/{model}",
        "pair_coeff      * * O H", "kspace_style none",
        "fix             1 all nve/kk", "run_style verlet/kk",
        "velocity        all create 300.0 20261112 mom yes rot no dist gaussian",
        "timestep        0.0005", "thermo_style custom step pe pxx pyy pzz",
        "thermo 1", "run 1", "run 0",
        f"write_dump all custom {OUT}/{tag}.dump id type q x y z fx fy fz "
        "modify sort id",
    ]
    p = f"{OUT}/in.{tag}.lammps"
    open(p, "w").write("\n".join(lines) + "\n")
    return p


def read_dump(path):
    lines = open(path).readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS"))
    cols = lines[start].split()[2:]
    ix = {c: i for i, c in enumerate(cols)}
    rows = {}
    for line in lines[start + 1:]:
        f = line.split()
        if len(f) < len(cols):
            continue
        rows[int(f[ix["id"]])] = (float(f[ix["fx"]]), float(f[ix["fy"]]),
                                  float(f[ix["fz"]]))
    return rows


def run(model, np_ranks, tag, extra):
    inp = deck(model, np_ranks, tag)
    env = dict(os.environ)
    env.update(ENV)
    env.update(extra)
    log = f"{OUT}/log.{tag}"
    if np_ranks == 1:
        env["CUDA_VISIBLE_DEVICES"] = "0"
        cmd = [LMP, "-k", "on", "g", "1", "-in", inp, "-log", log,
               "-screen", "none"]
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(np_ranks))
        cmd = [MPIRUN, "--oversubscribe", "-np", str(np_ranks), RANK_DEV, LMP,
               "-k", "on", "g", "1", "-in", inp, "-log", log, "-screen", "none"]
    p = subprocess.run(cmd, cwd=OUT, env=env, capture_output=True, text=True,
                       timeout=3600)
    txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
    pe = None
    for line in txt.splitlines():
        f = line.split()
        if len(f) >= 4 and f[0].isdigit():
            try:
                float(f[1]); pe = float(f[1])
            except ValueError:
                continue
    err = txt[txt.find("ERROR"):][:120].replace("\n", " ") if "ERROR" in txt else ""
    return pe, err


os.makedirs(OUT, exist_ok=True)
print(f"N={N}.  dF is np>1 against np=1 of the SAME mesh mode.\n")
for model in MODELS:
    for mode, extra in MODES:
        ref = None
        for np_ranks in (1, 2, 4):
            tag = f"{model[:-4]}.{mode}.np{np_ranks}"
            pe, err = run(model, np_ranks, tag, extra)
            if pe is None:
                print(f"{model:26s} {mode:4s} np={np_ranks}  FAIL {err}")
                break
            d = read_dump(f"{OUT}/{tag}.dump")
            if np_ranks == 1:
                ref = d
                print(f"{model:26s} {mode:4s} np=1  pe/atom={pe / N:.9f}")
                continue
            ids = sorted(ref)
            df = [math.dist(ref[i], d[i]) for i in ids]
            print(f"{model:26s} {mode:4s} np={np_ranks}  pe/atom={pe / N:.9f}  "
                  f"max|dF|={max(df):.2e}")
