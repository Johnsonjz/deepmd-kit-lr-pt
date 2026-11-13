#!/usr/bin/env python3
"""Tier-2 equivalence: dump one snapshot's per-atom q/F and the total energy.

A reference is a *directory* of dumps, not a live comparison: two builds of
libdeepmd_op_pt.so cannot coexist in one process, so a refactor is verified by
dumping before, dumping after and comparing the two directories with
equiv_compare.py.

usage:
  equiv_dump.py <outdir> [N] [model.pt2 ...]        # replicated path
  equiv_dump.py <outdir> [N] [model.pt2 ...] --slab # distributed (z-slab) path

The snapshot is one thermal step (velocity seed fixed) followed by `run 0`, so
it exercises a real force field rather than the perfect lattice.  All nodes are
charged in liquid water, which is the regime the deployed models run in.
"""
import os
import subprocess
import sys

BENCH = "/data/home/public/jiangzhen/dp/sog_work/bench"
FR = "/data/home/public/jiangzhen/dp/sog_work/mdsuite/frozen/dpa4_campaign"
LMP = "/data/home/public/jiangzhen/dp/lmp_kokkos_install/bin/lmp"
OMPI = "/data/app/openmpi-4.1.4/install/openmpi/4.1.4"
CONDA = "/data/home/jiangzhen/.conda/envs/lr212"
TORCH = f"{CONDA}/lib/python3.11/site-packages/torch"
NVD = f"{CONDA}/lib/python3.11/site-packages/nvidia"
INSTALL = "/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib"
KK_PLUGIN = "/tmp/kk_lrpt_plugin"

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

DEFAULT_MODELS = [
    "dpa4c.pt2", "dpa4-sog.pt2",                 # the SOG MP model, mid 4 ik
    "dpa4c-sog-bs6.pt2", "dpa4c-sog.pt2",        # B-spline 6, CubeS2 4 ik
    "dpa4c-sog-mid4ad.pt2", "dpa4c-sog-mid6ad.pt2", "dpa4c-sog-mid6ik.pt2",
    "dpa4c-sog-quads4ik.pt2", "dpa4c-sog-quads6ik.pt2",
    "dpa4c-sog-mid6ad-s102.pt2", "dpa4c-sog-mid6ik-s102.pt2",
    "dpa4c-sog-qs6ad-s102.pt2", "dpa4c-sog-qs6ik-s102.pt2",
]


def main():
    args = [a for a in sys.argv[1:] if a != "--slab"]
    slab = "--slab" in sys.argv
    out = args[0]
    N = int(args[1]) if len(args) > 1 else 1536
    models = args[2:] or DEFAULT_MODELS
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    env.update(ENV)
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU_BASE", "0")
    if slab:
        # One rank is one z-band, so the brick kernels run without a second GPU.
        env["SOG_MIDTOWN_SLAB"] = "1"
        env["SOG_SLAB_IK"] = "1"
    for m in models:
        tag = m.replace(".pt2", "")
        deck = f"""units metal
boundary p p p
atom_style charge/kk
atom_modify map yes
neighbor 2.0 bin
neigh_modify every 20 delay 0 check no
read_data {BENCH}/data_{N}.lmp
mass 1 16
mass 2 2
pair_style deepmd/kk {FR}/{m}
pair_coeff * * O H
kspace_style none
fix 1 all nve/kk
run_style verlet/kk
velocity all create 300.0 20261112 mom yes rot no dist gaussian
timestep 0.0005
thermo_style custom step pe pxx pyy pzz
thermo 1
run 1
run 0
write_dump all custom {out}/{tag}.dump id type q x y z fx fy fz modify sort id
"""
        inp = f"{out}/in.{tag}.lammps"
        open(inp, "w").write(deck)
        log = f"{out}/log.{tag}"
        p = subprocess.run([LMP, "-k", "on", "g", "1", "-in", inp, "-log", log,
                            "-screen", "none"], cwd=out, env=env,
                           capture_output=True, text=True, timeout=1800)
        txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
        pe = None
        for line in txt.splitlines():
            f = line.split()
            if len(f) >= 4 and f[0].isdigit():
                try:
                    float(f[1]); pe = float(f[1])
                except ValueError:
                    continue
        note = ""
        if "ERROR" in txt:
            note = txt[txt.find("ERROR"):][:120].replace("\n", " ")
        ok = os.path.exists(f"{out}/{tag}.dump")
        print(f"{tag:28s} rc={p.returncode} pe={pe} dump={'yes' if ok else 'NO'} {note}")


if __name__ == "__main__":
    main()
