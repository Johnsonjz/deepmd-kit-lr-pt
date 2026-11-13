#!/usr/bin/env python3
"""Tier-4b: the replicated and the z-slab solve on np = 1, 2 and 4.

Reports the step time and the peak device memory per GPU, which is what decides
which mode to deploy: the slab saves the replicated mesh but pays two transposes
and the collectives, so it wins only once the mesh no longer fits.

usage: perf_dist.py [N] [model.pt2 ...]
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time

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
OUT = "/tmp/dist_bench"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 98304
WARM = int(os.environ.get("WARM", "3"))
MEAS = int(os.environ.get("MEAS", "10"))

CONFIGS = sys.argv[2:] or [
    "dpa4c-sog.pt2",              # the paper's SOG configuration: CubeS2-4 ik
    "dpa4c-sog-mid6ik.pt2",
    "dpa4c-sog-mid6ik-s102.pt2",
    "dpa4c-sog-bs6.pt2",
]
MODES = [("rep", {}), ("slab", {"SOG_MIDTOWN_SLAB": "1", "SOG_SLAB_IK": "1"})]

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


class Peak(threading.Thread):
    def __init__(self, devices):
        super().__init__(daemon=True)
        self.devices, self.peak = set(devices), {}
        self.stop_evt = threading.Event()

    def run(self):
        while not self.stop_evt.is_set():
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
            for line in r.stdout.splitlines():
                i, m = [x.strip() for x in line.split(",")]
                if int(i) in self.devices:
                    self.peak[int(i)] = max(self.peak.get(int(i), 0), int(m))
            time.sleep(0.05)


def deck(model, np_ranks, tag):
    lines = ["units metal", "boundary p p p", "atom_style charge/kk",
             "atom_modify map yes"]
    if np_ranks > 1:
        lines.append(f"processors      1 1 {np_ranks}")
    lines += [
        "neighbor        2.0 bin", "neigh_modify every 20 delay 0 check no",
        f"read_data       {BENCH}/data_{N}.lmp", "mass 1 16", "mass 2 2",
        f"pair_style      deepmd/kk {FR}/{model}", "pair_coeff * * O H",
        "kspace_style none", "fix 1 all nve/kk", "run_style verlet/kk",
        "velocity        all create 300.0 12345 mom yes rot no dist gaussian",
        "thermo_style    custom step temp pe press vol", "thermo 10000",
        "timestep        0.0005", f"run {WARM}", f"run {MEAS}",
    ]
    p = f"{OUT}/in.{tag}.lammps"
    open(p, "w").write("\n".join(lines) + "\n")
    return p


def run(model, np_ranks, tag, extra):
    inp = deck(model, np_ranks, tag)
    env = dict(os.environ)
    env.update(ENV)
    env.update(extra)
    devs = list(range(np_ranks))
    log = f"{OUT}/log.{tag}"
    if np_ranks == 1:
        env["CUDA_VISIBLE_DEVICES"] = "0"
        cmd = [LMP, "-k", "on", "g", "1", "-in", inp, "-log", log,
               "-screen", "none"]
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devs))
        cmd = [MPIRUN, "--oversubscribe", "-np", str(np_ranks),
               "/tmp/lmp_rank_dev.sh", LMP, "-k", "on", "g", "1", "-in", inp,
               "-log", log, "-screen", "none"]
    pk = Peak(devs)
    pk.start()
    p = subprocess.run(cmd, cwd=OUT, env=env, capture_output=True, text=True,
                       timeout=3600)
    pk.stop_evt.set()
    pk.join(timeout=1)
    txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
    sec = None
    for val, steps in re.findall(
            r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", txt):
        if int(steps) == MEAS:
            sec = float(val)
    err = txt[txt.find("ERROR"):][:100].replace("\n", " ") if "ERROR" in txt else ""
    return (sec / MEAS * 1e3 if sec else None), (max(pk.peak.values()) if pk.peak else 0), err


os.makedirs(OUT, exist_ok=True)
print(f"N={N}, WARM={WARM} MEAS={MEAS}\n")
for model in CONFIGS:
    for mode, extra in MODES:
        for np_ranks in (1, 2, 4):
            tag = f"{model[:-4]}.{mode}.np{np_ranks}.N{N}"
            ms, mem, err = run(model, np_ranks, tag, extra)
            if ms is None:
                print(f"{model[:-4]:22s} {mode:4s} np={np_ranks}  FAIL {err}")
                continue
            print(f"{model[:-4]:22s} {mode:4s} np={np_ranks}  {ms:8.2f} ms/step  "
                  f"peak/GPU {mem:6d} MiB")
