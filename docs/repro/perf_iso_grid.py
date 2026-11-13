#!/usr/bin/env python3
"""Tier-4a: same-grid comparison of the mesh engines, with optional phases.

All backends are frozen at the SAME mesh spacing (0.98494 A, the deployed
B-spline-6 spacing), so the grid no longer confuses the comparison: what is
measured is the per-atom and per-mode implementation cost.  The "LO" row is the
short-range model, and LR is the difference against it.

usage:
  perf_iso_grid.py                 # step time, memory, peak
  TIMING=1 perf_iso_grid.py        # also print each engine's phase split
                                   # (needs >= 100 long-range calls: MEAS=110)

Sampling matters: a 0.1 s nvidia-smi poll misses (or catches by luck) the
one-time allocation spike of the first call, so this probe samples at 20 ms and
reports both the steady state and the peak.
"""
import os
import re
import statistics
import subprocess
import sys
import threading
import time

BENCH = "/data/home/public/jiangzhen/dp/sog_work/bench"
FR = "/data/home/public/jiangzhen/dp/sog_work/mdsuite/frozen/dpa4_campaign"
LMP = "/data/home/public/jiangzhen/dp/lmp_kokkos_install/bin/lmp"
OMPI = "/data/app/openmpi-4.1.4/install/openmpi/4.1.4"
CONDA = "/data/home/jiangzhen/.conda/envs/lr212"
TORCH = f"{CONDA}/lib/python3.11/site-packages/torch"
NVD = f"{CONDA}/lib/python3.11/site-packages/nvidia"
INSTALL = "/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib"
KK_PLUGIN = "/tmp/kk_lrpt_plugin"
OUT = os.environ.get("OUT", "/tmp/iso_grid")

N = int(os.environ.get("N", "98304"))
WARM = int(os.environ.get("WARM", "2"))
MEAS = int(os.environ.get("MEAS", "20"))
GPU = os.environ.get("GPU_BASE", "0")
TIMING = os.environ.get("TIMING", "0") != "0"

CONFIGS = [
    ("sr", "dpa4c.pt2"),
    ("bs6", "dpa4c-sog-bs6.pt2"),
    ("mid6ad", "dpa4c-sog-mid6ad-s102.pt2"),
    ("qs6ad", "dpa4c-sog-qs6ad-s102.pt2"),
    ("mid6ik", "dpa4c-sog-mid6ik-s102.pt2"),
    ("qs6ik", "dpa4c-sog-qs6ik-s102.pt2"),
]

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
    "SOG_MIDTOWN_DEBUG": "1",
}


def deck(model, tag):
    lines = [
        "units metal", "boundary p p p", "atom_style charge/kk",
        "atom_modify map yes", "neighbor 2.0 bin",
        "neigh_modify every 20 delay 0 check no",
        f"read_data {BENCH}/data_{N}.lmp", "mass 1 16", "mass 2 2",
        f"pair_style deepmd/kk {FR}/{model}", "pair_coeff * * O H",
        "kspace_style none", "fix 1 all nve/kk", "run_style verlet/kk",
        "velocity all create 300.0 12345 mom yes rot no dist gaussian",
        "thermo_style custom step temp pe press vol", "thermo 10000",
        "timestep 0.0005", f"run {WARM}", f"run {MEAS}",
    ]
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/in.{tag}.lammps"
    open(p, "w").write("\n".join(lines) + "\n")
    return p


def gpu_used():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits", "-i", GPU],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def run(label, model):
    inp = deck(model, label)
    env = dict(os.environ)
    env.update(ENV)
    env["CUDA_VISIBLE_DEVICES"] = GPU
    if TIMING:
        # both engines report under the same switch and the same phase names
        env["SOG_GPU_TIMING"] = "1"
    log = f"{OUT}/log.{label}"
    trace, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            m = gpu_used()
            if m is not None:
                trace.append((time.time(), m))
            time.sleep(0.02)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    p = subprocess.run([LMP, "-k", "on", "g", "1", "-in", inp, "-log", log,
                        "-screen", "none"], cwd=OUT, env=env,
                       capture_output=True, text=True, timeout=3600)
    stop.set()
    th.join(timeout=1)
    txt = (p.stdout or "") + "\n" + (p.stderr or "")
    if os.path.exists(log):
        txt += open(log, errors="ignore").read()
    loops = re.findall(r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", txt)
    sec = None
    for val, steps in loops:
        if int(steps) == MEAS:
            sec = float(val)
    grid = ""
    m = re.search(r"rebuild grid (\d+)x(\d+)x(\d+)", txt)
    if m:
        grid = "x".join(m.groups())
    mid = re.findall(r"SOG_GPU_TIMING \(n=\d+\): (.*?) grid", txt)
    bs = re.findall(r"SOG_BSPLINE_TIMING \(n=\d+\): (.*)", txt)
    base = trace[0][1] if trace else 0
    steady = statistics.median([m for t, m in trace if t > trace[-1][0] - 0.8])
    peak = max(m for _, m in trace)
    return sec / MEAS * 1e3 if sec else None, steady - base, peak - base, grid, \
        (mid[-1].strip() if mid else (bs[-1].strip() if bs else ""))


rows = []
for label, model in CONFIGS:
    if not os.path.exists(f"{FR}/{model}"):
        print("SKIP", label)
        continue
    ms, steady, peak, grid, ph = run(label, model)
    rows.append((label, grid, ms, steady, peak, ph))

sr = rows[0][2]
print(f"\n{'config':8s} {'grid':12s} {'ms/step':>9s} {'LR':>7s} "
      f"{'steady':>7s} {'peak':>7s}  phases")
for label, grid, ms, steady, peak, ph in rows:
    lr = f"{(ms - sr):6.2f}" if ms and sr else "   -  "
    print(f"{label:8s} {grid or '-':12s} {ms:9.3f} {lr} {steady:7.0f} "
          f"{peak:7.0f}  {ph}")
