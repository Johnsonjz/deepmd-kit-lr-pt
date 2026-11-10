# Standalone benchmark tool, not package code.
# ruff: noqa
#!/usr/bin/env python3
"""Spline-mode matrix on the device chain: replicated vs z-slab, speed and memory.

Ten frozen equal-accuracy variants of the same checkpoint -- B-spline 4/6,
CubeS2 4/6 (ad/ik) and QuadS 4/6 (ad/ik) -- run through ``pair deepmd/kk`` at
the extreme system sizes, each on the replicated mesh and on the distributed
(z-slab) mesh, reporting per-step time and peak device memory.

usage: python bench_spline_matrix.py [n ...]

Environment (this is the "is my build right" check): the tool prints the
toolchain it is about to use, then refuses to run if any artifact in the matrix
is missing.
"""

import csv
import json
import math
import os
import re
import subprocess
import sys
import threading
import time

BENCH = "/data/home/public/jiangzhen/dp/sog_work/bench"
FR = "/data/home/public/jiangzhen/dp/sog_work/mdsuite/frozen/midtown_vs_bspline"
LMP = "/data/home/public/jiangzhen/dp/lmp_kokkos_install/bin/lmp"
OMPI = "/data/app/openmpi-4.1.4/install/openmpi/4.1.4"
MPIRUN = f"{OMPI}/bin/mpirun"
CONDA = "/data/home/jiangzhen/.conda/envs/lr212"
TORCH = f"{CONDA}/lib/python3.11/site-packages/torch"
NVD = f"{CONDA}/lib/python3.11/site-packages/nvidia"
INSTALL = "/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib"
KK_PLUGIN = "/tmp/kk_lrpt_plugin"  # dpplugin.so -> the deepmd/kk build
OUT = os.environ.get(
    "OUT", "/data/home/public/jiangzhen/dp/sog_work/bench/matrix_spline_modes"
)
STEPS = int(os.environ.get("STEPS", "6"))
DEVS = [int(x) for x in os.environ.get("DEVS", "1,2").split(",")]

ENV = {
    "PATH": "/usr/local/cuda-12.4/bin:" + OMPI + "/bin:" + os.environ.get("PATH", ""),
    "LD_LIBRARY_PATH": ":".join(
        [
            OMPI + "/lib",
            f"{CONDA}/lib",
            f"{TORCH}/lib",
            f"{NVD}/cuda_runtime/lib",
            f"{NVD}/cuda_cupti/lib",
            f"{NVD}/nvjitlink/lib",
            f"{NVD}/cudnn/lib",
            f"{NVD}/cublas/lib",
            f"{NVD}/cusparse/lib",
            f"{NVD}/cusparselt/lib",
            f"{NVD}/curand/lib",
            f"{NVD}/cufft/lib",
            f"{NVD}/cufile/lib",
            f"{NVD}/nccl/lib",
            f"{NVD}/nvshmem/lib",
            "/data/home/public/jiangzhen/dp/lmp_kokkos_install/lib",
            INSTALL,
        ]
    ),
    "DP_BACKEND_PLUGIN_PATH": INSTALL,
    "OMP_NUM_THREADS": "1",
    "LAMMPS_PLUGIN_PATH": KK_PLUGIN,
}

# (label, frozen artifact stem)
CASES = [
    ("bspline-4 ad", "lr_bs4"),
    ("bspline-6 ad", "lr_bs6"),
    ("CubeS2-4 ad", "lr_mid4"),
    ("CubeS2-4 ik", "lr_mid4ik"),
    ("CubeS2-6 ad", "lr_mid6"),
    ("CubeS2-6 ik", "lr_mid6ik"),
    ("QuadS-4 ad", "lr_quads4"),
    ("QuadS-4 ik", "lr_quads4ik"),
    ("QuadS-6 ad", "lr_quads6"),
    ("QuadS-6 ik", "lr_quads6ik"),
]


class GpuSampler(threading.Thread):
    def __init__(self, devices):
        super().__init__(daemon=True)
        self.devices = list(devices)
        self.peak = {}
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            for dev, mib in self._used().items():
                self.peak[dev] = max(self.peak.get(dev, 0), mib)
            time.sleep(0.1)

    def _used(self):
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        out = {}
        for line in r.stdout.splitlines():
            i, m = [x.strip() for x in line.split(",")]
            if int(i) in self.devices:
                out[int(i)] = int(m)
        return out

    def stop(self):
        self._stop_evt.set()


def box_of(n):
    """Box length of the prepared data file, from its header."""
    with open(f"{BENCH}/data_{n}.lmp") as fh:
        for _ in range(20):
            line = fh.readline()
            f = line.split()
            if len(f) >= 4 and f[2] == "xlo":
                return float(f[1]) - float(f[0])
    raise RuntimeError(f"no xlo/xhi in data_{n}.lmp")


def _factorable(n):
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def spacing_of(stem):
    """Mesh spacing serialized in the artifact (metadata topology_params)."""
    meta = subprocess.run(
        ["unzip", "-p", f"{FR}/{stem}.pt2", "model/extra/metadata.json"],
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(meta)["topology_params"]["pme_mesh"]["spacing"]


def deck(n, model, np_ranks, slab):
    lines = [
        "units           metal",
        "boundary        p p p",
        "atom_style      charge/kk",
        "atom_modify     map yes",
    ]
    if np_ranks > 1:
        lines.append(f"processors      1 1 {np_ranks}")
    lines += [
        "neighbor        2.0 bin",
        "neigh_modify    every 20 delay 0 check no",
        f"read_data       {BENCH}/data_{n}.lmp",
        "mass            1 16",
        "mass            2 2",
        f"pair_style      deepmd/kk {model}",
        "pair_coeff      * * O H",
        "kspace_style    none",
        "fix             1 all nve/kk",
        "run_style       verlet/kk",
        "velocity        all create 300.0 12345 mom yes rot no dist gaussian",
        "thermo_style    custom step temp pe press vol",
        "thermo          1000",
        "timestep        0.0005",
        f"run             {STEPS}",
    ]
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/in.{'slab' if slab else 'rep'}.{n}.np{np_ranks}.lammps"
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def run(n, label, stem, np_ranks, slab):
    model = f"{FR}/{stem}.pt2"
    inp = deck(n, model, np_ranks, slab)
    tag = f"{stem}.{'slab' if slab else 'rep'}.{n}.np{np_ranks}"
    log = f"{OUT}/log.{tag}"
    if os.path.exists(log):
        os.remove(log)
    env = dict(os.environ)
    env.update(ENV)
    env["SOG_MIDTOWN_SLAB"] = "1" if slab else "0"
    devs = DEVS[:np_ranks]
    base = GpuSampler(devs)._used()
    samp = GpuSampler(devs)
    samp.start()
    if np_ranks == 1:
        env["CUDA_VISIBLE_DEVICES"] = str(devs[0])
        cmd = [LMP, "-k", "on", "g", "1", "-in", inp, "-log", log, "-screen", "none"]
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devs))
        cmd = [
            MPIRUN, "--oversubscribe", "-np", str(np_ranks),
            "/tmp/lmp_rank_dev.sh", LMP, "-k", "on", "g", "1",
            "-in", inp, "-log", log, "-screen", "none",
        ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=OUT, env=env, capture_output=True, text=True,
                              timeout=3600)
        rc, err = proc.returncode, proc.stderr or ""
    except subprocess.TimeoutExpired:
        rc, err = -9, "timeout"
    wall = time.time() - t0
    samp.stop()
    samp.join(timeout=2)
    txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
    m = re.search(r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", txt)
    pe = None
    for line in txt.splitlines():
        f = line.split()
        if len(f) >= 5 and f[0].isdigit():
            try:
                int(f[0])
                pe = float(f[2])
            except ValueError:
                continue
    mem = {d: samp.peak.get(d, 0) - base.get(d, 0) for d in devs}
    # The operator's own message (the AOTI shim prints it before the stack).
    msg = ""
    for ln in (txt + err).splitlines():
        if "[E" in ln or "ERROR" in ln:
            msg = ln.strip()[:220]
            break
    return dict(
        label=label, stem=stem, np=np_ranks, slab=slab,
        loop=float(m.group(1)) if m else None,
        pe=pe, mem_max=max(mem.values()) if mem else 0,
        mem_sum=sum(mem.values()), wall=wall, rc=rc,
        err="" if m else (msg or err.strip().splitlines()[-1][:220] if err.strip() else ""),
    )


def main():
    sizes = [int(x) for x in sys.argv[1:]] or [526848, 1119744]
    only = [x for x in os.environ.get("ONLY", "").split(",") if x]
    cases = [c for c in CASES if not only or c[1] in only]
    missing = [s for _, s in cases if not os.path.exists(f"{FR}/{s}.pt2")]
    if missing:
        sys.exit(f"missing artifacts: {missing}")
    print("toolchain:", flush=True)
    for probe in (
        ["/usr/local/cuda-12.4/bin/nvcc", "--version"],
        ["g++", "--version"],
        [f"{CONDA}/bin/python", "-c", "import torch;print('torch',torch.__version__)"],
    ):
        try:
            txt = subprocess.run(probe, capture_output=True, text=True).stdout
            print("  " + (txt.strip().splitlines()[-1] if probe[0].endswith("nvcc") else txt.strip().splitlines()[0]))
        except Exception as e:  # noqa: BLE001
            print("  probe failed:", e)
    rows = []
    for n in sizes:
        L = box_of(n)
        print(f"\n=== n={n}  L={L:.2f} A ===", flush=True)
        for label, stem in cases:
            sp = spacing_of(stem)
            grid = _factorable(math.ceil(L / sp))
            for slab, np_ranks in ((False, 1), (False, 2), (True, 2)):
                r = run(n, label, stem, np_ranks, slab)
                ms = f"{r['loop'] * 1000:8.1f}" if r["loop"] else "   FAIL"
                print(
                    f"{label:13s} sp={sp:.4f} {grid:4d}^3 "
                    f"{'slab' if slab else 'rep ':4s} np={np_ranks} "
                    f"ms={ms} mem={r['mem_max']:6d}/{r['mem_sum']:6d} MiB "
                    f"pe={r['pe']} {r['err'][:60] if not r['loop'] else ''}",
                    flush=True,
                )
                rows.append([n, f"{L:.2f}", label, stem, f"{sp:.6f}", grid,
                             "slab" if slab else "replicated", np_ranks,
                             "" if r["loop"] is None else f"{r['loop'] * 1000:.3f}",
                             r["mem_max"], r["mem_sum"], r["pe"], r["rc"], r["err"]])
    path = f"{OUT}/spline_matrix.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "L_A", "mode", "artifact", "spacing_A", "grid",
                    "mesh", "np", "ms_per_step", "mem_max_mib", "mem_sum_mib",
                    "pe", "rc", "error"])
        w.writerows(rows)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
