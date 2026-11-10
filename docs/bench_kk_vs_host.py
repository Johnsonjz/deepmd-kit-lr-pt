# Standalone benchmark tool, not package code.
# ruff: noqa
#!/usr/bin/env python3
"""Host `pair deepmd` vs device `pair deepmd/kk`, replicated vs z-slab.

Runs the same frozen lr-pt artifacts through both pair styles on the Kokkos
LAMMPS build and reports loop time, peak GPU memory and `pe`, so the pair
style's per-step data movement is separated from the mesh backend.

usage: python bench_kk_vs_host.py [n ...]
"""

import csv
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
HOST_PLUGIN = "/tmp/lmpplugin_lrpt"
OUT = os.environ.get(
    "OUT", "/data/home/public/jiangzhen/dp/sog_work/bench/matrix_kk_vs_host"
)
STEPS = int(os.environ.get("STEPS", "6"))

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
}


class GpuSampler(threading.Thread):
    def __init__(self, devices):
        super().__init__(daemon=True)
        self.devices, self.peak, self._stop_evt = set(devices), {}, threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            for dev, mib in self._used().items():
                self.peak[dev] = max(self.peak.get(dev, 0), mib)
            time.sleep(0.1)

    def _used(self):
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
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


def deck(n, model, np_ranks, kk):
    lines = [
        "units           metal",
        "boundary        p p p",
        "atom_style      " + ("charge/kk" if kk else "charge"),
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
        f"pair_style      {'deepmd/kk' if kk else 'deepmd'} {model}",
        "pair_coeff      * * O H",
        "kspace_style    none",
        f"fix             1 all {'nve/kk' if kk else 'nve'}",
        "velocity        all create 300.0 12345 mom yes rot no dist gaussian",
        "thermo_style    custom step temp pe press vol",
        "thermo          1000",
        "timestep        0.0005",
    ]
    if kk:
        lines.append("run_style       verlet/kk")
    lines.append(f"run             {STEPS}")
    os.makedirs(OUT, exist_ok=True)
    tag = "kk" if kk else "host"
    path = f"{OUT}/in.{tag}.{n}.np{np_ranks}.lammps"
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def run(n, model_name, np_ranks, kk, slab, devices):
    model = f"{FR}/{model_name}.pt2"
    inp = deck(n, model, np_ranks, kk)
    tag = f"{model_name}.{'kk' if kk else 'host'}.{'slab' if slab else 'rep'}.{n}.np{np_ranks}"
    log = f"{OUT}/log.{tag}"
    for p in (log,):
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ)
    env.update(ENV)
    env["SOG_MIDTOWN_SLAB"] = "1" if slab else "0"
    sampler = GpuSampler(devices)
    base = GpuSampler(devices)._used()
    sampler.start()
    if np_ranks == 1:
        env["CUDA_VISIBLE_DEVICES"] = str(devices[0])
        cmd = (
            [LMP]
            + (["-k", "on", "g", "1"] if kk else [])
            + ["-in", inp, "-log", log, "-screen", "none"]
        )
    else:
        # One GPU per rank: the wrapper picks this rank's entry out of the list.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        cmd = (
            [
                MPIRUN,
                "--oversubscribe",
                "-np",
                str(np_ranks),
                "/tmp/lmp_rank_dev.sh",
                LMP,
            ]
            + (["-k", "on", "g", "1"] if kk else [])
            + ["-in", inp, "-log", log, "-screen", "none"]
        )
    if kk:
        env["LAMMPS_PLUGIN_PATH"] = KK_PLUGIN
    else:
        env["LAMMPS_PLUGIN_PATH"] = HOST_PLUGIN
    proc = subprocess.run(
        cmd, cwd=OUT, env=env, capture_output=True, text=True, timeout=3600
    )
    sampler.stop()
    sampler.join(timeout=2)
    txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
    err = proc.stderr or ""
    m = re.search(r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", txt)
    pe = None
    for line in txt.splitlines():
        f = line.split()
        if len(f) >= 5 and f[0].isdigit():
            try:
                int(f[0])
                float(f[2])
            except ValueError:
                continue
            pe = float(f[2])
    warn = "SOG_MIDTOWN_SLAB" in txt and "Warning" in txt
    mem = max((sampler.peak.get(d, 0) - base.get(d, 0)) for d in devices)
    return dict(
        loop=float(m.group(1)) if m else None,
        pe=pe,
        mem_mib=mem,
        warn=warn,
        rc=proc.returncode,
        err=(txt[txt.find("ERROR") :][:160] if "ERROR" in txt else err[-160:]),
    )


def main():
    sizes = [int(x) for x in sys.argv[1:]] or [12288]
    rows = []
    cases = [("lr_bs4", True), ("lr_mid4ik", True)]
    for n in sizes:
        for model, is_ik in cases:
            for kk in (True, False):
                for slab, np_ranks, devs in ((False, 1, [2]), (True, 2, [2, 3])):
                    r = run(n, model, np_ranks, kk, slab, devs)
                    loop = f"{r['loop'] * 1000:9.1f}" if r["loop"] else "     fail"
                    print(
                        f"n={n:7d} {model:10s} {'kk' if kk else 'host':4s} "
                        f"{'slab' if slab else 'rep ':4s} np={np_ranks} "
                        f"ms={loop} mem={r['mem_mib']:6d} MiB pe={r['pe']} "
                        f"{'WARN ' if r['warn'] else ''}{r['err'][:70] if not r['loop'] else ''}",
                        flush=True,
                    )
                    rows.append(
                        [
                            n,
                            model,
                            "kk" if kk else "host",
                            "slab" if slab else "rep",
                            np_ranks,
                            r["loop"],
                            r["mem_mib"],
                            r["pe"],
                            r["warn"],
                            r["err"],
                        ]
                    )
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/kk_vs_host.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "n",
                "model",
                "pair",
                "mesh",
                "np",
                "loop_s",
                "mem_mib",
                "pe",
                "fallback_warn",
                "err",
            ]
        )
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
