# Standalone verification tool, not package code.
# ruff: noqa
"""End-to-end check that `force_mode` reaches the frozen canonical lower.

A periodic model is built twice -- the uncompressed reference evaluated on the
mesh, and the same model compressed into the `dpa4c_canonical` lower.  Both read
`long_range.force_mode`.  If the canonical path dropped the flag and silently
used AD, its force would not match an ik reference, and vice versa.

Set up exactly as `source/tests/pt_expt/model/test_long_range.py` builds it
(small DPA4C model, one charge channel, the SOG kernel the midtown mesh needs).

Environment (see docs/FREEZE_SOP.md): `module load gcc/13` and put the conda
env's libstdc++ first on LD_LIBRARY_PATH, or the torch Inductor wrapper fails
at dlopen with GLIBCXX_3.4.30.
"""
import copy
import sys

import numpy as np

sys.path.insert(0, "/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt")
import torch


def build(mesh_backend: str, force_mode: str):
    from deepmd.pt_expt.model.get_model import get_model

    config = {
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "dpa4c",
            "rcut": 3.0,
            "channels": 8,
            "lmax": 2,
            "n_radial": 8,
            "precision": "float32",
            "seed": 17,
        },
        "fitting_net": {
            "type": "ener",
            "neuron": [32, 32],
            "activation_function": "silu",
            "resnet_dt": False,
            "precision": "float32",
            "seed": 19,
        },
        "long_range": {
            "type": "latent_charge",
            "channels": 1,
            "charge_net": {"neuron": [16, 16], "precision": "float32", "seed": 23},
            "kernel": {
                "type": "sog",
                "amplitude": [0.7, -0.2, 0.1],
                "width": [0.8, 1.7, 3.2],
            },
            "charge_constraint": {"source": "neutral"},
            "periodic": True,
            # A deliberately coarse mesh: AD and ik agree to ~1e-5 on a fine
            # one, and the flag then cannot be told apart from compression
            # noise.  At this spacing they differ by orders more.
            "pme_spacing": 0.6,
            "mesh_backend": mesh_backend,
            "force_mode": force_mode,
        },
    }
    # One precision throughout: a mixed float32/float64 graph makes the
    # Inductor `pad_mm` autotuner compare a float and a double operand and
    # raise, which has nothing to do with the flag under test.
    return config, get_model(config).to("cuda").eval()


def check(mesh_backend: str, force_mode: str, tmp_path) -> None:
    from deepmd.pt_expt.utils.serialization import deserialize_to_file
    from deepmd.pt_expt.utils.env import DEVICE
    from deepmd.pt_expt.infer.deep_eval import DeepPot

    config, model = build(mesh_backend, force_mode)
    reference = copy.deepcopy(model)
    reference.atomic_model.long_range.use_mesh_evaluation()
    model.atomic_model.enable_compression(0.5)
    frozen = f"{tmp_path}/e2e_{mesh_backend}_{force_mode}.pt2"
    deserialize_to_file(
        frozen,
        {"model": model.serialize(), "model_def_script": config},
        do_atomic_virial=True,
        lower_kind="auto",
    )
    dp = DeepPot(frozen, auto_batch_size=False)
    lower = dp.deep_eval.metadata["lower_input_kind"]

    rng = np.random.default_rng(3)
    box = np.array([[[12.0, 0.0, 0.0], [0.8, 11.0, 0.0], [-0.5, 0.6, 13.0]]])
    coord = rng.uniform(0.0, 1.0, (1, 40, 3)) @ box[0]
    atype = rng.integers(0, 2, (1, 40))

    def tnp(a, dtype=None):
        t = torch.tensor(a, device=DEVICE)
        return t.to(dtype) if dtype is not None else t

    # The model differentiates its own energy for the force, so this call must
    # NOT run under no_grad.
    expected = reference(
        tnp(coord), tnp(atype, torch.int64), box=tnp(box), do_atomic_virial=True
    )
    actual = dp.eval(coord, box, atype, atomic=True)
    names = ("energy", "force", "virial", "atom_energy", "atom_virial")
    worst = 0.0
    for name, value in zip(names, actual, strict=True):
        target = expected[name].detach().cpu().numpy().reshape(value.shape)
        scale = max(np.abs(target).max(), 1e-12)
        rel = float(np.abs(value - target).max() / scale)
        worst = max(worst, rel)
    print(f"  {mesh_backend:8s} {force_mode}: lower={lower:18s} "
          f"worst rel |frozen - reference| = {worst:.2e}")
    return worst, actual[1]


def main():
    import tempfile

    # ad vs ik must give different forces on the same system, and each frozen
    # lower must reproduce its own reference.  The first claim is what proves
    # the flag is threaded; the second what proves it is not corrupting the run.
    worst, force = {}, {}
    for mode in ("ad", "ik"):
        with tempfile.TemporaryDirectory() as d:
            worst[mode], force[mode] = check("midtown", mode, d)
    both = np.concatenate([force["ad"].reshape(-1), force["ik"].reshape(-1)])
    scale = max(float(np.abs(both).max()), 1e-12)
    gap = float(np.abs(force["ad"] - force["ik"]).max() / scale)
    print(f"  |F_ad - F_ik| / max|F| = {gap:.2e}  "
          f"(must be well above the 1e-5 compression floor)")
    assert gap > 1e-4, "the frozen lowers ignore force_mode"
    print("done")


if __name__ == "__main__":
    main()
