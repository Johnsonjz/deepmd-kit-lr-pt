# SPDX-License-Identifier: LGPL-3.0-or-later
"""The long-range kernel training schedule of the pt_expt trainer.

``long_range_kernel_train_start_step`` holds the SOG kernel fixed until that
step and ``long_range_kernel_lr_scale_start`` / ``_lr_scale`` / ``_lr_ramp_steps``
then ramp its learning rate up from a fraction of the global rate.  These tests
read the recorded checkpoints, which is the only place the kernel state and its
own learning rate are visible: the display line reports the global group only.
"""

from pathlib import (
    Path,
)

import numpy as np
import pytest
import torch

from deepmd.pt_expt.entrypoints.main import (
    get_trainer,
)
from deepmd.utils import random as dp_random
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

KERNEL_AMPLITUDE = "model.Default.atomic_model.long_range.kernel.amplitude"
KERNEL_LOG_WIDTH = "model.Default.atomic_model.long_range.kernel.log_width"


def _write_system(path: Path) -> None:
    """A four-atom periodic O/H system, the smallest a periodic module takes."""
    path.mkdir()
    np.savetxt(path / "type.raw", [0, 1, 1, 1], fmt="%d")
    (path / "type_map.raw").write_text("O H\n")
    rng = np.random.RandomState(24)
    for index in range(2):
        target = path / f"set.{index:03d}"
        target.mkdir()
        np.save(target / "coord.npy", rng.uniform(0.5, 3.5, (2, 12)))
        np.save(target / "box.npy", np.tile(np.eye(3).reshape(1, 9) * 7, (2, 1)))
        np.save(target / "energy.npy", np.arange(2) * 0.01)
        np.save(target / "force.npy", rng.normal(size=(2, 12)) * 0.01)
        np.save(target / "virial.npy", rng.normal(size=(2, 9)) * 0.01)


def _config(
    path: Path,
    *,
    steps: int,
    train_start_step: int,
    lr_scale: float = 0.3,
    lr_scale_start: float = 0.01,
    lr_ramp_steps: int = 2,
) -> dict:
    model = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa4c",
            "rcut": 4.0,
            "channels": 8,
            "lmax": 2,
            "n_radial": 4,
            "use_amp": False,
            "seed": 1,
        },
        "fitting_net": {"type": "ener", "neuron": [8], "seed": 1},
        "data_stat_nbatch": 1,
        "long_range": {
            "type": "latent_charge",
            "channels": 1,
            "charge_net": {"neuron": [8], "precision": "float64", "seed": 1},
            "kernel": {"type": "sog", "n_gaussian": 4, "trainable": True},
            "periodic": True,
            "k_grid_mode": "ellipsoid",
            "k_grid_bounds": [4, 4, 4],
        },
    }
    loss = {
        "type": "ener",
        "start_pref_e": 1,
        "limit_pref_e": 1,
        "start_pref_f": 1,
        "limit_pref_f": 1,
        "start_pref_v": 1,
        "limit_pref_v": 1,
    }
    dataset = {"systems": [str(path)], "batch_size": 1}
    config = {
        "model": model,
        "loss": loss,
        "learning_rate": {
            "type": "exp",
            "start_lr": 1e-3,
            "stop_lr": 1e-5,
            "decay_steps": 50,
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "long_range_kernel_lr_scale": lr_scale,
            "long_range_kernel_lr_scale_start": lr_scale_start,
            "long_range_kernel_lr_ramp_steps": lr_ramp_steps,
            "long_range_kernel_train_start_step": train_start_step,
        },
        "training": {
            "training_data": dict(dataset),
            "validation_data": {**dataset, "numb_btch": 1},
            "numb_steps": steps,
            "save_freq": 1,
            "max_ckpt_keep": 100,
            "disp_freq": 1,
            "seed": 19,
        },
    }
    # ``optimizer.learning_rate`` is not a schema key; the schedule is the
    # learning-rate section.
    del config["optimizer"]["learning_rate"]
    return normalize(update_deepmd_input(config, warning=False), multi_task=False)


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs) -> dict[int, dict]:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    _write_system(data)
    config = _config(data, **kwargs)
    dp_random.seed(19)
    trainer = get_trainer(config)
    trainer.run()
    trainer._close_data_systems()
    checkpoints = {}
    for path in tmp_path.glob("model.ckpt-*.pt"):
        step = int(path.stem.rsplit("-", 1)[1])
        checkpoints[step] = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoints


def _kernel(state: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return state["model"][KERNEL_AMPLITUDE], state["model"][KERNEL_LOG_WIDTH]


def _group_lrs(state: dict) -> list[float]:
    return [group["lr"] for group in state["optimizer"]["param_groups"]]


def test_kernel_is_frozen_before_its_start_step(tmp_path, monkeypatch) -> None:
    """No kernel update at all while the schedule holds it fixed."""
    checkpoints = _run(
        tmp_path, monkeypatch, steps=4, train_start_step=3, lr_ramp_steps=2
    )
    early = _kernel(checkpoints[1])
    held = _kernel(checkpoints[2])
    for before, after in zip(early, held, strict=True):
        assert torch.equal(before, after)
    # The step at the start step trains, so the recorded kernel moves after it.
    later = _kernel(checkpoints[4])
    assert not torch.equal(held[1], later[1])


def test_kernel_learning_rate_follows_the_ramp(tmp_path, monkeypatch) -> None:
    """The kernel group ramps from the start scale to the end scale."""
    lr_scale, lr_scale_start, ramp = 0.3, 0.01, 2
    start = 3
    checkpoints = _run(
        tmp_path,
        monkeypatch,
        steps=6,
        train_start_step=start,
        lr_scale=lr_scale,
        lr_scale_start=lr_scale_start,
        lr_ramp_steps=ramp,
    )
    for step, state in sorted(checkpoints.items()):
        base, kernel_lr = _group_lrs(state)
        t = min(1.0, max(0.0, (step - start) / ramp))
        expected_scale = lr_scale_start * (lr_scale / lr_scale_start) ** t
        assert kernel_lr == pytest.approx(base * expected_scale, rel=1e-6)


def test_kernel_trains_from_the_first_step_by_default(tmp_path, monkeypatch) -> None:
    """A zero start step leaves the kernel training, as before the schedule."""
    checkpoints = _run(tmp_path, monkeypatch, steps=3, train_start_step=0)
    first, last = _kernel(checkpoints[1]), _kernel(checkpoints[3])
    assert not torch.equal(first[1], last[1])
