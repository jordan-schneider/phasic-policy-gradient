import argparse
from pathlib import Path
from typing import Optional

import GPUtil  # type: ignore
import torch
from mpi4py import MPI  # type: ignore
from procgen.env import ProcgenGym3Env

from . import logger, ppg
from . import torch_util as tu
from .envs import get_venv
from .impala_cnn import ImpalaEncoder


def make_model(env: ProcgenGym3Env, arch: ppg.Architecture) -> ppg.PhasicValueModel:
    enc_fn = lambda obtype: ImpalaEncoder(
        obtype.shape,
        outsize=256,
        chans=(16, 32, 32),
    )
    model = ppg.PhasicValueModel(env.ob_space, env.ac_space, enc_fn, arch=arch)
    return model


def train_fn(
    save_dir: Path,
    venv: Optional[ProcgenGym3Env] = None,
    env_name="coinrun",
    distribution_mode="hard",
    model_path: Optional[Path] = None,
    start_time: int = 0,
    arch: ppg.Architecture = "dual",
    interacts_total=100_000_000,
    num_envs=64,
    n_epoch_pi=1,
    n_epoch_vf=1,
    gamma=0.999,
    aux_lr=5e-4,
    lr=5e-4,
    n_minibatch=8,
    aux_mbsize=4,
    clip_param=0.2,
    kl_penalty=0.0,
    n_aux_epochs=6,
    n_pi=32,
    beta_clone=1.0,
    vf_true_weight=1.0,
    log_dir="/tmp/ppg",
    comm=None,
    port: int = 29500,
):
    """

    arch values:
    'shared' = shared policy and value networks
    'dual' = separate policy and value networks
    'detach' = shared policy and value networks, but with the value function gradient detached during the policy phase to avoid interference

    """
    if comm is None:
        comm = MPI.COMM_WORLD

    tu.setup_dist(
        comm=comm, start_port=port, gpu_offset=GPUtil.getFirstAvailable(order="load")[0]
    )
    tu.register_distributions_for_tree_util()

    is_master = comm.Get_rank() == 0

    if log_dir is not None:
        format_strs = ["csv", "stdout", "log", "tensorboard"] if is_master else []
        logger.configure(
            comm=comm,
            outdir=log_dir,
            format_strs=format_strs,
            append=model_path is not None,
        )

    if venv is None:
        venv = get_venv(
            num_envs=num_envs, env_name=env_name, distribution_mode=distribution_mode
        )

    model = make_model(venv, arch)
    model.to(tu.DEFAULT_DEVICE)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, tu.DEFAULT_DEVICE))

    logger.log(tu.format_model(model))
    tu.sync_params(model.parameters())

    name2coef = {"pol_distance": beta_clone, "vf_true": vf_true_weight}

    ppg.learn(
        venv=venv,
        model=model,
        interacts_total=interacts_total,
        ppo_hps=dict(
            lr=lr,
            γ=gamma,
            λ=0.95,
            nminibatch=n_minibatch,
            n_epoch_vf=n_epoch_vf,
            n_epoch_pi=n_epoch_pi,
            clip_param=clip_param,
            kl_penalty=kl_penalty,
            log_save_opts={
                "save_mode": "all",
                "init_timestep": start_time,
                "save_dir": save_dir,
            },
        ),
        aux_lr=aux_lr,
        aux_mbsize=aux_mbsize,
        n_aux_epochs=n_aux_epochs,
        n_pi=n_pi,
        name2coef=name2coef,
        comm=comm,
    )


def main():
    parser = argparse.ArgumentParser(description="Process PPG training arguments.")
    parser.add_argument("--env_name", type=str, default="coinrun")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--n_epoch_pi", type=int, default=1)
    parser.add_argument("--n_epoch_vf", type=int, default=1)
    parser.add_argument("--n_aux_epochs", type=int, default=6)
    parser.add_argument("--n_pi", type=int, default=32)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--kl_penalty", type=float, default=0.0)
    parser.add_argument(
        "--arch", type=str, default="dual"
    )  # 'shared', 'detach', or 'dual'

    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    train_fn(
        save_dir=Path.cwd(),
        env_name=args.env_name,
        num_envs=args.num_envs,
        n_epoch_pi=args.n_epoch_pi,
        n_epoch_vf=args.n_epoch_vf,
        n_aux_epochs=args.n_aux_epochs,
        n_pi=args.n_pi,
        arch=args.arch,
        comm=comm,
    )


if __name__ == "__main__":
    main()
