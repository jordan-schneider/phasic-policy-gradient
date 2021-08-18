import itertools
import operator
from typing import Callable, Dict, List, Literal, Tuple

import torch
import torch as th
from gym3.types import ValType  # type: ignore
from mpi4py import MPI  # type: ignore
from torch import distributions as td

from phasic_policy_gradient.impala_cnn import Encoder, ImpalaEncoder

from . import logger, ppo
from . import torch_util as tu
from .distr_builder import distr_builder
from .tree_util import tree_map, tree_reduce


def sum_nonbatch(logprob_tree):
    """
    sums over nonbatch dimensions and over all leaves of the tree
    use with nested action spaces, which require Product distributions
    """
    return tree_reduce(operator.add, tree_map(tu.sum_nonbatch, logprob_tree))


class PpoModel(th.nn.Module):
    def forward(
        self, ob, first, state_in
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @tu.no_grad
    def act(self, ob, first, state_in):
        pd, vpred, _, state_out = self(
            ob=tree_map(lambda x: x[:, None], ob),
            first=first[:, None],
            state_in=state_in,
        )
        ac = pd.sample()
        logp = sum_nonbatch(pd.log_prob(ac))
        return (
            tree_map(lambda x: x[:, 0], ac),
            state_out,
            dict(vpred=vpred[:, 0], logp=logp[:, 0]),
        )

    @tu.no_grad
    def v(self, ob, first, state_in):
        _pd, vpred, _, _state_out = self(
            ob=tree_map(lambda x: x[:, None], ob),
            first=first[:, None],
            state_in=state_in,
        )
        return vpred[:, 0]


class PhasicModel(PpoModel):
    def forward(
        self, ob, first, state_in
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def compute_aux_loss(self, aux, mb):
        raise NotImplementedError

    def initial_state(self, batchsize):
        raise NotImplementedError

    def aux_keys(self) -> List[str]:
        """Returns list of keys needed in mb dict for compute_aux_loss"""
        raise NotImplementedError

    def set_aux_phase(self, is_aux_phase: bool):
        "sometimes you want to modify the model, e.g. add a stop gradient"


Architecture = Literal["shared", "detach", "dual"]


class PhasicValueModel(PhasicModel):
    def __init__(
        self,
        obtype: ValType,
        actype: ValType,
        enc_fn: Callable[[ValType], ImpalaEncoder],
        arch: Architecture = "dual",
    ):
        super().__init__()

        detach_value_head = False
        vf_keys = None
        pi_key = "pi"

        if arch == "shared":
            self.true_vf_key = "pi"
        elif arch == "detach":
            self.true_vf_key = "pi"
            detach_value_head = True
        elif arch == "dual":
            self.true_vf_key = "vf"
        else:
            assert False

        self.vf_keys = vf_keys or [self.true_vf_key]
        self.pi_enc = enc_fn(obtype)
        self.pi_key = pi_key
        self.enc_keys = list(set([pi_key] + self.vf_keys))
        self.detach_value_head = detach_value_head
        pi_outsize, self.make_distr = distr_builder(actype)

        for k in self.enc_keys:
            self.set_encoder(k, enc_fn(obtype))

        for k in self.vf_keys:
            lastsize = self.get_encoder(k).codetype.size
            self.set_vhead(k, tu.NormedLinear(lastsize, 1, scale=0.1))

        lastsize = self.get_encoder(self.pi_key).codetype.size
        self.pi_head = tu.NormedLinear(lastsize, pi_outsize, scale=0.1)
        self.aux_vf_head = tu.NormedLinear(lastsize, 1, scale=0.1)

    def compute_aux_loss(self, aux, seg):
        vtarg = seg["vtarg"]
        return {
            "vf_aux": 0.5 * ((aux["vpredaux"] - vtarg) ** 2).mean(),
            "vf_true": 0.5 * ((aux["vpredtrue"] - vtarg) ** 2).mean(),
        }

    def reshape_x(self, x):
        b, t = x.shape[:2]
        x = x.reshape(b, t, -1)

        return x

    def get_encoder(self, key: str) -> ImpalaEncoder:
        return getattr(self, key + "_enc")

    def set_encoder(self, key: str, enc: ImpalaEncoder) -> None:
        setattr(self, key + "_enc", enc)

    def get_vhead(self, key: str) -> torch.nn.Linear:
        return getattr(self, key + "_vhead")

    def set_vhead(self, key: str, layer: torch.nn.Linear) -> None:
        setattr(self, key + "_vhead", layer)

    def forward(self, ob, first, state_in):
        state_out = {}
        x_out = {}

        for k in self.enc_keys:
            x_out[k], state_out[k] = self.get_encoder(k)(ob, first, state_in[k])
            x_out[k] = self.reshape_x(x_out[k])

        pi_x = x_out[self.pi_key]
        pivec = self.pi_head(pi_x)
        pd = self.make_distr(pivec)

        aux = {}
        for k in self.vf_keys:
            if self.detach_value_head:
                x_out[k] = x_out[k].detach()
            aux[k] = self.get_vhead(k)(x_out[k])[..., 0]
        vfvec = aux[self.true_vf_key]
        aux.update({"vpredaux": self.aux_vf_head(pi_x)[..., 0], "vpredtrue": vfvec})

        return pd, vfvec, aux, state_out

    @torch.no_grad()
    def value(self, obs: torch.Tensor):
        return self.get_vhead(self.true_vf_key)(
            self.get_encoder(self.true_vf_key).stateless_forward(obs)
        )

    def initial_state(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return {k: self.get_encoder(k).initial_state(batchsize) for k in self.enc_keys}

    def aux_keys(self) -> List[str]:
        return ["vtarg"]


def make_minibatches(segs, mbsize):
    """
    Yield one epoch of minibatch over the dataset described by segs
    Each minibatch mixes data between different segs
    """
    nenv = tu.batch_len(segs[0])
    nseg = len(segs)
    envs_segs = th.tensor(list(itertools.product(range(nenv), range(nseg))))
    for perminds in th.randperm(len(envs_segs)).split(mbsize):
        esinds = envs_segs[perminds]
        yield tu.tree_stack([tu.tree_slice(segs[segind], envind) for (envind, segind) in esinds])


def aux_train(*, model, segs, opt: torch.optim.Optimizer, mbsize, name2coef):
    """
    Train on auxiliary loss + policy KL + vf distance
    """
    needed_keys = {"ob", "first", "state_in", "oldpd"}.union(model.aux_keys())
    segs = [{k: seg[k] for k in needed_keys} for seg in segs]
    for mb in make_minibatches(segs, mbsize):
        mb = tree_map(lambda x: x.to(tu.dev()), mb)
        pd, _, aux, _state_out = model(mb["ob"], mb["first"], mb["state_in"])
        name2loss = {}
        name2loss["pol_distance"] = td.kl_divergence(mb["oldpd"], pd).mean()
        name2loss.update(model.compute_aux_loss(aux, mb))
        assert set(name2coef.keys()).issubset(name2loss.keys())
        loss = torch.zeros(())
        for name in name2loss.keys():
            unscaled_loss = name2loss[name]
            scaled_loss = unscaled_loss * name2coef.get(name, 1)
            logger.logkv_mean("unscaled/" + name, unscaled_loss)
            logger.logkv_mean("scaled/" + name, scaled_loss)
            loss += scaled_loss
        opt.zero_grad()
        loss.backward()
        tu.sync_grads(model.parameters())
        opt.step()


def compute_presleep_outputs(*, model, segs, mbsize, pdkey="oldpd", vpredkey="oldvpred"):
    def forward(ob, first, state_in):
        pd, vpred, _aux, _state_out = model.forward(ob.to(tu.dev()), first, state_in)
        return pd, vpred

    for seg in segs:
        seg[pdkey], seg[vpredkey] = tu.minibatched_call(
            forward, mbsize, ob=seg["ob"], first=seg["first"], state_in=seg["state_in"]
        )


def learn(
    *,
    model,
    venv,
    ppo_hps,
    aux_lr,
    aux_mbsize,
    n_aux_epochs=6,
    n_pi=32,
    kl_ewma_decay=None,
    interacts_total=float("inf"),
    name2coef=None,
    comm=None,
):
    """
    Run PPO for X iterations
    Then minimize aux loss + KL + value distance for X passes over data
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    ppo_state = None
    aux_state = th.optim.Adam(model.parameters(), lr=aux_lr)
    name2coef = name2coef or {}

    while True:
        store_segs = n_pi != 0 and n_aux_epochs != 0

        # Policy phase
        ppo_state = ppo.learn(
            venv=venv,
            model=model,
            learn_state=ppo_state,
            callbacks=[
                lambda _l: n_pi > 0 and _l["curr_iteration"] >= n_pi,
            ],
            interacts_total=interacts_total,
            store_segs=store_segs,
            comm=comm,
            **ppo_hps,
        )

        if ppo_state["curr_interact_count"] >= interacts_total:
            break

        if n_aux_epochs > 0:
            segs = ppo_state["seg_buf"]
            compute_presleep_outputs(model=model, segs=segs, mbsize=aux_mbsize)
            # Auxiliary phase
            for i in range(n_aux_epochs):
                logger.log(f"Aux epoch {i}")
                aux_train(
                    model=model,
                    segs=segs,
                    opt=aux_state,
                    mbsize=aux_mbsize,
                    name2coef=name2coef,
                )
                logger.dumpkvs()
            segs.clear()
