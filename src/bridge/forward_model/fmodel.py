import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.tree_util import Partial as partial

from .bias import (
    det_bias_dict,
    get_apply_cweb,
    make_bias_param_distro_dic,
    norm_factor_hard_cweb,
    norm_factor_no_cweb,
    sample_negbin,
    sample_poisson,
)
from .config import Constants, FMConfig
from .cosmic_web import get_cweb
from .ics import gen_input_arr, get_delta_in
from .lpt import displace_particles, get_delta_lpt, get_psi


class FModel:

    def __init__(self, fm_cfg: FMConfig):

        self.N = fm_cfg.N
        self.L = fm_cfg.L
        self.Z_I = fm_cfg.Z_I
        self.Z_F = fm_cfg.Z_F

        self.cte = Constants(fm_cfg.N, fm_cfg.L, fm_cfg.Z_I, fm_cfg.Z_F)
        self.fm_cfg = fm_cfg

    def input_arr(self, key):
        return gen_input_arr(key, cte=self.cte, fm_cfg=self.fm_cfg)

    def delta_in(self, x=None):
        f = lambda q: get_delta_in(q, cte=self.cte, fm_cfg=self.fm_cfg)
        return f if x is None else f(x)

    def psi(self, x=None):
        f = lambda q: get_psi(self.delta_in(q), cte=self.cte, fm_cfg=self.fm_cfg)
        return f if x is None else f(x)

    def lagr_pos(self, x=None):
        f = lambda q: displace_particles(self.psi(q), cte=self.cte, fm_cfg=self.fm_cfg)
        return f if x is None else f(x)

    def delta_lpt(self, x=None):
        f = lambda q: get_delta_lpt(self.delta_in(q), cte=self.cte, fm_cfg=self.fm_cfg)
        return f if x is None else f(x)

    def cweb(self, x=None):
        f = lambda y: get_cweb(y, cte=self.cte, fm_cfg=self.fm_cfg)
        return f if x is None else f(x)

    def n_tr_mean(self):

        if self.fm_cfg.global_N_TR:
            normalize = partial(norm_factor_no_cweb, N_TR=self.fm_cfg.N_TR)
        else:
            if not self.fm_cfg.soft_cweb:
                normalize = partial(
                    norm_factor_hard_cweb,
                    N_TR=self.fm_cfg.N_TR,
                    n_regions=self.fm_cfg.n_regions,
                )
            # else:
            #     normalize = partial(
            #         self.norm_factor_soft_cweb,
            #         N_TR=self.fm_cfg.bias_N_TR,
            #         n_regions=self.fm_cfg.n_regions,
            #     )

        apply_cweb = get_apply_cweb(self.fm_cfg)
        core, pnames = det_bias_dict[self.fm_cfg.det_bias_model]

        def f(delta_dm, params, cweb=None):
            kw = {name: apply_cweb(params[name], cweb) for name in pnames}
            n_tr_mean = core(delta_dm, **kw)
            return normalize(n_tr_mean)  # type: ignore

        return f

    def sample_n_tr(self, n_tr_mean, key, params=None, cweb=None):

        apply_cweb = get_apply_cweb(self.fm_cfg)

        stoch_bias = self.fm_cfg.stoch_bias_model
        if stoch_bias == "Poisson":  # cweb dos not enter
            return sample_poisson(key, n_tr_mean)
        elif stoch_bias == "NegBinomial":
            beta = jnp.array(params["beta"])  # type: ignore
            beta = apply_cweb(beta, cweb)
            return sample_negbin(key, n_tr_mean, beta)

    def build_model(self):

        n_regions = self.fm_cfg.n_regions

        get_delta_lpt = self.delta_lpt()
        get_cweb = self.cweb()
        apply_cweb = get_apply_cweb(self.fm_cfg)
        get_n_tr_mean = self.n_tr_mean()
        bias_distro_dic = make_bias_param_distro_dic(
            self.fm_cfg.det_bias_model, self.fm_cfg.stoch_bias_model, n_regions
        )

        def model(data):

            if self.fm_cfg.input_kind in ["WN", "CWN"]:
                q = numpyro.sample(
                    "q", dist.Normal(0, 1).expand([self.N, self.N, self.N]).to_event(3)  # type: ignore
                )
            elif self.fm_cfg.input_kind == "PHASES":
                q = numpyro.sample(
                    "q",
                    dist.Uniform(low=0, high=1)  # type: ignore
                    .expand(
                        [
                            self.N**3 // 2 - 4,  # type: ignore
                        ]
                    )
                    .to_event(1),
                )

            delta_lpt = get_delta_lpt(q)  # type: ignore
            cweb = get_cweb(delta_lpt)  # type: ignore

            params = {
                name: numpyro.sample(name, distro)  # type: ignore
                for name, distro in bias_distro_dic.items()
            }
            n_tr_mean = get_n_tr_mean(delta_lpt, params, cweb)

            if self.fm_cfg.stoch_bias_model == "Poisson":
                numpyro.sample("obs", dist.Poisson(n_tr_mean), obs=data)

            elif self.fm_cfg.stoch_bias_model == "NegBinomial":
                beta = params["beta"]
                beta = apply_cweb(beta, cweb)
                numpyro.sample("obs", dist.NegativeBinomial2(n_tr_mean, beta), obs=data)

        return model

    def _get_apply_cweb(self):
        if self.fm_cfg.cweb is None:
            f = lambda v, cweb: v
            return f
        else:
            if not self.fm_cfg.soft_cweb:
                f = lambda v, cweb: v[cweb]
                return f
            else:
                f = lambda v, cweb: jnp.einsum("ijkl,l->ijk", cweb, v)
                return f
