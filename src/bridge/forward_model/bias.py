import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from .config import FMConfig

# DETERMINISTIC BIAS FUNCTIONS


eps = 1e-1


def bias_l(delta_dm, b=1):
    return b * (1 + delta_dm)


def bias_pl(delta_dm, alpha=1):
    return jnp.power(1 + delta_dm + eps, alpha)


def bias_l_pl(delta_dm, alpha=1, b=1):
    return b * jnp.power(1 + delta_dm + eps, alpha)


def bias_pl_hp(delta_dm, alpha, e_hp, rho_hp):
    rho_dm = 1 + delta_dm + eps
    power_law = jnp.power(rho_dm, alpha)
    high_pass = jnp.exp(-jnp.power(((rho_dm + eps) / rho_hp + eps), -e_hp))
    return power_law * high_pass


def bias_pl_lp_hp(delta_dm, alpha, e_hp, rho_hp, e_lp, rho_lp):
    rho_dm = 1 + delta_dm + eps
    power_law = jnp.power(rho_dm, alpha)
    low_pass = jnp.exp(-jnp.power(((rho_dm + eps) / rho_lp + eps), e_lp))
    high_pass = jnp.exp(-jnp.power(((rho_dm + eps) / rho_hp + eps), -e_hp))
    return power_law * low_pass * high_pass


det_bias_dict = {
    "Linear": (bias_l, ("b",)),
    "PowerLaw": (bias_pl, ("alpha",)),
    "LinearPowerLaw": (bias_l_pl, ("alpha", "b")),
    "HighPassPowerLaw": (bias_pl_hp, ("alpha", "e_hp", "rho_hp")),
    "LowHighPassPowerLaw": (
        bias_pl_lp_hp,
        ("alpha", "e_hp", "rho_hp", "e_lp", "rho_lp"),
    ),
}


def make_bias_param_distro_dic(det_bias_model, stoch_bias_model, n_regions):

    out_dict = {}

    if det_bias_model == "Linear":
        b_distro = dist.LogNormal(0.0, 1.0).expand([n_regions]).to_event(1)  # type: ignore
        out_dict = {"b": b_distro}

    elif det_bias_model == "PowerLaw":
        alpha_distro = dist.LogNormal(0.273, 0.3).expand([n_regions]).to_event(1)  # type: ignore
        out_dict = {"alpha": alpha_distro}

    elif det_bias_model == "LinearPowerLaw":
        b_distro = dist.LogNormal(0.0, 1.0).expand([n_regions]).to_event(1)  # type: ignore
        alpha_distro = dist.LogNormal(0.0, 1.0).expand([n_regions]).to_event(1)  # type: ignore
        out_dict = {"b": b_distro, "alpha": alpha_distro}

    elif det_bias_model == "HighPassPowerLaw":
        alpha_distro = dist.LogNormal(0.0, 1.0).expand([n_regions]).to_event(1)  # type: ignore
        e_hp_distro = dist.Gamma(2.0, 2.0).expand([n_regions]).to_event(1)  # type: ignore
        rho_hp_distro = dist.LogNormal(0.0, 1.0).expand([n_regions]).to_event(1)  # type: ignore
        out_dict = {"alpha": alpha_distro, "e_hp": e_hp_distro, "rho_hp": rho_hp_distro}

    if stoch_bias_model == "NegBinomial":
        beta_distro = (
            dist.LogNormal(2.119, 0.2).expand([n_regions]).to_event(1)  # type: ignore
        )
        out_dict["beta"] = beta_distro

    return out_dict


# NORMALIZATION


def norm_factor_no_cweb(n_tr_mean, N_TR):
    norm = N_TR / jnp.sum(n_tr_mean)
    return norm * n_tr_mean


def norm_factor_hard_cweb(n_tr_mean, N_TR, cweb_arr, n_regions):
    region_sums = jnp.array(
        [jnp.sum(n_tr_mean * (cweb_arr == i)) for i in range(n_regions)]
    )
    norm_factors = jnp.where(region_sums > 0, N_TR / region_sums, 0.0)
    # print(region_sums / region_sums.sum())
    return norm_factors[cweb_arr] * n_tr_mean


def norm_factor_soft_cweb(n_tr_mean, N_TR, cweb_hard, cweb_soft, n_regions):
    region_sums = jnp.array(
        [jnp.sum(n_tr_mean * (cweb_hard == i)) for i in range(n_regions)]
    )
    norm_factors = jnp.where(region_sums > 0, N_TR / region_sums, 0.0)
    norm_factors = jnp.einsum("ijkl,l->ijk", cweb_soft, norm_factors)
    n_tr_mean = n_tr_mean * norm_factors
    return n_tr_mean * jnp.sum(N_TR) / jnp.sum(n_tr_mean)


# SAMPLING


def sample_poisson(key, n_tr_mean):
    return jax.random.poisson(key, n_tr_mean)


def sample_negbin(key, n_tr_mean, beta):

    sk1, sk2 = jax.random.split(key, 2)

    gamma_shape_param = beta
    gamma_rate_param = gamma_shape_param / n_tr_mean

    lambs = (
        jax.random.gamma(sk1, gamma_shape_param, shape=n_tr_mean.shape)
        / gamma_rate_param
    )

    return jax.random.poisson(sk2, lambs)


# HELPER FUNCTIONS


def manage_params(params):
    for key, val in params.items():
        val = jnp.atleast_1d(jnp.array(val)).astype(float)
        params[key] = val
    return params


def get_apply_cweb(fm_cfg: FMConfig):
    if fm_cfg.cweb is None:
        f = lambda v, cweb: v
        return f
    else:
        if not fm_cfg.soft_cweb:
            f = lambda v, cweb: v[cweb]
            return f
        else:
            f = lambda v, cweb: jnp.einsum("ijkl,l->ijk", cweb, v)
            return f
