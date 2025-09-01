import jax.numpy as jnp

from .algebra import eigenvals_of_hessian
from .config import Constants, FMConfig
from .fourier import my_fft


def get_web(field, fm_cfg: FMConfig, cte: Constants):
    lambdas = eigenvals_of_hessian(field, cte.kx, cte.ky, cte.kz, cte.INV_L3)
    web = lambda_classify(lambdas, fm_cfg)
    return web


def get_cweb(delta_dm, fm_cfg: FMConfig, cte: Constants):

    cweb = None
    if fm_cfg.cweb is None:
        return None
    else:
        if fm_cfg.cweb == "PhiWeb":
            cweb = get_phi_web(delta_dm, fm_cfg, cte)
        elif fm_cfg.cweb == "DeltaWeb":
            cweb = get_delta_web(delta_dm, fm_cfg, cte)
        elif fm_cfg.cweb == "PhiDeltaWeb":
            cweb = get_phi_delta_web(delta_dm, fm_cfg, cte)
        return cweb


def get_phi_delta_web(delta_dm, fm_cfg: FMConfig, cte: Constants):
    delta_dm = my_fft(delta_dm, cte.L3)
    delta_web = get_web(-delta_dm, fm_cfg, cte)
    phi_web = get_web(-delta_dm * fm_cfg.one_over_k_sq, fm_cfg, cte)
    if not fm_cfg.soft_cweb:
        phi_delta_web = 4 * phi_web + delta_web
        return phi_delta_web
    else:
        joint = phi_web[..., :, None] * delta_web[..., None, :]
        joint = joint.reshape(joint.shape[:-2] + (16,))  # flatten last‑2 axes
        joint /= joint.sum(axis=-1, keepdims=True)  # numerical safety
        return joint


def get_phi_web(delta_dm, fm_cfg: FMConfig, cte: Constants):
    delta_dm = my_fft(delta_dm, cte.L3)
    phi_web = get_web(-delta_dm * fm_cfg.one_over_k_sq, fm_cfg, cte)
    return phi_web


def get_delta_web(delta_dm, fm_cfg: FMConfig, cte: Constants):
    delta_dm = my_fft(delta_dm, cte.L3)
    phi_web = get_web(-delta_dm, fm_cfg, cte)
    return phi_web


def lambda_classify(lambdas, fm_cfg: FMConfig):
    if not fm_cfg.soft_cweb:
        return lambda_hard_classify(lambdas, fm_cfg.lambda_th)
    else:
        return lambda_smooth_classify(lambdas, fm_cfg.lambda_th, fm_cfg.cweb_sharpness)


def lambda_hard_classify(lambdas, lambda_th):
    cw_class = (lambdas > lambda_th).sum(axis=-1)
    return cw_class


def sigma(x, beta, th):
    return 1 / (1 + jnp.exp(jnp.clip(-beta * (x - th), -50, 50)))


def lambda_smooth_classify(lambdas, lambda_th, slope=10.0):

    p0 = sigma(lambdas[..., 0], slope, lambda_th)
    p1 = sigma(lambdas[..., 1], slope, lambda_th)
    p2 = sigma(lambdas[..., 2], slope, lambda_th)

    P_void = 1 - p0
    P_sheet = p0 - p1
    P_filament = p1 - p2
    P_knot = p2

    cw_probs = jnp.stack([P_void, P_sheet, P_filament, P_knot], axis=-1)

    return cw_probs
