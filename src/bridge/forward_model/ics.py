import jax
import jax.numpy as jnp

from .config import Constants, FMConfig
from .fourier import my_fft
from .fourier_aux import make_rfft3_arr_from_N_cube_numbers, make_rfft3_arr_from_phases


def get_delta_in(input_arr, cte: Constants, fm_cfg: FMConfig):
    delta = None
    if fm_cfg.input_kind == "WN":
        delta = my_fft(input_arr, cte.L3) * jnp.sqrt(fm_cfg.pow_spec * cte.INV_R3)
    elif fm_cfg.input_kind == "CWN":
        input_arr = make_rfft3_arr_from_N_cube_numbers(input_arr, cte.N)
        delta = input_arr * jnp.sqrt(fm_cfg.pow_spec * cte.L3 * 0.5)

    elif fm_cfg.input_kind == "PHASES":
        input_arr = jnp.mod(input_arr, 1)
        phases = 2 * jnp.pi * (input_arr - 0.5)
        phases = make_rfft3_arr_from_phases(phases, cte.N)
        z_unit = jnp.exp(1j * phases)
        fact = cte.L3 * jnp.sqrt(cte.INV_N3)
        delta = z_unit * jnp.sqrt(fm_cfg.pow_spec * cte.INV_R3) * fact
    return delta


def gen_input_arr(key, cte: Constants, fm_cfg: FMConfig):

    input_arr = None
    if fm_cfg.input_kind in ["WN", "CWN"]:
        input_arr = jax.random.normal(key, shape=(cte.N,) * 3)
    elif fm_cfg.input_kind == "PHASES":
        input_arr = jax.random.uniform(key, shape=(cte.N3 // 2 - 4,))  # type: ignore

    return input_arr
