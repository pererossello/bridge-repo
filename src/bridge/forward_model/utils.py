# ruff: noqa

import jax
import jax.numpy as jnp

from .cosmo import compute_or_load_D1_D2, compute_or_load_pow_spec_cube
from .config import Constants, FMConfig
from .fourier import my_fft, my_ifft
from .fourier_aux import get_polar_from_wn, make_N_cube_numbers_from_rfft3_arr


def apply_wiener_filter(
    n_tr_data,
    cte: Constants,
    sinc=True,
    fix_pk=True,
    return_kind="WN",
):

    D1, _ = compute_or_load_D1_D2(cte.Z_I, cte.Z_F)
    pk_lm = compute_or_load_pow_spec_cube(cte.N, cte.L, cte.Z_I, sinc=sinc)

    pk_lm *= D1**2

    pk_noise = 1 / (n_tr_data.sum() / cte.L**3)
    delta_n_tr_data = (cte.N**3 / n_tr_data.sum()) * n_tr_data - 1
    delta_hat_n_tr_data = my_fft(delta_n_tr_data, cte.L3)
    window = pk_lm / (pk_lm + pk_noise)

    delta_hat_wf = delta_hat_n_tr_data * window / D1
    delta_wf = my_ifft(delta_hat_wf, cte.INV_L3)

    if fix_pk:
        pk_wf = jnp.abs(delta_hat_wf) ** 2 * cte.INV_L3
        pk_wf = jnp.where(pk_wf != 0, pk_wf, 1.0)
        delta_hat_wf = delta_hat_wf * jnp.sqrt(pk_lm / D1**2) / jnp.sqrt(pk_wf)
        delta_wf = my_ifft(delta_hat_wf, cte.INV_L3)

    if return_kind == "WN":
        one_over_pk_lm = jnp.where(pk_lm != 0, 1 / (pk_lm / D1**2), 1.0)
        q_wf = my_ifft(delta_hat_wf * jnp.sqrt(one_over_pk_lm * cte.R3), cte.INV_L3)

        return delta_wf, q_wf

    if return_kind == "CWN":
        one_over_pk_lm = jnp.where(pk_lm != 0, 1 / (pk_lm / D1**2), 1.0)
        q_wf = make_N_cube_numbers_from_rfft3_arr(
            delta_hat_wf * jnp.sqrt(one_over_pk_lm), cte.N
        )
        q_wf *= jnp.sqrt(2 * cte.INV_L3)

        return delta_wf, q_wf

    if return_kind == "PHASES":
        one_over_pk_lm = jnp.where(pk_lm != 0, 1 / (pk_lm / D1**2), 1.0)
        q_wf = my_ifft(delta_hat_wf * jnp.sqrt(one_over_pk_lm * cte.R3), cte.INV_L3)
        input_phases, input_amps, input_nyq = get_polar_from_wn(q_wf, cte.N, cte.L)
        return delta_wf, input_phases

    if return_kind == "WN":
        one_over_pk_lm = jnp.where(pk_lm != 0, 1 / (pk_lm / D1**2), 1.0)
        q_wf = my_ifft(delta_hat_wf * jnp.sqrt(one_over_pk_lm * cte.R3), cte.INV_L3)

        return delta_wf, q_wf

    return delta_wf
