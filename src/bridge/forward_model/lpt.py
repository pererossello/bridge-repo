import jax
import jax.numpy as jnp

from .calculus import der_i_5ps, gradient_fd, gradient_hat, symmetric_gaussian_nonorm
from .config import Constants, FMConfig
from .fourier import get_k_sq, my_fft, my_fft_on_vec, my_ifft, my_ifft_on_vec
from .particle_mesh import particle_to_mesh, trilinear_interpolation


def get_delta_lpt(delta, cte: Constants, fm_cfg: FMConfig):

    psi = get_psi(delta, cte, fm_cfg)
    pos = displace_particles(psi, cte, fm_cfg)
    delta = particle_to_mesh(pos, cte)

    return delta


def displace_particles(psi, cte: Constants, fm_cfg: FMConfig):

    if jnp.iscomplexobj(psi):
        psi = my_ifft_on_vec(psi, cte.INV_L3)

    grid_shape = (cte.N, cte.N, cte.N)
    if fm_cfg.where_lagr_particles == "CORNER":
        grid = jnp.indices(grid_shape) * cte.R
    elif fm_cfg.where_lagr_particles == "CENTER":
        grid = (jnp.indices(grid_shape) + 0.5) * cte.R

    if fm_cfg.psi_trlilinear:
        positions = jnp.column_stack(
            [
                (grid[0]).reshape(-1),  # type: ignore
                (grid[1]).reshape(-1),  # type: ignore
                (grid[2]).reshape(-1),  # type: ignore
            ]
        )
        positions += trilinear_interpolation(psi, positions, cte)
    else:
        positions = jnp.column_stack(
            [
                (grid[0] + psi[0]).reshape(-1),  # type: ignore
                (grid[1] + psi[1]).reshape(-1),  # type: ignore
                (grid[2] + psi[2]).reshape(-1),  # type: ignore
            ]
        )
    return positions % cte.L


def get_psi(delta, cte: Constants, fm_cfg: FMConfig, **kwargs):
    if fm_cfg.lpt_method == "1LPT":
        return get_psi_lpt1(delta, cte=cte, fm_cfg=fm_cfg)
    elif fm_cfg.lpt_method == "2LPT":
        return get_psi_lpt2(delta, cte=cte, fm_cfg=fm_cfg)
    elif fm_cfg.lpt_method == "SC":
        return get_psi_sc(delta, cte=cte, fm_cfg=fm_cfg)
    elif fm_cfg.lpt_method == "ALPT":
        return get_psi_alpt(delta, cte=cte, fm_cfg=fm_cfg)


def get_psi_lpt1(delta_hat, cte: Constants, fm_cfg: FMConfig):
    phi_hat = -delta_hat * fm_cfg.one_over_k_sq

    psi = None
    if fm_cfg.der_method == "FR":
        psi = gradient_hat(-phi_hat, cte.kx, cte.ky, cte.kz)
    if fm_cfg.der_method == "FD":
        psi = gradient_fd(-my_ifft(phi_hat, cte.INV_L3), cte.N, cte.L)

    if fm_cfg.lpt_method == "1LPT":
        psi *= cte.D1
        if fm_cfg.rsd:
            if fm_cfg.rsd_type == "Radial":
                if fm_cfg.der_method == "FR":
                    psi = my_ifft_on_vec(psi, cte.INV_L3)
                dot = jnp.sum(psi * fm_cfg.r_vec_over_r, axis=0)
                psi += cte.f1 * dot * fm_cfg.r_vec_over_r * 1

            else:
                psi = psi.at[0, ...].multiply(1 + cte.f1)
    return psi


def get_psi_o2(psi1, cte: Constants, fm_cfg: FMConfig):

    L3, INV_L3 = cte.L3, cte.INV_L3
    kx, ky, kz = cte.kx, cte.ky, cte.kz

    if fm_cfg.der_method == "FR":
        phi1_dxx = 1j * kx * (-psi1[0])
        phi1_dxx = my_ifft(phi1_dxx, INV_L3)

        phi1_dyy = 1j * ky * (-psi1[1])
        phi1_dyy = my_ifft(phi1_dyy, INV_L3)

        phi1_dzz = 1j * kz * (-psi1[2])
        phi1_dzz = my_ifft(phi1_dzz, INV_L3)

        phi1_dxy = 1j * ky * (-psi1[0])
        phi1_dxy = my_ifft(phi1_dxy, INV_L3)

        phi1_dxz = 1j * kz * (-psi1[0])
        phi1_dxz = my_ifft(phi1_dxz, INV_L3)

        phi1_dyz = 1j * kz * (-psi1[1])
        phi1_dyz = my_ifft(phi1_dyz, INV_L3)

    elif fm_cfg.der_method == "FD":
        phi1_dxx = der_i_5ps(psi1[0], 0, cte.N, cte.L)
        phi1_dyy = der_i_5ps(psi1[1], 1, cte.N, cte.L)
        phi1_dzz = der_i_5ps(psi1[2], 2, cte.N, cte.L)
        phi1_dxy = der_i_5ps(psi1[0], 1, cte.N, cte.L)
        phi1_dxz = der_i_5ps(psi1[0], 2, cte.N, cte.L)
        phi1_dyz = der_i_5ps(psi1[1], 2, cte.N, cte.L)
    else:
        raise ValueError("Unknown derivative method")

    arr = (
        phi1_dyy * phi1_dzz
        + phi1_dxx * (phi1_dyy + phi1_dzz)
        - jnp.square(phi1_dxy)
        - jnp.square(phi1_dxz)
        - jnp.square(phi1_dyz)
    )

    arr = my_fft(arr, L3)

    phi2_hat = -arr * fm_cfg.one_over_k_sq

    if fm_cfg.der_method == "FR":
        psi2 = gradient_hat(phi2_hat, kx, ky, kz)
    elif fm_cfg.der_method == "FD":
        psi2 = gradient_fd(my_ifft(phi2_hat, cte.INV_L3), cte.N, cte.L)
    else:
        raise ValueError("Unknown derivative method")

    return psi2


def get_psi_lpt2(delta_hat, cte: Constants, fm_cfg: FMConfig):

    D1, D2 = cte.D1, cte.D2
    f1, f2 = cte.f1, cte.f2

    psi1 = get_psi_lpt1(delta_hat, cte, fm_cfg)
    psi2 = get_psi_o2(psi1, cte, fm_cfg)

    psi = psi1 * D1 + psi2 * D2

    if fm_cfg.rsd and (fm_cfg.lpt_method == "2LPT"):
        if fm_cfg.rsd_type == "Radial":
            if fm_cfg.der_method == "FR":
                psi = my_ifft_on_vec(psi, cte.INV_L3)
                psi1 = my_ifft_on_vec(psi1, cte.INV_L3)
                psi2 = my_ifft_on_vec(psi2, cte.INV_L3)

            dot1 = jnp.sum(psi1 * fm_cfg.r_vec_over_r, axis=0)
            dot2 = jnp.sum(psi2 * fm_cfg.r_vec_over_r, axis=0)
            psi += cte.D1 * cte.f1 * dot1 * fm_cfg.r_vec_over_r
            psi += cte.D2 * cte.f2 * dot2 * fm_cfg.r_vec_over_r

        else:
            psi_x = D1 * (1 + f1) * psi1[0, ...] + D2 * (1 + f2) * psi2[0, ...]  # type: ignore
            psi = psi.at[0, ...].set(psi_x)

    return psi


def get_psi_sc(delta_in, cte: Constants, fm_cfg: FMConfig):

    N, L, L3, INV_L3 = cte.N, cte.L, cte.L3, cte.INV_L3
    kx, ky, kz = cte.kx, cte.ky, cte.kz
    D1 = cte.D1

    delta_in *= D1
    threshold = 3 / 2

    sharpness = 1.0  # Controls how sharp the transition is
    bool_arr_soft = jax.nn.sigmoid(sharpness * (threshold - delta_in))
    thing = (
        jnp.sqrt(jnp.maximum(1 - 2 / 3 * delta_in * bool_arr_soft, 0)) * bool_arr_soft
    )

    # if fm_cfg.muscle:  # one iteration
    #     RES = L / N
    #     delta_in_hat = my_fft(delta_in, L3)
    #     r = (2**0) * RES
    #     one_over_var_k = r**2
    #     filt_k = symmetric_gaussian_nonorm(fm_cfg.k_sq, one_over_var_k)
    #     delta_smooth = my_ifft(filt_k * delta_in_hat, INV_L3)
    #     bool_arr_smooth = delta_smooth < 3 / 2
    #     bool_arr &= bool_arr_smooth

    div_psi = 3 * thing - 3

    div_psi = my_fft(div_psi, L3)
    phi_hat_sc = -div_psi * fm_cfg.one_over_k_sq

    if fm_cfg.der_method == "FR":
        psi = gradient_hat(phi_hat_sc, kx, ky, kz)
    if fm_cfg.der_method == "FD":
        psi = gradient_fd(my_ifft(phi_hat_sc, cte.INV_L3), cte.N, cte.L)

    return psi  # type: ignore


def get_psi_alpt(delta_in_hat, cte: Constants, fm_cfg: FMConfig):

    INV_L3 = cte.INV_L3

    delta_in = my_ifft(delta_in_hat, INV_L3)

    psi_sc = get_psi_sc(delta_in, cte, fm_cfg)
    psi_lpt2 = get_psi_lpt2(delta_in_hat, cte, fm_cfg)

    one_over_var_k = fm_cfg.r_s**2
    k_sq = get_k_sq(cte.N, cte.L)
    filt_k = symmetric_gaussian_nonorm(k_sq, one_over_var_k)

    if fm_cfg.der_method == "FD":
        psi_lpt2 = my_fft_on_vec(psi_lpt2, cte.L3)
        psi_sc = my_fft_on_vec(psi_sc, cte.L3)

    psi = filt_k * psi_lpt2 + (1 - filt_k) * psi_sc

    if fm_cfg.rsd:

        if fm_cfg.rsd_type == "Radial":

            psi = my_ifft_on_vec(psi, cte.INV_L3)

            psi_lpt1 = cte.D1 * gradient_hat(
                delta_in_hat * fm_cfg.one_over_k_sq, cte.kx, cte.ky, cte.kz
            )
            psi_lpt1 = my_ifft_on_vec(psi_lpt1, cte.INV_L3)
            psi_dif = psi - psi_lpt1

            dot1 = jnp.sum(psi_lpt1 * fm_cfg.r_vec_over_r, axis=0)
            dot2 = jnp.sum(psi_dif * fm_cfg.r_vec_over_r, axis=0)

            psi += cte.f1 * dot1 * fm_cfg.r_vec_over_r
            psi += cte.f2 * dot2 * fm_cfg.r_vec_over_r

        else:
            psi_lpt1_x = 1j * cte.kx * fm_cfg.one_over_k_sq * delta_in_hat * cte.D1
            psi = psi.at[0, ...].set(
                psi[0, ...] + cte.f1 * psi_lpt1_x + cte.f2 * (psi[0, ...] - psi_lpt1_x)
            )

    return psi
