# ruff: noqa

import jax
import jax.numpy as jnp

from .fourier import get_k, get_k_1D, get_k_rfft_1D, my_fft, my_ifft


def get_pdf(n_tr, bin_edges):
    hist, _ = jnp.histogram(n_tr, bins=bin_edges)
    return hist


def get_radial_distro(cube_domain, cube_array, bin_range, n_bins):

    min_val, max_val = bin_range
    bin_edges = jnp.linspace(min_val, max_val, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    array = cube_array.flatten()
    domain = cube_domain.flatten()

    counts_in_bins, _ = jnp.histogram(domain, bins=bin_edges)
    arr_1D, _ = jnp.histogram(domain, bins=bin_edges, weights=array)

    arr_1D = arr_1D / counts_in_bins

    return bin_centers, arr_1D


def get_pow_spec_1D(delta, L, n_bins, sphere_only=False):

    N = delta.shape[0]

    L3 = jnp.exp(3 * jnp.log(L))
    INV_L3 = jnp.exp(-3 * jnp.log(L))

    fact = 1 / jnp.sqrt(3) if sphere_only else 1.0

    delta_hat = my_fft(delta, L3)
    pk = jnp.abs(delta_hat * jnp.conj(delta_hat)) * INV_L3

    k = get_k(N, L)
    k_max = k.max() * fact

    return get_radial_distro(k, pk, bin_range=(0.0, k_max), n_bins=n_bins)


def get_pow_spec_quadrupole(
    delta, L, n_bins, sphere_only=False, direction="PlaneParallel"
):
    N = delta.shape[0]
    L3 = L**3
    INV_L3 = L3**-1

    fact = 1 / jnp.sqrt(3) if sphere_only else 1.0

    delta_hat = my_fft(delta, L3)

    k = get_k(N, L)
    k_max = k.max() * fact

    if direction == "PlaneParallel":  # LOS = +x
        kx = get_k_1D(N, L)[:, None, None]
        mu = kx / (k + 1e-20)
        Leg2 = 0.5 * (3.0 * mu**2 - 1.0)
        pk_quad = 5.0 * jnp.abs(delta_hat) ** 2 * Leg2 * INV_L3

    else:  # 'Radial'
        axis = (jnp.arange(N) + 0.5) * L / N - L / 2.0
        rx, ry, rz = jnp.meshgrid(axis, axis, axis, indexing="ij")
        r = jnp.sqrt(rx * rx + ry * ry + rz * rz)
        rinv = jnp.where(r > 0.0, 1.0 / r, 0.0)
        rhat = jnp.array([rx, ry, rz]) * rinv  # shape (3,N,N,N)

        # weighted fields  r̂_i r̂_j δ(x)
        w_xx = my_fft(rhat[0] * rhat[0] * delta, L3)
        w_yy = my_fft(rhat[1] * rhat[1] * delta, L3)
        w_zz = my_fft(rhat[2] * rhat[2] * delta, L3)
        w_xy = my_fft(rhat[0] * rhat[1] * delta, L3)
        w_xz = my_fft(rhat[0] * rhat[2] * delta, L3)
        w_yz = my_fft(rhat[1] * rhat[2] * delta, L3)

        # k-unit vectors
        k1d = get_k_1D(N, L)
        k1d_rfft = get_k_rfft_1D(N, L)
        kx, ky, kz = jnp.meshgrid(k1d, k1d, k1d_rfft, indexing="ij")
        kmag = jnp.sqrt(kx * kx + ky * ky + kz * kz) + 1e-20
        khat = [kx / kmag, ky / kmag, kz / kmag]

        # contraction  k̂_i k̂_j w_ij
        S = (
            khat[0] * khat[0] * w_xx
            + khat[1] * khat[1] * w_yy
            + khat[2] * khat[2] * w_zz
            + 2.0
            * (
                khat[0] * khat[1] * w_xy
                + khat[0] * khat[2] * w_xz
                + khat[1] * khat[2] * w_yz
            )
        )

        delta2_hat = 1.5 * S - 0.5 * delta_hat
        pk_quad = 5.0 * jnp.real(delta_hat * jnp.conj(delta2_hat)) * INV_L3

    pk = jnp.abs(delta_hat) ** 2 * INV_L3
    _, pk_1D = get_radial_distro(k, pk, (0.0, k_max), n_bins)
    k_c, pk_quad_1D = get_radial_distro(k, pk_quad, (0.0, k_max), n_bins)

    return k_c, pk_quad_1D / (pk_1D + 1e-10)


def get_cross_power_spec_1D(delta_1, delta_2, L, n_bins, sphere_only=False):

    N = delta_1.shape[0]

    fact = 1 / jnp.sqrt(3) if sphere_only else 1.0
    k = get_k(N, L)
    k_max = k.max() * fact

    L3 = jnp.exp(3 * jnp.log(L))
    INV_L3 = jnp.exp(-3 * jnp.log(L))

    delta_1_hat = my_fft(delta_1, L3)
    delta_2_hat = my_fft(delta_2, L3)

    pk1 = jnp.abs(delta_1_hat) ** 2 * INV_L3
    pk2 = jnp.abs(delta_2_hat) ** 2 * INV_L3

    k_c, pk1 = get_radial_distro(k, pk1, bin_range=(0.0, k_max), n_bins=n_bins)
    _, pk2 = get_radial_distro(k, pk2, bin_range=(0.0, k_max), n_bins=n_bins)
    denominator = jnp.sqrt(pk1 * pk2)

    numerator = (delta_1_hat * jnp.conj(delta_2_hat)) * INV_L3
    k_c, numerator = get_radial_distro(
        k, numerator, bin_range=(0.0, k_max), n_bins=n_bins
    )
    numerator = numerator.real

    cross = numerator / denominator

    return k_c, cross


def get_reduced_bispectrum(delta, L, k1, k2, thetas):

    L3 = jnp.exp(3 * jnp.log(L))
    INV_L3 = jnp.exp(-3 * jnp.log(L))

    delta_hat = my_fft(delta, L3)

    N = delta_hat.shape[0]
    NTHETA = thetas.shape[0]

    k3s = jnp.sqrt((k2 * jnp.sin(thetas)) ** 2 + (k2 * jnp.cos(thetas) + k1) ** 2)

    pow_spec = (delta_hat * jnp.conj(delta_hat)).real / L**3
    k = get_k(N, L)
    k_half_width = 2 * jnp.pi / L

    def pk_at_k(k_val):
        k_weights = jnp.where(
            (k < k_val + k_half_width) & (k > k_val - k_half_width), 1.0, 0.0
        )
        total_weight = jnp.sum(k_weights)
        pow_spec_val = jnp.sum(pow_spec * k_weights) / total_weight
        return pow_spec_val

    pk1 = pk_at_k(k1)
    pk2 = pk_at_k(k2)

    # Precompute pk3s
    pk3s = jax.vmap(pk_at_k)(k3s)

    k1_mask = (k < k1 + k_half_width) & (k > k1 - k_half_width)
    k2_mask = (k < k2 + k_half_width) & (k > k2 - k_half_width)

    delta_hat_k1 = delta_hat * k1_mask
    delta_k1 = my_ifft(delta_hat_k1, INV_L3)
    i_k1 = my_ifft(k1_mask, INV_L3)

    delta_hat_k2 = delta_hat * k2_mask
    delta_k2 = my_ifft(delta_hat_k2, INV_L3)
    i_k2 = my_ifft(k2_mask, INV_L3)

    def body_func(carry, i):
        red_bispec = carry

        k3 = k3s[i]
        pk3 = pk3s[i]

        k3_mask = (k < k3 + k_half_width) & (k > k3 - k_half_width)
        delta_hat_k3 = delta_hat * k3_mask
        delta_k3 = my_ifft(delta_hat_k3, INV_L3)
        i_k3 = my_ifft(k3_mask, INV_L3)

        bispec_val = jnp.nansum(delta_k1 * delta_k2 * delta_k3)
        denominator = pk1 * pk2 + pk1 * pk3 + pk2 * pk3
        NTRIANG = jnp.sum(i_k1 * i_k2 * i_k3)

        red_bispec_val = bispec_val / denominator
        red_bispec_val /= NTRIANG
        red_bispec_val *= INV_L3

        red_bispec = red_bispec.at[i].set(red_bispec_val)

        return red_bispec, None  # No need to accumulate outputs

    # Initialize the carry
    red_bispec_init = jnp.zeros(NTHETA)

    # Set up the indices over which to loop
    indices = jnp.arange(NTHETA)

    # Use lax.scan to perform the loop
    red_bispec_final, _ = jax.lax.scan(body_func, red_bispec_init, indices)

    return red_bispec_final
