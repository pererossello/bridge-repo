import jax.numpy as jnp


def der_i_5ps(array, axis, N, L):
    """
    Five point stencil derivative
    """

    fact = 1 / 12 * N / L

    arr_ip2 = jnp.roll(array, shift=-2, axis=axis)
    arr_ip1 = jnp.roll(array, shift=-1, axis=axis)
    arr_im1 = jnp.roll(array, shift=+1, axis=axis)
    arr_im2 = jnp.roll(array, shift=+2, axis=axis)

    result = fact * (arr_im2 - arr_ip2 + 8 * (arr_ip1 - arr_im1))

    return result


def gradient_hat(array_hat, kx, ky, kz):
    return 1j * jnp.asarray([kx * array_hat, ky * array_hat, kz * array_hat])


def gradient_fd(array, N, L):
    arr_dx = der_i_5ps(array, 0, N, L)
    arr_dy = der_i_5ps(array, 1, N, L)
    arr_dz = der_i_5ps(array, 2, N, L)
    return jnp.asarray([arr_dx, arr_dy, arr_dz])


def div_hat(array_vec, kx, ky, kz):
    divergence_hat = 1j * (array_vec[0] * kx + array_vec[1] * ky + array_vec[2] * kz)
    return divergence_hat


def div_fd(array_vec, N, L):
    arr_dx = der_i_5ps(array_vec[0], 0, N, L)
    arr_dy = der_i_5ps(array_vec[1], 1, N, L)
    arr_dz = der_i_5ps(array_vec[2], 2, N, L)
    return arr_dx + arr_dy + arr_dz


def symmetric_gaussian_nonorm(r_sq, one_over_var):
    return jnp.exp(-0.5 * r_sq * one_over_var)
