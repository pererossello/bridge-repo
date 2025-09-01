import jax.numpy as jnp

fourier_norm = "forward"


def my_fft(f, L3):
    f_k = L3 * jnp.fft.rfftn(f, norm=fourier_norm)
    return f_k


def my_ifft(f_k, INV_L3):
    f = INV_L3 * jnp.fft.irfftn(f_k, norm=fourier_norm)
    return f


def my_fft_on_vec(f_vec, L3):
    f_k = L3 * jnp.fft.rfftn(f_vec, norm=fourier_norm, axes=(1, 2, 3))
    return f_k


def my_ifft_on_vec(f_vec_k, INV_L3):
    f = INV_L3 * jnp.fft.irfftn(f_vec_k, norm=fourier_norm, axes=(1, 2, 3))
    return f


def get_k_1D(N, L):
    k_1D = jnp.fft.fftfreq(N) * N * 2 * jnp.pi / L
    return k_1D


def get_k_rfft_1D(N, L):
    k_1D = jnp.fft.rfftfreq(N) * N * 2 * jnp.pi / L
    return k_1D


def get_k(N, L):
    k_1D = get_k_1D(N, L)
    k_r_1D = get_k_rfft_1D(N, L)
    k_list = [k_1D] * 2 + [k_r_1D]
    k = jnp.linalg.norm(jnp.array(jnp.meshgrid(*k_list, indexing="ij")), axis=0)
    return k


def get_k_sq(N, L):
    return jnp.square(get_k(N, L))


def get_one_over_k_sq(N, L):
    eps = 1e-10
    k_sq = get_k_sq(N, L)
    one_over_k_sq = jnp.where(k_sq > eps, 1 / jnp.maximum(k_sq, eps), 0.0)
    return one_over_k_sq


def get_one_over_k(N, L):
    k = get_k(N, L)
    inv_k = jnp.where(k == 0, 1.0, 1 / k)
    return inv_k


def get_k_nyq(N, L):
    return jnp.pi * N / L


def get_win_ngp_hat(N):
    x, y, z = jnp.meshgrid(
        jnp.fft.fftfreq(N), jnp.fft.fftfreq(N), jnp.fft.rfftfreq(N), indexing="ij"
    )
    phase = jnp.exp(1j * jnp.pi * (x + y + z))
    return jnp.sinc(x) * jnp.sinc(y) * jnp.sinc(z) * phase


def get_win_cic_hat(N):
    return jnp.square(get_win_ngp_hat(N))


def get_win_ngp_hat_1D(N):
    x_r_1D = jnp.fft.rfftfreq(N)
    return jnp.exp(1j * jnp.pi * x_r_1D) * jnp.sinc(x_r_1D)


def get_win_ngp_hat_1D_2(k, N, L):
    x_r_1D = k / (N * 2 * jnp.pi / L)
    return jnp.exp(1j * jnp.pi * x_r_1D) * jnp.sinc(x_r_1D)
