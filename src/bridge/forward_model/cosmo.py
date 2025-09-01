import os

import jax.numpy as jnp
import jax_cosmo

from .fourier import (
    get_k,
    get_k_rfft_1D,
    get_win_ngp_hat,
    get_win_ngp_hat_1D,
    get_win_ngp_hat_1D_2,
)

directory = os.path.dirname(os.path.abspath(__file__))
dir_cache = directory + "/cache/"

if not os.path.exists(dir_cache):
    os.makedirs(dir_cache)

h = 0.6736
Omega_c = 0.1200 / h**2
Omega_b = 0.02237 / h**2
Omega_m = Omega_c + Omega_b
Omega_k = 0.0
sigma8 = 0.8225
n_s = 0.9649
w0 = -1.0
wa = 0.0


def get_cosmo():
    Cosmology = jax_cosmo.Cosmology(
        Omega_c=Omega_c,
        Omega_b=Omega_b,
        h=h,
        sigma8=sigma8,
        n_s=n_s,
        Omega_k=Omega_k,
        w0=w0,
        wa=wa,
    )
    return Cosmology


def get_D1(z_i, z_f):

    Cosmology = get_cosmo()

    a_i = jnp.atleast_1d(1.0 / (1 + z_i))
    a_f = jnp.atleast_1d(1.0 / (1 + z_f))

    D1_i = jax_cosmo.background.growth_factor(Cosmology, a_i)
    D1_f = jax_cosmo.background.growth_factor(Cosmology, a_f)

    D1 = (D1_f / D1_i)[0]

    return D1


def get_Omega_m(z):
    Cosmology = get_cosmo()
    a = jnp.atleast_1d(1.0 / (1 + z))
    return jax_cosmo.background.Omega_m_a(Cosmology, a)[0]


def get_H(z):
    Cosmology = get_cosmo()
    a = jnp.atleast_1d(1.0 / (1 + z))
    return jax_cosmo.background.H(Cosmology, a)[0] * h


def get_a(z):
    return jnp.atleast_1d(1.0 / (1 + z))[0]


def get_linear_power_cube(N, L, z_i, sinc=False):
    Cosmology = get_cosmo()
    a = 1 / (1 + z_i)
    k = get_k(N, L)
    fact = jnp.abs(get_win_ngp_hat(N)) ** 2 if sinc else 1.0
    power_spectra = jax_cosmo.power.linear_matter_power(Cosmology, k, a=a)
    return power_spectra * fact


def get_linear_power_1D(N, L, z_i, sinc=False):
    Cosmology = get_cosmo()
    a = 1 / (1 + z_i)
    k = get_k_rfft_1D(N, L)
    fact = jnp.abs(get_win_ngp_hat_1D(N)) ** 2 if sinc else 1.0
    power_spectra = jax_cosmo.power.linear_matter_power(Cosmology, k, a=a)
    return k, power_spectra * fact


def get_linear_power_1D_2(k, N, L, z_i, sinc=False):
    Cosmology = get_cosmo()
    a = 1 / (1 + z_i)
    fact = jnp.abs(get_win_ngp_hat_1D_2(k, N, L)) ** 2 if sinc else 1.0
    power_spectra = jax_cosmo.power.linear_matter_power(Cosmology, k, a=a)
    return power_spectra * fact


def compute_or_load_pow_spec_cube(N, L, Z_I, sinc=False):
    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)

    sinc_str = "_sinc" if sinc else ""
    pow_spec_path = dir_cache + f"pow_spec{sinc_str}_{N}_{L:0.3f}_{Z_I:0.3f}.npy"
    if os.path.exists(pow_spec_path):
        pow_spec = jnp.array(jnp.load(pow_spec_path))
    else:
        pow_spec = get_linear_power_cube(N, L, Z_I, sinc=sinc)
        jnp.save(pow_spec_path, pow_spec)
    return pow_spec


def compute_or_load_pow_spec_1D(N, L, Z_I, sinc=False):
    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)

    sinc_str = "_sinc" if sinc else ""
    pow_spec_path = dir_cache + f"pow_spec_1D{sinc_str}_{N}_{L:0.3f}_{Z_I:0.3f}.npy"
    if os.path.exists(pow_spec_path):
        pow_spec = jnp.array(jnp.load(pow_spec_path))
    else:
        pow_spec = get_linear_power_1D(N, L, Z_I, sinc=sinc)
        jnp.save(pow_spec_path, pow_spec)
    return pow_spec


def compute_or_load_D1_D2(Z_I, Z_F):

    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)

    D1_path = dir_cache + f"D1_{Z_I:0.3f}_{Z_F:0.3f}.npy"
    D2_path = dir_cache + f"D2_{Z_I:0.3f}_{Z_F:0.3f}.npy"

    if os.path.exists(D1_path) and os.path.exists(D2_path):
        D1 = jnp.array(jnp.load(D1_path))
        D2 = jnp.array(jnp.load(D2_path))
    else:
        D1 = get_D1(Z_I, Z_F)
        Omega_m = get_Omega_m(Z_F)
        D2 = -3 / 7 * Omega_m ** (-1 / 143) * D1**2
        jnp.save(D1_path, jnp.array(D1))
        jnp.save(D2_path, jnp.array(D2))

    return D1, D2


def compute_or_load_f1_f2(Z_F):
    f1_path = dir_cache + f"f1_{Z_F:0.3f}.npy"
    f2_path = dir_cache + f"f2_{Z_F:0.3f}.npy"

    if os.path.exists(f1_path) and os.path.exists(f2_path):
        f1 = jnp.array(jnp.load(f1_path))
        f2 = jnp.array(jnp.load(f2_path))
    else:
        Omega_m = get_Omega_m(Z_F)
        f1 = Omega_m ** (5 / 9)
        f2 = 2 * Omega_m ** (6 / 11)
        jnp.save(f1_path, jnp.array(f1))
        jnp.save(f2_path, jnp.array(f2))

    return f1, f2
