import h5py
import jax.numpy as jnp

from ..forward_model.config import Constants, FMConfig
from ..forward_model.fourier import my_ifft
from ..forward_model.ics import get_delta_in


def get_gellmanrubin_test_factor(
    rec_paths,
    cte: Constants,
    fm_cfg: FMConfig,
    hat=True,
    for_q=False,
):
    """ """

    N = cte.N

    shape = (N,) * 3

    if hat and not for_q:
        shape = (2, N, N, N // 2 + 1)  # type: ignore

    m = len(rec_paths)

    with h5py.File(rec_paths[0], "r") as f:
        n = f["Header"].attrs["M"]

    print(m, n)

    chain_mean = jnp.zeros((m,) + shape)
    chain_var = jnp.zeros((m,) + shape)

    for i, path in enumerate(rec_paths):
        acc = jnp.zeros(shape)
        with h5py.File(path, "r") as f:
            for j in range(n):  # type: ignore
                grp_name = f"{j:03d}"
                q = jnp.array(f[grp_name]["q"][:])  # type: ignore
                if not for_q:
                    if not hat:
                        q = my_ifft(get_delta_in(q, cte, fm_cfg), cte.INV_L3)
                    else:
                        q = get_delta_in(q, cte, fm_cfg)  # has hat
                        q = jnp.array([q.real, q.imag])  # type: ignore
                acc += q

        chain_mean = chain_mean.at[i].set(acc / n)

    for i, path in enumerate(rec_paths):
        acc = jnp.zeros(shape)
        with h5py.File(path, "r") as f:
            for j in range(n):  # type: ignore
                grp_name = f"{j:03d}"
                q = jnp.array(f[grp_name]["q"][:])  # type: ignore

                if not for_q:
                    if not hat:
                        q = my_ifft(get_delta_in(q, cte, fm_cfg), cte.INV_L3)
                    else:
                        q = get_delta_in(q, cte, fm_cfg)  # has hat
                        q = jnp.array([q.real, q.imag])  # type: ignore

                acc += (q - chain_mean[i]) ** 2
        chain_var = chain_var.at[i].set(acc / (n - 1))  # type: ignore

    B = n * jnp.var(chain_mean, axis=0, ddof=1)
    W = jnp.mean(chain_var, axis=0)
    var_hat = ((n - 1) / n) * W + (1 / n) * B  # type: ignore
    eps = 1e-12
    R_hat = jnp.sqrt(var_hat / jnp.clip(W, eps))

    return R_hat
