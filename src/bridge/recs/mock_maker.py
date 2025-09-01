import json
import os
import shutil

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ..forward_model.bias import manage_params
from ..forward_model.config import Constants, FMConfig, serialize_dic
from ..forward_model.fmodel import FModel
from ..forward_model.fourier import my_ifft
from ..forward_model.ics import gen_input_arr
from ..forward_model.plot_utils import compare_pow_spec, plot_cubes


def make_mock(
    fm_cfg: FMConfig, savedir, seed_int_q, params, seed_int_n_tr, saveplot=False
):

    cte = Constants(fm_cfg.N, fm_cfg.L, fm_cfg.Z_I, fm_cfg.Z_F)
    fmodel = FModel(fm_cfg)

    key_q = jax.random.PRNGKey(seed_int_q)
    key_n_tr = jax.random.PRNGKey(seed_int_n_tr)

    q_ref = gen_input_arr(key_q, cte, fm_cfg)
    din = my_ifft(fmodel.delta_in(q_ref), cte.INV_L3)
    dev = fmodel.delta_lpt(q_ref)
    cweb_arr = fmodel.cweb(dev)

    get_n_tr_mean = fmodel.n_tr_mean()
    params = manage_params(params)
    n_tr_mean = get_n_tr_mean(dev, params, cweb_arr)

    n_tr_data = fmodel.sample_n_tr(n_tr_mean, key_n_tr, params=params, cweb=cweb_arr)

    savedir = os.path.join(savedir, "data")
    if os.path.exists(savedir):
        shutil.rmtree(savedir)
    os.makedirs(savedir)

    filename = "mock.hdf5"
    datapath = os.path.join(savedir, filename)

    with h5py.File(datapath, "w") as h5_file:

        header = h5_file.create_group("Header")

        header.attrs["N"] = cte.N
        header.attrs["L"] = cte.L
        header.attrs["R"] = cte.R
        header.attrs["Z_I"] = cte.Z_I
        header.attrs["Z_F"] = cte.Z_F

        header.attrs["seed_int_q"] = seed_int_q
        header.attrs["seed_int_n_tr"] = seed_int_n_tr

        h5_file.create_dataset("input", data=q_ref, dtype="float32")
        h5_file.create_dataset("din", data=din, dtype="float32")
        h5_file.create_dataset("n_tr_data", data=n_tr_data, dtype="float32")
        h5_file.create_dataset("n_tr_mean", data=n_tr_mean, dtype="float32")

    filename = "fm_config.json"
    fmconfig_path = os.path.join(savedir, filename)
    fm_cfg.save(fmconfig_path)

    filename = "params.json"
    params_path = os.path.join(savedir, filename)
    params_serialized = serialize_dic(params)
    with open(params_path, "w") as f:
        json.dump(params_serialized, f)

    if saveplot:
        savedir = os.path.join(savedir, "plots")
        if os.path.exists(savedir):
            shutil.rmtree(savedir)
        os.makedirs(savedir)

        vlim_din = 5e-2

        lim_dev = 10 * jnp.std(dev)  # type: ignore
        vlim_dev = (-1, lim_dev)

        lim_n_tr = jnp.mean(n_tr_data) + 3 * jnp.std(n_tr_data)  # type: ignore
        vlim_n_tr = (0, lim_n_tr)

        titles = [f"{vlim_din:0.2e}", f"({vlim_dev[0]}, {vlim_dev[1]:0.2f})"] + [
            f"({vlim_n_tr[0]}, {vlim_n_tr[1]:0.2f})"
        ] * 2

        cmaps = ["seismic_r", "gnuplot"] + ["gnuplot2"] * 2
        vlims = [vlim_din, vlim_dev] + [vlim_n_tr] * 2
        fig, ax = plot_cubes(  # type: ignore
            [din, dev, n_tr_mean, n_tr_data],
            cmap=cmaps,  # type: ignore
            vlim=vlims,
            idx=cte.N // 2,
            axis=2,
            figsize=7,
            titles=titles,
        )
        fig.savefig(os.path.join(savedir, "ref_data_plot.png"), bbox_inches="tight")
        plt.close()

        fig, axs = compare_pow_spec(
            [din],
            cte.L,
            n_bins=40,
            pk_cube=fm_cfg.pow_spec,
            xlog=True,
            labels=["pk theory", "din"],
        )
        fig.savefig(os.path.join(savedir, "din_pk.png"), bbox_inches="tight")
        plt.close()

    return
