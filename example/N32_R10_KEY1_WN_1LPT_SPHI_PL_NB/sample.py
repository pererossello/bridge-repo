import os
import sys
import json
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import h5py
import jax

jax.config.update("jax_enable_x64", True)
print(jax.devices())
import jax.numpy as jnp
import h5py
import matplotlib.pyplot as plt

from numpyro.infer import MCMC, NUTS, HMC, init_to_value
from numpyro.infer.util import initialize_model
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to

from bridge.forward_model.config import (
    Constants,
    FMConfig,
    load_fm_config,
    serialize_dic,
)
from bridge.forward_model.fmodel import FModel
from bridge.forward_model.fourier import my_ifft
from bridge.forward_model.bias import manage_params
from bridge.forward_model.plot_utils import plot_cubes, compare_pow_spec
from bridge.forward_model.utils import apply_wiener_filter

from bridge.recs.utils import make_and_check_dir
from bridge.recs.plot_utils import compare_fields

this_file_dir = os.path.abspath(os.path.dirname(__file__))

lastrecfold = os.path.join(this_file_dir, f"00CH00/CH01/")

savefold = os.path.join(lastrecfold, f"S01")
make_and_check_dir(savefold)

figfold = os.path.join(savefold, "figures")
make_and_check_dir(figfold)

fm_cfg_path = os.path.join(this_file_dir, "data/fm_config.json")
fm_cfg = load_fm_config(fm_cfg_path)
cte = Constants(fm_cfg.N, fm_cfg.L, fm_cfg.Z_I, fm_cfg.Z_F)
fmodel = FModel(fm_cfg)

mock_path = os.path.join(this_file_dir, "data/mock.hdf5")
with h5py.File(mock_path, "r") as f:
    din_ref = jnp.array(f["din"][:])  # type: ignore
    n_tr_data = jnp.array(f["n_tr_data"][:])  # type: ignore

lastrec_path = os.path.join(lastrecfold, "sample.hdf5")
with h5py.File(lastrec_path, "r") as f:
    q_in = f["params"]["q"][:]  # type: ignore
    alpha_in = f["params"]["alpha"][:]  # type: ignore
    beta_in = f["params"]["beta"][:]  # type: ignore

init_bias_params_constrained = {
    "alpha": alpha_in,
    "beta": beta_in,
}
positive_transform = biject_to(constraints.positive)
init_params = {"q": q_in}
for k, v in init_bias_params_constrained.items():
    v_unconstr = positive_transform.inv(v)
    init_params[k] = v_unconstr

cfg_path = os.path.join(lastrecfold, "sample_cfg.json")
with open(cfg_path, "rb") as sample_cfg:
    cfg_loaded = pickle.load(sample_cfg)

model = fmodel.build_model()
init_inv_mass_matrix = cfg_loaded["sampling_cfg"]["inv_mass_matrix"]
init_step_size = cfg_loaded["sampling_cfg"]["step_size"]

init_step_size = 8e-4

max_tree_depth = 3
adapt_mass_matrix = False
dense_mass = False
adapt_step_size = False

num_warmup = 0
chunks = 20

num_samples = 10
thinning = num_samples

sample_path = os.path.join(savefold, "sample.hdf5")
with h5py.File(sample_path, "w") as f:
    header = f.create_group("Header")
    header.attrs["M"] = chunks
    header.attrs["thinning"] = thinning

for i in range(chunks):

    sample_seed = i
    key = jax.random.PRNGKey(sample_seed)
    kernel = NUTS(
        model,
        dense_mass=dense_mass,
        adapt_mass_matrix=adapt_mass_matrix,
        max_tree_depth=max_tree_depth,
        adapt_step_size=adapt_step_size,
        inverse_mass_matrix=init_inv_mass_matrix,
        step_size=init_step_size,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        thinning=thinning,
        progress_bar=True,
    )

    mcmc.run(key, data=n_tr_data, init_params=init_params)
    posterior = mcmc.get_samples()
    last = {name: arr[-1] for name, arr in mcmc.get_samples().items()}

    grp_name = f"{i:03d}"
    with h5py.File(sample_path, "a") as f:
        grp = f.create_group(grp_name)
        for name, values in last.items():
            grp.create_dataset(name, data=values)

    with h5py.File(sample_path, "r") as f:
        q_in = f[grp_name]["q"][:]  # type: ignore
        alpha_in = f[grp_name]["alpha"][:]  # type: ignore
        beta_in = f[grp_name]["beta"][:]  # type: ignore

    init_bias_params_constrained = {
        "alpha": alpha_in,
        "beta": beta_in,
    }

    positive_transform = biject_to(constraints.positive)
    init_params = {"q": q_in}
    for k, v in init_bias_params_constrained.items():
        v_unconstr = positive_transform.inv(v)
        init_params[k] = v_unconstr

    bias_params = {name: value for name, value in last.items() if name != "q"}
    text_lines = [
        f"{name}: [{', '.join(f'{v:.3f}' for v in value.ravel())}]"
        for name, value in bias_params.items()
    ]

    do_what = "w" if i == 0 else "a"
    txt_file_path = os.path.join(savefold, "last_params.txt")
    with open(txt_file_path, do_what) as f:
        text = "\n".join(text_lines)
        f.write(text + "\n")
        print(text)

    q_fit = last["q"]
    delta_in_fit = my_ifft(fmodel.delta_in(q_fit), cte.INV_L3)
    delta_lpt_fit = fmodel.delta_lpt(q_fit)
    cweb_arr = fmodel.cweb(delta_lpt_fit)

    get_n_tr_mean = fmodel.n_tr_mean()
    n_tr_mean_fit = get_n_tr_mean(delta_lpt_fit, bias_params, cweb_arr)

    key_sample = jax.random.PRNGKey(i + 100)
    n_tr_fit = fmodel.sample_n_tr(
        n_tr_mean_fit, key_sample, params=bias_params, cweb=cweb_arr
    )

    fig, axs = compare_fields(
        din_ref,
        delta_in_fit,
        n_tr_data,
        n_tr_fit,
        cte.L,
        n_bins=50,
        bispec=True,
        plot_maps=True,
        xlog=True,
        pk_rat_lim=(0.5, 1.5),
    )

    fig_path = os.path.join(figfold, f"fig_{i:03d}.png")
    print("Plotting...", end=" ")
    fig.savefig(fig_path, bbox_inches="tight")
    print("Done")

cfg = {
    "hmc_cfg": {
        "max_tree_depth": max_tree_depth,
        "adapt_mass_matrix": adapt_mass_matrix,
        "adapt_step_size": adapt_step_size,
        "dense_mass": dense_mass,
    },
    "seed": sample_seed,  # type: ignore
    "num_warmup": num_warmup,
    "num_samples": num_samples,
    "thinning": thinning,
    "chunks": chunks,
    "sampling_cfg": {
        "step_size": init_step_size,
        "inv_mass_matrix": init_inv_mass_matrix,
    },
}

cfg_path = os.path.join(savefold, "sample_cfg.json")
with open(cfg_path, "wb") as fp:
    pickle.dump(cfg, fp)
