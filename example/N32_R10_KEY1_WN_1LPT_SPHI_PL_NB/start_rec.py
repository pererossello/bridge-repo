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

from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, HMC, init_to_value
from numpyro.infer.util import initialize_model
from numpyro.distributions.transforms import biject_to
from numpyro import handlers

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

IDX = 0
CHAIN = 0
savefold = os.path.join(this_file_dir, f"{IDX:02d}CH{CHAIN:02d}")
make_and_check_dir(savefold)

fm_cfg_path = os.path.join(this_file_dir, "data/fm_config.json")
fm_cfg = load_fm_config(fm_cfg_path)
cte = Constants(fm_cfg.N, fm_cfg.L, fm_cfg.Z_I, fm_cfg.Z_F)
fmodel = FModel(fm_cfg)

mock_path = os.path.join(this_file_dir, "data/mock.hdf5")
with h5py.File(mock_path, "r") as f:
    din_ref = jnp.array(f["din"][:])  # type: ignore
    n_tr_data = jnp.array(f["n_tr_data"][:])  # type: ignore

delta_wf, q_wf = apply_wiener_filter(n_tr_data, cte, return_kind="WN")

# ######################

fig, axs = compare_fields(
    din_ref,
    delta_wf,
    n_tr_data,
    n_tr_data * 0.00001,
    cte.L,
    n_bins=50,
    bispec=True,
    plot_maps=True,
    xlog=True,
    pk_rat_lim=(0.5, 1.5),
)
fig_path = os.path.join(savefold, "fig_wf.png")
print("Plotting...", end=" ")
fig.savefig(fig_path, bbox_inches="tight")
print("Done")

# ######################


init_bias_params_constrained = {
    "alpha": jnp.array([1.0] * 4),
    "beta": jnp.array([10.0] * 4),
}
positive_transform = biject_to(constraints.positive)
init_params = {"q": q_wf}
for k, v in init_bias_params_constrained.items():
    v_unconstr = positive_transform.inv(v)
    init_params[k] = v_unconstr

model = fmodel.build_model()

max_tree_depth = 5
adapt_mass_matrix = True
dense_mass = False
adapt_step_size = True
init_step_size = 1e-3

num_warmup = 100
num_samples = 1
thinning = 1

sample_seed = 1
key = jax.random.PRNGKey(sample_seed)

kernel = NUTS(
    model,
    dense_mass=dense_mass,
    adapt_mass_matrix=adapt_mass_matrix,
    max_tree_depth=max_tree_depth,
    adapt_step_size=adapt_step_size,
    step_size=init_step_size,
)
mcmc = MCMC(
    kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    thinning=thinning,
    progress_bar=True,
)
mcmc.run(
    key,
    data=n_tr_data,
    init_params=init_params,
)


posterior = mcmc.get_samples()
first = {name: arr[0] for name, arr in mcmc.get_samples().items()}
last = {name: arr[-1] for name, arr in mcmc.get_samples().items()}

print(first["alpha"])
print(last["alpha"])
print(first["beta"])
print(last["beta"])

mcmc_last = mcmc._last_state
last_step_size = mcmc_last.adapt_state.step_size  # type: ignore
last_inv_mass_matrix = mcmc_last.adapt_state.inverse_mass_matrix  # type: ignore
last_num_steps = mcmc_last.num_steps  # type: ignore
mean_accepted_prob = mcmc_last.mean_accept_prob  # type: ignore
last_trajectory_length = mcmc_last.trajectory_length  # type: ignore

cfg = {
    "hmc_cfg": {
        "max_tree_depth": max_tree_depth,
        "adapt_mass_matrix": adapt_mass_matrix,
        "adapt_step_size": adapt_step_size,
        "dense_mass": dense_mass,
    },
    "seed": sample_seed,
    "num_warmup": num_warmup,
    "num_samples": num_samples,
    "thinning": thinning,
    "sampling_cfg": {
        "step_size": last_step_size,
        "num_steps": last_num_steps,
        "mean_accepted_prob": mean_accepted_prob,
        "trajectory_length": last_trajectory_length,
        "inv_mass_matrix": last_inv_mass_matrix,
    },
}

cfg_path = os.path.join(savefold, "sample_cfg.json")
with open(cfg_path, "wb") as fp:
    pickle.dump(cfg, fp)

sample_path = os.path.join(savefold, "sample.hdf5")
with h5py.File(sample_path, "w") as f:
    grp = f.create_group("params")
    for name, value in last.items():
        grp.create_dataset(name, data=value)

bias_params = {name: value for name, value in last.items() if name != "q"}
text_lines = [
    f"{name}: [{', '.join(f'{v:.3f}' for v in value.ravel())}]"
    for name, value in bias_params.items()
]

txt_file_path = os.path.join(savefold, "last_params.txt")
with open(txt_file_path, "w") as f:
    text = "\n".join(text_lines)
    f.write(text)

q_fit = last["q"]
delta_in_fit = my_ifft(fmodel.delta_in(q_fit), cte.INV_L3)
delta_lpt_fit = fmodel.delta_lpt(q_fit)
cweb_arr = fmodel.cweb(delta_lpt_fit)

get_n_tr_mean = fmodel.n_tr_mean()
n_tr_mean_fit = get_n_tr_mean(delta_lpt_fit, bias_params, cweb_arr)

key_sample = jax.random.PRNGKey(101)
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

fig_path = os.path.join(savefold, "fig.png")
print("Plotting...", end=" ")
fig.savefig(fig_path, bbox_inches="tight")
print("Done")
