import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


this_file_dir = os.path.abspath(os.path.dirname(__file__))
package_dir = os.path.join(this_file_dir, "..")
sys.path.insert(0, package_dir)

import jax

jax.config.update("jax_enable_x64", True)
print(jax.devices())
import jax.numpy as jnp
import matplotlib.pyplot as plt

from bridge.forward_model.fourier import my_ifft
from bridge.forward_model.config import Constants, FMConfig
from bridge.forward_model.fmodel import FModel
from bridge.forward_model.plot_utils import plot_cubes

from bridge.recs.mock_maker import make_mock

SEED_INT_Q = 1
SEED_INT_N_TR = 1

N, Z_I, Z_F = 32, 99, 0
R = 10
MND = 1e-2
L = R * N
N_TR = int(MND * L**3)
print(f"L: {L:0.2f} Mpc/h")
print(f"N_TR: {N_TR:0.2e}")

input_kind = "WN"
lpt_method = "1LPT"
rsd = False
# rsd_type = "Radial"
det_bias_model = "PowerLaw"
stoch_bias_model = "NegBinomial"
cweb = "PhiWeb"
soft_cweb = True
cweb_sharpness = 10

fm_cfg = FMConfig(
    N,
    L,
    Z_I,
    Z_F,
    N_TR=N_TR,
    input_kind=input_kind,
    lpt_method=lpt_method,
    rsd=rsd,
    # rsd_type=rsd_type,
    det_bias_model=det_bias_model,
    stoch_bias_model=stoch_bias_model,
    cweb=cweb,
    soft_cweb=soft_cweb,
    cweb_sharpness=cweb_sharpness,
)

params = {
    "alpha": jnp.array([1.05, 1.11, 1.23, 1.3]),
    "beta": jnp.array([7.1, 8.7, 8.1, 9.4]),
}


cweb_str = "S" if soft_cweb else "H"
if cweb == "PhiWeb":
    cweb_type_str = "PHI"
elif cweb == "PhiDeltaWeb":
    cweb_type_str = "PHI-D"
else:
    cweb_type_str = ""

bias_str = "PL" if det_bias_model == "PowerLaw" else "PLT"
distr_str = "NB" if stoch_bias_model == "NegBinomial" else "POS"

SAVEDIR = f"N{N}_R{R:0.0f}_KEY{SEED_INT_Q}_{input_kind}_{lpt_method}_{cweb_str}{cweb_type_str}_{bias_str}_{distr_str}"
SAVEDIR = os.path.join(this_file_dir, SAVEDIR)
print(SAVEDIR)

make_mock(fm_cfg, SAVEDIR, SEED_INT_Q, params, SEED_INT_N_TR, saveplot=True)
