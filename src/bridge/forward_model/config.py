import inspect
import json
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .cosmo import (
    compute_or_load_D1_D2,
    compute_or_load_f1_f2,
    compute_or_load_pow_spec_cube,
)
from .fourier import (
    get_k_1D,
    get_k_rfft_1D,
    get_one_over_k_sq,
)


@dataclass
class Constants:
    N: int
    L: float
    Z_I: float
    Z_F: float

    SHAPE: tuple = field(init=False)
    R: float = field(init=False)
    INV_R: float = field(init=False)
    L3: float = field(init=False)
    INV_L3: float = field(init=False)
    N3: float = field(init=False)
    INV_N3: float = field(init=False)
    R3: float = field(init=False)
    INV_R3: float = field(init=False)

    D1: jnp.ndarray = field(init=False)
    D2: jnp.ndarray = field(init=False)
    f1: jnp.ndarray = field(init=False)
    f2: jnp.ndarray = field(init=False)

    kx: jnp.ndarray = field(init=False)
    ky: jnp.ndarray = field(init=False)
    kz: jnp.ndarray = field(init=False)

    def __post_init__(self):
        self.SHAPE = (self.N,) * 3
        self.R = self.L / self.N
        self.INV_R = 1 / self.R
        self.L3 = jnp.exp(3 * jnp.log(self.L))  # type: ignore
        self.INV_L3 = jnp.exp(-3 * jnp.log(self.L))  # type: ignore
        self.N3 = self.N**3
        self.INV_N3 = jnp.exp(-3 * jnp.log(self.N))  # type: ignore
        self.R3 = jnp.exp(-3 * jnp.log(self.N) + 3 * jnp.log(self.L))  # type: ignore
        self.INV_R3 = jnp.exp(3 * jnp.log(self.N) - 3 * jnp.log(self.L))  # type: ignore

        self.D1, self.D2 = compute_or_load_D1_D2(self.Z_I, self.Z_F)
        self.f1, self.f2 = compute_or_load_f1_f2(self.Z_F)

        self.kx = get_k_1D(self.N, self.L)[:, None, None]
        self.ky = get_k_1D(self.N, self.L)[None, :, None]
        self.kz = get_k_rfft_1D(self.N, self.L)[None, None, :]


@dataclass
class FMConfig:
    N: int
    L: float
    Z_I: float
    Z_F: float

    pk_with_sinc: bool = True
    where_lagr_particles: str = "CENTER"
    psi_trlilinear: bool = False

    # INPUT KIND
    input_kind: str = "WN"  # CWN

    # MESH SETTINGS
    der_method: str = "FR"
    pm_method: str = "CIC"

    # LPT SETTINGS
    lpt_method: str = "1LPT"
    rsd: bool = False
    rsd_type: str = "Radial"
    r_s = 4

    # BIAS SETTINGS
    det_bias_model: str = "Linear"
    stoch_bias_model: str = "Poisson"
    N_TR: int = None  # type: ignore

    # CWEB SETTINGS
    cweb: None | str = None
    lambda_th: float = 0.05
    soft_cweb: bool = False
    cweb_sharpness: float = 10.0

    def __post_init__(self):

        self._set_aux_cts()
        self._set_heavy_cts()
        self._check_N_TR()
        self._set_num_parameters()

    def _set_aux_cts(self):
        self.psi_has_hat = True if self.der_method == "FR" else False
        self.n_mesh = self.N

    def _check_N_TR(self):

        if self.N_TR is None:
            print("N_TR is not set. No bias in forward model.")
        else:
            self.N_TR = jnp.array(jnp.atleast_1d(self.N_TR))  # type: ignore

            if self.N_TR.shape == (1,):  # type: ignore
                self.global_N_TR = True
            else:
                self.global_N_TR = False
                if self.N_TR.shape != (self.n_regions):  # type: ignore
                    raise ValueError(
                        "Shape of N_TR should be equal to number of regions!"
                    )

    def _set_num_parameters(self) -> int:  # type: ignore
        base_params = {
            "Linear": 1,
            "PowerLaw": 1,
            "LinearPowerLaw": 2,
            "HighPassPowerLaw": 3,
            "LowHighPassPowerLaw": 5,
        }[self.det_bias_model]

        if self.stoch_bias_model == "NegBinomial":
            base_params += 1

        multiplier = {
            None: 1,
            "PhiWeb": 4,
            "DeltaWeb": 4,
            "PhiDeltaWeb": 16,
        }[self.cweb]

        self.n_regions = multiplier
        self.num_bias_parameters = multiplier * base_params

    def _set_heavy_cts(self):

        self.one_over_k_sq = get_one_over_k_sq(self.N, self.L)
        self.pow_spec = compute_or_load_pow_spec_cube(
            self.N, self.L, self.Z_I, sinc=self.pk_with_sinc
        )

        if self.rsd and self.rsd_type == "Radial":
            axis = (jnp.arange(self.N) + 0.5) * self.L / self.N - self.L / 2.0
            rx, ry, rz = jnp.meshgrid(axis, axis, axis, indexing="ij")
            r = jnp.sqrt(rx * rx + ry * ry + rz * rz)

            eps = 1e-10
            rinv = jnp.where(r > eps, 1 / r, 0.0)
            self.r_vec_over_r = jnp.array([rx, ry, rz]) * rinv

    def to_dict(self):
        """Convert only initialization arguments to a dictionary, excluding computed fields."""
        init_args = inspect.signature(self.__init__).parameters.keys()
        data = {key: getattr(self, key) for key in init_args if hasattr(self, key)}
        data = serialize_dic(data)
        return data

    def save(self, filename):
        """Save the instance to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)


def serialize_dic(obj):
    if isinstance(obj, np.integer | np.floating):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, jax.Array):
        return np.array(obj).tolist()
    elif isinstance(obj, dict):
        return {k: serialize_dic(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_dic(v) for v in obj]
    else:
        return obj


def load_fm_config(filename):
    """Load an FMConfig instance from a JSON file, ensuring large arrays are recomputed."""
    with open(filename) as f:
        data = json.load(f)
    for key, value in data.items():
        if isinstance(value, list) and all(isinstance(v, int | float) for v in value):
            data[key] = jnp.array(value)
    instance = FMConfig(**data)
    return instance
