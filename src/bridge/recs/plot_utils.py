import jax.numpy as jnp
import matplotlib.pyplot as plt

from ..forward_model.plot_utils import get_projection
from ..forward_model.stats import (
    get_cross_power_spec_1D,
    get_pow_spec_1D,
    get_reduced_bispectrum,
)


def compare_fields(
    din_ref,
    din,
    n_tr_ref,
    n_tr,
    L,
    n_bins=50,
    bispec=False,
    n_thetas=50,
    plot_maps=False,
    width=1,
    map_axis=2,
    xlog=False,
    sphere_only=False,
    pk_rat_lim=(0.8, 1.2),
    title=None,
    cmap_n_tr="gnuplot2",
):

    N = din_ref.shape[0]

    lw = 0.35
    lbs = 1.5

    hspace = 0.075
    nrows = 2

    if not plot_maps:
        x = 0
        if not bispec:
            ncols = 2
            fs, r = 7, 1.5
            fig, axs = plt.subplots(nrows, ncols, figsize=(r * fs, fs))
            plt.subplots_adjust(wspace=0.15, hspace=hspace)
        else:
            ncols = 3
            fs, r = 7, 2
            fig, axs = plt.subplots(nrows, ncols, figsize=(r * fs, fs))
            plt.subplots_adjust(wspace=0.15, hspace=hspace)
    else:
        x = 2
        if not bispec:
            ncols = 4
            fs, r = 7, 2.5
            fig, axs = plt.subplots(nrows, ncols, figsize=(r * fs, fs))
            plt.subplots_adjust(wspace=0.15, hspace=hspace)
        else:
            ncols = 5
            fs, r = 7, 2.5
            fig, axs = plt.subplots(nrows, ncols, figsize=(r * fs, fs))
            plt.subplots_adjust(wspace=0.15, hspace=hspace)

    lw = lw * fs
    lbs *= fs

    if title is not None:
        y = 0.95
        fig.suptitle(title, y=y, fontsize=lbs * 1.1)

    for j in range(ncols - x):
        axs[0, j + x].tick_params(labelbottom=False)

    for ax in axs.ravel():
        ax.tick_params(labelsize=lbs)
        ax.grid(
            which="major",
            linewidth=0.5,
            color="k",
            alpha=0.25,
        )

    if xlog:
        for i in range(nrows):
            for j in range(2):
                axs[i, j + x].set_xscale("log")

    for i in range(nrows):
        # axs[i, 0 + x].set_ylim(pk_rat_lim[0], pk_rat_lim[1])
        axs[i, 1 + x].set_ylim(-0.05, 1.05)

    axs[0, -1 + x].text(1.02, 0.5, "n_tr", rotation=-90, transform=axs[0, -1].transAxes)
    axs[1, -1 + x].text(
        1.02, 0.5, "delta_in", rotation=-90, transform=axs[1, -1].transAxes
    )

    if plot_maps:

        for i in range(2):
            for j in range(2):
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        def get_proj(arr):
            arr_slice, _, _ = get_projection(
                arr, idx=N // 2, axis=map_axis, width=width
            )
            return arr_slice

        lim_n_tr = jnp.mean(n_tr_ref) + jnp.std(n_tr_ref)
        vlim_n_tr = (0, lim_n_tr)
        n_tr_ref_slice = get_proj(n_tr_ref)
        axs[0, 0].imshow(
            n_tr_ref_slice.T,
            vmin=vlim_n_tr[0],
            vmax=vlim_n_tr[1],
            origin="lower",
            cmap=cmap_n_tr,
        )
        add_text(axs[0, 0], "N_TR (REF)", lbs, alpha=1)

        n_tr_slice = get_proj(n_tr)
        axs[0, 1].imshow(
            n_tr_slice.T,
            vmin=vlim_n_tr[0],
            vmax=vlim_n_tr[1],
            origin="lower",
            cmap=cmap_n_tr,
        )
        add_text(axs[0, 1], "N_TR (REC)", lbs, alpha=1)

        din_ref_slice = get_proj(din_ref)
        axs[1, 0].imshow(
            din_ref_slice.T,
            vmin=-0.75e-1,
            vmax=0.75e-1,
            origin="lower",
            cmap="seismic_r",
        )
        add_text(axs[1, 0], r"$\delta_0$ (REF)", lbs, alpha=1)

        din_slice = get_proj(din)
        axs[1, 1].imshow(
            din_slice.T,
            vmin=-0.75e-1,
            vmax=0.75e-1,
            origin="lower",
            cmap="seismic_r",
        )
        add_text(axs[1, 1], r"$\delta_0$ (REC)", lbs, alpha=1)

    n_tr_ref = n_tr_ref * N**3 / n_tr_ref.sum() - 1
    n_tr = n_tr * N**3 / n_tr.sum() - 1

    # PK RAT N_TR
    k_c, n_tr_ref_pk = get_pow_spec_1D(
        n_tr_ref, L, n_bins=n_bins, sphere_only=sphere_only
    )
    k_c, n_tr_pk = get_pow_spec_1D(n_tr, L, n_bins=n_bins, sphere_only=sphere_only)

    n_tr_pk_rat = n_tr_pk / n_tr_ref_pk
    axs[0, 0 + x].plot(k_c, n_tr_pk_rat, c="k", lw=lw)

    y_min_n_tr_pk_rat = jnp.min(jnp.array([jnp.min(n_tr_pk_rat), pk_rat_lim[0]]))
    y_max_n_tr_pk_rat = jnp.max(jnp.array([jnp.max(n_tr_pk_rat), pk_rat_lim[1]]))
    axs[0, 0 + x].set_ylim(y_min_n_tr_pk_rat, y_max_n_tr_pk_rat)

    # PK RAT DIN
    k_c, din_ref_pk = get_pow_spec_1D(
        din_ref, L, n_bins=n_bins, sphere_only=sphere_only
    )
    k_c, din_pk = get_pow_spec_1D(din, L, n_bins=n_bins, sphere_only=sphere_only)
    din_pk_rat = din_pk / din_ref_pk
    axs[1, 0 + x].plot(k_c, din_pk_rat, c="k", lw=lw)

    y_min_din_pk_rat = jnp.min(jnp.array([jnp.min(din_pk_rat), pk_rat_lim[0]]))
    y_max_din_pk_rat = jnp.max(jnp.array([jnp.max(din_pk_rat), pk_rat_lim[1]]))
    axs[1, 0 + x].set_ylim(y_min_din_pk_rat, y_max_din_pk_rat)

    # CROSS N_TR
    k_c, n_tr_cpk = get_cross_power_spec_1D(
        n_tr_ref, n_tr, L, n_bins=n_bins, sphere_only=sphere_only
    )
    axs[0, 1 + x].plot(k_c, n_tr_cpk, c="k", lw=lw)

    # CROSS DIN
    k_c, din_ref_cpk = get_cross_power_spec_1D(
        din_ref, din, L, n_bins=n_bins, sphere_only=sphere_only
    )
    k_c, prop = get_cross_power_spec_1D(
        din_ref, n_tr_ref, L, n_bins=n_bins, sphere_only=sphere_only
    )
    axs[1, 1 + x].plot(k_c, din_ref_cpk, c="k", lw=lw)
    axs[1, 1 + x].plot(k_c, prop, c="b", lw=lw, ls="--")

    if bispec:
        # BISPEC CONFIG
        thetas = jnp.linspace(0, jnp.pi, n_thetas)
        k1, k2 = 0.1, 0.2

        din_ref_bsp = get_reduced_bispectrum(din_ref, L, k1, k2, thetas)
        din_bsp = get_reduced_bispectrum(din, L, k1, k2, thetas)
        axs[1, -1].plot(thetas, din_bsp, c="k", ls="-")
        axs[1, -1].plot(thetas, din_ref_bsp, c="r", ls="--")

        n_tr_ref_bsp = get_reduced_bispectrum(n_tr_ref, L, k1, k2, thetas)
        n_tr_bsp = get_reduced_bispectrum(n_tr, L, k1, k2, thetas)
        axs[0, -1].plot(thetas, n_tr_bsp, c="k", ls="-")
        axs[0, -1].plot(thetas, n_tr_ref_bsp, c="r", ls="--")

        # axs[2].plot(thetas, bispec_ref, c=c_ref)
        # for i, delta in enumerate(delta_list):
        #     bispec = get_reduced_bispectrum(delta, L, k1, k2, thetas)
        #     axs[2].plot(thetas, bispec, c=cs[i], ls="--")

        #     axs_[2].plot(thetas, bispec / bispec_ref, ls="--", label=labels[i], c=cs[i])

    # axs[0].tick_params(labelsize=labelsize, top=True)

    return fig, axs


def add_text(ax, text, ts, bbox=True, c="k", alpha=0.75):
    if bbox:
        bbox = dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.1",
            alpha=alpha,
        )
    else:
        bbox = None

    ax.text(0.02, 0.05, text, c=c, fontsize=ts, transform=ax.transAxes, bbox=bbox)
