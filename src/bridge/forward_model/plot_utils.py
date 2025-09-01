# ruff: noqa

import sys

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from .fourier import get_k
from .stats import (
    get_pow_spec_1D,
    get_cross_power_spec_1D,
    get_radial_distro,
)


def plot_cubes(
    list_of_arrays,
    cmap="seismic_r",
    figsize=5,
    axis=0,
    idx=0,
    width=1,
    vlim=None,
    wspace=0.05,
    return_ims=False,
    titles=None,
    title=None,
    ts=1.25,
    interpolation="nearest",
):

    if not isinstance(list_of_arrays, list):
        list_of_arrays = [list_of_arrays]
    M = len(list_of_arrays)

    if not isinstance(cmap, list):
        cmap = [cmap] * M

    if vlim is not None:
        if isinstance(vlim, (float, int)):
            vlim = [(-vlim, vlim)] * M
        elif isinstance(vlim, list):
            for i, vlim_item in enumerate(vlim):
                if isinstance(vlim_item, (int, float)):
                    vlim[i] = (-vlim_item, vlim_item)
        elif isinstance(vlim, tuple):
            vlim = [(vlim[0], vlim[1])] * M

    if M < 4:
        n_col = M
        n_row = 1
    elif M == 4:
        n_col = 2
        n_row = 2

    ratio = n_col / n_row  # type: ignore
    figsize_fact = figsize / np.sqrt(ratio)
    fig, axs = plt.subplots(n_row, n_col, figsize=(figsize_fact * ratio, figsize_fact))  # type: ignore
    plt.subplots_adjust(wspace=wspace, hspace=0.05)

    ims = []
    axs_flat = axs.flatten() if M > 1 else [axs]
    for i, ax in enumerate(axs_flat):
        ax.set_xticks([])
        ax.set_yticks([])

        arr = list_of_arrays[i]
        arr, min_val, max_val = get_projection(arr, axis, idx, width)

        vmin = min_val if vlim is None else vlim[i][0]
        vmax = max_val if vlim is None else vlim[i][1]

        im = ax.imshow(
            arr.T,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            cmap=cmap[i],
            interpolation=interpolation,
        )
        ims.append(im)

        if titles is not None:
            ax.text(
                0.04,
                0.04,
                titles[i],
                ha="left",
                va="bottom",
                transform=ax.transAxes,
                fontsize=figsize * ts,
                c="k",
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="round,pad=0.25",
                    alpha=1,
                ),
            )
        if title is not None:
            fig.suptitle(title, fontsize=figsize * ts, y=0.92)

    if return_ims:
        return fig, axs, ims

    return fig, axs


def get_projection(array, axis, idx, width):

    N = array.shape[0]

    if idx > N - 1:
        raise ValueError("idx should be smaller than N-1")

    idx_min = idx
    idx_max = idx + width

    if axis == 0:
        matrix = array[idx_min:idx_max, :, :]
    elif axis == 1:
        matrix = array[:, idx_min:idx_max, :]
    elif axis == 2:
        matrix = array[:, :, idx_min:idx_max]

    matrix = np.sum(matrix, axis=axis) / width  # type: ignore
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    return matrix, min_val, max_val


def compare_pow_spec(
    delta_list,
    L,
    n_bins=50,
    labels=None,
    title=None,
    xlog=False,
    no_labels=False,
    sphere_only=False,
    lw=0.5,
    x_lim=None,
    cross=True,
    alpha=0.75,
    y2lim=None,
    ls=None,
    colors=None,
    pk_cube=None,
    k_min=None,
):

    # grid_sizes = [delta.shape[0] for delta in delta_list]
    # deltas_same_gridsize = all([delta.shape==delta_list[0].shape for delta in delta_list])

    M = len(delta_list)

    if pk_cube is not None:
        N = delta_list[0].shape[0]
        print("pk_cube is not None, forcing no cross")
        cross = False
        M += 1

        k = get_k(N, L)
        fact = 1 / jnp.sqrt(3) if sphere_only else 1.0
        bin_range = (0.0, k.max() * fact)
        k, pk_ref = get_radial_distro(k, pk_cube, bin_range=bin_range, n_bins=n_bins)

    else:
        k, pk0 = get_pow_spec_1D(
            delta_list[0], L, n_bins=n_bins, sphere_only=sphere_only
        )

    if labels == None:
        labels = [f"{i}" for i in range(M)]
    if ls == None:
        ls = ["-."] + ["-"] * (M - 1)
    if colors == None:
        cmap = plt.get_cmap("nipy_spectral")
        cs = [cmap(i) for i in np.linspace(0.1, 0.9, M - 1, endpoint=True)]
        cs = ["k"] + cs
    else:
        cs = colors
        if pk_cube is not None:
            cs = ["k"] + cs

    figsize = 5
    lw = lw * figsize

    labelsize = 11

    if cross:
        fig, axs = plt.subplots(1, 3, figsize=(3 * figsize, figsize))
        plt.subplots_adjust(wspace=0.15, hspace=None)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(2 * figsize, figsize))
        plt.subplots_adjust(wspace=0.15, hspace=None)

    if pk_cube is not None:
        pk0 = pk_ref  # type: ignore

    for i, delta in enumerate(delta_list):
        if pk_cube is not None:
            i += 1
        k, pk = get_pow_spec_1D(delta, L, n_bins=n_bins, sphere_only=sphere_only)

        axs[0].plot(k, pk, c=cs[i], ls=ls[i], label=labels[i], lw=lw, alpha=alpha)
        axs[1].plot(k, pk / pk0, c=cs[i], ls=ls[i], lw=lw, alpha=alpha)  # type: ignore

        if cross:
            k, pk_cross = get_cross_power_spec_1D(
                delta_list[0], delta_list[i], L, n_bins, sphere_only=sphere_only
            )
            axs[2].plot(k, pk_cross, c=cs[i], ls=ls[i], lw=lw, alpha=alpha)

    if pk_cube is not None:
        axs[0].plot(
            k,
            pk_ref,  # type: ignore
            c=cs[0],
            ls=ls[0],
            label=labels[0],
            lw=lw,
            alpha=alpha,
            zorder=100,
        )

    axs[0].set_yscale("log")

    if not sphere_only:
        for ax in axs:
            ax.axvline(k.max() / jnp.sqrt(3), color="grey", ls="--")

    if x_lim is not None:
        axs[0].set_xlim(x_lim[0], x_lim[1])
    # axs[0].set_xscale('log')

    if not no_labels:
        axs[0].legend(fontsize=labelsize)

    axs[0].tick_params(labelsize=labelsize, top=True)

    # axs[1].set_ylim(0.5, 2)
    axs[1].tick_params(
        labelsize=labelsize,
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
        top=True,
    )

    if cross:
        axs[2].set_ylim(-0.2, 1.1)
        axs[2].tick_params(
            labelsize=labelsize,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
            top=True,
        )

    for ax in axs:

        ax.grid(
            which="major",
            linewidth=0.5,
            color="k",
            alpha=0.25,
        )
        ax.set_xlabel("$k$ [$h$/Mpc]", fontsize=labelsize)

        if xlog:
            ax.set_xscale("log")
        else:
            ax.set_xlim(0, None)
            xticks = ax.get_xticks()
            ax2 = ax.twiny()
            ax2.set_xlim(axs[0].get_xlim())  # Ensure the limits of top and
            ax2.xaxis.set_major_locator(FixedLocator(xticks))
            ax2.set_xticklabels(
                [""] + [f"{2*np.pi/label:0.2f}" for label in xticks[1:]]
            )
            ax2.set_xlabel(r"$\lambda$ [Mpc/$h$]", fontsize=labelsize, labelpad=7)
            ax2.tick_params(labelsize=labelsize)

    if y2lim is not None:
        axs[1].set_ylim(y2lim[0], y2lim[1])

    if xlog:
        y = 0.95
    else:
        y = 1.05
    if title is not None:
        fig.suptitle(title, y=y, fontsize=labelsize * 1.3)

    return fig, axs


####################################################


def stats_plot_for_article(bk_rat_y_lim=(0.5, 1.5)):

    fig_size = 7
    ratio = 2.25
    hspace = 0.075
    wspace = 0.21

    pk_rat_y_lim = (0.9, 1.1)
    cross_y_lim = (-0.05, 1.05)

    fs, r = fig_size, ratio
    labelsize_factor = 2.75  # 1.75

    sep_fact = 0.5

    nrows = 2
    ncols = 2
    row_ratios = [1.5, 1]
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(r * fs, fs), gridspec_kw={"height_ratios": row_ratios}
    )
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    labelsize = labelsize_factor * fs

    axs[0, 0].tick_params(labelbottom=False)
    axs[0, 1].tick_params(labelbottom=False)

    ax00_pos = axs[0, 0].get_position()
    ax01_pos = axs[0, 1].get_position()
    ax11_pos = axs[1, 1].get_position()

    separation = ax01_pos.x0 - ax00_pos.x1
    separation = separation * sep_fact
    total_height = ax00_pos.y1 - ax11_pos.y0

    ax_c = fig.add_axes(
        [ax01_pos.x1 + separation, ax11_pos.y0, ax01_pos.width, total_height]  # type: ignore
    )

    axs_all = list(axs.ravel()) + [ax_c]
    for ax in axs_all:
        ax.tick_params(labelsize=labelsize)
        ax.grid(
            which="major",
            linewidth=0.5,
            color="k",
            alpha=0.25,
        )

    sw = 0.25 * fs
    tick_length = sw * 5
    for ax in axs_all:  # use .flat to iterate even if it's 2D
        for spine in ax.spines.values():
            spine.set_linewidth(sw)
        ax.tick_params(which="major", direction="inout", width=sw, length=tick_length)
        ax.tick_params(
            which="minor", direction="inout", width=sw, length=tick_length * 0.6
        )

    axs_log_x = [axs[0, 0], axs[1, 0], ax_c]
    for ax in axs_log_x:
        ax.set_xscale("log")
        ax.set_xlabel(r"$k \, [h \, \text{Mpc}^{-1}]$ ", fontsize=labelsize * 1.2)

    axs[0, 0].set_yscale("log")

    axs[1, 0].set_ylim(pk_rat_y_lim[0], pk_rat_y_lim[1])
    axs[1, 1].set_ylim(bk_rat_y_lim[0], bk_rat_y_lim[1])

    ax_c.set_ylim(cross_y_lim[0], cross_y_lim[1])

    axs[1, 1].set_xlabel(r"$\theta$ ", fontsize=labelsize * 1.2)
    axs[1, 1].set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    axs[1, 1].set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])

    yl_00 = r"$P_{\text{rec}} (k)$"
    yl_10 = r"$P_{\text{rec}}/P_{\text{ref}}$"
    axs[0, 0].set_ylabel(yl_00, fontsize=labelsize * 1.25)
    axs[1, 0].set_ylabel(yl_10, fontsize=labelsize * 1.25)

    yl_01 = r"$Q_{\text{rec}} (\theta)$"
    yl_11 = r"$Q_{\text{rec}}/Q_{\text{ref}}$"
    lp = -1
    axs[0, 1].set_ylabel(yl_01, fontsize=labelsize * 1.25, labelpad=lp)
    axs[1, 1].set_ylabel(yl_11, fontsize=labelsize * 1.25, labelpad=lp)

    ax_c.yaxis.set_label_position("right")  # Move the label to the right
    ax_c.yaxis.tick_right()
    yl_c = r"$C(k)$"
    lp = -1
    ax_c.set_ylabel(yl_c, fontsize=labelsize * 1.25, labelpad=lp)

    return fig, axs_all


def stats_plot_for_article_2(bk_rat_y_lim=(0.5, 1.5), quad_rat_y_lim=(-5, 5)):

    fig_size = 8
    ratio = 3.2
    hspace = 0.075
    wspace = 0.5

    pk_rat_y_lim = (0.9, 1.1)
    cross_y_lim = (-0.05, 1.05)

    fs, r = fig_size, ratio
    labelsize_factor = 3.5  # 2.75#1.75

    sep_fact = 0.05

    nrows = 2
    ncols = 3
    row_ratios = [1.8, 1]
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(r * fs, fs),
        gridspec_kw={"height_ratios": row_ratios},
    )
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    labelsize = labelsize_factor * fs

    axs[0, 0].tick_params(labelbottom=False)
    axs[0, 1].tick_params(labelbottom=False)
    axs[0, 2].tick_params(labelbottom=False)

    ax00_pos = axs[0, 0].get_position()
    ax02_pos = axs[0, 2].get_position()
    ax12_pos = axs[1, 2].get_position()

    separation = ax02_pos.x0 - ax00_pos.x1
    separation = separation * sep_fact
    total_height = ax00_pos.y1 - ax12_pos.y0

    ax_c = fig.add_axes(
        [ax02_pos.x1 + separation * 4.5, ax12_pos.y0, ax02_pos.width, total_height]  # type: ignore
    )

    tick_label_size_fact = 1.2
    axs_all = list(axs.ravel()) + [ax_c]
    for ax in axs_all:
        ax.tick_params(labelsize=labelsize * tick_label_size_fact)
        ax.grid(
            which="major",
            linewidth=0.5,
            color="k",
            alpha=0.25,
        )

    sw = 0.35 * fs
    tick_length = sw * 5
    for ax in axs_all:  # use .flat to iterate even if it's 2D
        for spine in ax.spines.values():
            spine.set_linewidth(sw)
        ax.tick_params(which="major", direction="inout", width=sw, length=tick_length)
        ax.tick_params(
            which="minor", direction="inout", width=sw, length=tick_length * 0.6
        )

    ls_fact = 1.25

    axs_log_x = [axs[0, 0], axs[1, 0], axs[0, 1], axs[1, 1], ax_c]
    for ax in axs_log_x:
        ax.set_xscale("log")
        ax.set_xlabel(r"$k \, [h \, \text{Mpc}^{-1}]$ ", fontsize=labelsize * ls_fact)

    axs[0, 0].set_yscale("log")
    # axs[0, 1].set_yscale("log")

    axs[1, 0].set_ylim(pk_rat_y_lim[0], pk_rat_y_lim[1])
    axs[1, 1].set_ylim(quad_rat_y_lim[0], quad_rat_y_lim[1])
    axs[1, 2].set_ylim(bk_rat_y_lim[0], bk_rat_y_lim[1])

    ax_c.set_ylim(cross_y_lim[0], cross_y_lim[1])

    axs[1, 2].set_xlabel(r"$\theta$ ", fontsize=labelsize * ls_fact)
    axs[1, 2].set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    axs[1, 2].set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])

    lp = fs * 2

    yl_00 = r"$P_{\text{rec}} (k)$"
    yl_10 = r"$P_{\text{rec}}/P_{\text{ref}}$"
    axs[0, 0].set_ylabel(yl_00, fontsize=labelsize * ls_fact, labelpad=lp)
    axs[1, 0].set_ylabel(yl_10, fontsize=labelsize * ls_fact, labelpad=lp)

    yl_01 = r"$Q_{\text{rec}} (\theta)$"
    yl_11 = r"$Q_{\text{rec}}/Q_{\text{ref}}$"
    axs[0, 2].set_ylabel(yl_01, fontsize=labelsize * ls_fact, labelpad=lp)
    axs[1, 2].set_ylabel(yl_11, fontsize=labelsize * ls_fact, labelpad=lp)

    yl_02 = r"$P_{2,\text{rec}} (k)$"
    yl_12 = r"$P_{2,\text{rec}}/P_{2,\text{ref}}$"
    axs[0, 1].set_ylabel(yl_02, fontsize=labelsize * ls_fact, labelpad=lp)
    axs[1, 1].set_ylabel(yl_12, fontsize=labelsize * ls_fact, labelpad=lp)

    # ax_c.yaxis.set_label_position("right")
    # ax_c.yaxis.tick_right()

    yl_c = r"$C(k)$"
    lp = -1
    ax_c.set_ylabel(yl_c, fontsize=labelsize * ls_fact, labelpad=lp)

    return fig, axs_all


############################


def scatter_density(
    ax,
    x,
    y,
    step=1,
    n_bins=250,
    cmap="turbo",
    color="b",
    alpha=0.25,
    s=1.0,
    zorder=1,
    x_lim=None,
    y_lim=None,
    label=None,
    log_bins_x=False,
    log_bins_y=False,
):
    x = x.ravel()[::step]
    y = y.ravel()[::step]

    if cmap is False:
        ax.scatter(x, y, s=s, lw=0.0, c=color, alpha=alpha, label=label)
        return

    # Apply limits
    if x_lim is None:
        min_x, max_x = np.min(x), np.max(x)
    else:
        min_x, max_x = x_lim
    if y_lim is None:
        min_y, max_y = np.min(y), np.max(y)
    else:
        min_y, max_y = y_lim

    # Define bin edges
    if log_bins_x:
        if min_x <= 0:
            raise ValueError("x values must be > 0 for log bins")
        edges1 = np.logspace(np.log10(min_x), np.log10(max_x), n_bins)
    else:
        edges1 = np.linspace(min_x, max_x, n_bins)

    if log_bins_y:
        if min_y <= 0:
            raise ValueError("y values must be > 0 for log bins")
        edges2 = np.logspace(np.log10(min_y), np.log10(max_y), n_bins)
    else:
        edges2 = np.linspace(min_y, max_y, n_bins)

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[edges1, edges2])

    # Digitize points into bin indices
    idxs1 = np.searchsorted(edges1, x, side="right") - 1
    idxs2 = np.searchsorted(edges2, y, side="right") - 1

    # Mask out-of-bounds indices
    valid = (idxs1 >= 0) & (idxs1 < n_bins - 1) & (idxs2 >= 0) & (idxs2 < n_bins - 1)
    idxs1 = idxs1[valid]
    idxs2 = idxs2[valid]
    x = x[valid]
    y = y[valid]

    density = hist[idxs1, idxs2]
    density = density / np.max(density) if np.max(density) > 0 else density

    ax.scatter(x, y, s=s, lw=0.0, c=density, cmap=cmap, label=label, zorder=zorder)


def contour_density(
    ax,
    x,
    y,
    step=1,
    n_bins=250,
    levels=(0.5, 0.75, 0.9, 0.99),
    cmap="Blues",
    linewidths=1.5,
    linestyles="-",
    zorder=1,
    x_lim=None,
    y_lim=None,
    log_bins_x=False,
    log_bins_y=False,
    fill=True,
    alpha=0.5,
    legend=True,
    labelsize=1,
    loc="best",
):

    x = x.ravel()[::step]
    y = y.ravel()[::step]

    if x_lim is None:
        min_x, max_x = np.min(x), np.max(x)
    else:
        min_x, max_x = x_lim
    if y_lim is None:
        min_y, max_y = np.min(y), np.max(y)
    else:
        min_y, max_y = y_lim

    # Bin edges
    if log_bins_x:
        if min_x <= 0:
            raise ValueError("x values must be > 0 for log binning")
        edges1 = np.logspace(np.log10(min_x), np.log10(max_x), n_bins)
    else:
        edges1 = np.linspace(min_x, max_x, n_bins)

    if log_bins_y:
        if min_y <= 0:
            raise ValueError("y values must be > 0 for log binning")
        edges2 = np.logspace(np.log10(min_y), np.log10(max_y), n_bins)
    else:
        edges2 = np.linspace(min_y, max_y, n_bins)

    # Histogram and normalize
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[edges1, edges2])
    hist = hist.T  # for imshow/contour compatibility

    # Convert histogram to probability density
    hist = hist / jnp.sum(hist)

    flat = hist.flatten()
    idx_sort = jnp.argsort(flat)[::-1]
    cumsum = jnp.cumsum(flat[idx_sort])

    # Compute contour levels for the given cumulative levels
    thresholds = []
    for p in levels:
        try:
            threshold = float(flat[idx_sort][jnp.searchsorted(cumsum, p)])
        except IndexError:
            threshold = 0.0
        thresholds.append(threshold)

    # Enforce strictly increasing order and drop duplicates
    thresholds = np.asarray(thresholds)
    thresholds = np.sort(thresholds)
    thresholds = np.unique(thresholds)

    # If only one level remains, matplotlib will still error. Guard against that:
    if len(thresholds) < 2:
        print("Not enough unique contour levels — skipping contour plot.")
        return
    # Grid centers for contouring
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers)

    # Plot contours
    cs = ax.contour(
        X,
        Y,
        hist,
        levels=thresholds,
        cmap=cmap,
        linewidths=linewidths,
        linestyles=linestyles,
        zorder=zorder,
    )

    if fill:
        # Add a lower bound for contourf to fill from
        fill_levels = [0.0] + list(thresholds) + [hist.max()]
        vmax = fill_levels[-1]

        colors = plt.cm.get_cmap(cmap, len(thresholds) + 1)
        cmap = ListedColormap(colors(np.linspace(0, 1, len(thresholds) + 1)))
        norm = BoundaryNorm(boundaries=fill_levels, ncolors=len(fill_levels) - 1)

        ax.contourf(
            X,
            Y,
            hist,
            levels=fill_levels,
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            zorder=zorder,
            vmin=0.0,
            vmax=vmax,
        )

    if legend:
        handles = []
        labels = []
        a = len(levels)
        for i, level in enumerate(levels):
            color = cmap((a - i) / a)  # type: ignore
            handles.append(Patch(facecolor=color, edgecolor="none", alpha=alpha))
            labels.append(f"{int(level * 100)}%")

        ax.legend(handles, labels, loc=loc, frameon=True, fontsize=labelsize)

    return cs
