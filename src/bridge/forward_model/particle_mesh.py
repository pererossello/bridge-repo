import jax.numpy as jnp

from .config import Constants


def particle_to_mesh(positions, cte: Constants):

    density = cic(positions, cte.N, cte.L)
    delta = density - 1

    return delta


def cic(pos, N, L):

    pos = (pos * N / L) % N

    ix_f = pos[:, 0]
    iy_f = pos[:, 1]
    iz_f = pos[:, 2]

    ix0 = jnp.floor(ix_f).astype(int)
    iy0 = jnp.floor(iy_f).astype(int)
    iz0 = jnp.floor(iz_f).astype(int)

    dx = ix_f - ix0
    dy = iy_f - iy0
    dz = iz_f - iz0

    ix1 = (ix0 + 1) % N
    iy1 = (iy0 + 1) % N
    iz1 = (iz0 + 1) % N

    w000 = (1 - dx) * (1 - dy) * (1 - dz)
    w100 = dx * (1 - dy) * (1 - dz)
    w010 = (1 - dx) * dy * (1 - dz)
    w001 = (1 - dx) * (1 - dy) * dz
    w110 = dx * dy * (1 - dz)
    w101 = dx * (1 - dy) * dz
    w011 = (1 - dx) * dy * dz
    w111 = dx * dy * dz

    density = jnp.zeros((N, N, N))

    density = (
        density.at[ix0, iy0, iz0]
        .add(w000)
        .at[ix1, iy0, iz0]
        .add(w100)
        .at[ix0, iy1, iz0]
        .add(w010)
        .at[ix0, iy0, iz1]
        .add(w001)
        .at[ix1, iy1, iz0]
        .add(w110)
        .at[ix1, iy0, iz1]
        .add(w101)
        .at[ix0, iy1, iz1]
        .add(w011)
        .at[ix1, iy1, iz1]
        .add(w111)
    )

    return density


def trilinear_interpolation(field, pos_lagr, cte: Constants):

    scaled_pos = pos_lagr * cte.INV_R

    i = jnp.floor(scaled_pos[:, 0]).astype(jnp.int32)
    j = jnp.floor(scaled_pos[:, 1]).astype(jnp.int32)
    k = jnp.floor(scaled_pos[:, 2]).astype(jnp.int32)

    dx = scaled_pos[:, 0] - i
    dy = scaled_pos[:, 1] - j
    dz = scaled_pos[:, 2] - k

    i0 = i
    i1 = (i + 1) % cte.N
    j0 = j
    j1 = (j + 1) % cte.N
    k0 = k
    k1 = (k + 1) % cte.N

    c000 = field[:, i0, j0, k0]
    c100 = field[:, i1, j0, k0]
    c010 = field[:, i0, j1, k0]
    c110 = field[:, i1, j1, k0]
    c001 = field[:, i0, j0, k1]
    c101 = field[:, i1, j0, k1]
    c011 = field[:, i0, j1, k1]
    c111 = field[:, i1, j1, k1]

    c00 = c000 * (1.0 - dx) + c100 * dx
    c10 = c010 * (1.0 - dx) + c110 * dx
    c01 = c001 * (1.0 - dx) + c101 * dx
    c11 = c011 * (1.0 - dx) + c111 * dx

    c0 = c00 * (1.0 - dy) + c10 * dy
    c1 = c01 * (1.0 - dy) + c11 * dy

    shifts = c0 * (1.0 - dz) + c1 * dz

    return shifts.T
