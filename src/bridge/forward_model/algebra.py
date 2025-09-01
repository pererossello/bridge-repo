import jax.numpy as jnp

from .fourier import my_ifft


def eigenvals_of_hessian(array_hat, kx, ky, kz, INV_L3):

    m00 = my_ifft(-array_hat * kx**2, INV_L3)
    m11 = my_ifft(-array_hat * ky**2, INV_L3)
    m22 = my_ifft(-array_hat * kz**2, INV_L3)

    m01 = my_ifft(-array_hat * kx * ky, INV_L3)
    m02 = my_ifft(-array_hat * kx * kz, INV_L3)
    m12 = my_ifft(-array_hat * ky * kz, INV_L3)

    a = -jnp.ones_like(m00)

    # trace
    b = m00 + m11 + m22

    # Sum of the products of 2x2 minors
    min01 = m00 * m11 - m01**2
    min02 = m00 * m22 - m02**2
    min12 = m11 * m22 - m12**2
    c = -(min01 + min02 + min12)

    # determinant
    d = (
        m00 * (m11 * m22 - m12**2)
        - m01 * (m01 * m22 - m12 * m02)
        + m02 * (m01 * m12 - m11 * m02)
    )

    roots = roots_cubic_equation(a, b, c, d)

    roots = jnp.sort(roots, axis=-1)[..., ::-1]
    roots = roots.real

    return roots


def roots_cubic_equation(a, b, c, d):

    Delta0 = jnp.square(b) - 3 * a * c
    Delta1 = 2 * jnp.power(b, 3) - 9 * a * b * c + 27 * jnp.square(a) * d

    discriminant = (jnp.square(Delta1) - 4 * jnp.power(Delta0, 3)).astype(jnp.complex64)

    C = jnp.power(((Delta1 + jnp.sqrt(discriminant)) / 2), (1 / 3))

    xis = jnp.array([1, (-1 + jnp.sqrt(3) * 1j) / 2, (-1 - jnp.sqrt(3) * 1j) / 2])

    C_xis = C[..., None] * xis[..., :]

    roots = (
        -1
        / (3 * a[..., None] * C_xis)
        * (b[..., None] * C_xis + jnp.square(C_xis) + Delta0[..., None])
    )

    return roots
