import jax.numpy as jnp

from .fourier import my_fft, my_ifft


def make_rfft2_arr_from_phases(numbers, N):
    """
    numbers should be shape (N^2 / 2 - 2, )
    """

    arr_hat_r = jnp.zeros((N, N // 2 + 1))

    if numbers.shape != (round(N**2 // 2 - 2),):
        raise ValueError(f"Wrong shape = {numbers.shape}, Should be (N^2 /2 -2,)")

    # make adequate shape arrs

    M1 = N // 2 - 1
    M2 = 2 * M1**2 + M1

    ####

    H_line = numbers[:M1]
    arr_hat_r = arr_hat_r.at[0, 1 : M1 + 1].set(H_line)
    numbers = numbers[M1:]

    ####

    V_line_DC = numbers[:M1]
    arr_hat_r = arr_hat_r.at[1 : M1 + 1, 0].set(V_line_DC)
    # set conj
    arr_hat_r = arr_hat_r.at[-M1:, 0].set(-V_line_DC[::-1])

    numbers = numbers[M1:]

    V_line_NYQ = numbers[:M1]
    arr_hat_r = arr_hat_r.at[1 : M1 + 1, N // 2].set(V_line_NYQ)
    # set conj
    arr_hat_r = arr_hat_r.at[-M1:, N // 2].set(-V_line_NYQ[::-1])
    numbers = numbers[M1:]

    ####

    rectangle = numbers[:M2]
    rectangle = rectangle.reshape((N - 1, M1))
    arr_hat_r = arr_hat_r.at[1:, 1 : M1 + 1].set(rectangle)

    # print(jnp.where(arr_hat_r == 1))

    return arr_hat_r


def make_rfft2_arr_from_amps(numbers, N):
    """
    numbers should be shape (N^2 / 2 - 2, )
    """

    arr_hat_r = jnp.zeros((N, N // 2 + 1))

    if numbers.shape != (round(N**2 // 2 - 2),):
        raise ValueError(f"Wrong shape = {numbers.shape}, Should be (N^2 /2 -2,)")

    # make adequate shape arrs

    M1 = N // 2 - 1
    M2 = 2 * M1**2 + M1

    ####

    H_line = numbers[:M1]
    arr_hat_r = arr_hat_r.at[0, 1 : M1 + 1].set(H_line)
    numbers = numbers[M1:]

    ####

    V_line_DC = numbers[:M1]
    arr_hat_r = arr_hat_r.at[1 : M1 + 1, 0].set(V_line_DC)
    # set conj
    arr_hat_r = arr_hat_r.at[-M1:, 0].set(V_line_DC[::-1])

    numbers = numbers[M1:]

    V_line_NYQ = numbers[:M1]
    arr_hat_r = arr_hat_r.at[1 : M1 + 1, N // 2].set(V_line_NYQ)
    # set conj
    arr_hat_r = arr_hat_r.at[-M1:, N // 2].set(V_line_NYQ[::-1])
    numbers = numbers[M1:]

    ####

    rectangle = numbers[:M2]
    rectangle = rectangle.reshape((N - 1, M1))
    arr_hat_r = arr_hat_r.at[1:, 1 : M1 + 1].set(rectangle)

    # print(jnp.where(arr_hat_r == 1))

    return arr_hat_r


def from_rfft2_arr_to_fft2_arr_phases(arr_hat_r, N):
    arr_hat_rec = jnp.zeros((N, N))
    if arr_hat_r.shape != (N, N // 2 + 1):
        raise ValueError("Wrong shape! Should be (N,N//2+1)")
    arr_hat_rec = arr_hat_rec.at[:, : N // 2 + 1].set(arr_hat_r)
    arr_hat_rec = arr_hat_rec.at[0, -N // 2 + 1 :].set(-arr_hat_r[0, 1 : N // 2][::-1])
    arr_hat_rec = arr_hat_rec.at[1:, -N // 2 + 1 :].set(
        -arr_hat_r[1:, 1 : N // 2][::-1, ::-1]
    )
    return arr_hat_rec


def from_rfft2_arr_to_fft2_arr_amps(arr_hat_r, N):
    arr_hat_rec = jnp.zeros((N, N))
    if arr_hat_r.shape != (N, N // 2 + 1):
        raise ValueError("Wrong shape! Should be (N,N//2+1)")
    arr_hat_rec = arr_hat_rec.at[:, : N // 2 + 1].set(arr_hat_r)
    arr_hat_rec = arr_hat_rec.at[0, -N // 2 + 1 :].set(arr_hat_r[0, 1 : N // 2][::-1])
    arr_hat_rec = arr_hat_rec.at[1:, -N // 2 + 1 :].set(
        arr_hat_r[1:, 1 : N // 2][::-1, ::-1]
    )
    return arr_hat_rec


def make_rfft3_arr_from_amps(numbers, N):
    """
    phases should be shape N**3/2 - 4
    """

    arr_hat_r = jnp.zeros((N, N, N // 2 + 1))

    M_PLANE = N**2 // 2 - 2
    M_CUBOID = N**3 // 2 - N**2

    cuboid_vals = numbers[:M_CUBOID]
    cuboid_vals = cuboid_vals.reshape((N, N, N // 2 - 1))
    numbers = numbers[M_CUBOID:]

    DC_plane_vals = numbers[:M_PLANE]
    numbers = numbers[M_PLANE:]

    NYQ_plane_vals = numbers[:M_PLANE]

    # ASSIGN
    arr_hat_r = arr_hat_r.at[:, :, 1 : N // 2].set(cuboid_vals[...])

    DC_plane = make_rfft2_arr_from_amps(DC_plane_vals, N)
    DC_plane = from_rfft2_arr_to_fft2_arr_amps(DC_plane, N)
    arr_hat_r = arr_hat_r.at[:, :, 0].set(DC_plane)

    NYQ_plane = make_rfft2_arr_from_amps(NYQ_plane_vals, N)
    NYQ_plane = from_rfft2_arr_to_fft2_arr_amps(NYQ_plane, N)
    arr_hat_r = arr_hat_r.at[:, :, N // 2].set(NYQ_plane)

    return arr_hat_r


def make_rfft3_arr_from_phases(numbers, N):
    """
    phases should be shape N**3/2 - 4
    """

    arr_hat_r = jnp.zeros((N, N, N // 2 + 1))

    M_PLANE = N**2 // 2 - 2
    M_CUBOID = N**3 // 2 - N**2

    cuboid_vals = numbers[:M_CUBOID]
    cuboid_vals = cuboid_vals.reshape((N, N, N // 2 - 1))
    numbers = numbers[M_CUBOID:]

    DC_plane_vals = numbers[:M_PLANE]
    numbers = numbers[M_PLANE:]

    NYQ_plane_vals = numbers[:M_PLANE]

    # ASSIGN
    arr_hat_r = arr_hat_r.at[:, :, 1 : N // 2].set(cuboid_vals[...])

    DC_plane = make_rfft2_arr_from_phases(DC_plane_vals, N)
    DC_plane = from_rfft2_arr_to_fft2_arr_phases(DC_plane, N)
    arr_hat_r = arr_hat_r.at[:, :, 0].set(DC_plane)

    NYQ_plane = make_rfft2_arr_from_phases(NYQ_plane_vals, N)
    NYQ_plane = from_rfft2_arr_to_fft2_arr_phases(NYQ_plane, N)
    arr_hat_r = arr_hat_r.at[:, :, N // 2].set(NYQ_plane)

    return arr_hat_r


def make_rfft3_arr_from_N_cube_numbers(numbers, N):

    numbers = numbers.reshape(-1)

    arr_hat_r = jnp.zeros((N, N, N // 2 + 1), dtype=complex)

    if numbers.shape != (N**3,):
        # now this is redundant because of reshape above
        raise ValueError("Wrong shape! Should be (N^3,)")

    M_main = N**3 - 2 * N**2
    blue_vals = numbers[:M_main]
    blue_vals = blue_vals.reshape((N, N, N // 2 - 1, 2))
    numbers = numbers[M_main:]

    plane_1_vals = numbers[: N**2]
    numbers = numbers[N**2 :]

    plane_2_vals = numbers[: N**2]

    # ASSIGN
    arr_hat_r = arr_hat_r.at[:, :, 1 : N // 2].set(
        blue_vals[..., 0] + 1j * blue_vals[..., 1]
    )

    plane_1 = make_rfft2_arr_from_N_sq_numbers(plane_1_vals, N)
    plane_1 = from_rfft2_arr_to_fft2_arr(plane_1, N)
    arr_hat_r = arr_hat_r.at[:, :, 0].set(plane_1)

    plane_2 = make_rfft2_arr_from_N_sq_numbers(plane_2_vals, N)
    plane_2 = from_rfft2_arr_to_fft2_arr(plane_2, N)
    arr_hat_r = arr_hat_r.at[:, :, N // 2].set(plane_2)

    return arr_hat_r


def make_N_cube_numbers_from_rfft3_arr(arr_hat, N):
    """ "
    :arr: is a valid rfft3 array
    the inverse of 'make_rfft3_arr_from_N_cube_numbers'
    """

    numbers = jnp.zeros(N**3)

    # blue vals
    M_main = N**3 - 2 * N**2
    blue_vals_real = arr_hat[:, :, 1 : N // 2].real
    blue_vals_imag = arr_hat[:, :, 1 : N // 2].imag
    blue_vals = jnp.stack([blue_vals_real, blue_vals_imag], axis=-1)
    blue_vals = blue_vals.reshape(M_main)
    numbers = numbers.at[:M_main].set(blue_vals)

    # DC plane
    M_plane = N**2
    DC_plane_vals = arr_hat[:, :, 0]
    DC_plane_vals = DC_plane_vals[:, : N // 2 + 1]  # extracting the rrft
    idx_i = M_main
    idx_f = M_main + M_plane
    numbers_DC_plane = make_N_sq_numbers_from_rfft2_arr(DC_plane_vals, N)
    numbers = numbers.at[idx_i:idx_f].set(numbers_DC_plane)

    # NYQ plane
    NYQ_plane_vals = arr_hat[:, :, N // 2]
    NYQ_plane_vals = NYQ_plane_vals[:, : N // 2 + 1]  # extracting the rrft
    idx_i = M_main + M_plane
    idx_f = M_main + M_plane * 2
    numbers_NYQ_plane = make_N_sq_numbers_from_rfft2_arr(NYQ_plane_vals, N)
    numbers = numbers.at[idx_i:idx_f].set(numbers_NYQ_plane)

    numbers = numbers.reshape((N,) * 3)

    return numbers


def make_N_sq_numbers_from_rfft2_arr(arr_hat, N):

    numbers = jnp.zeros(N**2)

    # set nyquist modes
    M_nyq = 4
    numbers = numbers.at[0].set(arr_hat[0, 0].real)
    numbers = numbers.at[1].set(arr_hat[N // 2, 0].real)
    numbers = numbers.at[2].set(arr_hat[0, N // 2].real)
    numbers = numbers.at[3].set(arr_hat[N // 2, N // 2].real)

    # place yellow modes (complex)
    M_yellow = N - 2
    yellow_vals = arr_hat[0, 1 : N // 2]
    yellow_vals_real = yellow_vals.real
    yellow_vals_imag = yellow_vals.imag
    yellow_vals = jnp.stack([yellow_vals_real, yellow_vals_imag], axis=-1)
    yellow_vals = yellow_vals.reshape(M_yellow)
    idx_i = M_nyq
    idx_f = idx_i + M_yellow
    numbers = numbers.at[idx_i:idx_f].set(yellow_vals)

    # place green modes (complex)
    M_green1 = N - 2
    green1_vals = arr_hat[1 : N // 2, 0]
    green1_vals_real = green1_vals.real
    green1_vals_imag = green1_vals.imag
    green1_vals = jnp.stack([green1_vals_real, green1_vals_imag], axis=-1)

    M_green2 = N - 2
    green2_vals = arr_hat[1 : N // 2, N // 2]
    green2_vals_real = green2_vals.real
    green2_vals_imag = green2_vals.imag
    green2_vals = jnp.stack([green2_vals_real, green2_vals_imag], axis=-1)

    M_green = M_green1 + M_green2
    green_vals = jnp.stack([green1_vals, green2_vals], axis=-1)
    green_vals = green_vals.reshape(M_green)
    idx_i = M_nyq + M_yellow
    idx_f = idx_i + M_green
    numbers = numbers.at[idx_i:idx_f].set(green_vals)

    M_blue = N**2 - M_nyq - M_yellow - M_green
    blue_vals = arr_hat[1:, 1 : N // 2]
    blue_vals_real = blue_vals.real
    blue_vals_imag = blue_vals.imag
    blue_vals = jnp.stack([blue_vals_real, blue_vals_imag], axis=-1)
    blue_vals = blue_vals.reshape(M_blue)
    numbers = numbers.at[-M_blue:].set(blue_vals)

    return numbers


def make_rfft2_arr_from_N_sq_numbers(numbers, N):

    arr_hat_r = jnp.zeros((N, N // 2 + 1), dtype=complex)

    if numbers.shape != (N**2,):
        raise ValueError("Wrong shape! Should be (N^2,)")

    # make adequate shape arrs

    NYQ_vals = numbers[:4]
    numbers = numbers[4:]

    yellow_vals = numbers[: N - 2]
    yellow_vals = yellow_vals.reshape((N // 2 - 1, 2))
    numbers = numbers[N - 2 :]

    green_vals = numbers[: 2 * (N - 2)]
    green_vals = green_vals.reshape((N // 2 - 1, 2, 2))
    numbers = numbers[2 * (N - 2) :]

    blue_vals = numbers
    blue_vals = blue_vals.reshape((N - 1, N // 2 - 1, 2))

    # ASSIGN VALUES

    # place nyq modes (four of them and all real)
    nyq_idxs = jnp.array([(0, 0), (N // 2, 0), (0, N // 2), (N // 2, N // 2)])
    arr_hat_r = arr_hat_r.at[tuple(nyq_idxs.T)].set(NYQ_vals)

    # place yellow modes (complex)
    arr_hat_r = arr_hat_r.at[0, 1 : N // 2].set(
        yellow_vals[:, 0] + 1j * yellow_vals[:, 1]
    )

    # place green modes (complex)
    arr_hat_r = arr_hat_r.at[1 : N // 2, 0].set(
        green_vals[:, 0, 0] + 1j * green_vals[:, 1, 0]
    )
    arr_hat_r = arr_hat_r.at[1 : N // 2, N // 2].set(
        green_vals[:, 0, 1] + 1j * green_vals[:, 1, 1]
    )

    # place pink modes (copmlex conjugates of green modes)
    arr_hat_r = arr_hat_r.at[-N // 2 + 1 :, 0].set(
        jnp.conj(arr_hat_r[1 : N // 2, 0][::-1])
    )
    arr_hat_r = arr_hat_r.at[-N // 2 + 1 :, N // 2].set(
        jnp.conj(arr_hat_r[1 : N // 2, N // 2][::-1])
    )

    # blue modes
    arr_hat_r = arr_hat_r.at[1:, 1 : N // 2].set(
        blue_vals[:, :, 0] + 1j * blue_vals[:, :, 1]
    )

    return arr_hat_r


def from_rfft2_arr_to_fft2_arr(arr_hat_r, N):
    arr_hat_rec = jnp.zeros((N, N), dtype=complex)
    if arr_hat_r.shape != (N, N // 2 + 1):
        raise ValueError("Wrong shape! Should be (N,N//2+1)")
    arr_hat_rec = arr_hat_rec.at[:, : N // 2 + 1].set(arr_hat_r)
    arr_hat_rec = arr_hat_rec.at[0, -N // 2 + 1 :].set(
        jnp.conj(arr_hat_r[0, 1 : N // 2][::-1])
    )
    arr_hat_rec = arr_hat_rec.at[1:, -N // 2 + 1 :].set(
        jnp.conj(arr_hat_r[1:, 1 : N // 2][::-1, ::-1])
    )
    return arr_hat_rec


def get_polar_from_wn(arr, N, L):
    "arr is (N,N,N) of white noise"
    fact = L**3 * jnp.sqrt(1 / N**3) / jnp.sqrt(2)

    # this way real and imag part of arr have std = 1
    arr = my_fft(arr, L**3) / fact

    # print('a', arr.imag.std())

    # M_PLANE = N**2 // 2 - 2
    M_CUBOID = N**3 // 2 - N**2

    cuboid_vals = arr[:, :, 1 : N // 2]
    DC_vals = arr[:, :, 0]
    NYQ_vals = arr[:, :, N // 2 + 1]

    def get_rfft2_vals(arr, N):

        M1 = N // 2 - 1
        M2 = 2 * M1**2 + M1

        h_line = arr[0, 1 : M1 + 1]
        v_line = arr[1 : M1 + 1, 0]
        v_line_nyq = arr[1 : M1 + 1, N // 2]
        rectangle = arr[1:, 1 : M1 + 1].reshape((M2,))

        numbers = jnp.concatenate((h_line, v_line, v_line_nyq, rectangle))

        return numbers

    cuboid_numbers = cuboid_vals.reshape((M_CUBOID,))
    DC_numbers = get_rfft2_vals(DC_vals, N)
    NYQ_numbers = get_rfft2_vals(NYQ_vals, N)

    numbers = jnp.concatenate((cuboid_numbers, DC_numbers, NYQ_numbers))

    input_amps = jnp.abs(numbers)
    input_phases = (jnp.angle(numbers)) / (2 * jnp.pi) + 0.5

    nyq_idx = N // 2
    input_nyq = jnp.zeros(7)

    input_nyq = input_nyq.at[0].set(arr[nyq_idx, 0, 0].real)
    input_nyq = input_nyq.at[1].set(arr[0, nyq_idx, 0].real)
    input_nyq = input_nyq.at[2].set(arr[0, 0, nyq_idx].real)
    input_nyq = input_nyq.at[3].set(arr[nyq_idx, nyq_idx, 0].real)
    input_nyq = input_nyq.at[4].set(arr[0, nyq_idx, nyq_idx].real)
    input_nyq = input_nyq.at[5].set(arr[nyq_idx, 0, nyq_idx].real)
    input_nyq = input_nyq.at[6].set(arr[nyq_idx, nyq_idx, nyq_idx].real)

    return input_phases, input_amps, input_nyq


def get_wn_from_polar(x, N, L):

    M1 = N**3 // 2 - 4
    # M2 = 7

    x_ph = x[:M1]
    x_amp = x[M1 : 2 * M1]
    x_nyq = x[-7:]

    # phases
    phases = 2 * jnp.pi * (x_ph - 0.5)
    phases = make_rfft3_arr_from_phases(phases, N)
    z_unit = jnp.exp(1j * phases)

    # amps
    amps = make_rfft3_arr_from_amps(x_amp, N)

    ##
    u = z_unit * amps

    # place nyq modes
    nyq_idx = N // 2
    u = u.at[nyq_idx, 0, 0].set(x_nyq[0])
    u = u.at[0, nyq_idx, 0].set(x_nyq[1])
    u = u.at[0, 0, nyq_idx].set(x_nyq[2])

    u = u.at[nyq_idx, nyq_idx, 0].set(x_nyq[3])
    u = u.at[0, nyq_idx, nyq_idx].set(x_nyq[4])
    u = u.at[nyq_idx, 0, nyq_idx].set(x_nyq[5])

    u = u.at[nyq_idx, nyq_idx, nyq_idx].set(x_nyq[6])

    fact = L**3 * jnp.sqrt(1 / N**3) / jnp.sqrt(2)
    wn = my_ifft(u, 1 / L**3) * fact

    return wn
