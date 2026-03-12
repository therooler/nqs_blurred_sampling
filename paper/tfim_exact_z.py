import sys
import os
import numpy as np
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.append("../src")
from fgo import FGOState
# TFIM model in terms of Majorana fermions.
# see arXiv:2511.02907 for the details of the mapping between TFIM and Majorana fermions

def gen_tfim(n, J, h, PX=1):
    H = np.zeros([2 * n, 2 * n], dtype=np.complex128)
    idx = np.arange(n)
    H[2 * idx, 2 * idx + 1] = -1j * h

    H[2 * idx[:-1] + 1, 2 * idx[:-1] + 2] = -1j * J
    H[2 * n - 1, 0] = 1j * J * PX
    return 2 * (H - H.T)


def gen_tfim_flip_x(n, J, h, flip_idx, PX=1):
    H = np.zeros([2 * n, 2 * n], dtype=np.complex128)
    idx = np.arange(n)
    H[2 * idx, 2 * idx + 1] = -1j * h
    H[2 * flip_idx, 2 * flip_idx + 1] = 1j * h

    H[2 * idx[:-1] + 1, 2 * idx[:-1] + 2] = -1j * J
    H[2 * n - 1, 0] = 1j * J * PX
    return 2 * (H - H.T)

# \ket{+} state in Majorana representation.
def gen_plus_jax(n):
    N = 2 * n
    H = jnp.zeros((N, N), dtype=jnp.complex128)
    idx = 2 * jnp.arange(n)
    H = H.at[idx, idx + 1].set(1.0j)
    Hsym = H - H.T
    _, v = jnp.linalg.eigh(Hsym)
    return v[:, :n]


# ket{-} state in Majorana representation. X_i = 1, X_i = -1 for the last site.
def gen_minus_jax(n):
    N = 2 * n
    H = jnp.zeros((N, N), dtype=jnp.complex128)
    idx = 2 * jnp.arange(n - 1)
    H = H.at[idx, idx + 1].set(1.0j)
    H = H.at[2 * (n - 1), 2 * (n - 1) + 1].set(-1.0j)
    Hsym = H - H.T
    _, v = jnp.linalg.eigh(Hsym)
    return v[:, :n]


# ZZ stablizer state in Majorana representation. zz is the eigenvalue of Z_i Z_{i+1}.
def gen_v_from_zz(zz, PX=1):
    zz = jnp.asarray(zz)
    n = zz.shape[-1]
    N = 2 * n

    sgn = jnp.sign(zz)
    sgn = sgn.at[..., -1].set(-PX * sgn[..., -1])  # parity convention

    norm = 1.0 / jnp.sqrt(2.0)
    top = (-1j * sgn) * norm
    bot = (1.0 + 0j) * norm

    a_idx = 2 * jnp.arange(n) + 1  # (n,)
    b_idx = (2 * jnp.arange(n) + 2) % N
    col_idx = jnp.arange(n)

    batch_shape = zz.shape[:-1]
    v = jnp.zeros(batch_shape + (N, n), dtype=jnp.complex128)
    v = v.at[..., a_idx, col_idx].set(top)
    v = v.at[..., b_idx, col_idx].set(bot)
    return v


# \ket{s} + PX\ket{-s} state in Majorana representation.
def v_from_s(s, PX=1):
    zz_val = s * jnp.roll(s, -1, axis=-1)
    return gen_v_from_zz(zz_val, PX)

# calculate the total Z magnetization for the state exp(-iHt) |s0> where |s0> is a product state in the Z basis.
# s0_i(<s0+|expih(x_i->-x_i)t exp-iht |s0+> + <s0-|expih(x_i->-x_i)t exp-iht |s0->)
def sz_peaked_state(n, J, h, t, s0):
    s0_plus = v_from_s(s0, PX=1)
    s0_minus = v_from_s(s0, PX=-1)
    fgo_s0_plus = FGOState(0.0, s0_plus, s0_plus)
    fgo_s0_minus = FGOState(0.0, s0_minus, s0_minus)

    H_plus = -1.0j * t * gen_tfim(n, J, h, PX=1)
    H_minus = -1.0j * t * gen_tfim(n, J, h, PX=-1)
    fgo_expmih_plus = fgo_s0_plus.apply_quadratic(H_plus)
    fgo_expmih_minus = fgo_s0_minus.apply_quadratic(H_minus)

    total_z = 0.0
    for i in range(n):
        H_flip_plus = 1.0j * t * gen_tfim_flip_x(n, J, h, i, PX=1)
        H_flip_minus = 1.0j * t * gen_tfim_flip_x(n, J, h, i, PX=-1)

        fgo_plus = fgo_expmih_plus.apply_quadratic(H_flip_plus)
        fgo_minus = fgo_expmih_minus.apply_quadratic(H_flip_minus)

        total_z += s0[i] / 2 * (jnp.exp(fgo_plus.logeta) + jnp.exp(fgo_minus.logeta))

    return jnp.real(total_z)

def get_ed(J,h,N, T):
    tlist = np.arange(0, T, T/100)

    s0 = jnp.array([-1] * N)
    total_z = []
    save_path = f"./data/TFIM_exact_Z/"
    path = save_path + f"times_{N}_h{h:1.3f}_J{J:1.2f}_T{T:1.2f}.npy"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(path):
        for i, t in enumerate(tlist):
            if i % 50 == 0:
                print(f"t={t:.3f}")
            total_z.append(sz_peaked_state(N, J, h, t, s0))
        np.save(save_path + f"times_{N}_h{h:1.3f}_J{J:1.2f}_T{T:1.2f}.npy", tlist)
        np.save(save_path + f"exactZ_{N}_h{h:1.3f}_J{J:1.2f}_T{T:1.2f}.npy", total_z )

    tlist = np.load(save_path + f"times_{N}_h{h:1.3f}_J{J:1.2f}_T{T:1.2f}.npy")
    pZlist = np.load(save_path + f"exactZ_{N}_h{h:1.3f}_J{J:1.2f}_T{T:1.2f}.npy")
    return tlist, pZlist