# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from math import pi, sqrt, log
from functools import reduce
import jax.numpy as jnp
from jax import Array, vmap
import jax.scipy.signal


def abeles_constant_smearing(
    q: Array,
    thickness: Array,
    roughness: Array,
    sld: Array,
    dq: Array = None,
    gauss_num: int = 51,
    xrr_dq: bool = True,
    abeles_func=None,
):

    assert dq is not None

    dq = jnp.atleast_2d(dq)

    batch_size = thickness.shape[0]

    if dq.shape[0] != batch_size:
        dq = jnp.broadcast_to(dq, (batch_size, dq.shape[-1]))

    abeles_func = abeles_func or abeles

    q_lin = _get_q_axes(q, dq, gauss_num, xrr_dq=xrr_dq)
    kernels = _get_t_gauss_kernels(dq, gauss_num)

    curves = abeles_func(q_lin, thickness, roughness, sld)

    padding = (kernels.shape[-1] - 1) // 2
    
    # JAX doesn't have conv1d with groups, use convolve instead
    smeared_curves = jnp.zeros_like(curves)
    curves_padded = jnp.pad(curves, ((0, 0), (padding, padding)), mode='reflect')
    
    for i in range(kernels.shape[0]):
        smeared_curves = smeared_curves.at[i].set(
            jax.scipy.signal.convolve(curves_padded[i], kernels[i], mode='valid')
        )

    if q.shape[0] != smeared_curves.shape[0]:
        q = jnp.broadcast_to(q, (smeared_curves.shape[0], *q.shape[1:]))

    smeared_curves = _batch_linear_interp1d(q_lin, smeared_curves, q)

    return smeared_curves


_FWHM = 2 * sqrt(2 * log(2.0))
_2PI_SQRT = 1.0 / sqrt(2 * pi)


def _batch_linspace(start: Array, end: Array, num: int):
    return (
        jnp.linspace(0, 1, int(num))[None]
        * (end - start)
        + start
    )


def _torch_gauss(x, s):
    return _2PI_SQRT / s * jnp.exp(-0.5 * x**2 / s / s)


def _get_t_gauss_kernels(resolutions: Array, gaussnum: int = 51):
    gauss_x = _batch_linspace(-1.7 * resolutions, 1.7 * resolutions, gaussnum)
    gauss_y = (
        _torch_gauss(gauss_x, resolutions / _FWHM)
        * (gauss_x[:, 1] - gauss_x[:, 0])[:, None]
    )
    return gauss_y


def _get_q_axes(
    q: Array, resolutions: Array, gaussnum: int = 51, xrr_dq: bool = True
):
    if xrr_dq:
        return _get_q_axes_for_constant_dq(q, resolutions, gaussnum)
    else:
        return _get_q_axes_for_linear_dq(q, resolutions, gaussnum)


def _get_q_axes_for_linear_dq(q: Array, resolutions: Array, gaussnum: int = 51):
    gaussgpoint = (gaussnum - 1) / 2

    lowq = jnp.maximum(q.min(1), 1e-6)[..., None]
    highq = q.max(1)[..., None]

    start = jnp.log10(lowq) - 6 * resolutions / _FWHM
    end = jnp.log10(highq * (1 + 6 * resolutions / _FWHM))

    interpnums = (
        jnp.abs((jnp.abs(end - start)) / (1.7 * resolutions / _FWHM / gaussgpoint))
        .round()
        .astype(int)
    )

    q_lin = 10 ** _batch_linspace_with_padding(start, end, interpnums)

    return q_lin


def _get_q_axes_for_constant_dq(
    q: Array, resolutions: Array, gaussnum: int = 51
) -> Array:
    gaussgpoint = (gaussnum - 1) / 2

    start = q.min(1)[:, None] - resolutions * 1.7
    end = q.max(1)[:, None] + resolutions * 1.7

    interpnums = (
        jnp.abs((jnp.abs(end - start)) / (1.7 * resolutions / gaussgpoint))
        .round()
        .astype(int)
    )

    q_lin = _batch_linspace_with_padding(start, end, interpnums)
    q_lin = jnp.maximum(q_lin, 1e-6)

    return q_lin


def _batch_linspace_with_padding(start: Array, end: Array, nums: Array) -> Array:
    max_num = nums.max().astype(int)

    deltas = 1 / (nums - 1)

    x = jnp.maximum(
        _batch_linspace(deltas * (nums - max_num), jnp.ones_like(deltas), max_num), 0
    )

    x = x * (end - start) + start

    return x


def _batch_linear_interp1d(x: Array, y: Array, x_new: Array) -> Array:
    eps = jnp.finfo(y.dtype).eps

    # Handle searchsorted per batch element since JAX doesn't support batched searchsorted
    def interp_single(x_single, y_single, x_new_single):
        ind = jnp.searchsorted(x_single, x_new_single)
        ind = jnp.clip(ind - 1, 0, x_single.shape[-1] - 2)
        slopes = (y_single[1:] - y_single[:-1]) / (eps + (x_single[1:] - x_single[:-1]))
        y_new = y_single[ind] + slopes[ind] * (x_new_single - x_single[ind])
        return y_new
    
    y_new = vmap(interp_single)(x, y, x_new)
    
    return y_new


def abeles(
    q: Array,
    thickness: Array,
    roughness: Array,
    sld: Array,
):
    c_dtype = jnp.complex128 if q.dtype == jnp.float64 else jnp.complex64

    batch_size, num_layers = thickness.shape

    if sld.shape[-1] == num_layers + 1:
        # add zero ambient sld
        sld = jnp.concatenate([jnp.zeros((batch_size, 1)), sld], -1)
    if sld.shape[-1] != num_layers + 2:
        raise ValueError(
            "Number of SLD values does not equal to num_layers + 2 (substrate + ambient)."
        )

    sld = sld[:, None]

    # add zero thickness for ambient layer:
    thickness = jnp.concatenate([jnp.zeros((batch_size, 1)), thickness], -1)[:, None]

    roughness = roughness[:, None] ** 2

    sld = (sld - sld[..., :1]) * 1e-6 + 1e-36j

    k_z0 = (q / 2).astype(c_dtype)

    if k_z0.ndim == 1:
        k_z0 = jnp.expand_dims(k_z0, 0)

    if k_z0.ndim == 2:
        k_z0 = jnp.expand_dims(k_z0, -1)

    k_n = jnp.sqrt(k_z0**2 - 4 * pi * sld)

    # k_n.shape - (batch, q, layers)

    k_n, k_np1 = k_n[..., :-1], k_n[..., 1:]

    beta = 1j * thickness * k_n

    exp_beta = jnp.exp(beta)
    exp_m_beta = jnp.exp(-beta)

    rn = (k_n - k_np1) / (k_n + k_np1) * jnp.exp(-2 * k_n * k_np1 * roughness)

    c_matrices = jnp.stack(
        [
            jnp.stack([exp_beta, rn * exp_m_beta], -1),
            jnp.stack([rn * exp_beta, exp_m_beta], -1),
        ],
        -1,
    )

    # Move layer axis to front and split into list
    c_matrices = jnp.moveaxis(c_matrices, -3, 0)
    c_matrices = [c_matrices[i] for i in range(c_matrices.shape[0])]

    m = reduce(jnp.matmul, c_matrices)

    r = jnp.abs(m[..., 1, 0] / m[..., 0, 0]) ** 2
    r = jnp.minimum(r, 1.0)

    return r

def abeles_compiled(
    q: Array,
    thickness: Array,
    roughness: Array,
    sld: Array,
):
    return abeles(q, thickness, roughness, sld)