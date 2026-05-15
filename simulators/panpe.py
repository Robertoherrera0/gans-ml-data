# -*- coding: utf-8 -*-
# Standalone PyTorch Abeles implementation
# Licensed under the MIT license

from __future__ import annotations

import math
from functools import reduce
from math import pi, sqrt, log

import torch
from torch import Tensor
from torch.nn.functional import conv1d, pad


def abeles(
    q: Tensor,
    thickness: Tensor,
    roughness: Tensor,
    sld: Tensor,
):
    c_dtype = torch.complex128 if q.dtype is torch.float64 else torch.complex64

    batch_size, num_layers = thickness.shape

    if sld.shape[-1] == num_layers + 1:
        # add zero ambient sld
        sld = torch.cat([torch.zeros(batch_size, 1).to(sld), sld], -1)
    if sld.shape[-1] != num_layers + 2:
        raise ValueError(
            "Number of SLD values does not equal to num_layers + 2 (substrate + ambient)."
        )

    sld = sld[:, None]

    # add zero thickness for ambient layer:
    thickness = torch.cat([torch.zeros(batch_size, 1).to(thickness), thickness], -1)[
        :, None
    ]

    roughness = roughness[:, None] ** 2

    sld = (sld - sld[..., :1]) * 1e-6 + 1e-36j

    k_z0 = (q / 2).to(c_dtype)

    if k_z0.dim() == 1:
        k_z0.unsqueeze_(0)

    if k_z0.dim() == 2:
        k_z0.unsqueeze_(-1)

    k_n = torch.sqrt(k_z0**2 - 4 * math.pi * sld)

    # k_n.shape - (batch, q, layers)

    k_n, k_np1 = k_n[..., :-1], k_n[..., 1:]

    beta = 1j * thickness * k_n

    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)

    rn = (k_n - k_np1) / (k_n + k_np1) * torch.exp(-2 * k_n * k_np1 * roughness)

    c_matrices = torch.stack(
        [
            torch.stack([exp_beta, rn * exp_m_beta], -1),
            torch.stack([rn * exp_beta, exp_m_beta], -1),
        ],
        -1,
    )

    # maybe faster to swap axes and provide a single tensor to reduce
    c_matrices = [c.squeeze(-3) for c in c_matrices.split(1, -3)]

    m = reduce(torch.matmul, c_matrices)

    r = (m[..., 1, 0] / m[..., 0, 0]).abs() ** 2
    r = torch.clamp_max_(r, 1.0)

    return r


def abeles_constant_smearing(
    q: Tensor,
    thickness: Tensor,
    roughness: Tensor,
    sld: Tensor,
    dq: Tensor = None,
    gauss_num: int = 51,
    xrr_dq: bool = True,
    abeles_func=None,
):

    assert dq is not None

    dq = torch.atleast_2d(dq)

    batch_size = thickness.shape[0]

    if dq.shape[0] != batch_size:
        dq = dq.expand(batch_size, dq.shape[-1])

    abeles_func = abeles_func or abeles

    q_lin = _get_q_axes(q, dq, gauss_num, xrr_dq=xrr_dq)
    kernels = _get_t_gauss_kernels(dq, gauss_num)

    curves = abeles_func(q_lin, thickness, roughness, sld)

    padding = (kernels.shape[-1] - 1) // 2
    smeared_curves = conv1d(
        pad(curves[None], (padding, padding), "reflect"),
        kernels[:, None],
        groups=kernels.shape[0],
    )[0]

    if q.shape[0] != smeared_curves.shape[0]:
        q = q.expand(smeared_curves.shape[0], *q.shape[1:])

    smeared_curves = _batch_linear_interp1d(q_lin, smeared_curves, q)

    return smeared_curves


_FWHM = 2 * sqrt(2 * log(2.0))
_2PI_SQRT = 1.0 / sqrt(2 * pi)


def _batch_linspace(start: Tensor, end: Tensor, num: int):
    return (
        torch.linspace(0, 1, int(num), device=end.device, dtype=end.dtype)[None]
        * (end - start)
        + start
    )


def _torch_gauss(x, s):
    return _2PI_SQRT / s * torch.exp(-0.5 * x**2 / s / s)


def _get_t_gauss_kernels(resolutions: Tensor, gaussnum: int = 51):
    gauss_x = _batch_linspace(-1.7 * resolutions, 1.7 * resolutions, gaussnum)
    gauss_y = (
        _torch_gauss(gauss_x, resolutions / _FWHM)
        * (gauss_x[:, 1] - gauss_x[:, 0])[:, None]
    )
    return gauss_y


def _get_q_axes(
    q: Tensor, resolutions: Tensor, gaussnum: int = 51, xrr_dq: bool = True
):
    if xrr_dq:
        return _get_q_axes_for_constant_dq(q, resolutions, gaussnum)
    else:
        return _get_q_axes_for_linear_dq(q, resolutions, gaussnum)


def _get_q_axes_for_linear_dq(q: Tensor, resolutions: Tensor, gaussnum: int = 51):
    gaussgpoint = (gaussnum - 1) / 2

    lowq = torch.clamp_min_(q.min(1).values, 1e-6)[..., None]
    highq = q.max(1).values[..., None]

    start = torch.log10(lowq) - 6 * resolutions / _FWHM
    end = torch.log10(highq * (1 + 6 * resolutions / _FWHM))

    interpnums = (
        torch.abs((torch.abs(end - start)) / (1.7 * resolutions / _FWHM / gaussgpoint))
        .round()
        .to(int)
    )

    q_lin = 10 ** _batch_linspace_with_padding(start, end, interpnums)

    return q_lin


def _get_q_axes_for_constant_dq(
    q: Tensor, resolutions: Tensor, gaussnum: int = 51
) -> Tensor:
    gaussgpoint = (gaussnum - 1) / 2

    start = q.min(1).values[:, None] - resolutions * 1.7
    end = q.max(1).values[:, None] + resolutions * 1.7

    interpnums = (
        torch.abs((torch.abs(end - start)) / (1.7 * resolutions / gaussgpoint))
        .round()
        .to(int)
    )

    q_lin = _batch_linspace_with_padding(start, end, interpnums)
    q_lin = torch.clamp_min_(q_lin, 1e-6)

    return q_lin


def _batch_linspace_with_padding(start: Tensor, end: Tensor, nums: Tensor) -> Tensor:
    max_num = nums.max().int().item()

    deltas = 1 / (nums - 1)

    x = torch.clamp_min_(
        _batch_linspace(deltas * (nums - max_num), torch.ones_like(deltas), max_num), 0
    )

    x = x * (end - start) + start

    return x


def _batch_linear_interp1d(x: Tensor, y: Tensor, x_new: Tensor) -> Tensor:
    eps = torch.finfo(y.dtype).eps

    ind = torch.searchsorted(x.contiguous(), x_new.contiguous())

    ind = torch.clamp_(ind - 1, 0, x.shape[-1] - 2)
    slopes = (y[..., 1:] - y[..., :-1]) / (eps + (x[..., 1:] - x[..., :-1]))
    ind_y = (
        ind + torch.arange(slopes.shape[0], device=slopes.device)[:, None] * y.shape[1]
    )
    ind_slopes = (
        ind
        + torch.arange(slopes.shape[0], device=slopes.device)[:, None] * slopes.shape[1]
    )

    y_new = y.flatten()[ind_y] + slopes.flatten()[ind_slopes] * (
        x_new - x.flatten()[ind_y]
    )

    return y_new