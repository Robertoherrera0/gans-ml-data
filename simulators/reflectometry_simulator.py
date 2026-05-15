# -*- coding: utf-8 -*-
"""
TensorFlow implementation of Abeles reflectivity calculation with resolution smearing.
Port from PyTorch panpe implementation.
"""

import tensorflow as tf
import math


def abeles(q, thickness, roughness, sld):
    """
    Calculate neutron/X-ray reflectivity using Abeles matrix formalism.
    
    Args:
        q: (batch, q_points) or (q_points,) momentum transfer
        thickness: (batch, num_layers) layer thicknesses
        roughness: (batch, num_layers+1) interfacial roughness
        sld: (batch, num_layers+1) or (batch, num_layers+2) scattering length density
        
    Returns:
        r: (batch, q_points) reflectivity values
    """
    c_dtype = tf.complex128 if q.dtype == tf.float64 else tf.complex64
    
    batch_size = tf.shape(thickness)[0]
    num_layers = tf.shape(thickness)[1]
    
    # Handle SLD padding
    if sld.shape[-1] == num_layers + 1:
        # Add zero ambient SLD
        sld = tf.concat([tf.zeros([batch_size, 1], dtype=sld.dtype), sld], axis=-1)
    
    if sld.shape[-1] != num_layers + 2:
        raise ValueError(
            f"SLD shape mismatch: expected {num_layers + 2}, got {sld.shape[-1]}"
        )
    
    sld = sld[:, None, :]  # (batch, 1, layers)
    
    # Add zero thickness for ambient layer
    thickness = tf.concat(
        [tf.zeros([batch_size, 1], dtype=thickness.dtype), thickness], axis=-1
    )
    thickness = thickness[:, None, :]  # (batch, 1, layers)
    
    roughness = roughness[:, None, :] ** 2  # (batch, 1, layers)
    
    # Convert SLD to complex with small imaginary part for numerical stability
    sld = tf.cast((sld - sld[..., :1]) * 1e-6, c_dtype) + tf.cast(1e-36, c_dtype) * 1j
    
    # Wave vectors
    k_z0 = tf.cast(q / 2, c_dtype)
    if len(k_z0.shape) == 1:
        k_z0 = k_z0[None, :]
    if len(k_z0.shape) == 2:
        k_z0 = k_z0[..., None]  # (batch, q, 1)
    
    k_n = tf.sqrt(k_z0**2 - 4 * math.pi * sld)  # (batch, q, layers)
    
    k_n_curr = k_n[..., :-1]
    k_n_next = k_n[..., 1:]
    
    beta = tf.cast(1j, c_dtype) * tf.cast(thickness, c_dtype) * k_n_curr
    exp_beta = tf.exp(beta)
    exp_m_beta = tf.exp(-beta)
    
    # Fresnel coefficients with roughness - cast roughness to complex
    roughness_factor = tf.exp(-2 * k_n_curr * k_n_next * tf.cast(roughness, c_dtype))
    rn = (k_n_curr - k_n_next) / (k_n_curr + k_n_next) * roughness_factor
    
    # Characteristic matrices: shape (batch, q, layers-1, 2, 2)
    c11 = exp_beta
    c12 = rn * exp_m_beta
    c21 = rn * exp_beta
    c22 = exp_m_beta
    
    # Stack into matrices
    c_matrices = tf.stack([
        tf.stack([c11, c12], axis=-1),
        tf.stack([c21, c22], axis=-1)
    ], axis=-1)  # (batch, q, layers-1, 2, 2)
    
    # Matrix multiplication across layers
    # Split along layer dimension and reduce
    c_list = tf.unstack(c_matrices, axis=2)
    m = c_list[0]
    for c in c_list[1:]:
        m = tf.matmul(m, c)
    # Reflectivity from transfer matrix
    r = tf.abs(m[..., 1, 0] / m[..., 0, 0]) ** 2
    r = tf.minimum(r, 1.0)
    
    return r

def abeles_constant_smearing(
    q,
    thickness,
    roughness,
    sld,
    dq,
    gauss_num=51,
    xrr_dq=True,
):
    """
    Calculate smeared reflectivity with constant dQ/Q resolution.
    
    Args:
        q: (batch, q_points) or (q_points,) momentum transfer
        thickness: (batch, num_layers) layer thicknesses  
        roughness: (batch, num_layers+1) interfacial roughness
        sld: (batch, num_layers+1 or num_layers+2) scattering length density
        dq: (batch, q_points) or (1, q_points) resolution
        gauss_num: number of points in Gaussian kernel (default 51)
        xrr_dq: if True, use constant dQ; if False, use constant dQ/Q
        
    Returns:
        smeared_curves: (batch, q_points) smeared reflectivity
    """
    if dq is None:
        raise ValueError("dq must be provided for constant smearing")
    
    if len(dq.shape) == 1:
        dq = dq[None, :]
    
    batch_size = int(tf.shape(thickness)[0].numpy())
    
    if tf.shape(dq)[0] != batch_size:
        dq = tf.tile(dq, [batch_size, 1])
    
    # Generate high-resolution Q axis
    q_lin = _get_q_axes(q, dq, gauss_num, xrr_dq=xrr_dq)
    
    # Generate Gaussian kernels
    kernels = _get_t_gauss_kernels(dq, gauss_num)
    
    # Calculate reflectivity on high-res grid
    curves = abeles(q_lin, thickness, roughness, sld)
    
    # Apply Gaussian convolution
    padding = (tf.shape(kernels)[-1] - 1) // 2
    curves_padded = tf.pad(curves, [[0, 0], [padding, padding]], mode='REFLECT')
    
    # Manual convolution per batch item
    smeared_curves = []
    for i in range(batch_size):
        # kernel shape: (51,) -> (51, 1, 1) for conv1d
        kernel = kernels[i, :, None, None]
        # curve shape: (N,) -> (1, N, 1) for conv1d
        curve_input = curves_padded[i:i+1, :, None]
        
        conv_result = tf.nn.conv1d(
            curve_input,
            kernel,
            stride=1,
            padding='VALID'
        )
        smeared_curves.append(conv_result[0, :, 0])
    
    smeared_curves = tf.stack(smeared_curves, axis=0)
    
    # Interpolate back to original Q grid
    if len(q.shape) == 1:
        q = tf.tile(q[None, :], [batch_size, 1])
    elif tf.shape(q)[0] != batch_size:
        q = tf.tile(q, [batch_size, 1])
    
    smeared_curves = _batch_linear_interp1d(q_lin, smeared_curves, q)
    
    return smeared_curves


# Helper functions
_FWHM = 2 * math.sqrt(2 * math.log(2.0))
_2PI_SQRT = 1.0 / math.sqrt(2 * math.pi)


def _batch_linspace(start, end, num):
    """Batched linspace from start to end."""
    steps = tf.linspace(0.0, 1.0, num)
    steps = tf.cast(steps[None, :], start.dtype)  # Match dtype
    return steps * (end - start) + start

def _torch_gauss(x, s):
    """Gaussian function."""
    return tf.cast(_2PI_SQRT, x.dtype) / s * tf.exp(-0.5 * x**2 / s / s)

def _get_t_gauss_kernels(resolutions, gaussnum=51):
    """Generate Gaussian convolution kernels."""
    # resolutions should be (batch, 1) - one value per batch
    gauss_x = _batch_linspace(-1.7 * resolutions, 1.7 * resolutions, gaussnum)
    dx = gauss_x[:, 1:2] - gauss_x[:, 0:1]
    gauss_y = _torch_gauss(gauss_x, resolutions / _FWHM) * dx
    return gauss_y

def _get_q_axes(q, resolutions, gaussnum=51, xrr_dq=True):
    """Generate high-resolution Q axis for convolution."""
    if xrr_dq:
        return _get_q_axes_for_constant_dq(q, resolutions, gaussnum)
    else:
        return _get_q_axes_for_linear_dq(q, resolutions, gaussnum)


def _get_q_axes_for_constant_dq(q, resolutions, gaussnum=51):
    """Q axis for constant dQ resolution."""
    gaussgpoint = (gaussnum - 1) / 2
    
    start = tf.reduce_min(q, axis=1, keepdims=True) - resolutions * 1.7
    end = tf.reduce_max(q, axis=1, keepdims=True) + resolutions * 1.7
    
    interpnums = tf.cast(
        tf.round(tf.abs((end - start)) / (1.7 * resolutions / gaussgpoint)),
        tf.int32
    )
    
    q_lin = _batch_linspace_with_padding(start, end, interpnums)
    q_lin = tf.maximum(q_lin, 1e-6)
    
    return q_lin


def _get_q_axes_for_constant_dq(q, resolutions, gaussnum=51):
    """Q axis for constant dQ resolution."""
    gaussgpoint = (gaussnum - 1) / 2
    
    start = tf.reduce_min(q, axis=1, keepdims=True) - resolutions * 1.7
    end = tf.reduce_max(q, axis=1, keepdims=True) + resolutions * 1.7
    
    interpnums = tf.cast(
        tf.round(tf.abs((end - start)) / (1.7 * resolutions / gaussgpoint)),
        tf.int32
    )
    
    q_lin = _batch_linspace_with_padding(start, end, interpnums)
    q_lin = tf.maximum(q_lin, 1e-6)
    
    return q_lin

def _batch_linspace_with_padding(start, end, nums):
    """Batched linspace with variable lengths (padded to max)."""
    max_num = int(tf.reduce_max(nums).numpy())
    batch_size = int(tf.shape(start)[0].numpy())
    
    result = []
    for i in range(batch_size):
        line = tf.linspace(start[i, 0], end[i, 0], max_num)
        result.append(line)
    
    return tf.stack(result, axis=0)

def _batch_linear_interp1d(x, y, x_new):
    """Batched 1D linear interpolation."""
    eps = tf.experimental.numpy.finfo(y.dtype).eps
    
    # Find indices
    ind = tf.searchsorted(x, x_new)
    ind = tf.clip_by_value(ind - 1, 0, tf.shape(x)[-1] - 2)
    
    # Calculate slopes
    slopes = (y[:, 1:] - y[:, :-1]) / (eps + (x[:, 1:] - x[:, :-1]))
    
    # Gather values
    batch_size = tf.shape(y)[0]
    batch_indices = tf.range(batch_size)[:, None]
    
    y_gathered = tf.gather_nd(y, tf.stack([
        tf.tile(batch_indices, [1, tf.shape(x_new)[1]]),
        ind
    ], axis=-1))
    
    slopes_gathered = tf.gather_nd(slopes, tf.stack([
        tf.tile(batch_indices, [1, tf.shape(x_new)[1]]),
        ind
    ], axis=-1))
    
    x_gathered = tf.gather_nd(x, tf.stack([
        tf.tile(batch_indices, [1, tf.shape(x_new)[1]]),
        ind
    ], axis=-1))
    
    y_new = y_gathered + slopes_gathered * (x_new - x_gathered)
    
    return y_new