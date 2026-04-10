"""
Standalone test: verifies that build_sld_stack + abeles + apply_smearing
matches calculate_reflectivity (refl1d) for all 3 contrasts.
"""

import math
import numpy as np
import torch
from functools import reduce
from refl1d.names import QProbe, Slab, Experiment, SLD, Parameter

TEST_PARAMS = {
    "substrate_roughness": 5.0,
    "sio2_thickness":      20.0,
    "sio2_sld":            3.41,
    "sio2_roughness":      3.0,
    "head1_thickness":     10.0,
    "head1_sld":           2.0,
    "head1_roughness":     2.0,
    "tail_thickness":      30.0,
    "tail_sld":            0.5,
    "tail_roughness":      3.0,
    "head2_thickness":     10.0,
    "head2_sld":           2.0,
    "head2_roughness":     2.0,
    "medicine_thickness":  50.0,
    "medicine_sld":        2.5,
    "medicine_roughness":  4.0,
}

PARAMS_PHYS = np.array([
    TEST_PARAMS["substrate_roughness"],
    TEST_PARAMS["sio2_thickness"],     TEST_PARAMS["sio2_sld"],     TEST_PARAMS["sio2_roughness"],
    TEST_PARAMS["head1_thickness"],    TEST_PARAMS["head1_sld"],    TEST_PARAMS["head1_roughness"],
    TEST_PARAMS["tail_thickness"],     TEST_PARAMS["tail_sld"],     TEST_PARAMS["tail_roughness"],
    TEST_PARAMS["head2_thickness"],    TEST_PARAMS["head2_sld"],    TEST_PARAMS["head2_roughness"],
    TEST_PARAMS["medicine_thickness"], TEST_PARAMS["medicine_sld"], TEST_PARAMS["medicine_roughness"],
], dtype=np.float32)

CONTRAST_SLDS  = [-0.56, 6.36, 2.07]
SLD_SUBSTRATE  = 2.07
WAVELENGTH_RES = 0.019959062306768447
q              = np.logspace(np.log10(0.005), np.log10(0.25), 250).astype(np.float32)
PASS_THRESHOLD = 1e-2


def refl1d_reflectivity(q, solvent_sld, smearing=False):
    dq    = WAVELENGTH_RES * q / 2.355 if smearing else np.zeros_like(q)
    zeros = np.zeros(len(q))
    probe = QProbe(q, dq, data=(zeros, zeros))
    p     = TEST_PARAMS
    layers = [
        dict(name="substrate", sld=SLD_SUBSTRATE,      isld=0, thickness=0,                       roughness=p["substrate_roughness"]),
        dict(name="sio2",      sld=p["sio2_sld"],      isld=0, thickness=p["sio2_thickness"],      roughness=p["sio2_roughness"]),
        dict(name="head1",     sld=p["head1_sld"],     isld=0, thickness=p["head1_thickness"],     roughness=p["head1_roughness"]),
        dict(name="tail",      sld=p["tail_sld"],      isld=0, thickness=p["tail_thickness"],      roughness=p["tail_roughness"]),
        dict(name="head2",     sld=p["head2_sld"],     isld=0, thickness=p["head2_thickness"],     roughness=p["head2_roughness"]),
        dict(name="medicine",  sld=p["medicine_sld"],  isld=0, thickness=p["medicine_thickness"],  roughness=p["medicine_roughness"]),
        dict(name="medium",    sld=solvent_sld,        isld=0, thickness=0,                        roughness=0.0),
    ]
    sample = Slab(material=SLD(name=layers[0]["name"], rho=layers[0]["sld"]), interface=layers[0]["roughness"])
    for l in layers[1:]:
        sample = sample | Slab(material=SLD(name=l["name"], rho=l["sld"], irho=l["isld"]),
                               thickness=l["thickness"], interface=l["roughness"])
    probe.background = Parameter(value=0, name="background")
    _, r = Experiment(probe=probe, sample=sample).reflectivity()
    return r


def abeles(q, thickness, roughness, sld):
    c_dtype    = torch.complex128 if q.dtype == torch.float64 else torch.complex64
    batch_size = thickness.shape[0]
    sld        = sld * 1e-6
    sld        = sld[:, None].to(c_dtype)
    thickness  = torch.cat([torch.zeros(batch_size, 1), thickness], -1)[:, None]
    roughness  = (roughness[:, None] ** 2).to(c_dtype)
    sld        = (sld - sld[..., :1]) + 1e-36j
    k_z0       = (q / 2).to(c_dtype)[None, :, None]
    k_n        = torch.sqrt(k_z0 ** 2 - 4 * math.pi * sld)
    k_n, k_np1 = k_n[..., :-1], k_n[..., 1:]
    beta       = 1j * thickness * k_n
    rn         = (k_n - k_np1) / (k_n + k_np1 + 1e-30) * torch.exp(-2 * k_n * k_np1 * roughness)
    c_matrices = torch.stack([
        torch.stack([torch.exp(beta),      rn * torch.exp(-beta)], -1),
        torch.stack([rn * torch.exp(beta), torch.exp(-beta)],      -1),
    ], -1)
    c_matrices = [c.squeeze(-3) for c in c_matrices.split(1, -3)]
    m          = reduce(torch.matmul, c_matrices)
    r          = (m[..., 1, 0] / (m[..., 0, 0] + 1e-30)).abs() ** 2
    return r.clamp(0.0, 1.0)


def apply_smearing(R_sim, q, wavelength_res=WAVELENGTH_RES):
    nodes   = torch.tensor([-2.0202, -0.9586, 0.0, 0.9586, 2.0202], device=q.device, dtype=q.dtype)
    weights = torch.tensor([ 0.1995,  0.3936, 0.9454, 0.3936, 0.1995], device=q.device, dtype=q.dtype)
    weights = weights / weights.sum()
    sigma   = wavelength_res * q / 2.355
    R_out   = torch.zeros_like(R_sim)
    for node, weight in zip(nodes, weights):
        q_shift  = (q + node * sigma).clamp(min=q[0], max=q[-1])
        idx      = torch.searchsorted(q.contiguous(), q_shift.contiguous()).clamp(1, len(q) - 1)
        q_lo, q_hi = q[idx - 1], q[idx]
        t        = ((q_shift - q_lo) / (q_hi - q_lo + 1e-30)).clamp(0, 1)
        R_interp = (1 - t) * R_sim[:, idx - 1] + t * R_sim[:, idx]
        R_out   += weight * R_interp
    return R_out


def build_sld_stack(params_phys, solvent_sld):
    B             = params_phys.shape[0]
    solvent_col   = torch.full((B, 1), solvent_sld)
    substrate_col = torch.full((B, 1), SLD_SUBSTRATE)
    layer_slds    = params_phys[:, [2, 5, 8, 11, 14]]
    sld           = torch.cat([solvent_col, layer_slds, substrate_col], dim=1)
    thickness     = params_phys[:, [1, 4, 7, 10, 13]]
    roughness     = params_phys[:, [0, 3, 6, 9, 12, 15]]
    return sld, thickness, roughness


def run_test():
    q_t      = torch.tensor(q)
    p_tensor = torch.tensor(PARAMS_PHYS).unsqueeze(0)

    print("--- without smearing (abeles vs refl1d dq=0) ---")
    print(f"{'Contrast':<10} {'Max err':>12} {'Pass?':>8}")
    print("-" * 35)
    for solvent_sld in CONTRAST_SLDS:
        r_ref    = refl1d_reflectivity(q, solvent_sld, smearing=False)
        sld, th, ro = build_sld_stack(p_tensor, solvent_sld)
        r_abeles = abeles(q_t, th, ro, sld).squeeze().numpy()
        err      = np.max(np.abs(r_abeles - r_ref))
        print(f"{solvent_sld:<10} {err:>12.6f} {'YES' if err < PASS_THRESHOLD else 'NO':>8}")

    print()
    print("--- with smearing (abeles+smearing vs refl1d with dq) ---")
    print(f"{'Contrast':<10} {'Max err':>12} {'Pass?':>8}")
    print("-" * 35)
    all_pass = True
    for solvent_sld in CONTRAST_SLDS:
        r_ref    = refl1d_reflectivity(q, solvent_sld, smearing=True)
        sld, th, ro = build_sld_stack(p_tensor, solvent_sld)
        R_sim    = abeles(q_t, th, ro, sld)
        R_sim    = apply_smearing(R_sim, q_t)
        r_abeles = R_sim.squeeze().numpy()
        err      = np.max(np.abs(r_abeles - r_ref))
        passed   = err < PASS_THRESHOLD
        print(f"{solvent_sld:<10} {err:>12.6f} {'YES' if passed else 'NO':>8}")
        if not passed:
            all_pass = False

    print("-" * 35)
    print("All tests passed!" if all_pass else "FAILED")


if __name__ == "__main__":
    run_test()