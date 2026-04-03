import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.append('/c/Users/rdhc54/GANS-AI/panpe')
from panpe.simulator.reflectivity.abeles import abeles
from panpe.simulator.reflectivity.smearing import abeles_constant_smearing

DATA_DIR = Path("fixed-medium-dataset")
X_mem = np.load(DATA_DIR / "gans_flowcell_reflectivity_data.npy")
Y_mem = np.load(DATA_DIR / "gans_flowcell_sample_parameters_norm.npy")
q_grid = np.load(DATA_DIR / "gans_flowcell_q_grid.npy")
with open(DATA_DIR / "gans_flowcell_metadata.json") as f:
    metadata = json.load(f)

param_bounds_list = [[p["bounds"][0], p["bounds"][1]] for p in metadata["parameters"]]
contrast_slds = [-0.56, 6.36, 2.07]

def parratt_torch(params_norm, q_grid, param_bounds_list, contrast_slds):
    B = params_norm.shape[0]
    device = params_norm.device
    bounds = torch.tensor(param_bounds_list, dtype=torch.float32, device=device)
    lo, hi = bounds[:, 0], bounds[:, 1]
    params_phys = ((params_norm + 1.0) / 2.0) * (hi - lo) + lo

    # reversed order: medium → Medicine → Head2 → Tail → Head1 → SiO2 → Si
    thickness = torch.stack([
        params_phys[:, 13],  # Medicine
        params_phys[:, 10],  # Head2
        params_phys[:, 7],   # Tail
        params_phys[:, 4],   # Head1
        params_phys[:, 1],   # SiO2
    ], dim=1)

    roughness = torch.stack([
        params_phys[:, 0],
        torch.min(params_phys[:, 15], params_phys[:, 13]),  # Medicine
        torch.min(params_phys[:, 12], params_phys[:, 10]),  # Head2
        torch.min(params_phys[:, 9],  params_phys[:, 7]),   # Tail
        torch.min(params_phys[:, 6],  params_phys[:, 4]),   # Head1
        torch.min(params_phys[:, 3],  params_phys[:, 1]),   # SiO2
    ], dim=1)

    q = torch.tensor(q_grid, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)
    dq = (0.019959 * q / 2.355).mean(dim=1, keepdim=True)

    all_contrasts = []
    for medium_sld in contrast_slds:
        sld = torch.stack([
            torch.full((B,), medium_sld, device=device),  # medium
            params_phys[:, 14],  # Medicine
            params_phys[:, 11],  # Head2
            params_phys[:, 8],   # Tail
            params_phys[:, 5],   # Head1
            params_phys[:, 2],   # SiO2
            torch.full((B,), 2.07, device=device),  # Si substrate
        ], dim=1)

        refl = abeles_constant_smearing(
            q=q, thickness=thickness, roughness=roughness,
            sld=sld, dq=dq, gauss_num=17
        )
        all_contrasts.append(refl)

    refl_all = torch.stack(all_contrasts, dim=1)
    return torch.log(torch.clamp(refl_all, min=1e-10))

y_true = torch.tensor(Y_mem[:16], dtype=torch.float32)
x_log = torch.tensor(np.log(np.clip(X_mem[:16], 1e-10, None)), dtype=torch.float32)

with torch.no_grad():
    refl_pred = parratt_torch(y_true, q_grid, param_bounds_list, contrast_slds)

print("Parratt range:", refl_pred[0].min().item(), refl_pred[0].max().item())
print("Input range:  ", x_log[0, :3, :].min().item(), x_log[0, :3, :].max().item())
print("MSE with true params:", torch.nn.functional.mse_loss(refl_pred, x_log[:, :3, :]).item())