import json
import math
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import random_split, TensorDataset
from refl1d.names import QProbe, Slab, Experiment, SLD

# ========== FNO BLOCKS ==========
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weights_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def compl_mul1d(self, x_ft, w_real, w_imag):
        out_real = torch.einsum("bim,iom->bom", x_ft.real, w_real) - torch.einsum("bim,iom->bom", x_ft.imag, w_imag)
        out_imag = torch.einsum("bim,iom->bom", x_ft.real, w_imag) + torch.einsum("bim,iom->bom", x_ft.imag, w_real)
        return torch.complex(out_real, out_imag)

    def forward(self, x):
        batchsize, n = x.shape[0], x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-1), device=x.device, dtype=torch.cfloat)
        modes = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights_real[:, :, :modes], self.weights_imag[:, :, :modes])
        return torch.fft.irfft(out_ft, n=n, dim=-1)

class FNOBlock1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.BatchNorm1d(width)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.spectral(x) + self.pointwise(x)))

# ========== CONFIG ==========
data_dir = "dataset-with-priors/medium-dataset-with-noise-and-priors"
config = "gans_flowcell"
SLD_SUBSTRATE = 2.07
WAVELENGTH_RES = 0.019959062306768447
PLOT_ORDER = [("H2O", 2.07, 2), ("D2O", -0.56, 1), ("Mix", 6.36, 0)]

with open(f"{data_dir}/{config}_metadata.json", "r") as f:
    metadata = json.load(f)
q_grid = np.load(f"{data_dir}/{config}_q_grid.npy")

# ========== LOAD DATASET ==========
print("Loading dataset...")
reflectivity_obs_all = np.load(f"{data_dir}/{config}_reflectivity_data.npy", mmap_mode="r")
params_phys_all = np.load(f"{data_dir}/{config}_sample_parameters.npy")
prior_bounds_all = np.load(f"{data_dir}/{config}_prior_bounds.npy")

TRAIN_FRAC, VAL_FRAC, RANDOM_SEED = 0.90, 0.05, 42
N = len(reflectivity_obs_all)
n_train, n_val = int(TRAIN_FRAC * N), int(VAL_FRAC * N)
n_test = N - n_train - n_val

dummy_dataset = TensorDataset(torch.arange(N))
train_ds, val_ds, test_ds = random_split(dummy_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(RANDOM_SEED))

test_sample_idx = 156  # CHANGE THIS to try different samples
original_idx = test_ds.indices[test_sample_idx]
print(f"Test sample {test_sample_idx} → original {original_idx}")

true_params = params_phys_all[original_idx]
synthetic_prior = prior_bounds_all[original_idx]
obs_raw = reflectivity_obs_all[original_idx]  # (3, 250) - Dataset order: [D2O, Mix, H2O]

bounds_array = np.array([p["bounds"] for p in metadata["parameters"]])
global_lo, global_hi = bounds_array[:, 0], bounds_array[:, 1]

n_params = len(metadata["parameters"])
prior_lo_norm, prior_hi_norm = synthetic_prior[:n_params], synthetic_prior[n_params:]
prior_lo_phys = (prior_lo_norm + 1.0) / 2.0 * (global_hi - global_lo) + global_lo
prior_hi_phys = (prior_hi_norm + 1.0) / 2.0 * (global_hi - global_lo) + global_lo

param_names = ["sub_rough", "sio2_thick", "sio2_sld", "sio2_rough", "head1_thick", "head1_sld", "head1_rough",
               "tail_thick", "tail_sld", "tail_rough", "head2_thick", "head2_sld", "head2_rough", "med_thick", "med_sld", "med_rough"]

print("\nTrue parameters:")
for name, val in zip(param_names, true_params):
    print(f"  {name:12s}: {val:6.2f}")

# ========== ABELES ==========
def abeles(q, thickness, roughness, sld):
    c_dtype = torch.complex64
    batch_size = thickness.shape[0]
    sld = sld * 1e-6
    sld = sld[:, None].to(c_dtype)
    thickness = torch.cat([torch.zeros(batch_size, 1, device=q.device), thickness], dim=-1)[:, None]
    roughness = (roughness[:, None] ** 2).to(c_dtype)
    sld = (sld - sld[..., :1]) + 1e-36j
    k_z0 = (q / 2).to(c_dtype)[None, :, None]
    k_n = torch.sqrt(k_z0**2 - 4 * math.pi * sld)
    k_n, k_np1 = k_n[..., :-1], k_n[..., 1:]
    beta = 1j * thickness * k_n
    exp_beta, exp_m_beta = torch.exp(beta), torch.exp(-beta)
    rn = (k_n - k_np1) / (k_n + k_np1 + 1e-30) * torch.exp(-2 * k_n * k_np1 * roughness)
    c_matrices = torch.stack([torch.stack([exp_beta, rn * exp_m_beta], dim=-1), torch.stack([rn * exp_beta, exp_m_beta], dim=-1)], dim=-1)
    c_matrices = [c.squeeze(-3) for c in c_matrices.split(1, dim=-3)]
    m = reduce(torch.matmul, c_matrices)
    return (m[..., 1, 0] / (m[..., 0, 0] + 1e-30)).abs() ** 2

def apply_smearing(R_sim, q):
    nodes = torch.tensor([-2.0202, -0.9586, 0.0, 0.9586, 2.0202], device=q.device, dtype=q.dtype)
    weights = torch.tensor([0.1995, 0.3936, 0.9454, 0.3936, 0.1995], device=q.device, dtype=q.dtype)
    weights = weights / weights.sum()
    sigma, R_out = WAVELENGTH_RES * q / 2.355, torch.zeros_like(R_sim)
    for node, weight in zip(nodes, weights):
        q_shift = (q + node * sigma).clamp(min=q[0], max=q[-1])
        idx = torch.searchsorted(q.contiguous(), q_shift.contiguous()).clamp(1, len(q) - 1)
        q_lo, q_hi = q[idx - 1], q[idx]
        t = ((q_shift - q_lo) / (q_hi - q_lo + 1e-30)).clamp(0, 1)
        R_interp = torch.exp((1 - t) * torch.log(R_sim[:, idx - 1] + 1e-12) + t * torch.log(R_sim[:, idx] + 1e-12))
        R_out += weight * R_interp
    return R_out

def calc_reflectivity(params_phys_tensor, solvent_sld, q_tensor, device):
    if params_phys_tensor.ndim == 1:
        params_phys_tensor = params_phys_tensor.unsqueeze(0)
    B = params_phys_tensor.shape[0]
    solvent_col = torch.full((B, 1), solvent_sld, device=device)
    substrate_col = torch.full((B, 1), SLD_SUBSTRATE, device=device)
    layer_slds = params_phys_tensor[:, [2, 5, 8, 11, 14]]
    sld = torch.cat([solvent_col, layer_slds, substrate_col], dim=1)
    thickness, roughness = params_phys_tensor[:, [1, 4, 7, 10, 13]], params_phys_tensor[:, [0, 3, 6, 9, 12, 15]]
    R_sim = abeles(q_tensor, thickness, roughness, sld)
    R_sim = apply_smearing(R_sim, q_tensor)
    return R_sim.squeeze(0).detach().cpu().numpy()

def get_sld_profile(params_phys, medium_sld=2.07):
    layers = [
        dict(name="substrate", sld=2.07, roughness=float(params_phys[0]), thickness=0),
        dict(name="sio2", sld=float(params_phys[2]), roughness=float(params_phys[3]), thickness=float(params_phys[1])),
        dict(name="head1", sld=float(params_phys[5]), roughness=float(params_phys[6]), thickness=float(params_phys[4])),
        dict(name="tail", sld=float(params_phys[8]), roughness=float(params_phys[9]), thickness=float(params_phys[7])),
        dict(name="head2", sld=float(params_phys[11]), roughness=float(params_phys[12]), thickness=float(params_phys[10])),
        dict(name="medicine", sld=float(params_phys[14]), roughness=float(params_phys[15]), thickness=float(params_phys[13])),
        dict(name="medium", sld=float(medium_sld), roughness=0.0, thickness=0),
    ]
    zeros, dq = np.zeros(len(q_grid)), WAVELENGTH_RES * q_grid / 2.355
    probe = QProbe(q_grid, dq, data=(zeros, zeros))
    sample = Slab(material=SLD(name=layers[0]["name"], rho=layers[0]["sld"]), interface=layers[0]["roughness"])
    for layer in layers[1:]:
        sample = sample | Slab(material=SLD(name=layer["name"], rho=layer["sld"]), thickness=layer["thickness"], interface=layer["roughness"])
    z, sld, _ = Experiment(probe=probe, sample=sample).smooth_profile()
    return z, sld

# ========== FNO REGRESSOR MODEL ==========
class FNORegressor1d(nn.Module):
    def __init__(self, in_channels=6, width=128, modes=16, n_layers=5, out_dim=16, prior_dim=32):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.Sequential(*[FNOBlock1d(width, modes) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(width + prior_dim, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, out_dim), nn.Tanh(),
        )
        bounds = torch.tensor([[p["bounds"][0], p["bounds"][1]] for p in metadata["parameters"]], dtype=torch.float32)
        self.register_buffer("bounds", bounds)
        self.t_indices = [1, 4, 7, 10, 13]
        self.r_indices = [3, 6, 9, 12, 15]

    def apply_hard_constraint(self, out):
        for t_idx, r_idx in zip(self.t_indices, self.r_indices):
            t_lo, t_hi = self.bounds[t_idx, 0], self.bounds[t_idx, 1]
            r_lo, r_hi = self.bounds[r_idx, 0], self.bounds[r_idx, 1]
            t_phys = (out[:, t_idx] + 1.0) / 2.0 * (t_hi - t_lo) + t_lo
            r_phys = (out[:, r_idx] + 1.0) / 2.0 * (r_hi - r_lo) + r_lo
            r_phys_clamped = torch.min(r_phys, t_phys * 0.95)
            r_norm = 2.0 * (r_phys_clamped - r_lo) / (r_hi - r_lo) - 1.0
            r_norm = r_norm.clamp(-1.0, 1.0)
            out = torch.cat([out[:, :r_idx], r_norm.unsqueeze(1), out[:, r_idx+1:]], dim=1)
        return out

    def forward(self, x, p):
        x = self.lift(x)
        x = self.blocks(x)
        x = x.mean(dim=-1)
        x = torch.cat([x, p], dim=-1)
        out = self.head(x)
        return self.apply_hard_constraint(out)

# ========== LOAD MODEL & PREDICT ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nLoading FNO Regressor model...")
model = FNORegressor1d(in_channels=6, width=128, modes=16, n_layers=5, out_dim=16, prior_dim=32).to(device)
checkpoint = torch.load("fno_with_priors.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

# Dataset order: [D2O, Mix, H2O]
# FNO Regressor expects: [R, log10(R)] for each contrast, interleaved with q
# Build input: [R_D2O, q, R_Mix, q, R_H2O, q] then [log_D2O, q, log_Mix, q, log_H2O, q]
q_norm = (q_grid - q_grid.min()) / (q_grid.max() - q_grid.min())
q_channel = np.tile(q_norm, (3, 1))  # (3, 250)

# Arrange in dataset order for model
x_sample = obs_raw  # Already in [D2O, Mix, H2O] order
x_log = np.log10(x_sample + 1e-10)
x_input = np.vstack([x_sample, x_log])  # (6, 250)

x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)
p_tensor = torch.tensor(synthetic_prior, dtype=torch.float32).unsqueeze(0).to(device)

print("Getting model prediction...")
with torch.no_grad():
    pred_params_norm = model(x_tensor, p_tensor).squeeze(0).cpu().numpy()

# Convert to physical space
pred_params_phys = (pred_params_norm + 1.0) / 2.0 * (global_hi - global_lo) + global_lo

# CLAMP to prior bounds
pred_params_phys = np.clip(pred_params_phys, prior_lo_phys, prior_hi_phys)

print("\nComparison:")
for name, true_val, pred_val in zip(param_names, true_params, pred_params_phys):
    print(f"  {name:12s}: true={true_val:6.2f}, pred={pred_val:6.2f}, diff={abs(true_val-pred_val):6.2f}")

# ========== CALC REFLECTIVITY ==========
q_tensor = torch.tensor(q_grid, dtype=torch.float32, device=device)
true_tensor = torch.tensor(true_params, dtype=torch.float32, device=device)
pred_tensor = torch.tensor(pred_params_phys, dtype=torch.float32, device=device)

# For plotting, rearrange to [H2O, D2O, Mix] order
obs_reflectivity_plot = np.array([obs_raw[2], obs_raw[0], obs_raw[1]])  # [H2O, D2O, Mix]
pred_reflectivity_plot = np.array([calc_reflectivity(pred_tensor, solvent_sld, q_tensor, device) for _, solvent_sld, _ in PLOT_ORDER])

log_obs_plot = np.log10(obs_reflectivity_plot + 1e-10)
log_pred_plot = np.log10(pred_reflectivity_plot + 1e-10)
rmse_plot = np.sqrt(np.mean((log_obs_plot - log_pred_plot) ** 2, axis=1))
mae_plot = np.mean(np.abs(log_obs_plot - log_pred_plot), axis=1)

print("\nMetrics:")
for i, (name, _, _) in enumerate(PLOT_ORDER):
    print(f"  {name}: RMSE(log)={rmse_plot[i]:.4f}, MAE(log)={mae_plot[i]:.4f}")

# ========== PLOT REFLECTIVITY ==========
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8})
fig = plt.figure(figsize=(11, 6))
gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
obs_color, pred_color = "#6BAED6", "tab:orange"

for i, (name, _, _) in enumerate(PLOT_ORDER):
    ax_top = fig.add_subplot(gs[0, i])
    ax_top.plot(q_grid, log_obs_plot[i], "o", color=obs_color, markersize=2.5, alpha=0.65, markeredgewidth=0, label="Observed")
    ax_top.plot(q_grid, log_pred_plot[i], "-", color=pred_color, linewidth=1.5, label="Predicted")
    ax_top.set_title(f"{name}\nRMSE(log)={rmse_plot[i]:.4f}, MAE(log)={mae_plot[i]:.4f}", fontsize=9)
    ax_top.grid(True, alpha=0.25, linewidth=0.4)
    ax_top.tick_params(labelsize=8)
    ax_top.set_ylim([-8, 0])
    ax_top.set_yticks([0, -1, -2, -3, -4, -5, -6, -7, -8])
    ax_top.set_yticklabels([r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$', r'$10^{-6}$', r'$10^{-7}$', r'$10^{-8}$'])
    if i == 0:
        ax_top.set_ylabel(r"$\log R(q)$", fontsize=10)
    ax_top.legend(fontsize=7, loc="upper right", framealpha=0.9)
    
    ax_bot = fig.add_subplot(gs[1, i], sharex=ax_top)
    residuals_log = log_obs_plot[i] - log_pred_plot[i]
    ax_bot.plot(q_grid, residuals_log, "-", color=obs_color, linewidth=1.0)
    ax_bot.axhline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax_bot.grid(True, alpha=0.25, linewidth=0.4)
    ax_bot.tick_params(labelsize=8)
    ax_bot.set_xlabel(r"$q$ ($\AA^{-1}$)", fontsize=9)
    if i == 0:
        ax_bot.set_ylabel(r"$\Delta \log R(q)$", fontsize=9)

fig.suptitle("FNO Regressor: Observed vs Predicted Reflectivity", fontsize=11, y=0.98)
plt.savefig("fno_regressor_observed_vs_predicted.png", dpi=300, bbox_inches="tight")
print("\nSaved: fno_regressor_observed_vs_predicted.png")
plt.show()

# ========== PLOT SLD ==========
z_true, sld_true = get_sld_profile(true_params)
z_pred, sld_pred = get_sld_profile(pred_params_phys)
z_lo, sld_lo = get_sld_profile(prior_lo_phys)
z_hi, sld_hi = get_sld_profile(prior_hi_phys)

fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
ax.plot(z_true, sld_true, "k-", linewidth=2.5, label="True", zorder=4)
ax.plot(z_pred, sld_pred, "r-", linewidth=2.5, label="Predicted", zorder=3)
ax.plot(z_lo, sld_lo, "--", color="tab:blue", linewidth=1.8, alpha=0.75, label="Prior lower", zorder=2)
ax.plot(z_hi, sld_hi, "--", color="tab:green", linewidth=1.8, alpha=0.75, label="Prior upper", zorder=2)
ax.set_xlabel("z (Å)", fontsize=13)
ax.set_ylabel("SLD (10⁻⁶ Å⁻²)", fontsize=13)
ax.set_title(f"FNO Regressor SLD Profile — Sample {original_idx}", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="best", framealpha=0.95)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.tick_params(labelsize=11)
plt.savefig("fno_regressor_sld_profile.png", dpi=300, bbox_inches="tight")
print("Saved: fno_regressor_sld_profile.png")
plt.show()

print("\n✓ Done!")