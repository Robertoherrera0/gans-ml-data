import json
import math
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
import torch

from refl1d.names import QProbe, Slab, Experiment, SLD
# Cell 7: FNO Summary Network for BayesFlow

import torch
import torch.nn as nn

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes        = modes
        scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weights_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def compl_mul1d(self, x_ft, w_real, w_imag):
        out_real = torch.einsum("bim,iom->bom", x_ft.real, w_real) - torch.einsum("bim,iom->bom", x_ft.imag, w_imag)
        out_imag = torch.einsum("bim,iom->bom", x_ft.real, w_imag) + torch.einsum("bim,iom->bom", x_ft.imag, w_real)
        return torch.complex(out_real, out_imag)

    def forward(self, x):
        batchsize = x.shape[0]
        n         = x.shape[-1]
        x_ft      = torch.fft.rfft(x, dim=-1)
        out_ft    = torch.zeros(batchsize, self.out_channels, x_ft.size(-1), device=x.device, dtype=torch.cfloat)
        modes     = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :modes] = self.compl_mul1d(
            x_ft[:, :, :modes],
            self.weights_real[:, :, :modes],
            self.weights_imag[:, :, :modes]
        )
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNOBlock1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral  = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.norm      = nn.BatchNorm1d(width)
        self.act       = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.spectral(x) + self.pointwise(x)))


class FNOSummaryNetwork(nn.Module):
    """
    Takes reflectivity (B, 3, 250) and prior_bounds (B, 32)
    and produces a summary embedding (B, embedding_dim).

    This plugs into BayesFlow as the summary network.
    """
    def __init__(self,
                 n_contrasts=3,
                 n_q=250,
                 width=128,
                 modes=32,
                 n_layers=5,
                 prior_dim=32,
                 embedding_dim=256):
        super().__init__()

        # q channel — fixed, not learned
        q_norm = torch.tensor(
            (Q_GRID - Q_GRID.min()) / (Q_GRID.max() - Q_GRID.min()),
            dtype=torch.float32
        )
        self.register_buffer("q_norm", q_norm)

        # input: per contrast we have log-R + q = 2 channels
        # but we process all contrasts together
        # so in_channels = n_contrasts * 2 (log-R for each + q repeated)
        in_channels = n_contrasts * 2  # 6

        self.lift   = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.Sequential(*[FNOBlock1d(width, modes) for _ in range(n_layers)])

        # after global pooling: width + prior_dim -> embedding_dim
        self.head = nn.Sequential(
            nn.Linear(width + prior_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )

        self.embedding_dim = embedding_dim

    def forward(self, reflectivity, prior_bounds):
        """
        reflectivity:  (B, 3, 250) — log-reflectivity for H2O, D2O, MIX
        prior_bounds:  (B, 32)
        returns:       (B, embedding_dim)
        """
        B = reflectivity.shape[0]

        # build q channel (B, 1, 250) repeated for each contrast
        q = self.q_norm.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, 250)

        # stack: for each contrast interleave log-R and q
        # result shape: (B, 6, 250) = [R_H2O, q, R_D2O, q, R_MIX, q]
        channels = []
        for c in range(3):
            channels.append(reflectivity[:, c:c+1, :])  # (B, 1, 250)
            channels.append(q)                           # (B, 1, 250)
        x = torch.cat(channels, dim=1)  # (B, 6, 250)

        # FNO encoding
        x = self.lift(x)       # (B, width, 250)
        x = self.blocks(x)     # (B, width, 250)
        x = x.mean(dim=-1)     # (B, width) — global average pooling

        # concatenate prior bounds and project
        x = torch.cat([x, prior_bounds], dim=-1)  # (B, width + 32)
        x = self.head(x)                           # (B, embedding_dim)

        return x

# ========== CONFIGURATION ==========
data_dir = "dataset-with-priors/medium-dataset-with-noise-and-priors"
config = "gans_flowcell"

SLD_SUBSTRATE = 2.07
WAVELENGTH_RES = 0.019959062306768447

# Visual mapping
CONTRASTS = [
    ("Mix", 6.36, 0),
    ("D2O", -0.56, 1),
    ("H2O", 2.07, 2),
]

PLOT_ORDER = [
    ("H2O", 2.07, 2),
    ("D2O", -0.56, 1),
    ("Mix", 6.36, 0),
]
# ====================================

# Load metadata for bounds info
with open(f"{data_dir}/{config}_metadata.json", "r") as f:
    metadata = json.load(f)

q_grid = np.load(f"{data_dir}/{config}_q_grid.npy")

# ========== CREATE SYNTHETIC SAMPLE ==========
print("Creating synthetic presentation sample...")

# Hand-picked parameters for nice visualization
# Hand-picked parameters for nice visualization
true_params = np.array([
    4.5,    # substrate roughness - moderate
    15.0,   # sio2 thickness - thinner layer
    2.5,    # sio2 sld - lower contrast
    3.5,    # sio2 roughness - moderate
    12.0,   # head1 thickness - thin head
    4.5,    # head1 sld - high contrast
    2.0,    # head1 roughness - very smooth
    45.0,   # tail thickness - thick hydrophobic
    0.3,    # tail sld - low positive
    3.0,    # tail roughness - smooth
    12.0,   # head2 thickness - match head1
    4.5,    # head2 sld - match head1
    2.0,    # head2 roughness - very smooth
    120.0,  # medicine thickness - very thick layer
    2.8,    # medicine sld - moderate
    6.0,    # medicine roughness - rougher interface
])

# Create prior bounds around these parameters (±20% window)
bounds_array = np.array([p["bounds"] for p in metadata["parameters"]])
global_lo = bounds_array[:, 0]
global_hi = bounds_array[:, 1]

# Create tight priors around true values
prior_lo_phys = np.maximum(true_params * 0.8, global_lo)
prior_hi_phys = np.minimum(true_params * 1.2, global_hi)

# Normalize priors to [-1, 1]
prior_lo_norm = 2.0 * (prior_lo_phys - global_lo) / (global_hi - global_lo) - 1.0
prior_hi_norm = 2.0 * (prior_hi_phys - global_lo) / (global_hi - global_lo) - 1.0
synthetic_prior = np.concatenate([prior_lo_norm, prior_hi_norm]).astype(np.float32)

print("\nParameters:")
param_names = [
    "sub_rough", "sio2_thick", "sio2_sld", "sio2_rough",
    "head1_thick", "head1_sld", "head1_rough",
    "tail_thick", "tail_sld", "tail_rough",
    "head2_thick", "head2_sld", "head2_rough",
    "med_thick", "med_sld", "med_rough"
]
for name, val in zip(param_names, true_params):
    print(f"  {name:12s}: {val:6.2f}")

# ========== ABELES REFLECTIVITY ==========
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
    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)
    rn = (k_n - k_np1) / (k_n + k_np1 + 1e-30)
    rn = rn * torch.exp(-2 * k_n * k_np1 * roughness)
    c_matrices = torch.stack([
        torch.stack([exp_beta, rn * exp_m_beta], dim=-1),
        torch.stack([rn * exp_beta, exp_m_beta], dim=-1),
    ], dim=-1)
    c_matrices = [c.squeeze(-3) for c in c_matrices.split(1, dim=-3)]
    m = reduce(torch.matmul, c_matrices)
    r = (m[..., 1, 0] / (m[..., 0, 0] + 1e-30)).abs() ** 2
    return r.clamp(0.0, 1.0)

def apply_smearing(R_sim, q):
    nodes = torch.tensor([-2.0202, -0.9586, 0.0, 0.9586, 2.0202], device=q.device, dtype=q.dtype)
    weights = torch.tensor([0.1995, 0.3936, 0.9454, 0.3936, 0.1995], device=q.device, dtype=q.dtype)
    weights = weights / weights.sum()
    sigma = WAVELENGTH_RES * q / 2.355
    R_out = torch.zeros_like(R_sim)
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
    thickness = params_phys_tensor[:, [1, 4, 7, 10, 13]]
    roughness = params_phys_tensor[:, [0, 3, 6, 9, 12, 15]]
    R_sim = abeles(q_tensor, thickness, roughness, sld)
    R_sim = apply_smearing(R_sim, q_tensor)
    return R_sim.squeeze(0).detach().cpu().numpy()

# ========== SLD PROFILE ==========
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
    zeros = np.zeros(len(q_grid))
    dq = WAVELENGTH_RES * q_grid / 2.355
    probe = QProbe(q_grid, dq, data=(zeros, zeros))
    sample = Slab(material=SLD(name=layers[0]["name"], rho=layers[0]["sld"]), interface=layers[0]["roughness"])
    for layer in layers[1:]:
        sample = sample | Slab(material=SLD(name=layer["name"], rho=layer["sld"]),
                              thickness=layer["thickness"], interface=layer["roughness"])
    z, sld, _ = Experiment(probe=probe, sample=sample).smooth_profile()
    return z, sld

# ========== MODEL ARCHITECTURE ==========
import torch.nn as nn
import zuko

# Hyperparameters from training
WIDTH = 128
N_MODES = 32
N_FNO_LAYERS = 5
EMBEDDING_DIM = 256
PRIOR_DIM = 32
FLOW_TRANSFORMS = 8
FLOW_HIDDEN = 256

class FNOEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        q_norm = torch.tensor((q_grid - q_grid.min()) / (q_grid.max() - q_grid.min()), dtype=torch.float32)
        self.register_buffer("q_norm", q_norm)
        
        # 3 contrasts x 4 channels: raw logR, standardized logR, q, q^2
        self.lift = nn.Conv1d(12, WIDTH, kernel_size=1)
        self.blocks = nn.Sequential(*[FNOBlock1d(WIDTH, N_MODES) for _ in range(N_FNO_LAYERS)])
        self.head = nn.Sequential(
            nn.Linear(WIDTH, EMBEDDING_DIM), nn.GELU(),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM), nn.GELU(),
        )

    def forward(self, x):
        B = x.shape[0]
        q = self.q_norm.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        q2 = q ** 2
        channels = []
        for c in range(3):
            x_c = x[:, c:c+1, :]
            x_mean = x_c.mean(dim=-1, keepdim=True)
            x_std = x_c.std(dim=-1, keepdim=True).clamp_min(1e-6)
            x_standardized = (x_c - x_mean) / x_std
            channels.append(x_c)
            channels.append(x_standardized)
            channels.append(q)
            channels.append(q2)
        x = torch.cat(channels, dim=1)
        x = self.lift(x)
        x = self.blocks(x)
        x = x.mean(dim=-1)
        return self.head(x)

class FNOFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FNOEncoder()
        context_dim = EMBEDDING_DIM + PRIOR_DIM
        self.flow = zuko.flows.NSF(features=16, context=context_dim, 
                                   transforms=FLOW_TRANSFORMS,
                                   hidden_features=[FLOW_HIDDEN, FLOW_HIDDEN])

    def forward(self, reflectivity, prior_bounds):
        emb = self.encoder(reflectivity)
        context = torch.cat([emb, prior_bounds], dim=-1)
        return self.flow(context)

    def sample(self, reflectivity, prior_bounds, n_samples=500):
        return self.forward(reflectivity, prior_bounds).sample((n_samples,))

# ========== CALCULATE REFLECTIVITIES ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained FNO Flow model
print("\nLoading trained FNO Flow model...")
model_path = "fno_flow_online_shape_q2.pt"
model = FNOFlow().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Calculate observed reflectivity from true params
q_tensor = torch.tensor(q_grid, dtype=torch.float32, device=device)
true_tensor = torch.tensor(true_params, dtype=torch.float32, device=device)

# After calculating obs_reflectivity_raw, ADD NOISE:
obs_reflectivity_raw = np.array([
    calc_reflectivity(true_tensor, solvent_sld, q_tensor, device)
    for _, solvent_sld, _ in PLOT_ORDER
])

# ADD REALISTIC NOISE (matching training data)
np.random.seed(42)  # for reproducibility
for i in range(3):
    # Add relative noise (5-30% of signal)
    rel_sigma = np.random.uniform(0.05, 0.15, size=obs_reflectivity_raw[i].shape)
    noise = np.random.randn(*obs_reflectivity_raw[i].shape) * rel_sigma * obs_reflectivity_raw[i]
    obs_reflectivity_raw[i] = obs_reflectivity_raw[i] + noise
    
    # Add background
    log_background = np.random.uniform(np.log10(1e-9), np.log10(1e-5))
    obs_reflectivity_raw[i] = obs_reflectivity_raw[i] + 10**log_background
    
    # Add scale uncertainty
    scale = np.random.uniform(0.95, 1.05)
    obs_reflectivity_raw[i] = obs_reflectivity_raw[i] * scale
    
    # Clip to reasonable range
    obs_reflectivity_raw[i] = np.clip(obs_reflectivity_raw[i], 1e-10, 10)

# Prepare model input - Flow model expects LOG reflectivity
# Order for model: [H2O, D2O, Mix] based on PLOT_ORDER
log_refl_for_model = np.log(obs_reflectivity_raw + 1e-10)  # (3, 250)

refl_tensor = torch.tensor(log_refl_for_model, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 3, 250)
prior_tensor = torch.tensor(synthetic_prior, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 32)

# Get MODEL prediction - sample from posterior
print("Sampling from posterior...")
with torch.no_grad():
    samples = model.sample(refl_tensor, prior_tensor, n_samples=500)  # (500, 1, 16)
    samples = samples[:, 0, :].cpu().numpy()  # (500, 16)

# Take posterior mean as prediction
pred_params_norm = samples.mean(axis=0)
pred_params_phys = (pred_params_norm + 1.0) / 2.0 * (global_hi - global_lo) + global_lo
# CLAMP to prior bounds
pred_params_phys = np.clip(pred_params_phys, prior_lo_phys, prior_hi_phys)

print("\nComparison:")
for name, true_val, pred_val in zip(param_names, true_params, pred_params_phys):
    print(f"  {name:12s}: true={true_val:6.2f}, pred={pred_val:6.2f}, diff={abs(true_val-pred_val):6.2f}")

# Calculate predicted reflectivity
pred_tensor = torch.tensor(pred_params_phys, dtype=torch.float32, device=device)

# Calculate observed (from true params) and predicted reflectivity
obs_reflectivity_plot = np.array([
    calc_reflectivity(true_tensor, solvent_sld, q_tensor, device)
    for _, solvent_sld, _ in PLOT_ORDER
])

pred_reflectivity_plot = np.array([
    calc_reflectivity(pred_tensor, solvent_sld, q_tensor, device)
    for _, solvent_sld, _ in PLOT_ORDER
])

# Calculate metrics
log_obs_plot = np.log10(obs_reflectivity_plot + 1e-10)
log_pred_plot = np.log10(pred_reflectivity_plot + 1e-10)
rmse_plot = np.sqrt(np.mean((log_obs_plot - log_pred_plot) ** 2, axis=1))
mae_plot = np.mean(np.abs(log_obs_plot - log_pred_plot), axis=1)

print("\nMetrics:")
for i, (name, _, _) in enumerate(PLOT_ORDER):
    print(f"  {name}: RMSE(log)={rmse_plot[i]:.4f}, MAE(log)={mae_plot[i]:.4f}")

# ========== PLOT 1: REFLECTIVITY ==========
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

fig = plt.figure(figsize=(11, 6))
gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

obs_color = "#6BAED6"
pred_color = "tab:orange"

for i, (name, _, _) in enumerate(PLOT_ORDER):
    ax_top = fig.add_subplot(gs[0, i])
    
    ax_top.plot(q_grid, log_obs_plot[i], "o", color=obs_color, markersize=2.5, 
                alpha=0.65, markeredgewidth=0, label="Observed")
    ax_top.plot(q_grid, log_pred_plot[i], "-", color=pred_color, linewidth=1.5, label="Predicted")
    
    ax_top.set_title(f"{name}\nRMSE(log)={rmse_plot[i]:.4f}, MAE(log)={mae_plot[i]:.4f}", fontsize=9)
    # Add explicit y-axis ticks
    ax_top.set_ylim([-8, 0])
    ax_top.set_yticks([0, -1, -2, -3, -4, -5, -6, -7, -8])
    ax_top.set_yticklabels([r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', 
                            r'$10^{-4}$', r'$10^{-5}$', r'$10^{-6}$', r'$10^{-7}$', r'$10^{-8}$'])
    ax_top.grid(True, alpha=0.25, linewidth=0.4)
    ax_top.tick_params(labelsize=8)
    
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

fig.suptitle("Observed and Predicted Reflectivity Curves Across Three Contrasts", fontsize=11, y=0.98)
plt.savefig("presentation_observed_vs_predicted.png", dpi=300, bbox_inches="tight")
print("\nSaved: presentation_observed_vs_predicted.png")
plt.show()

# ========== PLOT 2: SLD PROFILE ==========
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
ax.set_title("SLD Profile — Sample 2889", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="best", framealpha=0.95)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.tick_params(labelsize=11)

plt.savefig("presentation_sld_profile.png", dpi=300, bbox_inches="tight")
print("Saved: presentation_sld_profile.png")
plt.show()

print("\n✓ Generated presentation plots!")