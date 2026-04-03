import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import copy
from refl1d.names import QProbe, Slab, Experiment, SLD, Parameter

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = Path("fixed-medium-dataset")
Q_PATH = DATA_DIR / "gans_flowcell_q_grid.npy"
X_PATH = DATA_DIR / "gans_flowcell_reflectivity_data.npy"
Y_PATH = DATA_DIR / "gans_flowcell_sample_parameters_norm.npy"
META_PATH = DATA_DIR / "gans_flowcell_metadata.json"

BATCH_SIZE = 256
NUM_WORKERS = 0
TRAIN_FRAC = 0.90
VAL_FRAC = 0.05
TEST_FRAC = 0.05
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WIDTH = 128
N_MODES = 16
N_FNO_LAYERS = 5
OUT_DIM = 16
MAX_SAMPLES = 100000

with open(META_PATH, "r") as f:
    metadata = json.load(f)

q_grid = np.load(Q_PATH)
contrast_names = [c["name"] for c in metadata["contrasts"]]
param_labels = []
for p in metadata["parameters"]:
    if p["index"] == 0:
        param_labels.append(f"surface_{p['name']}")
    else:
        param_labels.append(f"L{p['layer']}_{p['name']}")

X_mem = np.load(X_PATH)
Y_mem = np.load(Y_PATH)
N = min(X_mem.shape[0], MAX_SAMPLES)
print(f"Reflectivity shape: {X_mem.shape}")
print(f"Using samples: {N}")

class ReflectometryFNODataset(Dataset):
    def __init__(self, x_mem, y_mem, q_grid, n_samples=None, eps=1e-10):
        self.x_mem = x_mem
        self.y_mem = y_mem
        self.n_samples = len(x_mem) if n_samples is None else n_samples
        self.eps = eps
        q = np.asarray(q_grid, dtype=np.float32)
        self.q_channels = np.stack([q, q, q], axis=0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = np.array(self.x_mem[idx], dtype=np.float32)
        y = np.array(self.y_mem[idx], dtype=np.float32)
        x = np.log(np.clip(x, self.eps, None))
        x = np.concatenate([x, self.q_channels], axis=0)
        return torch.from_numpy(x), torch.from_numpy(y)

dataset = ReflectometryFNODataset(X_mem, Y_mem, q_grid, n_samples=N)

n_train = int(TRAIN_FRAC * len(dataset))
n_val = int(VAL_FRAC * len(dataset))
n_test = len(dataset) - n_train - n_val

train_ds, val_ds, test_ds = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

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
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(x.shape[0], self.out_channels, x_ft.size(-1), device=x.device, dtype=torch.cfloat)
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
        y = self.spectral(x) + self.pointwise(x)
        return self.act(self.norm(y))

class FNORegressor1d(nn.Module):
    def __init__(self, in_channels=6, width=128, modes=16, n_layers=5, out_dim=16):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.Sequential(*[FNOBlock1d(width, modes) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, out_dim), nn.Tanh(),
        )

    def forward(self, x):
        x = self.lift(x)
        x = self.blocks(x)
        x = x.mean(dim=-1)
        return self.head(x)

# Parratt
param_bounds_list = [[p["bounds"][0], p["bounds"][1]] for p in metadata["parameters"]]
contrast_slds = [-0.56, 6.36, 2.07]

def parratt_torch(params_norm, q_grid, param_bounds_list, contrast_slds):
    B = params_norm.shape[0]
    device = params_norm.device
    bounds = torch.tensor(param_bounds_list, dtype=torch.float32, device=device)
    lo, hi = bounds[:, 0], bounds[:, 1]
    params_phys = ((params_norm + 1.0) / 2.0) * (hi - lo) + lo
    surf_rough = params_phys[:, 0]
    thickness = torch.stack([
        torch.zeros(B, device=device), params_phys[:, 1], params_phys[:, 4],
        params_phys[:, 7], params_phys[:, 10], params_phys[:, 13],
        torch.zeros(B, device=device),
    ], dim=1) * 1e-10
    sld_layers = torch.stack([
        torch.full((B,), 2.07, device=device), params_phys[:, 2], params_phys[:, 5],
        params_phys[:, 8], params_phys[:, 11], params_phys[:, 14],
    ], dim=1) * 1e14
    roughness = torch.stack([
        surf_rough,
        torch.min(params_phys[:, 3],  params_phys[:, 1]),   # roughness < thickness
        torch.min(params_phys[:, 6],  params_phys[:, 4]),
        torch.min(params_phys[:, 9],  params_phys[:, 7]),
        torch.min(params_phys[:, 12], params_phys[:, 10]),
        torch.min(params_phys[:, 15], params_phys[:, 13]),
    ], dim=1) * 1e-10
    q = torch.tensor(q_grid, dtype=torch.float32, device=device) * 1e10
    q_ang = torch.tensor(q_grid, dtype=torch.float32, device=device)
    all_contrasts = []
    for medium_sld in contrast_slds:
        med = torch.full((B, 1), medium_sld * 1e14, device=device)
        slds = torch.cat([sld_layers, med], dim=1)
        kz0 = (q / 2).unsqueeze(0).unsqueeze(0)
        kz = torch.sqrt(torch.clamp(kz0**2 - 4 * torch.pi * slds.unsqueeze(2), min=1e-30).to(torch.cfloat))
        r = torch.zeros(B, len(q_grid), dtype=torch.cfloat, device=device)
        for i in range(slds.shape[1] - 2, -1, -1):
            ki, ki1 = kz[:, i, :], kz[:, i+1, :]
            rough = roughness[:, i].unsqueeze(1).to(torch.cfloat)
            rij = (ki - ki1) / (ki + ki1 + 1e-30)
            rij = rij * torch.exp(-2 * ki * ki1 * rough**2)
            phase = torch.exp(2j * ki1 * thickness[:, i+1].unsqueeze(1).to(torch.cfloat))
            r = (rij + r * phase) / (1 + rij * r * phase + 1e-30)
        all_contrasts.append(torch.abs(r) ** 2)
    refl_all = torch.stack(all_contrasts, dim=1)
    dq = 0.019959 * q_ang / 2.355
    smeared = torch.zeros_like(refl_all)
    for qi in range(len(q_grid)):
        weights = torch.exp(-0.5 * ((q_ang - q_ang[qi]) / (dq[qi] + 1e-10))**2)
        weights = weights / weights.sum()
        smeared[:, :, qi] = (refl_all * weights.unsqueeze(0).unsqueeze(0)).sum(dim=2)
    return torch.log(torch.clamp(smeared, min=1e-10))

# physics consistency using refl1d
wavelength_resolution = 0.019959062306768447

def calculate_reflectivity(q, model_description):
    zeros = np.zeros(len(q))
    dq = wavelength_resolution * q / 2.355
    probe = QProbe(q, dq, data=(zeros, zeros))
    layers = model_description["layers"]
    sample = Slab(material=SLD(name=layers[0]["name"], rho=layers[0]["sld"]), interface=layers[0]["roughness"])
    for layer in layers[1:]:
        sample = sample | Slab(material=SLD(name=layer["name"], rho=layer["sld"], irho=layer["isld"]),
                               thickness=layer["thickness"], interface=layer["roughness"])
    probe.background = Parameter(value=model_description["background"], name="background")
    return Experiment(probe=probe, sample=sample).reflectivity()[1]

def get_model_description(params, metadata):
    base = {
        "layers": [
            {"sld": 2.07, "isld": 0, "thickness": 0,  "roughness": 11.1, "name": "substrate"},
            {"sld": 3.41, "isld": 0, "thickness": 60,  "roughness": 3.0,  "name": "sio2"},
            {"sld": 2.0,  "isld": 0, "thickness": 20,  "roughness": 3.0,  "name": "head1"},
            {"sld": 0.0,  "isld": 0, "thickness": 40,  "roughness": 3.0,  "name": "tail"},
            {"sld": 2.0,  "isld": 0, "thickness": 20,  "roughness": 3.0,  "name": "head2"},
            {"sld": 2.0,  "isld": 0, "thickness": 10,  "roughness": 3.0,  "name": "medicine"},
            {"sld": 0.0,  "isld": 0, "thickness": 0,   "roughness": 0.0,  "name": "medium"},
        ],
        "scale": 1, "background": 0,
    }
    param_defs = [
        (0,"roughness"),(1,"thickness"),(1,"sld"),(1,"roughness"),
        (2,"thickness"),(2,"sld"),(2,"roughness"),
        (3,"thickness"),(3,"sld"),(3,"roughness"),
        (4,"thickness"),(4,"sld"),(4,"roughness"),
        (5,"thickness"),(5,"sld"),(5,"roughness"),
    ]
    desc = copy.deepcopy(base)
    for i, (layer_idx, par_name) in enumerate(param_defs):
        value = float(params[i])
        if par_name == "roughness" and layer_idx != 0:
            thickness = desc["layers"][layer_idx]["thickness"]
            if value > thickness:
                lo, hi = metadata["parameters"][i]["bounds"]
                value = np.random.uniform(lo, min(hi, thickness))
        desc["layers"][layer_idx][par_name] = value
    return desc

def physics_consistency_score(preds_phys, x_input_raw, q_grid, metadata):
    contrasts = [c["medium_sld"] for c in metadata["contrasts"]]
    errors = []
    for i in range(len(preds_phys)):
        desc = get_model_description(preds_phys[i], metadata)
        for j, medium_sld in enumerate(contrasts):
            desc_c = copy.deepcopy(desc)
            desc_c["layers"][-1]["sld"] = medium_sld
            r_pred = calculate_reflectivity(q_grid, desc_c)
            r_true = x_input_raw[i, j, :]
            errors.append(np.mean((np.log(r_pred + 1e-10) - np.log(r_true + 1e-10))**2))
    return np.mean(errors)

# model
torch.manual_seed(42)
np.random.seed(42)

model = FNORegressor1d(in_channels=6, width=WIDTH, modes=N_MODES,
                       n_layers=N_FNO_LAYERS, out_dim=OUT_DIM).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_count = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)
    return total_loss / total_count

# training
train_losses, val_losses = [], []
best_val = float("inf")
best_path = "best_fno.pt"

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(model, train_loader, criterion, optimizer=optimizer)
    val_loss = run_epoch(model, val_loader, criterion, optimizer=None)
    scheduler.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), best_path)
    print(f"Epoch {epoch:02d} | train={train_loss:.6f} | val={val_loss:.6f}")

# evaluation
model.load_state_dict(torch.load(best_path, map_location=device))
test_loss = run_epoch(model, test_loader, criterion, optimizer=None)

xb, yb = next(iter(test_loader))
with torch.no_grad():
    preds = model(xb.to(device)).cpu().numpy()
yb = yb.numpy()

bounds = np.array(param_bounds_list)
lo, hi = bounds[:, 0], bounds[:, 1]
preds_phys = ((preds + 1.0) / 2.0) * (hi - lo) + lo
truth_phys = ((yb + 1.0) / 2.0) * (hi - lo) + lo

# per parameter MSE
mse_per_param = ((preds - yb)**2).mean(axis=0)
mape_per_param = (np.abs(preds_phys - truth_phys) / (np.abs(truth_phys) + 1e-8) * 100).mean(axis=0)

print(f"\n{'='*60}")
print(f"Test MSE: {test_loss:.4f}")
print(f"\n{'Parameter':<25} {'MSE(norm)':>10} {'MAPE(%)':>10} {'Type':>12}")
print("-"*60)
for name, mse, mape in zip(param_labels, mse_per_param, mape_per_param):
    ptype = 'SLD' if 'sld' in name else ('thickness' if 'thickness' in name else 'roughness')
    print(f"{name:<25} {mse:>10.4f} {mape:>10.2f} {ptype:>12}")
print("-"*60)
print(f"{'OVERALL':<25} {mse_per_param.mean():>10.4f} {mape_per_param.mean():>10.2f}")

# physics consistency
x_raw = X_mem[:16].astype(np.float32)
y_raw = Y_mem[:16].astype(np.float32)
true_phys_16 = ((y_raw + 1.0) / 2.0) * (hi - lo) + lo
x_log = np.log(np.clip(x_raw, 1e-10, None))
q_ch = np.tile(np.stack([q_grid.astype(np.float32)]*3, axis=0), (16,1,1))
x_inp = np.concatenate([x_log, q_ch], axis=1)
with torch.no_grad():
    preds_16 = model(torch.tensor(x_inp).to(device)).cpu().numpy()
preds_phys_16 = ((preds_16 + 1.0) / 2.0) * (hi - lo) + lo

pred_score = physics_consistency_score(preds_phys_16, x_raw, q_grid, metadata)
true_score = physics_consistency_score(true_phys_16, x_raw, q_grid, metadata)
print(f"\nPhysics consistency (true):      {true_score:.4f}")
print(f"Physics consistency (predicted): {pred_score:.4f}")
print(f"Physics gap:                     {pred_score - true_score:.4f}")

# plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(train_losses, label='train', color='#378ADD', linewidth=2)
axes[0].plot(val_losses, label='val', color='#E24B4A', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training vs validation loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

colors = ['#1D9E75' if 'sld' in n else '#E24B4A' for n in param_labels]
axes[1].scatter(range(16), mse_per_param, c=colors, s=80, zorder=3)
axes[1].set_xticks(range(16))
axes[1].set_xticklabels(param_labels, rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('MSE (normalized)')
axes[1].set_title('Per-parameter MSE')
axes[1].grid(True, alpha=0.3)
import matplotlib.patches as mpatches
axes[1].legend(handles=[
    mpatches.Patch(color='#1D9E75', label='SLD'),
    mpatches.Patch(color='#E24B4A', label='thickness/roughness')
], fontsize=10)

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nDone. Results saved to results.png")