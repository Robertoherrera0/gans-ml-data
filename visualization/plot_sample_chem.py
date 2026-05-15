import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load data
DATA_DIR = Path("small-dataset-chemistry")

q_grid = np.load(DATA_DIR / "gans_flowcell_q_grid.npy")
X_data = np.load(DATA_DIR / "gans_flowcell_reflectivity_data.npy")
Y_params = np.load(DATA_DIR / "gans_flowcell_sample_parameters.npy")
Y_norm = np.load(DATA_DIR / "gans_flowcell_sample_parameters_norm.npy")
prior_bounds = np.load(DATA_DIR / "gans_flowcell_prior_bounds.npy")

with open(DATA_DIR / "gans_flowcell_metadata.json", "r") as f:
    metadata = json.load(f)

print("Data shapes:")
print(f"  Q grid: {q_grid.shape}")
print(f"  Reflectivity: {X_data.shape}")
print(f"  Parameters (physical): {Y_params.shape}")
print(f"  Parameters (normalized): {Y_norm.shape}")
print(f"  Prior bounds: {prior_bounds.shape}")

print("\nContrasts:", [c["name"] for c in metadata["contrasts"]])
print("\nChemistry models:")
for key, val in metadata["chemistry_models"].items():
    print(f"  {key}: {val}")

# ========== Plot 1: Sample reflectivity curves ==========
sample_idx = 0

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

contrast_names = [c["name"] for c in metadata["contrasts"]]  # Use actual order from metadata

for c_idx, c_name in enumerate(contrast_names):
    ax = axes[c_idx]
    ax.semilogy(q_grid, X_data[sample_idx, c_idx, :], 'o-', markersize=3)
    ax.set_xlabel("Q (Å⁻¹)")
    ax.set_ylabel("R(Q)")
    ax.set_title(f"{c_name} contrast")
    ax.grid(True, alpha=0.3)

plt.suptitle(f"Sample {sample_idx} - Reflectivity Curves (Chemistry-Based)")
plt.tight_layout()
plt.savefig("chemistry_data_reflectivity_sample.png", dpi=150)
plt.show()

# ========== Plot 2: SLD distribution ==========
param_labels = [
    "substrate_roughness",
    "sio2_thickness", "sio2_sld", "sio2_roughness",
    "head1_thickness", "head1_sld", "head1_roughness",
    "tail_thickness", "tail_sld", "tail_roughness",
    "head2_thickness", "head2_sld", "head2_roughness",
    "medicine_thickness", "medicine_sld", "medicine_roughness",
]

# Plot SLD distributions
sld_indices = [2, 5, 8, 11, 14]  # sio2, head1, tail, head2, medicine
sld_names = ["sio2_sld", "head1_sld", "tail_sld", "head2_sld", "medicine_sld"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for i, (idx, name) in enumerate(zip(sld_indices, sld_names)):
    ax = axes[i]
    
    sld_values = Y_params[:, idx]
    
    ax.hist(sld_values, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel("SLD (×10⁻⁶ Å⁻²)")
    ax.set_ylabel("Count")
    ax.set_title(name)
    ax.grid(alpha=0.3)
    
    # Add mean line
    mean_val = sld_values.mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.legend()

# Remove extra subplot
axes[-1].remove()

plt.suptitle("SLD Distributions (Chemistry-Based Sampling)", fontsize=14)
plt.tight_layout()
plt.savefig("chemistry_data_sld_distributions.png", dpi=150)
plt.show()

# ========== Plot 3: Parameter statistics ==========
print("\n=== SLD Statistics ===")
print(f"{'Parameter':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 60)

for idx, name in zip(sld_indices, sld_names):
    sld_values = Y_params[:, idx]
    
    print(f"{name:<20} {sld_values.mean():>8.2f} "
          f"{sld_values.std():>8.2f} {sld_values.min():>8.2f} {sld_values.max():>8.2f}")

print("\nData saved to:")
print("  - chemistry_data_reflectivity_sample.png")
print("  - chemistry_data_sld_distributions.png")