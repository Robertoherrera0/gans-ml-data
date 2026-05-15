# complete_lipid_database.py
import pandas as pd
from periodictable import formula, neutron_sld

# Step 1: Create the lipid catalog (hardcoded - no external file needed)
lipid_data = [
    {"name": "DPPC", "formula": "C40H80NO8P", "mw_g_mol": 734.039, "density_g_cm3": 1.03, "source": "Avanti"},
    {"name": "DMPC", "formula": "C36H72NO8P", "mw_g_mol": 677.933, "density_g_cm3": 1.01, "source": "Avanti"},
    {"name": "DOPC", "formula": "C44H84NO8P", "mw_g_mol": 786.113, "density_g_cm3": 1.02, "source": "literature"},
    {"name": "DSPC", "formula": "C44H88NO8P", "mw_g_mol": 790.145, "density_g_cm3": 1.05, "source": "Avanti"},
    {"name": "D2O", "formula": "D2O", "mw_g_mol": 20.03, "density_g_cm3": 1.11, "source": "Sigma"},
    {"name": "H2O", "formula": "H2O", "mw_g_mol": 18.015, "density_g_cm3": 1.00, "source": "Sigma"},
]

# Step 2: Calculate SLD for each
results = []
for lipid in lipid_data:
    compound = formula(lipid['formula'])
    sld, sld_imag, xs_inc = neutron_sld(compound, density=lipid['density_g_cm3'])
    
    results.append({
        'name': lipid['name'],
        'formula': lipid['formula'],
        'density_g_cm3': lipid['density_g_cm3'],
        'sld_real_1e6_A2': round(sld, 3),
        'source': lipid['source']
    })

# Step 3: Create dataframe and save
df = pd.DataFrame(results)
df.to_csv("lipids_with_sld.csv", index=False)

# Step 4: Print results
print("=" * 60)
print("LIPID SLD DATABASE")
print("=" * 60)
print(df.to_string(index=False))
print("\nSaved to: lipids_with_sld.csv")