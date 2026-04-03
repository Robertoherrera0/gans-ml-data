import numpy as np
import os

DATA_DIR = "large-dataset" 

print(DATA_DIR)

def main():
    q = np.load(os.path.join(DATA_DIR, "gans_flowcell_q_grid.npy"))
    refl = np.load(os.path.join(DATA_DIR, "gans_flowcell_reflectivity_data.npy"))
    params = np.load(os.path.join(DATA_DIR, "gans_flowcell_sample_parameters.npy"))
    params_norm = np.load(os.path.join(DATA_DIR, "gans_flowcell_sample_parameters_norm.npy"))

    n_samples = refl.shape[0]

    idx = np.random.randint(0, n_samples) + 8

    print("sample #", idx)

    print("Q shape:", q.shape)

    print(f"Reflectivy curves for sample # {idx}", refl[idx].shape)
    print(f"Paramertes of sample # {idx}")
    print(params[idx])

    print("Training data shape:",refl.shape)
    print("samples shape", params.shape)



if __name__ == "__main__":
    main()