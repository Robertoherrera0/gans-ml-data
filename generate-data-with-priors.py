import os
import json
import copy
import numpy as np
from refl1d.names import QProbe, Slab, Experiment, SLD, Parameter

np.random.seed(42)
wavelength_resolution = 0.019959062306768447

def calculate_reflectivity(q, model_description, q_resolution=wavelength_resolution):
    zeros = np.zeros(len(q))
    dq = q_resolution * q / 2.355
    probe = QProbe(q, dq, data=(zeros, zeros))
    layers = model_description["layers"]
    sample = Slab(
        material=SLD(name=layers[0]["name"], rho=layers[0]["sld"]),
        interface=layers[0]["roughness"],
    )
    for layer in layers[1:]:
        sample = sample | Slab(
            material=SLD(name=layer["name"], rho=layer["sld"], irho=layer["isld"]),
            thickness=layer["thickness"],
            interface=layer["roughness"],
        )
    probe.background = Parameter(value=model_description["background"], name="background")
    experiment = Experiment(probe=probe, sample=sample)
    q_out, r = experiment.reflectivity()
    return r

def add_noise(r, rel_sigma_min=0.05, rel_sigma_max=0.30):
    rel_sigmas = np.random.uniform(rel_sigma_min, rel_sigma_max, size=r.shape)
    sigmas     = rel_sigmas * r
    r_noisy    = np.random.normal(loc=r, scale=sigmas)
    log_b      = np.random.uniform(np.log10(1e-9), np.log10(1e-4))
    r_noisy    = r_noisy + 10 ** log_b
    r_noisy    = r_noisy * np.random.uniform(0.95, 1.05)
    r_noisy    = np.clip(r_noisy, 1e-10, None)
    return r_noisy.astype(np.float32)

class Contrast:
    def __init__(self, name, sld):
        self.name = name
        self.medium_sld = sld

class ReflectivityModels:

    model_description = dict(
        layers=[
            dict(sld=2.07, isld=0, thickness=0,   roughness=11.1, name="substrate"),
            dict(sld=3.41, isld=0, thickness=60,  roughness=3.0,  name="sio2"),
            dict(sld=2.0,  isld=0, thickness=20,  roughness=3.0,  name="head1"),
            dict(sld=0.0,  isld=0, thickness=40,  roughness=3.0,  name="tail"),
            dict(sld=2.0,  isld=0, thickness=20,  roughness=3.0,  name="head2"),
            dict(sld=2.0,  isld=0, thickness=10,  roughness=3.0,  name="medicine"),
            dict(sld=0.0,  isld=0, thickness=0,   roughness=0.0,  name="medium"),
        ],
        scale=1,
        background=0,
    )

    parameters = [
        dict(i=0, par="roughness",  bounds=[1, 20]),
        dict(i=1, par="thickness",  bounds=[5, 30]),
        dict(i=1, par="sld",        bounds=[2.07, 4.1]),
        dict(i=1, par="roughness",  bounds=[1, 30]),
        dict(i=2, par="thickness",  bounds=[3, 20]),
        dict(i=2, par="sld",        bounds=[1, 5]),
        dict(i=2, par="roughness",  bounds=[1, 20]),
        dict(i=3, par="thickness",  bounds=[5, 50]),
        dict(i=3, par="sld",        bounds=[-1, 3]),
        dict(i=3, par="roughness",  bounds=[1, 30]),
        dict(i=4, par="thickness",  bounds=[3, 50]),
        dict(i=4, par="sld",        bounds=[1, 5]),
        dict(i=4, par="roughness",  bounds=[1, 30]),
        dict(i=5, par="thickness",  bounds=[5, 200]),
        dict(i=5, par="sld",        bounds=[1, 5]),
        dict(i=5, par="roughness",  bounds=[1, 50]),
    ]

    contrasts = [
        Contrast("H2O", -0.56),
        Contrast("D2O", 6.36),
        Contrast("MIX", 2.07),
    ]

    def __init__(self, q=None, name="gans_flowcell"):
        self.name = name
        self.params_norm       = None
        self.params            = None
        self.prior_bounds      = None 
        self.reflectivity_data = None
        if q is None:
            self.q = np.logspace(np.log10(0.005), np.log10(0.25), 250)
        else:
            self.q = q

    def sample_prior_bounds(self, theta_phys):
        """
        Following Munteanu 2024:
        For each parameter:
          1. sample width w uniformly in [0.01 * range, full range]
          2. sample center c uniformly such that [c-w/2, c+w/2] stays within global bounds
             AND theta_phys is inside the window
          3. return lo and hi normalized to [-1, 1] w.r.t. global bounds

        theta_phys: [16] physical parameter values
        returns: [32] normalized prior bounds (lo_0, lo_1, ..., lo_15, hi_0, ..., hi_15)
        """
        n = len(self.parameters)
        lo_norm = np.empty(n)
        hi_norm = np.empty(n)

        for k, p in enumerate(self.parameters):
            g_lo, g_hi = p["bounds"]
            g_range    = g_hi - g_lo
            theta_k    = theta_phys[k]

            # sample width — between 1% and 100% of global range
            w = np.random.uniform(0.01 * g_range, g_range)

            # center must satisfy: c - w/2 >= g_lo, c + w/2 <= g_hi, AND theta_k in [c-w/2, c+w/2]
            c_min = max(g_lo + w / 2, theta_k - w / 2 + 1e-9)
            c_max = min(g_hi - w / 2, theta_k + w / 2 - 1e-9)

            if c_min > c_max:
                # fallback: window centered on theta_k, clamped to global bounds
                c = theta_k
            else:
                c = np.random.uniform(c_min, c_max)

            lo = np.clip(c - w / 2, g_lo, g_hi)
            hi = np.clip(c + w / 2, g_lo, g_hi)

            # normalize to [-1, 1]
            lo_norm[k] = 2 * (lo - g_lo) / g_range - 1
            hi_norm[k] = 2 * (hi - g_lo) / g_range - 1

        return np.concatenate([lo_norm, hi_norm]).astype(np.float32)

    def generate(self, n_samples):
        n_params    = len(self.parameters)
        n_contrasts = len(self.contrasts)
        n_q         = len(self.q)

        params_norm = np.random.uniform(-1, 1, size=(n_samples, n_params))
        params      = self.to_physical_parameters(params_norm)

        reflectivity_data = np.empty((n_samples, n_contrasts, n_q), dtype=np.float32)
        params_corrected  = np.empty_like(params)
        prior_bounds      = np.empty((n_samples, 2 * n_params), dtype=np.float32)  # NEW

        for i, p in enumerate(params):
            if i % 50000 == 0:
                print(f"{i} / {n_samples} samples")

            desc = self.get_model_description(p)

            for k, par in enumerate(self.parameters):
                params_corrected[i, k] = desc["layers"][par["i"]][par["par"]]

            # generate prior bounds from corrected physical params
            prior_bounds[i] = self.sample_prior_bounds(params_corrected[i])  # NEW

            for j, contrast in enumerate(self.contrasts):
                desc_c = self.get_model_description(p)
                desc_c["layers"][-1]["sld"] = contrast.medium_sld
                r = calculate_reflectivity(self.q, desc_c)
                reflectivity_data[i, j, :] = add_noise(r)

        self.params_norm       = params_norm
        self.params            = params_corrected
        self.prior_bounds      = prior_bounds
        self.reflectivity_data = reflectivity_data

    def to_physical_parameters(self, params_norm):
        params = np.zeros_like(params_norm)
        for i, p in enumerate(self.parameters):
            lo, hi = p["bounds"]
            a = (hi - lo) / 2
            b = (hi + lo) / 2
            params[:, i] = params_norm[:, i] * a + b
        return params

    def get_model_description(self, params):
        desc = copy.deepcopy(self.model_description)
        for i, p in enumerate(self.parameters):
            if p["par"] == "thickness":
                desc["layers"][p["i"]]["thickness"] = params[i]
        for i, p in enumerate(self.parameters):
            if p["par"] == "thickness":
                continue
            layer_index    = p["i"]
            parameter_name = p["par"]
            value          = params[i]
            if parameter_name == "roughness" and layer_index != 0:
                thickness = desc["layers"][layer_index]["thickness"]
                if value > thickness:
                    lo, hi    = self.parameters[i]["bounds"]
                    hi_valid  = min(hi, thickness)
                    value     = np.random.uniform(lo, hi_valid)
            desc["layers"][layer_index][parameter_name] = value
        return desc

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{self.name}_q_grid.npy"),                    self.q)
        np.save(os.path.join(output_dir, f"{self.name}_reflectivity_data.npy"),         self.reflectivity_data)
        np.save(os.path.join(output_dir, f"{self.name}_sample_parameters_norm.npy"),    self.params_norm)
        np.save(os.path.join(output_dir, f"{self.name}_sample_parameters.npy"),         self.params)
        np.save(os.path.join(output_dir, f"{self.name}_prior_bounds.npy"),              self.prior_bounds)  # NEW

        meta = {
            "parameters": [
                {"index": i, "layer": p["i"], "name": p["par"], "bounds": p["bounds"]}
                for i, p in enumerate(self.parameters)
            ],
            "contrasts": [
                {"name": c.name, "medium_sld": c.medium_sld}
                for c in self.contrasts
            ],
            "q_points": len(self.q),
            "parameter_sampling": "uniform in [-1,1]",
            "noise": "Gaussian 5-30% per Q point + log-uniform background [1e-9,1e-4] + normalization misalignment [0.95,1.05]",
            "prior_bounds": "shape (N, 32) — normalized lo and hi for each of 16 parameters, following Munteanu et al. 2024",
        }
        with open(os.path.join(output_dir, f"{self.name}_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)


def main():
    n_samples  = 3000000
    output_dir = "large-dataset-with-noise-and-priors"
    model = ReflectivityModels()
    print(f"Generating {n_samples} samples...")
    model.generate(n_samples)
    print("Saving dataset...")
    model.save(output_dir)
    print("Done.")

if __name__ == "__main__":
    main()