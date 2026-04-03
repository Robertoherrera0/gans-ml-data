import numpy as np
import matplotlib.pyplot as plt
from refl1d.names import *

q = np.logspace(np.log10(0.005), np.log10(0.25), 250)

contrasts = [
    ("H2O", -0.56, "blue"),
    ("D2O",  6.36, "red"),
    ("MIX",  2.07, "gold"),
]

layers = [
    dict(name="substrate", sld=2.07, thickness=0, roughness=4),

    dict(name="sio2", thickness=10, sld=3.4, roughness=3.0),

    dict(name="head1", thickness=11, sld=1.2, roughness=3.0),
    dict(name="tail", thickness=20, sld=-0.2, roughness=3.0),
    dict(name="head2", thickness=15, sld=2.4, roughness=3.0),

    dict(name="medicine", thickness=30, sld=4.0, roughness=3.0),
]

profiles = {}

wavelength_resolution = 0.019959062306768447

for name, medium, color in contrasts:

    probe = QProbe(q, wavelength_resolution*q/2.355,
                   data=(np.zeros(len(q)), np.zeros(len(q))))

    sample = Slab(
        material=SLD(name="substrate", rho=layers[0]["sld"]),
        interface=layers[0]["roughness"]
    )

    for l in layers[1:]:

        sample = sample | Slab(
            material=SLD(name=l["name"], rho=l["sld"]),
            thickness=l["thickness"],
            interface=l["roughness"]
        )

    sample = sample | Slab(
        material=SLD(name="medium", rho=medium),
        thickness=0,
        interface=0
    )

    expt = Experiment(probe=probe, sample=sample)

    z, sld, _ = expt.smooth_profile()

    profiles[name] = sld

profiles_array = np.vstack([
    profiles["H2O"],
    profiles["D2O"],
    profiles["MIX"]
])

diff = np.max(np.abs(profiles_array - profiles_array[0]), axis=0)

mask = diff > 1e-3
diverge_idx = np.where(mask)[0][0]

fig, ax = plt.subplots(figsize=(10,6))

# structural region
ax.plot(
    z[:diverge_idx],
    profiles["MIX"][:diverge_idx],
    color="black",
    linewidth=3
)

# medium region
for name, _, color in contrasts:
    ax.plot(
        z[diverge_idx:],
        profiles[name][diverge_idx:],
        color=color,
        linewidth=2
    )

# layer boundaries
boundaries = []
ymin, ymax = ax.get_ylim()

# create large empty space above plot
extra_space = (ymax - ymin) * 1.5
ax.set_ylim(ymin, ymax + extra_space)

label_height = ymax + extra_space *0.8

for x, name in boundaries:
    ax.text(
        x,
        label_height,
        name,
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=11
    )

ax.set_xlabel("z (Å)")
ax.set_ylabel("SLD (10⁻⁶ Å⁻²)")
ax.set_title(" Structural Model")

ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
