import numpy as np
import matplotlib.pyplot as plt
from refl1d.names import *

directory = "medium-dataset"
config = "gans_flowcell"

sample = 99 

reflectivity = np.load(
    f"{directory}/{config}_reflectivity_data.npy",
    mmap_mode="r"
)
params = np.load(f"{directory}/{config}_sample_parameters.npy")
q = np.load(f"{directory}/{config}_q_grid.npy")

p = params[sample]


param_names = [
    "sub_rough",

    "sio2_thick","sio2_sld","sio2_rough",

    "head1_thick","head1_sld","head1_rough",

    "tail_thick","tail_sld","tail_rough",

    "head2_thick","head2_sld","head2_rough",

    "med_thick","med_sld","med_rough"
]

contrasts = [
    ("H2O",-0.56,"blue"),
    ("D2O",6.36,"red"),
    ("MIX",2.07,"gold"),
]


layers = [
    dict(name="substrate",sld=2.07,thickness=0,roughness=p[0]),

    dict(name="sio2",thickness=p[1],sld=p[2],roughness=p[3]),
    dict(name="head1",thickness=p[4],sld=p[5],roughness=p[6]),
    dict(name="tail",thickness=p[7],sld=p[8],roughness=p[9]),
    dict(name="head2",thickness=p[10],sld=p[11],roughness=p[12]),
    dict(name="medicine",thickness=p[13],sld=p[14],roughness=p[15]),
]

profiles = {}

for name,medium,color in contrasts:

    probe = QProbe(q,0.02*q/2.355,data=(np.zeros(len(q)),np.zeros(len(q))))

    sample = Slab(
        material=SLD(name="substrate",rho=layers[0]["sld"]),
        interface=layers[0]["roughness"]
    )

    for l in layers[1:]:

        sample = sample | Slab(
            material=SLD(name=l["name"],rho=l["sld"]),
            thickness=l["thickness"],
            interface=l["roughness"]
        )

    sample = sample | Slab(
        material=SLD(name="medium",rho=medium),
        thickness=0,
        interface=0
    )

    expt = Experiment(probe=probe,sample=sample)

    z,sld,_ = expt.smooth_profile()

    profiles[name] = sld

profiles_array = np.vstack([
    profiles["H2O"],
    profiles["D2O"],
    profiles["MIX"]
])

diff = np.max(np.abs(profiles_array - profiles_array[0]),axis=0)

diverge_idx = np.argmax(diff > 1e-6)

fig,ax = plt.subplots(1,3,figsize=(18,5))

for i,(name,_,color) in enumerate(contrasts):

    ax[0].semilogy(
        q,
        reflectivity[sample,i],
        color=color,
        label=name,
        linewidth=2
    )

ax[0].set_xlabel("Q (Å⁻¹)")
ax[0].set_ylabel("R(Q)")
ax[0].set_title("Reflectivity")
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot(
    z[:diverge_idx],
    profiles["MIX"][:diverge_idx],
    color="black",
    linewidth=3,
    label="Structure"
)

for name,_,color in contrasts:

    ax[1].plot(
        z[diverge_idx:],
        profiles[name][diverge_idx:],
        color=color,
        label=name
    )

# layer boundaries
zpos = 0
for l in layers[1:]:

    zpos += l["thickness"]

    ax[1].axvline(zpos,color="gray",linestyle="--",alpha=0.5)

    ax[1].text(
        zpos,
        ax[1].get_ylim()[1],
        l["name"],
        rotation=90,
        verticalalignment="bottom",
        fontsize=9
    )

ax[1].set_xlabel("z (Å)")
ax[1].set_ylabel("SLD (10⁻⁶ Å⁻²)")
ax[1].set_title("Structural Model")
ax[1].legend()
ax[1].grid(alpha=0.3)

ax[2].axis("off")

text = "\n".join(
    [f"{name:12s}: {val:7.3f}" for name,val in zip(param_names,p)]
)

ax[2].text(
    0.05,
    0.95,
    f"Sample {sample}\n\n{text}",
    va="top",
    family="monospace",
    fontsize=11
)

plt.tight_layout()
plt.show()
