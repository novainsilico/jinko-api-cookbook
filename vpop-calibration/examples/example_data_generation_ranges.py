# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import sys
import numpy as np
import pandas as pd

sys.path.append("../")
from src.ode_model.pk_two_compartments_equations import pk_two_compartments_model

from IPython.display import display

# %load_ext autoreload
# %autoreload 2

# %%
my_model = pk_two_compartments_model
nb_timesteps = 15
tmax = 24.0
initial_conditions = np.array([10.0, 0.0])
time_steps = np.linspace(0.0, tmax, nb_timesteps)

# %%
# Testing the simulate method
timesteps = [0, 1, 2, 3]
n = len(timesteps)
df = pd.DataFrame(
    {
        "id": ["foo"] * n,
        "protocol_arm": ["A"] * n,
        "k_12": [0.5] * n,
        "k_21": [0.1] * n,
        "k_el": [0.1] * n,
        "output_name": ["A0"] * n,
        "time": timesteps,
        "A0_0": [10] * n,
        "A1_0": [0] * n,
    }
)

my_model.simulate_model(df)

# %%
# Testing the simulate on ranges method
log_nb_patients = 5
param_ranges = {
    "k_12": {"low": 0.02, "high": 0.07, "log": False},
    "k_21": {"low": 0.1, "high": 0.3, "log": False},
    "k_el": {"low": 0.1, "high": 0.3, "log": False},
}

protocol_design = pd.DataFrame({"protocol_arm": ["A", "B"], "k_el": [0.1, 0.5]})
nb_protocols = len(protocol_design)
print(f"Simulating {2**log_nb_patients} patients on {nb_protocols} scenario arms")
dataset = my_model.simulate_wide_dataset_from_ranges(
    log_nb_patients,
    param_ranges,
    initial_conditions,
    protocol_design,
    np.array([0.05, 0.0002]),
    "additive",
    time_steps,
)
display(dataset)

# %%
from plotnine import *

p1 = (
    ggplot(dataset, aes(x="time", y="value", color="id"))
    + geom_line()
    + facet_grid(rows="protocol_arm", cols="output_name")
    + theme(legend_position="none")
)

p1.show()


