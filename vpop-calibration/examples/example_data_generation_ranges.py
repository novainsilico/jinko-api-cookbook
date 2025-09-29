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

sys.path.append("../src")
from DataGenerator import DataGenerator
from pk_two_compartments_equations import *

# %%
nb_outputs = 2
nb_parameters = 3
nb_timesteps = 15
tmax = 24.0
initial_conditions = np.array([5.0, 3.0])
output_names = ["A0", "A1"]
param_names = ["k_12", "k_21", "k_el"]
time_steps = np.linspace(0.0, tmax, nb_timesteps)

data_generator = DataGenerator(
    pk_two_compartments_model,
    output_names,
    param_names,
)

# %%
log_nb_patients = 3
print(f"Simulating {2**log_nb_patients} patients")
param_ranges = {
    "k_12": {"low": 0.02, "high": 0.07, "log": False},
    "k_21": {"low": 0.1, "high": 0.3, "log": False},
    "k_el": {"low": -4, "high": 0, "log": True},
}
dataset = data_generator.simulate_wide_dataset_from_ranges(
    log_nb_patients,
    param_ranges,
    initial_conditions,
    np.array([0.05, 0.0002]),
    "additive",
    time_steps,
)

# %%
from plotnine import *

p1 = (
    ggplot(dataset, aes(x="time", y="A0", color="ind_id"))
    + geom_line()
    + theme(legend_position="none")
)


p2 = (
    ggplot(dataset, aes(x="time", y="A1", color="ind_id"))
    + geom_line()
    + theme(legend_position="none", figure_size=[10, 4])
)

(p1 | p2)
