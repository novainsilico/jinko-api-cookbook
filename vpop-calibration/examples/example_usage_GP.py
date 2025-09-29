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
import torch
import sys
import numpy as np

# %load_ext autoreload
# %autoreload 2

sys.path.append("../src")
from GP import *
from DataGenerator import *
from pk_two_compartments_equations_with_absorption import *

# %%
nb_outputs = 3
nb_parameters = 4
output_names = ["A0", "A1", "A2"]
param_names = ["k_a", "k_12", "k_21", "k_el"]
data_generator = DataGenerator(
    pk_two_compartments_model,
    output_names,
    param_names,
)

# %%
# time
nb_time_steps = 20
tmax = 24.0
time_steps = np.linspace(0.0, tmax, nb_time_steps)
initial_conditions = np.array([10.0, 0.0, 0.0])
observation_noise = np.array([0.1,0.05,0.03])
log2_nb_patients = 6
nb_patients = 2**log2_nb_patients
param_ranges = {
    "k_a": {"low": -2, "high": 0, "log":True},
    "k_12": {"low": 0.02, "high": 0.07, "log": False},
    "k_21": {"low": 0.1, "high": 0.3, "log": False},
    "k_el": {"low": -4, "high": 0, "log": True},
}


print(f"Simulating {2**log2_nb_patients} patients")
dataset = data_generator.simulate_wide_dataset_from_ranges(
    log2_nb_patients,
    param_ranges,
    initial_conditions,
    observation_noise,
    "additive",
    time_steps,
)

# %%
display(dataset)

# %%
# Transform the inputs for easier fitting
dataset[param_names] = dataset[param_names].map(np.log10)

# Convert to a torch tensor
training_data = torch.tensor(dataset.drop(["ind_id"], axis=1).values, dtype=torch.float32)
display(training_data)

# %%
# initiate our GP class
myGP = GP(
    nb_parameters + 1, # time is an additional input for the GP
    param_names,
    nb_outputs,
    output_names,
    training_data,
    var_strat="IMV",  # either IMV (Independent Multitask Variational) or LMCV (Linear Model of Coregionalization Variational)
    kernel="RBF",  # Either RBF or SMK
    data_already_normalized=False,  # default
    nb_inducing_points=100,
    mll="ELBO",  # default, otherwise PLL
    nb_training_iter=500,
    training_proportion=0.7,
    learning_rate=0.01,
    num_mixtures=3,
    jitter=1e-4,
)

# %%
myGP.train(mini_batching=False, mini_batch_size=None)

# %%
myGP.plot_loss()
myGP.eval_perf()

# %%
myGP.plot_obs_vs_predicted(data_set="training")
myGP.plot_obs_vs_predicted(data_set="validation")

# %%
j = torch.randint(nb_patients, (1,))[0]
myGP.plot_individual_solution(j)

# %%
myGP.plot_all_solutions("training")
# myGP.plot_all_solutions("validation")
