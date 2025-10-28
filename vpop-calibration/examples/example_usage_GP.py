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
from IPython.display import display

# %load_ext autoreload
# %autoreload 2

sys.path.append("../")
from src.gp_surrogate import *
from src.ode_model.pk_two_compartments_equations_with_absorption import *

# %%
my_model = pk_two_compartments_abs_model
nb_timesteps = 15
tmax = 24.0
initial_conditions = np.array([10.0, 0.0, 0.0])
time_steps = np.linspace(0.0, tmax, nb_timesteps)

# %%
# Generate training data using an ODE model
log_nb_patients = 5
nb_patients = 2**log_nb_patients
param_ranges = {
    "k_12": {"low": 0.02, "high": 0.07, "log": False},
    "k_21": {"low": 0.1, "high": 0.3, "log": False},
    "k_a": {"low": -2.0, "high": 0.0, "log": True},
}

protocol_design = pd.DataFrame({"protocol_arm": ["arm-A", "arm-B"], "k_el": [0.1, 0.5]})
nb_protocols = len(protocol_design)
print(f"Simulating {nb_patients} patients on {nb_protocols} scenario arms")
dataset = my_model.simulate_wide_dataset_from_ranges(
    log_nb_patients,
    param_ranges,
    initial_conditions,
    protocol_design,
    None,
    None,
    time_steps,
)
display(dataset)

# %%
learned_ode_params = list(param_ranges.keys())
descriptors = learned_ode_params + ["time"]
print(descriptors)

# initiate our GP class
myGP = GP(
    dataset,
    descriptors,
    var_strat="IMV",  # either IMV (Independent Multitask Variational) or LMCV (Linear Model of Coregionalization Variational)
    kernel="RBF",  # Either RBF or SMK
    data_already_normalized=False,  # default
    nb_inducing_points=100,
    mll="ELBO",  # default, otherwise PLL
    nb_training_iter=500,
    training_proportion=0.7,
    learning_rate=0.05,
    jitter=1e-6,
    log_inputs=learned_ode_params,
)

# %%
myGP.train(mini_batching=False, mini_batch_size=None)

# %%
myGP.plot_loss()
myGP.eval_perf()

# %%
myGP.plot_obs_vs_predicted(data_set="training")

# %%
j = torch.randint(nb_patients, (1,))[0]
myGP.plot_individual_solution(j)

# %%
myGP.plot_all_solutions("training")
