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
import pandas as pd
from IPython.display import display

from vpop_calibration import OdeModel, simulate_dataset_from_ranges, GP


# %%
# Define the ode model
def equations(t, y, k_a, k_12, k_21, k_el):
    # y[0] is A_absorption, y[1] is A_central, y[2] is A_peripheral
    A_absorption, A_central, A_peripheral = y[0], y[1], y[2]
    dA_absorption_dt = -k_a * A_absorption
    dA_central_dt = (
        k_a * A_absorption + k_21 * A_peripheral - k_12 * A_central - k_el * A_central
    )
    dA_peripheral_dt = k_12 * A_central - k_21 * A_peripheral

    ydot = [dA_absorption_dt, dA_central_dt, dA_peripheral_dt]
    return ydot


variable_names = ["A0", "A1", "A2"]
parameter_names = ["k_a", "k_12", "k_21", "k_el"]

pk_two_compartments_model = OdeModel(equations, variable_names, parameter_names)

# %%
# Generate training data using an ODE model
log_nb_patients = 5
nb_patients = 2**log_nb_patients
param_ranges = {
    "k_12": {"low": 0.02, "high": 0.07, "log": False},
    "k_21": {"low": 0.1, "high": 0.3, "log": False},
    "k_a": {"low": -2.0, "high": 0.0, "log": True},
}

initial_conditions = np.array([10.0, 0.0, 0.0])

nb_timesteps = 15
tmax = 24.0
time_steps = np.linspace(0.0, tmax, nb_timesteps)

protocol_design = pd.DataFrame({"protocol_arm": ["arm-A", "arm-B"], "k_el": [0.1, 0.5]})
nb_protocols = len(protocol_design)
print(f"Simulating {nb_patients} patients on {nb_protocols} scenario arms")
dataset = simulate_dataset_from_ranges(
    pk_two_compartments_model,
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
    nb_training_iter=200,
    training_proportion=0.7,
    learning_rate=0.1,
    lr_decay=0.99,
    jitter=1e-6,
    log_inputs=learned_ode_params,
)

# %%
myGP.train()

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
