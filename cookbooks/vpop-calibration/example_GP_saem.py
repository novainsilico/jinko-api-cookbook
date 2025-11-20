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

# %% [markdown]
# This notebook demonstrates how to use the full tensorized implementation of NLMEs and PySAEM.
#
# Basically, it is the same as PySAEM and NLME, only the internal logic changes.
#

# %%
# import simulated data, PySAEM and the PK two compartments model
import torch
import pickle
import sys
import numpy as np
import pandas as pd
from IPython.display import display
from plotnine import *
import uuid

from vpop_calibration import (
    OdeModel,
    GP,
    StructuralGp,
    NlmeModel,
    PySAEM,
    simulate_dataset_from_ranges,
    simulate_dataset_from_omega,
)


# %% [markdown]
# # Step 1: Train the GP
#

# %% [markdown]
# ### Generate a training data set using an ODE model
#
# First we initiate an ODE model that will allow us to generate training data for the GP and synthetic real world data for running SAEM later.
#

# %%
# Define the reference PK model we will use for training the GP and generating a synthetic data set
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

ode_model = OdeModel(equations, variable_names, parameter_names)
print(ode_model.variable_names)
time_span = (0, 24)
nb_steps = 10
time_steps = np.linspace(time_span[0], time_span[1], nb_steps).tolist()

protocol_design = pd.DataFrame({"protocol_arm": ["arm-A", "arm-B"], "k_el": [0.5, 10]})
nb_protocols = len(protocol_design)

initial_conditions = np.array([10.0, 0.0, 0.0])

# %% [markdown]
# ### Train the GP, or load an existing pickle
#

# %%
model_file = "gp_surrogate_pk_model.pkl"
folder_path = "./"

model_full_path = folder_path + model_file

use_pickle = True
override_existing_pickle = True

# %%
myGP = GP.__new__(GP)  # this is just to silence pylance's unbound variable warnings
if (override_existing_pickle) or (not use_pickle):
    # Simulate a training data set using parameters sampled via Sobol sequences
    log_nb_patients = 8
    param_ranges = {
        "k_12": {"low": -2.0, "high": 0.0, "log": True},
        "k_21": {"low": -1.0, "high": 1.0, "log": True},
        "k_a": {"low": 0.0, "high": 1.0, "log": True},
    }

    print(f"Simulating {2**log_nb_patients} patients on {nb_protocols} scenario arms")
    dataset = simulate_dataset_from_ranges(
        ode_model,
        log_nb_patients,
        param_ranges,
        initial_conditions,
        protocol_design,
        None,
        None,
        time_steps,
    )

    learned_ode_params = list(param_ranges.keys())
    descriptors = learned_ode_params + ["time"]

    # Instantiate a GP
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
    # Train the GP
    myGP.train()

    if use_pickle and override_existing_pickle:
        with open(model_full_path, "wb") as file:
            pickle.dump(myGP, file)
        print(f"Model saved to {model_file}")
elif use_pickle and (not override_existing_pickle):
    try:
        with open(model_full_path, "rb") as file:
            myGP = pickle.load(file)

        print("Model loaded successfully!")

    except FileNotFoundError:
        print(
            f"File not found. Please make sure '{model_full_path}' exists and is in the correct directory."
        )

# %% [markdown]
# # Step 2: Generate a real-world data set
#

# %%
time_span_rw = (0, 24)
nb_steps_rw = 10

# For each output and for each patient, give a list of time steps to be simulated
time_steps_rw = np.linspace(time_span_rw[0], time_span_rw[1], nb_steps_rw).tolist()

# Parameter definitions
true_log_MI = {"k_21": 0.0}
true_log_PDU = {
    "k_12": {"mean": -1.0, "sd": 0.25},
}
error_model_type = "additive"
true_res_var = [0.5, 0.02, 0.1]
true_covariate_map = {
    "k_12": {"foo": {"coef": "cov_foo_k12", "value": 0.2}},
}

# Create a patient data frame
# It should contain at the very minimum one `id` per patient
nb_patients = 50
patients_df = pd.DataFrame({"id": [str(uuid.uuid4()) for _ in range(nb_patients)]})
rng = np.random.default_rng()
patients_df["protocol_arm"] = rng.binomial(1, 0.5, nb_patients)
patients_df["protocol_arm"] = patients_df["protocol_arm"].apply(
    lambda x: "arm-A" if x == 0 else "arm-B"
)
patients_df["k_a"] = rng.lognormal(-1, 0.1, nb_patients)
patients_df["foo"] = rng.lognormal(0, 0.1, nb_patients)
display(patients_df)

print(f"Simulating {nb_patients} patients on {nb_protocols} protocol arms")
obs_df = simulate_dataset_from_omega(
    ode_model,
    protocol_design,
    time_steps,
    initial_conditions,
    true_log_MI,
    true_log_PDU,
    error_model_type,
    true_res_var,
    true_covariate_map,
    patients_df,
)

display(obs_df)

# %%
p1 = (
    ggplot(obs_df, aes(x="time", y="value", color="id"))
    + geom_line()
    + facet_grid(rows="protocol_arm", cols="output_name")
    + theme(legend_position="none")
)

p1.show()

# %% [markdown]
# # Step 3: Train the surrogate model on real-world data
#

# %%
# Initial pop estimates
# Parameter definitions
init_log_MI = {"k_21": -1.0}
# init_log_MI = {}
init_log_PDU = {
    "k_12": {"mean": -0.1, "sd": 0.1},
    # "k_21": {"mean": -1.0, "sd": 0.2},
}
error_model_type = "additive"
init_res_var = [0.1, 0.05, 0.5]
init_covariate_map = {
    "k_12": {"foo": {"coef": "cov_foo_k12", "value": -0.1}},
    # "k_21": {},
}

# Create a structural model
structural_gp = StructuralGp(myGP)
# Create a NLME moedl
nlme_surrogate = NlmeModel(
    structural_gp,
    patients_df,
    init_log_MI,
    init_log_PDU,
    init_res_var,
    init_covariate_map,
    error_model_type,
)
# Create an optimizer: here we use SAEM
optimizer = PySAEM(
    nlme_surrogate,
    obs_df,
    mcmc_burn_in=0,
    mcmc_first_burn_in=0,
    mcmc_nb_samples=1,
    mcmc_proposal_var_scaling_factor=0.5,
    nb_phase1_iterations=10,
    nb_phase2_iterations=None,
    convergence_threshold=1e-4,
    patience=5,
    learning_rate_power=0.8,
    annealing_factor=0.95,
    verbose=False,
)

# %%
# %%prun -s cumulative
optimizer.run()

# %%
optimizer.plot_map_estimates()

# %%
true_MI = {name: np.exp(val) for name, val in true_log_MI.items()}
true_mus = {name: np.exp(val["mean"]) for name, val in true_log_PDU.items()}
true_sd = {name: val["sd"] for name, val in true_log_PDU.items()}
true_covs = {
    str(cov["coef"]): float(cov["value"])
    for item in true_covariate_map.values()
    for cov in item.values()
}
true_betas = true_mus | true_covs
print(true_betas)
true_sigmas = {
    name: float(true_res_var[j]) for j, name in enumerate(ode_model.variable_names)
}
print(true_sigmas)

# %%
optimizer.plot_convergence_history(
    true_MI=true_MI,
    true_betas=true_betas,
    true_residual_var=true_sigmas,
    true_sd=true_sd,
)


