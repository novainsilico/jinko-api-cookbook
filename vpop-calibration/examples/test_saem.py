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
# import simulated data, PySAEM and the PK two compartments model
import sys
import numpy as np
import pandas as pd
from IPython.display import display

sys.path.append("../")
from src.ode_model.pk_two_compartments_equations_with_absorption import *
from src.nlme import *
from src.pysaem import *

# %load_ext autoreload
# %autoreload 2

# %load_ext cProfile

# %%
# Create an NLME model
# First the structural model
# We need an ODE model, initial conditions and time steps

ode_model = pk_two_compartments_abs_model
print(ode_model.variable_names)

protocol_design = pd.DataFrame(
    {"protocol_arm": ["arm-A", "arm-B"]}
)  # this is just a dummy design
time_span = (0, 24)
nb_steps = 20

# For each output and for each patient, give a list of time steps to be simulated
time_steps = np.linspace(time_span[0], time_span[1], nb_steps).tolist()

initial_conditions = np.array([10.0, 0.0, 0.0])
structural_model = StructuralOdeModel(
    ode_model, protocol_design, initial_conditions, time_steps
)

# Parameter definitions
init_log_MI = {"k_a": 0.1}
init_log_PDU = {
    "k_12": {"mean": -1, "sd": 0.25},
    "k_21": {"mean": 0, "sd": 0.25},
    "k_el": {"mean": -1, "sd": 0.25},
}
error_model_type = "additive"
init_res_var = [0.05, 0.02, 0.1]
covariate_map = {
    "k_12": {
        "foo": {"coef": "cov_foo_k12", "value": 0.5},
        "bar": {"coef": "cov_bar_k12", "value": 1.5},
    },
    "k_21": {},
    "k_el": {"foo": {"coef": "cov_foo_kel", "value": 0.2}},
}  # list for each PDU, which covariate influences it, and the name of the coefficient

patient_covariates = pd.DataFrame(
    {
        "id": ["a", "b", "c", "d"],
        "protocol_arm": ["arm-A", "arm-A", "arm-B", "arm-A"],
        "foo": [1, 2, 3, 4],
        "bar": [1, 1, 1, 1],
    }
)

# %%
nlme_model = NLMEModel(
    structural_model,
    patient_covariates,
    init_log_MI,
    init_log_PDU,
    init_res_var,
    covariate_map,
    error_model_type,
)

# %%
observations_df = nlme_model.generate_dataset_from_omega()
display(observations_df)

# %%
nlme_model.init_mcmc_sampler(observations_df, verbose=True)

# %%
eta_init = nlme_model.sample_individual_etas()
nlme_model._log_posterior_etas(eta_init, nlme_model.patients)

# %%
# eta_init = nlme_model.sample_individual_etas()
eta_init = torch.zeros((nlme_model.nb_patients, nlme_model.nb_PDU))
eta, mean_PDU, pred_df = nlme_model.mcmc_sample(
    init_eta_for_all_ind=eta_init,
    nb_samples=1,
    nb_burn_in=0,
    proposal_var_eta=torch.diag(torch.Tensor([0.05, 0.05, 0.05])),
)
display(mean_PDU)

# %%
log_MI = nlme_model.log_MI
log_MI_expanded = log_MI.unsqueeze(0).repeat((nlme_model.nb_patients, 1))
theta = torch.exp(torch.cat((log_MI_expanded, mean_PDU), dim=1))
patient_descriptors = pd.DataFrame(data=theta.numpy(), columns=nlme_model.descriptors)

# %%
predicted = nlme_model.generate_dataset_from_omega().rename(
    columns={"value": "predicted_value"}
)
display(predicted)
nlme_model.sum_sq_residuals(predicted)

# %%
optimizer = PySAEM(
    nlme_model,
    observations_df,
    mcmc_burn_in=1,
    mcmc_first_burn_in=0,
    mcmc_nb_samples=1,
    mcmc_proposal_var_scaling_factor=0.5,
    nb_phase1_iterations=10,
    nb_phase2_iterations=None,
    convergence_threshold=1e-4,
    patience=5,
    learning_rate_power=0.8,
    annealing_factor=0.95,
    verbose=True,
)
optimizer.run()
