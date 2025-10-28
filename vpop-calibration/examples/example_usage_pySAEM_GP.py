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
# Let us use PySAEM on a PK two compartments model.
#

# %%
# import simulated data, PySAEM and the PK two compartments model
import torch
import pickle
import sys
import numpy as np
from IPython.display import display
from plotnine import *
import uuid

sys.path.append("../")
from src.pysaem import *
from src.ode_model.pk_two_compartments_equations_with_absorption import *
from src.gp_surrogate import *
from src.nlme import *

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Step 1: Train the GP
#

# %% [markdown]
# ### Generate a training data set using an ODE model
#
# First we initiate an ODE model that will allow us to generate training data for the GP and synthetic real world data for running SAEM later.
#

# %%
# We need an ODE model, a protocol design, initial conditions and time steps
ode_model = pk_two_compartments_abs_model
print(ode_model.variable_names)

protocol_design = pd.DataFrame(
    {"protocol_arm": ["arm-A", "arm-B"], "k_el": [0.5, 10]}
)  # this is just a dummy design
time_span = (0, 24)
nb_steps = 20

# For each output and for each patient, give a list of time steps to be simulated
time_steps = np.linspace(time_span[0], time_span[1], nb_steps).tolist()

initial_conditions = np.array([10.0, 0.0, 0.0])

# Simulate a training data set using parameters sampled via Sobol sequences
log_nb_patients = 8
param_ranges = {
    "k_12": {"low": -2.0, "high": 0.0, "log": True},
    "k_21": {"low": -1.0, "high": 1.0, "log": True},
    "k_a": {"low": 0.0, "high": 1.0, "log": True},
}
learned_ode_params = list(param_ranges.keys())
nb_protocols = len(protocol_design)
print(f"Simulating {2**log_nb_patients} patients on {nb_protocols} scenario arms")
dataset = ode_model.simulate_wide_dataset_from_ranges(
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
p1 = (
    ggplot(dataset, aes(x="time", y="value", color="id"))
    + geom_line()
    + facet_grid(rows="protocol_arm", cols="output_name")
    + theme(legend_position="none")
)

p1.show()

# %% [markdown]
# ### Train the GP, or load an existing pickle
#

# %%
reuse_existing_pickled_GP = False  # pass it to true once you have trained the GP and saved it in a .pkl file (next cells). This way, you can reuse it!
save_GP_for_reuse = False  # pass it to False if you don't want a dowloaded pickel file of your serialized GP

pickled_GP_name = "my_gp_save.pkl"  # your trained GP will be saved under this name

# %%
# we first need to create our trained GP to pass to our NLMEModel
# data would be our observations. here we simulate some data with our DataGenerator
# the data should be similar to what we are going to use our GP on. Here we are using Sobol sequences to explore the space around the true parameters used to simulate the data that the GP will be used on.

gp_descriptors = learned_ode_params + ["time"]


myGP = GP(
    dataset,
    gp_descriptors,
    var_strat="IMV",  # either IMV (Independent Multitask Variational) or LMCV (Linear Model of Coregionalization Variational)
    kernel="RBF",  # Either RBF or SMK
    nb_inducing_points=50,
    nb_training_iter=500,
    training_proportion=0.9,
    learning_rate=0.1,
    jitter=1e-5,
    log_inputs=learned_ode_params,
    lr_decay=0.99,
)
if reuse_existing_pickled_GP == False:  # then train GP

    print("Training the GP")
    myGP.train(mini_batching=False, mini_batch_size=None)
    myGP.plot_loss()
else:  # reuse serialized GP (overrides the above definition)
    filepath = pickled_GP_name

    try:
        with open(filepath, "rb") as file:
            myGP = pickle.load(file)

        print("Model loaded successfully!")
        myGP.plot_loss()

    except FileNotFoundError:
        print(
            f"File not found. Please make sure '{filepath}' exists and is in the correct directory."
        )

# %%
if (reuse_existing_pickled_GP == False) and (save_GP_for_reuse == True):
    import pickle

    filepath = pickled_GP_name
    with open(filepath, "wb") as file:
        pickle.dump(myGP, file)
    print(f"Model saved to {filepath}")

# %%
myGP.eval_perf()
myGP.plot_obs_vs_predicted(data_set="training")

# %% [markdown]
# # Step 2: Generate a real-world data set
#

# %%
# Set up an NLME model to simulate a real-world data set
structural_model = StructuralOdeModel(
    ode_model, protocol_design, initial_conditions, time_steps
)

# Parameter definitions
true_log_MI = {}
true_log_PDU = {
    "k_12": {"mean": -1, "sd": 0.25},
    "k_21": {"mean": 0, "sd": 0.25},
}
error_model_type = "additive"
true_res_var = [0.5, 0.02, 0.1]
# list for each PDU, which covariate influences it, and the name of the coefficient
covariate_map = {
    "k_12": {"foo": {"coef": "cov_foo_k12", "value": 0.1}},
    "k_21": {},
}

# Create a patient data frame
# It should contain at the very minimum one `id` per patient
nb_patients = 52
patients_df = pd.DataFrame({"id": [str(uuid.uuid4()) for _ in range(nb_patients)]})
rng = np.random.default_rng()
patients_df["protocol_arm"] = rng.binomial(1, 0.5, nb_patients)
patients_df["protocol_arm"] = patients_df["protocol_arm"].apply(
    lambda x: "arm-A" if x == 0 else "arm-B"
)
patients_df["k_a"] = rng.normal(1, 0.2, nb_patients)
patients_df["foo"] = rng.normal(0, 1, nb_patients)
display(patients_df)

# %%
# we create our NLMEModel (data structure defined in PysAEM) to pass to PySAEM
pk_model_true = NLMEModel(
    structural_model,
    patients_df,
    true_log_MI,
    true_log_PDU,
    true_res_var,
    covariate_map,
    error_model_type,
)

# %%
# Generating a synthetic real life data set

observations_df = pk_model_true.generate_dataset_from_omega()
display(observations_df)

# %%
p1 = (
    ggplot(observations_df, aes(x="time", y="value", color="id"))
    + geom_line()
    + facet_grid(rows="protocol_arm", cols="output_name")
    + theme(legend_position="none")
)

p1.show()

# %% [markdown]
# # Step 3: Train the surrogate model on real-world data
#

# %%
gp_structural_model = StructuralGp(gp_model=myGP, time_steps=time_steps)

# Initial values for the NLME model
# Different than the true values
init_log_MI = {}
init_log_PDU = {
    "k_12": {"mean": -0.2, "sd": 0.25},
    "k_21": {"mean": 0.2, "sd": 0.25},
}
init_res_var = [0.1, 0.1, 0.2]
init_covariate_map = {
    "k_12": {"foo": {"coef": "cov_foo_k12", "value": 0.1}},
    "k_21": {},
}

# %%
# Create the model to be optimized
nlme_surrogate = NLMEModel(
    gp_structural_model,
    patients_df,
    init_log_MI,
    init_log_PDU,
    init_res_var,
    init_covariate_map,
    error_model_type,
)

# Create the optimizer
optimizer = PySAEM(
    nlme_surrogate,
    observations_df,
    mcmc_burn_in=1,
    mcmc_first_burn_in=5,
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

# %%
# %%prun -s cumulative
optimizer.run()

# %%
map_per_patient = optimizer.plot_map_estimates()

# %%
true_MI = {name: np.exp(val) for name, val in true_log_MI.items()}
true_mus = {name: np.exp(val["mean"]) for name, val in true_log_PDU.items()}
true_sd = {name: val["sd"] for name, val in true_log_PDU.items()}
true_covs = {
    str(cov["coef"]): float(cov["value"])
    for item in covariate_map.values()
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


