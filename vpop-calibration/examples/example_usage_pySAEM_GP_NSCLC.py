# -*- coding: utf-8 -*-
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
import torch
import pickle
import sys
import numpy as np

sys.path.append("../src")
from PySAEM_for_one_timepoint import *
from pk_two_compartments_equations_with_absorption import *
from GP import *
from DataGenerator import *


torch.set_default_dtype(torch.float32)

# %%
# Retrieve the GP
import pickle

filepath = "trained_gp_model.pkl"
try:
    with open(filepath, "rb") as file:
        myGP = pickle.load(file)
    myGP.plot_loss()

except FileNotFoundError:
    print(
        f"File not found. Please make sure '{filepath}' exists and is in the correct directory."
    )

# %%
# let us define the arguments asked by PySAEM, corresponding to our model
MI_names: List[str] = []
PDU_names: List[str] = [
    "Tumor1:apparitionEgfrT790m",
    "Tumor2:apparitionEgfrT790m",
    "Tumor3:apparitionEgfrT790m",
    "Tumor4:apparitionEgfrT790m",
    "Tumor5:apparitionEgfrT790m",
]
cov_coeffs_names: List[str] = (
    []
)  # if other PDKs can impact the PDUs I want to calibrate
output_names: List[str] = ["Line1:TimeToProgression"]
error_model_type: str = "additive"
covariate_map: Dict[str, List[str]] = (
    {}
)  # list for each PDU, which covariate influences it

# %%
# we first need to create our trained GP to pass to our NLMEModel
# data would be our observations. here we simulate some data in the import data_pk_two_compartments_absorption.py
# the data should be similar to what we are going to use our GP on. Here we are using Sobol sequences to explore the space around the true parameters used to simulate the data that the GP will be used on.
reuse_existing_pickled_GP = True  # pass it to true once you have trained the GP and saved it in a .pkl file (next cells). This way, you can reuse it!
save_GP_for_reuse = True  # pass it to False if you don't want a dowloaded pickel file of your serialized GP
pickled_GP_name = (
    "trained_gp_model.pkl"  # your trained GP will be saved under this name
)


# instantiate our DataGenerator, which we will use to generate data both to train our GP on and to execute SAEM on
param_names = MI_names + PDU_names
initial_conditions: torch.Tensor = torch.Tensor([10.0, 4.0, 0.0])
data_generator = DataGenerator(
    pk_two_compartments_model,
    output_names,
    param_names,
    initial_conditions,
)

if reuse_existing_pickled_GP == False:  # then train GP
    # define the parameter space we want to explore
    # should be a range in which you think your parameters will be
    # here I reference the rates that I will reuse later to simulate the data
    k_a = 0.05
    k12 = 0.67
    k21 = 0.20
    k_el = 0.15
    bounds_low: torch.Tensor = torch.tensor(
        [0.05 * 0.5, k12 * 0.5, k21 * 0.5, k_el * 0.5]
    )
    bounds_high: torch.Tensor = torch.tensor([k_a * 2, k12 * 2, k21 * 2, k_el * 2])
    nb_steps: int = 20
    time_span: Tuple[float, float] = (0.0, 24.0)
    time_steps: torch.Tensor = torch.linspace(time_span[0], time_span[1], nb_steps)
    dataset = data_generator.simulate_wide_dataset_from_ranges(
        1024,
        bounds_low,
        bounds_high,
        torch.Tensor((0.01, 0.01)),
        "additive",  # "additive" or "proportional"
        time_steps,
    )
    torch.manual_seed(42)
    myGP = GP(
        len(MI_names)
        + len(PDU_names)
        + 1,  # + 1 for the time. in the backlog of the GP, I suggest to change the input to avoid this weird parameters number computation
        param_names,
        len(output_names),
        output_names,
        dataset,
        var_strat="IMV",  # either IMV (Independent Multitask Variational) or LMCV (Linear Model of Coregionalization Variational)
        kernel="RBF",  # Either RBF or SMK
        data_already_normalized=False,  # default
        nb_inducing_points=100,  # 500,
        mll="ELBO",  # default, otherwise PLL
        nb_training_iter=10,  # 700,
        training_proportion=0.7,
        learning_rate=0.01,
        num_mixtures=3,
        jitter=1e-4,
    )
    myGP.train(mini_batching=False, mini_batch_size=None)
    myGP.plot_loss()
else:  # reuse serialized GP
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
if reuse_existing_pickled_GP == False and save_GP_for_reuse:
    import pickle

    filepath = pickled_GP_name
    with open(filepath, "wb") as file:
        pickle.dump(myGP, file)
    print(f"Model saved to {filepath}")

# %%
myGP.eval_perf()
myGP.plot_obs_vs_predicted(data_set="training")

# %%
# we create our NLMEModel (data structure defined in PysAEM) to pass to PySAEM
pk_model = NLMEModel_from_GP(
    MI_names=MI_names,
    PDU_names=PDU_names,
    cov_coeffs_names=cov_coeffs_names,
    outputs_names=output_names,
    GP_model=myGP,
    GP_error_threshold=0.7,
    error_model_type=error_model_type,
    covariate_map=covariate_map,
)

# %%
# # we have to pass timeseries of the outputs for each individual to PySAEM, under the data structure List[IndividualData]
# # here we simulate the data, with the method generate_data_for_pySAEM from data_pysaem_example_usage
# # in this method, an ODE solver is used on lists of parameters centered around the same parameters the GP was trained on, with simulated etas and covariate effects.
# # the true parameters will of course be hidden to pySAEM which will try to infer them
# # normally, all_individual_data would be constructed from experimental observations

# # chose arbitrary theoretical values
# nb_individuals: int = 30
# # time
# time_span: Tuple[int, int] = (0, 24)
# nb_steps1: int = 20
# nb_steps2: int = 10

# time_steps1: torch.Tensor = torch.linspace(time_span[0], time_span[1], nb_steps1)
# time_steps2: torch.Tensor = torch.linspace(time_span[0], time_span[1], nb_steps2)

# list_intermediate = [time_steps1, time_steps2]
# list_time_steps: List[List[torch.Tensor]] = [list_intermediate] * nb_individuals


# # true pop parameters
# V1: float = 15.0  # volume of compartment 1
# V2: float = 50.0
# Q: float = 10.0  # intercompartmental clearance
# true_k_a: float = 0.05  # absorption rate for both compartments
# true_k_el: float = 0.15  # elimination rate of compartment 1
# true_k12: float = Q / V1
# true_k21: float = Q / V2
# true_coeff_MEMBRANE12_k12: float = 0
# true_coeff_MEMBRANE12_k21: float = 0
# true_MI: torch.Tensor = torch.Tensor([true_k_a]).unsqueeze(-1)
# true_betas: torch.Tensor = torch.Tensor(
#     [
#         log(true_k12),
#         true_coeff_MEMBRANE12_k12,
#         log(true_k21),
#         true_coeff_MEMBRANE12_k21,
#         log(true_k_el),
#     ]
# )
# true_omega: torch.Tensor = torch.tensor(
#     [[0.1**2, 0.0, 0.0], [0.0, 0.1**2, 0.0], [0.0, 0.0, 0.2**2]]
# )  # Variance of eta_k12, eta_k21, eta_k_el
# true_residual_var: torch.Tensor = torch.Tensor([0.05, 0.07]).unsqueeze(
#     -1
# )  # residual error variance

# list_covariates_dict = [{"name":"MEMBRANE_12", "mean":0, "std": 0}]
# # generate the data
# observations_df, covariates_df = data_generator.simulate_dataset_from_omega(
#     nb_individuals,
#     true_MI,
#     true_betas,
#     true_omega,
#     true_residual_var,
#     list_covariates_dict
#     list_time_steps,
#     pk_model,
# )

# %%
# Load data
import pandas as pd

dataStudy = pd.read_csv("ttp_censored.csv")

dataStudy["ind_id"] = [f"id_{i}" for i in range(dataStudy.shape[0])]
dataStudy["output_name"] = "Line1:TimeToProgression"
dataStudy = dataStudy.rename(columns={"time": "value"})

print(dataStudy)


# %%
# PySAEM (as any SAEM algorithm) requires initial guesses of the population parameters
# here we can start close to our true parameters used for simulation
# normally the biomodeler has to put in some initial guess
initial_pop_MI: torch.Tensor = torch.Tensor([]).unsqueeze(-1)
initial_pop_betas = torch.Tensor(
    [
        log(5),
        log(6),
        log(5),
        log(6),
        log(5),
    ]
).unsqueeze(-1)
initial_pop_omega = torch.diag(
    torch.Tensor([1, 1, 1, 1, 3])
)  # covariance matrix of individual effects
initial_res_var = torch.Tensor([4]).unsqueeze(
    -1
)  # residual error variance (depends on the number of outputs)

# %%
#  run PySAEM

saem = PySAEM_for_one_timepoint(
    model=pk_model,
    observations_df=dataStudy,
    covariates_df=None,
    initial_pop_MI=initial_pop_MI,
    initial_pop_betas=initial_pop_betas,
    initial_pop_omega=initial_pop_omega,
    initial_res_var=initial_res_var,
    mcmc_first_burn_in=30,  # used at the first iteration, bigger because the initial etas start at zeros (afterwards they start at the last etas)
    mcmc_burn_in=5,  # used for the rest of the iterations
    mcmc_nb_samples=3,  # nb of collected samples in MCMC per chain, after burn-in
    mcmc_proposal_var_scaling_factor=0.2,  # the variance of the multivariate normal distribution that the next eta from the Markov Chain is sampled from is scaling_factor * omega
    nb_phase1_iterations=20,
    nb_phase2_iterations=10,  # phase 1 is the exploration phase, the learning_rate is nill and there is simulated annealing for omega and the residual error
    convergence_threshold=1e-4,
    patience=5,
    learning_rate_power=0.8,
    annealing_factor=0.98,
    verbose=False,
)

(
    estimated_MI_mus,
    estimated_betas,
    estimated_omega,
    estimated_var_res,
    history,
) = saem.run()

# here we can compare with the true values we used to simulate the data
# print(f"True population MI: k_a: {true_k_a:.4f}")
# print(
#     f"True population mus: k12: {true_k12:.4f}, k21: {true_k21:.4f}, k_el: {true_k_el:.4f}"
# )
# print(
#     f"True population betas: {", ".join([f"{true_beta.item():.4f}" for true_beta in true_betas])}"
# )

# print(
#     f"True omega (diagonal): {", ".join([f"{val.item():.4f}" for val in torch.diag(true_omega)])}"
# )

# print(
#     f"True residual var: {", ".join([f"{val.item():.4f}" for val in true_residual_var.flatten()])}"
# )

# %%
saem.history

# %%
import matplotlib.pyplot as plt
from math import exp

for i in range(0, 5):
    beta_history = [exp(h[i].item()) for h in history["population_betas"]]
    plt.plot(
        beta_history,
        label=f"Mutation on tumor {i+1}",
    )

plt.legend()

# %%
saem.plot_convergence_history()
