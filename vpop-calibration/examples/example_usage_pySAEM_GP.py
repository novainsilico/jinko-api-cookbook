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

# %% [markdown]
# Let us use PySAEM on a PK two compartments model.
#
# We assume individual parameters theta*i = cat (MI, PDU = mu_pop * exp(eta*i) * exp(coeffs*pop * covariates_i))
# MI is a Model Intrinsic parameter with no InterIndividual Variance (as opposed to Patient Descriptors Unknown).
# eta_i is an individual effect, we assume eta_i follows a normal distribution N(0,omega)
# We assume a normal residual error between the predictions and the observations : Yij = f(theta_ij) + epsilon_ij.
# epsilon_ij following N(0,sigma²)
# We want to estimate mu_pop (here k_12, k_21, k_el), coeffs_pop (here coeff_MEMBRANE_12_k12 and coeff_MEMBRANE_12_k21) and omega. We also want to estimate sigma².
# We set betas_pop as the array [log(mu1), coeff_cov1_having_effect_on_mu1, coeff_cov2_having_effect_on_mu1, ... log(mu2), ...], here [log(k_12), coeff_MEMBRANE_12_k12, log(k_21), coeff_MEMBRANE_12_k21, log(k_el)].
#

# %%
# import simulated data, PySAEM and the PK two compartments model
import torch
import pickle
import sys
import numpy as np
from IPython.display import display

sys.path.append("../")
from src.PySAEM import *
from pk_two_compartments_equations_with_absorption import *
from src.GP import *
from src.DataGenerator import *

# %load_ext autoreload
# %autoreload 2

torch.set_default_dtype(torch.float64)

# %%
# let us define the arguments asked by PySAEM, corresponding to our model
MI_names: List[str] = ["k_a"]
PDU_names: List[str] = ["k_12", "k_21", "k_el"]
cov_coeffs_names: List[str] = []
output_names: List[str] = ["A0", "A1", "A2"]
nb_outputs = len(output_names)
error_model_type: str = "additive"
covariate_map: Dict[str, List[str]] = {
    "k_12": [],
    "k_21": [],
    "k_el": [],
}  # list for each PDU, which covariate influences it

# %%
# we first need to create our trained GP to pass to our NLMEModel
# data would be our observations. here we simulate some data with our DataGenerator
# the data should be similar to what we are going to use our GP on. Here we are using Sobol sequences to explore the space around the true parameters used to simulate the data that the GP will be used on.

reuse_existing_pickled_GP = False  # pass it to true once you have trained the GP and saved it in a .pkl file (next cells). This way, you can reuse it!
save_GP_for_reuse = False  # pass it to False if you don't want a dowloaded pickel file of your serialized GP
pickled_GP_name = "my_gp_save.pkl"  # your trained GP will be saved under this name


# instantiate our DataGenerator, which we will use to generate data both to train our GP on and to execute SAEM on
param_names = MI_names + PDU_names
data_generator = DataGenerator(
    pk_two_compartments_model,
    output_names,
    param_names,
)

if reuse_existing_pickled_GP == False:  # then train GP
    nb_time_steps = 10
    tmax = 24.0
    time_steps = np.linspace(0.0, tmax, nb_time_steps)
    initial_conditions = np.array([10.0, 0.0, 0.0])
    observation_noise = np.array([0.1, 0.05, 0.03])
    log2_nb_patients = 8
    nb_patients = 2**log2_nb_patients
    param_ranges = {
        "k_a": {"low": -2, "high": 0, "log": True},
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

    # Transform the inputs for easier fitting
    dataset[param_names] = dataset[param_names].map(np.log10)
    descriptors = param_names + ["time"]
    outputs = output_names
    myGP = GP(
        dataset,
        descriptors,
        outputs,
        var_strat="IMV",  # either IMV (Independent Multitask Variational) or LMCV (Linear Model of Coregionalization Variational)
        kernel="RBF",  # Either RBF or SMK
        data_already_normalized=False,  # default
        nb_inducing_points=100,
        mll="ELBO",  # default, otherwise PLL
        nb_training_iter=10,
        training_proportion=0.7,
        learning_rate=0.01,
        num_mixtures=3,
        jitter=1e-4,
    )
    print("Training the GP")
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
pk_model_true = NLMEModel(
    MI_names=MI_names,
    PDU_names=PDU_names,
    cov_coeffs_names=cov_coeffs_names,
    outputs_names=output_names,
    structural_model=pk_two_compartments_model,
    error_model_type=error_model_type,
    covariate_map=covariate_map,
)

# %%
# Generating a synthetic real life data set

# we have to pass timeseries of the outputs for each individual to PySAEM, under the data structure List[IndividualData]
# in this method, an ODE solver is used on lists of parameters centered around the same parameters the GP was trained on, with simulated etas and covariate effects.
# the true parameters will of course be hidden to pySAEM which will try to infer them
# normally, all_individual_data would be constructed from experimental observations

# chose arbitrary theoretical values
nb_individuals: int = 50
# time
time_span: Tuple[int, int] = (0, 24)
nb_steps: int = 10

# For each output and for each patient, give a list of time steps to be simulated
time_steps_obs: torch.Tensor = torch.linspace(time_span[0], time_span[1], nb_steps)
list_intermediate = [time_steps_obs] * nb_outputs
list_time_steps: List[List[torch.Tensor]] = [list_intermediate] * nb_individuals

initial_conditions = np.array([10.0, 5.0, 0.0])
# true pop parameters
V1: float = 15.0  # volume of compartment 1
V2: float = 50.0
Q: float = 10.0  # intercompartmental clearance
true_k_a: float = 0.05  # absorption rate for both compartments
true_k_el: float = 0.15  # elimination rate of compartment 1
true_k12: float = Q / V1
true_k21: float = Q / V2
true_MI: torch.Tensor = torch.Tensor([true_k_a]).unsqueeze(-1)
true_coeff_MEMBRANE12_k12 = 0.5
true_betas: torch.Tensor = torch.Tensor(
    [
        np.log(true_k12),
        np.log(true_k21),
        np.log(true_k_el),
    ]
)
true_omega: torch.Tensor = torch.tensor(
    [[0.1**2, 0.0, 0.0], [0.0, 0.1**2, 0.0], [0.0, 0.0, 0.2**2]]
)  # Variance of eta_k12, eta_k21, eta_k_el
true_residual_var: torch.Tensor = torch.Tensor([0.01, 0.02, 0.03]).unsqueeze(
    -1
)  # residual error variance
list_covariates_dict = None  # otherwise of the form list_covariates_dict = [{"name":"_", "mean":_, "var": 0._},{"name":"_", "mean":_, "var": 0._}...]
# generate the data
observations_df, covariates_df = data_generator.simulate_dataset_from_omega(
    nb_individuals,
    true_MI,
    true_betas,
    true_omega,
    true_residual_var,
    list_covariates_dict,
    list_time_steps,
    pk_model_true,
    initial_conditions,
)

# %%
display(observations_df)

# %%
nlme_surrogate = NLMEModel(
    structural_model=myGP,
    MI_names=MI_names,
    PDU_names=PDU_names,
    cov_coeffs_names=cov_coeffs_names,
    outputs_names=output_names,
    GP_error_threshold=0.7,
    error_model_type=error_model_type,
    covariate_map=covariate_map,
)

# %%
# PySAEM (as any SAEM algorithm) requires initial guesses of the population parameters
# here we can start close to our true parameters used for simulation
# normally the biomodeler has to put in some initial guess
initial_pop_MI: torch.Tensor = torch.Tensor([0.04]).unsqueeze(-1)
initial_pop_betas = torch.log(
    torch.Tensor(
        [
            0.8,
            # if there were a covariate influencing the first PDU (of initial guess 0.8), here we would put the coeff of its influence on the PDU (etc. for other PDUs)
            0.18,
            0.20,
        ]
    )
).unsqueeze(-1)
initial_pop_omega = torch.diag(torch.Tensor([0.2**2, 0.2**2, 0.5**2]))
initial_res_var = torch.Tensor([0.04, 0.04, 0.04]).unsqueeze(
    -1
)  # residual error variance

# %%
#  run PySAEM
saem = PySAEM(
    model=nlme_surrogate,
    observations_df=observations_df,
    covariates_df=None,  # None if no covariates
    initial_pop_MI=initial_pop_MI,
    initial_pop_betas=initial_pop_betas,
    initial_pop_omega=initial_pop_omega,
    initial_res_var=initial_res_var,
    mcmc_first_burn_in=30,  # used at the first iteration, bigger because the initial etas start at zeros (afterwards they start at the last etas)
    mcmc_burn_in=5,  # used for the rest of the iterations
    mcmc_nb_samples=3,  # nb of collected samples in MCMC per chain, after burn-in
    mcmc_proposal_var_scaling_factor=0.2,  # the variance of the multivariate normal distribution that the next eta from the Markov Chain is sampled from is scaling_factor * omega
    nb_phase1_iterations=10,
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
print(f"True population MI: k_a: {true_k_a:.4f}")
print(
    f"True population mus: k12: {true_k12:.4f}, k21: {true_k21:.4f}, k_el: {true_k_el:.4f}"
)
print(
    f"True population betas: {", ".join([f"{true_beta.item():.4f}" for true_beta in true_betas])}"
)

print(
    f"True omega (diagonal): {", ".join([f"{val.item():.4f}" for val in torch.diag(true_omega)])}"
)

print(
    f"True residual var: {", ".join([f"{val.item():.4f}" for val in true_residual_var.flatten()])}"
)

# %%
saem.plot_convergence_history(true_MI, true_betas, true_omega, true_residual_var)


