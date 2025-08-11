# %% [markdown]
# Let us use PySAEM on a PK two compartments model.
#
# We assume individual parameters theta_i = cat (MI, PDU = mu_pop * exp(eta*i) * exp(coeffs_pop * covariates_i))
# MI is a Model Intrinsic parameter with no InterIndividual Variance (as opposed to Patient Descriptors Unknown).
# eta_i is an individual effect, we assume eta_i follows a normal distribution N(0,omega)
# We assume a normal residual error between the predictions and the observations : Yij = f(theta_ij) + epsilon_ij.
# epsilon_ij following N(0,sigma²)
# We want to estimate mu_pop (here k_12, k_21, k_el), coeffs_pop (here coeff_MEMBRANE_12_k12 and coeff_MEMBRANE_12_k21)  and omega. We also want to estimate sigma².
# We set betas_pop as the array [log(mu1), coeff_cov1_having_effect_on_mu1, coeff_cov2_having_effect_on_mu1, ... log(mu2), ...], here [log(k_12), coeff_MEMBRANE_12_k12, log(k_21), coeff_MEMBRANE_12_k21, log(k_el)].
#


# %%
# import simulated data, PySAEM and the PK two compartments model
import os, sys
import torch
from math import log
from scipy.integrate import solve_ivp
from multiprocessing import Pool

current_directory = os.getcwd()
print(current_directory)
module_directory = current_directory + "/vpop-calibration/"
print(module_directory)
# Add the current directory to sys.path if it's not already there
if module_directory not in sys.path:
    sys.path.append(module_directory)

from PySAEM import *
from data_pysaem_example_usage import *
from pk_two_compartments_equations_with_absorption import pk_two_compartments_model

torch.set_default_dtype(torch.float32)

# %%
# let us define the arguments asked by PySAEM, corresponding to our model
MI_names: List[str] = ["k_a"]
PDU_names: List[str] = ["k12", "k21", "k_el"]
cov_coeffs_names: List[str] = ["coeff_MEMBRANE_12_k12", "coeff_MEMBRANE_12_k21"]
outputs_names: List[str] = ["A1", "A2"]
error_model_type: str = "additive"
covariate_map: Dict[str, List[str]] = {
    "k12": ["MEMBRANE_12"],
    "k21": ["MEMBRANE_12"],
    "k_el": [],
}  # list for each PDU, which covariate influences it

# %%
# We need to pass an NLMEModel (which translates our assumptions upon the biological phenomenae at play) to PySAEM
# For this, we need a structural_model, which corresponds to the underlying equations we assume
# it respect the signature
# structural_model(time_steps_list: Union[torch.Tensor, List[torch.Tensor]],initial_conditions: torch.Tensor,list_params: List[torch.Tensor] -> List[torch.Tensor]


# Here we use an ODE solver but it could be anything, including Jinko solving
def structural_model_brick(
    time_steps: np.ndarray,
    initial_conditions: np.ndarray,
    params: np.ndarray,
) -> torch.Tensor:
    """
    PK two compartments model.
    Takes time, initial conditions and parameters
    Returns the timeseries of the outputs according to the equations of pk_two_compartments_model: torch.Tensor[nb_outputs x nb_time_points].
    """
    # convert params to a tuple of parameters to pass it to solve_ivp
    # compute time_span, required by solve_ivp
    time_span = (time_steps[0], time_steps[-1])
    sol = solve_ivp(
        pk_two_compartments_model,
        time_span,
        initial_conditions,
        method="LSODA",
        t_eval=time_steps,
        rtol=1e-6,
        atol=1e-6,
        args=params,
    )
    if not sol.success:
        raise ValueError(f"ODE integration failed: {sol.message}")
    return torch.Tensor(sol.y[1:])


# here to parallelize calls to the ODE solver, we use starmap from Python. But it could be a nicer solution, for example the Jinko parallelization when working with jinko
def structural_model(
    list_time_steps: List[torch.Tensor],
    list_params: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Calls structural_model_brick for each pair of (time_steps, params).
    Returns a list of the solutions (List[torch.Tensors [nb_outputs x time_steps]]), i.e. the predicted outputs corresponding to the given time steps, initial conditions and parameters, according to the equations modeling the phenomena.
    """
    # extend time_steps_list if there is only one time_steps by repeating it
    if len(list_time_steps) == 1:
        list_time_steps = list_time_steps * len(list_params)

    list_time_steps_np = [ts.detach().cpu().numpy() for ts in list_time_steps]
    list_params_np = [p.detach().cpu().numpy() for p in list_params]
    initial_conditions_np = np.array([10.0, 4.0, 0.0])

    pool = Pool()
    starmap_args: List[
        Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []
    for i in range(len(list_params)):
        starmap_args.append(
            (list_time_steps_np[i], initial_conditions_np, list_params_np[i])
        )

    list_sol: List[torch.Tensor] = pool.starmap(structural_model_brick, starmap_args)
    pool.close()
    pool.join()
    return list_sol


# %%
# now we can create our NLMEModel (data structure defined in PysAEM) to pass to PySAEM
pk_model = NLMEModel_from_struct_model(
    MI_names=MI_names,
    PDU_names=PDU_names,
    cov_coeffs_names=cov_coeffs_names,
    outputs_names=outputs_names,
    structural_model=structural_model,
    error_model_type=error_model_type,
    covariate_map=covariate_map,
)

# %%
# we have to pass timeseries of the outputs for each individual to PySAEM, under the data structure List[IndividualData]
# here we simulate the data, with the method generate_data_for_pySAEM from data_pysaem_example_usage
# in this method, an ODE solver is used on lists of parameters centered around the same parameters the GP was trained on, with simulated etas and covariate effects.
# the true parameters will of course be hidden to pySAEM which will try to infer them
# normally, all_individual_data would be constructed from experimental observations

# chose arbitrary theoretical values
nb_individuals: int = 20
# time
time_span: Tuple[int, int] = (0, 24)
nb_steps: int = 20
time_steps: torch.Tensor = torch.linspace(time_span[0], time_span[1], nb_steps)
list_time_steps: List[torch.Tensor] = [time_steps] * nb_individuals
# true pop parameters
V1: float = 15.0  # volume of compartment 1
V2: float = 50.0
Q: float = 10.0  # intercompartmental clearance
true_k_a: float = 0.1  # absorption rate for both compartments
true_k_el: float = 0.15  # elimination rate of compartment 1
true_k12: float = Q / V1
true_k21: float = Q / V2
true_coeff_MEMBRANE12_k12: float = -0.4
true_coeff_MEMBRANE12_k21: float = -0.3
true_MI: torch.Tensor = torch.Tensor([true_k_a]).unsqueeze(-1)
true_betas: torch.Tensor = torch.Tensor(
    [
        log(true_k12),
        true_coeff_MEMBRANE12_k12,
        log(true_k21),
        true_coeff_MEMBRANE12_k21,
        log(true_k_el),
    ]
)
true_omega: torch.Tensor = torch.tensor(
    [[0.1**2, 0.0, 0.0], [0.0, 0.1**2, 0.0], [0.0, 0.0, 0.2**2]]
)  # Variance of eta_k12, eta_k21, eta_k_el
true_residual_var: torch.Tensor = torch.Tensor([0.1, 0.1]).unsqueeze(
    -1
)  # residual error variance
true_MEMBRANE_12: int = 0  # covariate
true_MEMBRANE_12_var: int = 1

# generate the data
all_individual_data: List[IndividualData] = generate_data_for_pySAEM(
    nb_individuals,
    true_MI,
    true_betas,
    true_omega,
    true_residual_var,
    true_MEMBRANE_12,
    true_MEMBRANE_12_var,
    list_time_steps,
    pk_model,
)

# %%
# PySAEM (as any SAEM algorithm) requires initial guesses of the population parameters
# here we can start close to our true parameters used for simulation
# normally the biomodeler has to put in some initial guess
initial_pop_MI: torch.Tensor = torch.Tensor([0.15]).unsqueeze(-1)
initial_pop_betas = torch.Tensor(
    [
        log(0.9),
        -0.15,
        log(0.1),
        -0.08,
        log(0.33),
    ]
).unsqueeze(-1)
initial_pop_omega = torch.diag(torch.Tensor([0.2**2, 0.2**2, 0.2**2]))
initial_res_var = torch.Tensor([0.4, 0.8]).unsqueeze(-1)  # residual error variance

# %%
#  run PySAEM
saem = PySAEM(
    model=pk_model,
    list_individual_data=all_individual_data,
    initial_pop_MI=initial_pop_MI,
    initial_pop_betas=initial_pop_betas,
    initial_pop_omega=initial_pop_omega,
    initial_res_var=initial_res_var,
    mcmc_first_burn_in=30,  # used at the first iteration, bigger because the initial etas start at zeros (afterwards they start at the last etas)
    mcmc_burn_in=2,  # used for the rest of the iterations
    mcmc_nb_samples=5,  # nb of collected samples in MCMC per chain, after burn-in
    mcmc_proposal_var_scaling_factor=0.3,  # the variance of the multivariate normal distribution that the next eta from the Markov Chain is sampled from is scaling_factor * omega
    nb_saem_iterations=10,
    saem_phase1_iterations=5,  # phase 1 is the exploration phase, the learning_rate is nill and there is simulated annealing for omega and the residual error
    saem_annealing_factor=0.95,
    verbose=True,
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
