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
import os, sys
import torch
from scipy.integrate import solve_ivp
from multiprocessing import Pool
import numpy as np
from PySAEM import IndividualData, NLMEModel
from pk_two_compartments_equations_with_absorption import pk_two_compartments_model

# %%
current_directory = os.getcwd()
print(current_directory)
module_directory = current_directory + "/vpop-calibration/"
print(module_directory)
# Add the current directory to sys.path if it's not already there
if module_directory not in sys.path:
    sys.path.append(module_directory)

from PySAEM import *

torch.set_default_dtype(torch.float32)


# %%
# Prepare simulated data
# This data would normally be an experimental data set (converted to the data structure List[IndividualData] for PySAEM), from which we want to infer population parameters' distribution with PySAEM
# To simulate a dataset with individual effects and covariates, we will repurpose the methods of PySAEM (create_design_matrix, individual_parameters) with the true values we chose earlier and add noise
# first, let us create some structural model using an ODE solver to transform individual parameters and timesteps into outputs values
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
    if len(time_steps) == 1:
        # no integration needed, return the initial conditions
        return torch.Tensor(initial_conditions[1:].reshape(-1, 1))
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


def generate_data_for_pySAEM(
    nb_individuals: int,
    true_MI: torch.Tensor,
    true_betas: torch.Tensor,
    true_omega: torch.Tensor,
    true_residual_var: torch.Tensor,
    true_MEMBRANE_12: Union[int, float],
    true_MEMBRANE_12_var: Union[int, float],
    list_time_steps: List[torch.Tensor],
    pk_model: NLMEModel,
) -> List[IndividualData]:
    torch.manual_seed(32)
    list_individual_data = []
    list_individual_params = []

    # sample individual effects, eta_i ~ N(0, Omega)
    distrib_etas = torch.distributions.MultivariateNormal(
        loc=torch.zeros(true_omega.shape[0]), covariance_matrix=true_omega
    )
    etas = torch.Tensor(distrib_etas.sample((nb_individuals,)))
    # sample values for the covariate
    MEMBRANE_12: torch.Tensor = torch.normal(
        mean=torch.Tensor([true_MEMBRANE_12] * nb_individuals),
        std=torch.Tensor([true_MEMBRANE_12_var]),
    )

    # create the IndData data structure, with time, covariate and the design matrix for each individual
    # compute also individual parameters
    for i in range(nb_individuals):
        time_steps = list_time_steps[i]
        # create the data structure that PySAEM is expecting
        ind_data: IndividualData = IndividualData(
            id=f"ID_{i+1}",
            time=time_steps,
            observations=None,
            covariates={"MEMBRANE_12": MEMBRANE_12[i]},
        )
        # create a design matrix to compute individual parameters
        # The rows of X_i correspond to the MIs then PDUs.
        # The columns of X_i correspond to the elements of the beta vector for the multiplication.
        # Each row (corresponding to one PDU) contains 1 in the column corresponding to log(mu) in betas. For each covariate influencing the PDU, the value of this covariate for this individual in the column corresponding to the coeff of this covariate in betas. Else zeros.
        ind_data.create_design_matrix(pk_model)
        design_matrix_X_i: torch.Tensor = (
            ind_data.design_matrix_X_i
        )  # torch.Tensor [(nb_PDU) x nb_betas]

        # compute individual parameters, theta_i = cat (MI, PDU=mu_pop * exp(eta_i)* exp (coeffs*covariates_i))
        true_individual_params: torch.Tensor = torch.Tensor(
            pk_model.individual_parameters(
                true_MI, true_betas, etas[i].unsqueeze(0), design_matrix_X_i
            )[0]
        )  # [0] because individual_parameters returns a list of 1 tensor
        list_individual_params.append(true_individual_params)
        list_individual_data.append(ind_data)

    # call the structural model to get the predicted outputs (that will become our fake observations)
    list_true_concentrations: List[torch.Tensor] = structural_model(
        list_time_steps, list_individual_params
    )

    # redistribute the results of the structural model on the individuals, adding noise in the process
    for i in range(nb_individuals):
        # add noise
        noise: torch.Tensor = torch.normal(
            torch.zeros_like(list_true_concentrations[i]),
            (torch.sqrt(true_residual_var)).expand(
                -1, list_true_concentrations[i].shape[1]
            ),
        )
        observed_concentrations: torch.Tensor = list_true_concentrations[i] + noise
        list_individual_data[i].observations = observed_concentrations

    return list_individual_data
