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
from PySAEM import IndividualData, NLMEModel


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
def generate_data(
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

    # get individual true parameters for each individual, via the true MIs, mus and individual effects

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

    # call the structural model once with all the individual parameters to optimize
    list_true_concentrations: List[torch.Tensor] = pk_model.structural_model(
        list_time_steps, list_individual_params
    )

    # reditribute the results of the structural model on the individuals, adding noise in the process
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
