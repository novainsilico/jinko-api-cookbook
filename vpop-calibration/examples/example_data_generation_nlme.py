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
import torch
import pickle
import sys
import numpy as np

sys.path.append("../src")
from pk_two_compartments_equations_with_absorption import *
from DataGenerator import *
from PySAEM import *


torch.set_default_dtype(torch.float32)

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
# we create our NLMEModel (data structure defined in PysAEM) to pass to PySAEM
pk_model = NLMEModel_from_struct_model(
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
nb_individuals: int = 30
# time
time_span: Tuple[int, int] = (0, 24)
nb_steps: int = 20

# For each output and for each patient, give a list of time steps to be simulated
time_steps: torch.Tensor = torch.linspace(time_span[0], time_span[1], nb_steps)
list_intermediate = [time_steps] * nb_outputs
list_time_steps: List[List[torch.Tensor]] = [list_intermediate] * nb_individuals


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
    [[1.3**2, 0.0, 0.0], [0.0, 0.5**2, 0.0], [0.0, 0.0, 0.2**2]]
)  # Variance of eta_k12, eta_k21, eta_k_el
true_residual_var: torch.Tensor = torch.Tensor([0.01, 0.02, 0.01]).unsqueeze(
    -1
)  # residual error variance
list_covariates_dict = None  # otherwise of the form list_covariates_dict = [{"name":"_", "mean":_, "var": 0._},{"name":"_", "mean":_, "var": 0._}...]
# generate the data
initial_conditions = np.array([10.0, 5.0, 0.0])
# instantiate our DataGenerator, which we will use to generate data both to train our GP on and to execute SAEM on
param_names = MI_names + PDU_names
data_generator = DataGenerator(
    pk_two_compartments_model,
    output_names,
    param_names,
)

nlme_model = NLMEModel_from_struct_model(
    MI_names=MI_names,
    PDU_names=PDU_names,
    cov_coeffs_names=cov_coeffs_names,
    outputs_names=output_names,
    structural_model=pk_two_compartments_model,
    error_model_type=error_model_type,
    covariate_map=covariate_map,
)

observations_df, covariates_df = data_generator.simulate_dataset_from_omega(
    nb_individuals,
    true_MI,
    true_betas,
    true_omega,
    true_residual_var,
    list_covariates_dict,
    list_time_steps,
    pk_model,
    initial_conditions,
)

# %%
from plotnine import *


(
    ggplot(observations_df, aes(x="time", y="value", color="ind_id"))
    + geom_line()
    + facet_wrap("output_name", ncol=3)
    + scale_color_discrete(guide=None)
    + theme(figure_size=(16, 8))
)
