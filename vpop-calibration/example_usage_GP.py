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

current_directory = os.getcwd()
print(current_directory)
module_directory = current_directory + "/vpop-calibration/"
print(module_directory)
# Add the current directory to sys.path if it's not already there
if module_directory not in sys.path:
    sys.path.append(module_directory)

from GP import *
from data_pk_two_compartments import *

# %%
torch.set_default_dtype(torch.float32)

# %%
# import the data from pk_two_compartments_data
# our data tensor must be a tensor with columns param1 param2 ... time output1 output2 ...
# in this two-compartments PK model, we have columns k12 k21 k_el t A1 A2
data, nb_parameters, nb_outputs, data, time_steps, param_names, output_names = (
    get_data()
)


# %%
# initiate our GP class
myGP = GP(
    nb_parameters,
    param_names,
    nb_outputs,
    output_names,
    data,
    var_strat="IMV",  # either IMV (Independent Multitask Variational) or LMCV (Linear Model of Coregionalization Variational)
    kernel="RBF",  # Either RBF or SMK
    data_already_normalized=False,  # default
    nb_inducing_points=100,
    mll="ELBO",  # default, otherwise PLL
    nb_training_iter=500,
    training_proportion=0.7,
    learning_rate=0.01,
    num_mixtures=3,
    jitter=1e-4,
)

# %%
myGP.train(mini_batching=False, mini_batch_size=None)

# %%
myGP.plot_loss()
myGP.eval_perf()

# %%
myGP.plot_obs_vs_predicted(data_set="training")
myGP.plot_obs_vs_predicted(data_set="validation")

# %%
j = torch.randint(30, (1,))[0]
myGP.plot_individual_solution(j)

# %%
myGP.plot_all_solutions("training")
myGP.plot_all_solutions("validation")
