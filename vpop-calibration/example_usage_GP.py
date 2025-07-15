# %%
from scipy.integrate import solve_ivp
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
from pk_two_compartments_model import pk_two_compartments_model_equations

# %%
torch.set_default_dtype(torch.float32)

# %%
# STEP 1: set-up the pk_two_compartments_model
nb_outputs = 2  # A1,A2
nb_parameters = 3 + 1  # k12, k21, k_el and time
param_names = ["k12", "k21", "k_el"]
output_names = ["A1", "A2"]
V1 = 15.0  # volume of compartment 1
V2 = 50.0
Q = 10.0  # intercompartmental clearance
k_el = 0.15  # elimination rate of compartment 1
time_span = (0, 24)
k12 = Q / V1
k21 = Q / V2
# initial conditions
A1_initial = 4
A2_initial = 0
initial_conditions = [A1_initial, A2_initial]
# time
nb_steps = 30
time_steps = torch.linspace(time_span[0], time_span[1], nb_steps)

# %%
# STEP 2: create a dataframe with our real data (that will be split into a training and validating set)
# the dataframes must have columns param1, param2, ... paramTime, output1, output2, ... each line will correspond to a set of parameters and its outputs

# create a multivariate normal distribution to sample different sets of ks
ks_means = torch.tensor([k12, k21, k_el])
ks_variances = 0.1 * ks_means
ks_covariance_matrix = torch.diag(ks_variances)
ks_distribution = torch.distributions.MultivariateNormal(
    ks_means.float(), covariance_matrix=ks_covariance_matrix
)
# create a tensor with different values of the k parameters, sampling them from the distribution
nb_ks_samples = 30
ks_tensor = abs(
    ks_distribution.sample(sample_shape=torch.Size([nb_ks_samples]))
)  # rate constants cannot be negative
# each line of the ks_tensor is a set of values for k12, k21, k_el

# solve numerically for each set of ks and store the solution in the final df data
sol_list = []
for i in range(0, ks_tensor.shape[0]):
    sol = solve_ivp(
        pk_two_compartments_model_equations,
        time_span,
        initial_conditions,
        method="BDF",
        t_eval=time_steps,
        rtol=1e-7,
        atol=1e-7,
        args=(ks_tensor[i, :]),
    )
    ks_expanded = ks_tensor[i].expand(time_steps.shape[0], ks_tensor[i].shape[0])
    time_steps_expanded = time_steps.unsqueeze(1)
    outputs = torch.stack([torch.from_numpy(sol.y[0]), torch.from_numpy(sol.y[1])], -1)
    solution = torch.cat((ks_expanded, time_steps_expanded, outputs), dim=1)
    sol_list.append(solution)
data = torch.cat(
    sol_list, dim=0
)  # data is a 5400 x 6 tensor, each line is a composed of k12 k21 k_el t A1 A2 in the case of the 2 compartments PK model
data = data.to(torch.float32)

# Add noise to the data
observational_noise_sigma = 0.05
data[:, nb_parameters:] += (
    torch.randn(data.shape[0], nb_outputs) * observational_noise_sigma
)
print(data.shape)

# %%
# initiate our GP class
myGP = GP(
    nb_parameters,
    nb_outputs,
    data,
    time_steps,
    strategy="IMV",  # either IMV (Independent Multitask Variational) or LMCV (Linear Model of Coregionalization Variational)
    data_already_normalized=False,  # default
    nb_inducing_points=200,
    nb_latents=None,  # not needed since we use IMV
    mll="ELBO",  # default, otherwise PLL
    nb_training_iter=1000,
    training_proportion=0.7,
    learning_rate=0.005,
)

# %%
myGP.set_up()

# %%
myGP.train(mini_batching=False, mini_batch_size=None)

# %%
myGP.eval(
    given_data=None, time_steps_eval=None
)  # this way the GP will use the rest of the first given data, sampled according to the training_proportion
myGP.plot_loss()

# %%
myGP.plot_solution(param_names, output_names, all_at_once=True)
