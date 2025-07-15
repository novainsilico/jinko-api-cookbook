# %%
import numpy as np
import torch


# %%
def pk_two_compartments_model_equations(t, y, k12, k21, k_el):
    y = np.array(y)
    # differential equation
    ydot = [  # y[0] is A1, y[1] is A2
        k21 * y[1] - k12 * y[0] - k_el * y[0],
        k12 * y[0] - k21 * y[1],
    ]
    return ydot


# # %%
# nb_outputs = 2  # A1,A2

# %%
# initiate parameters
V1 = 15.0  # volume of compartment 1
V2 = 50.0
Q = 10.0  # intercompartmental clearance
k_el = 0.15  # elimination rate of compartment 1
time_span = (0, 24)
k12 = Q / V1
k21 = Q / V2
# initial conditions
A1_initial = 2
A2_initial = 0
initial_conditions = [A1_initial, A2_initial]
# time
nb_steps = 200
time_steps = torch.linspace(time_span[0], time_span[1], nb_steps)

# %%
# create a distribution for parameters' exploration
ks_means = torch.tensor(
    [k12, k21, k_el] + np.random.normal(0, 0.01, 3)
)  # means for the distributions are close to the real values
ks_variances = [0.1 * k12, 0.05 * k21, 0.08 * k_el]
ks_covariance_matrix = torch.diag(torch.tensor(ks_variances).float())
ks_distribution = torch.distributions.MultivariateNormal(
    ks_means.float(), covariance_matrix=ks_covariance_matrix
)  # problem: we need to ensure that this distribution has no negative values
