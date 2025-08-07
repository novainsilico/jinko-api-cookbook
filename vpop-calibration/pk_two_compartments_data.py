# %% [markdown]
# This notenook generates data, solving pk_two_compartments_equations for outputs A1 and A2 from sets of parameters (k12,k21,k_el) following a normal distribution.
# The final data is a tensor with columns k12 k21 k_el t A1 A2. It is used in the example_usage_GP notebook.

# %%
import torch
from scipy.integrate import solve_ivp
from pk_two_compartments_equations import pk_two_compartments_model

# %%
torch.set_default_dtype(torch.float32)


# %%
def get_data():
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
    observational_noise_sigma = 0.05
    data = generate_data(
        ks_tensor,
        time_span,
        time_steps,
        initial_conditions,
        observational_noise_sigma,
        nb_parameters,
        nb_outputs,
    )
    return data, nb_parameters, nb_outputs, data, time_steps, param_names, output_names


# %%
def generate_data(
    ks_tensor,
    time_span,
    time_steps,
    initial_conditions,
    observational_noise_sigma,
    nb_parameters,
    nb_outputs,
):

    # solve numerically for each set of ks and store the solution in the final df data
    sol_list = []
    for i in range(0, ks_tensor.shape[0]):
        sol = solve_ivp(
            pk_two_compartments_model,
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
        outputs = torch.stack(
            [torch.from_numpy(sol.y[0]), torch.from_numpy(sol.y[1])], -1
        )
        solution = torch.cat((ks_expanded, time_steps_expanded, outputs), dim=1)
        sol_list.append(solution)
    data = torch.cat(
        sol_list, dim=0
    )  # data is a tensor with columns k12 k21 k_el t A1 A2
    data = data.to(torch.float32)

    # Add noise to the data
    data[:, nb_parameters:] += (
        torch.randn(data.shape[0], nb_outputs) * observational_noise_sigma
    )
    return data
