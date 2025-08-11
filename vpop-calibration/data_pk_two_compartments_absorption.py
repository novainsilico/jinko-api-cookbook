# %% [markdown]
# This notenook generates data, solving pk_two_compartments_equations for outputs A1 and A2 from sets of parameters (k_a, k12, k21, k_el) in a Sobol sequence.
# The final data is a tensor with columns k_a k12 k21 k_el t A1 A2. It is used in example usage notebooks for GP and pySAEM.

# %%
import torch
from scipy.integrate import solve_ivp
from scipy.stats.qmc import Sobol
from pk_two_compartments_equations_with_absorption import pk_two_compartments_model
from typing import List, Tuple

# %%
torch.set_default_dtype(torch.float32)


# %%
def get_data():
    nb_outputs: int = 2  # A1,A2
    nb_parameters: int = 4 + 1  # k_a, k12, k21, k_el and time
    param_names: List[str] = ["k_a", "k12", "k21", "k_el"]
    output_names: List[str] = ["A1", "A2"]
    V1: float = 15.0  # volume of compartment 1
    V2: float = 50.0
    Q: float = 10.0  # intercompartmental clearance
    k_a: float = 0.1
    k_el: float = 0.15  # elimination rate of compartment 1
    k12: float = Q / V1
    k21: float = Q / V2
    # initial conditions
    A0_initial: float = 10
    A1_initial: float = 4
    A2_initial: float = 0
    initial_conditions: torch.Tensor = torch.Tensor(
        [A0_initial, A1_initial, A2_initial]
    )
    # create parameters ks sets with a Sobol sequence
    nb_ks_samples: int = 128  # power of two so that sobol sequences are more stable
    bounds_low: torch.Tensor = torch.tensor(
        [k_a * 0.5, k12 * 0.5, k21 * 0.5, k_el * 0.5]
    )
    bounds_high: torch.Tensor = torch.tensor([k_a * 2, k12 * 2, k21 * 2, k_el * 2])
    sobol_engine = Sobol(d=4, scramble=True)
    sobol_sequence = sobol_engine.random(nb_ks_samples)
    ks_tensor: torch.Tensor = (
        torch.from_numpy(sobol_sequence) * (bounds_high - bounds_low) + bounds_low
    )  # each line of the ks_tensor is a set of values for k_a, k12, k21, k_el
    # time
    nb_steps: int = (
        24  # here all time_steps are of the same length but it could be different
    )
    time_span: Tuple[float, float] = (0.0, 24.0)
    time_steps: torch.Tensor = torch.linspace(time_span[0], time_span[1], nb_steps)
    list_time_steps: List[torch.Tensor] = [time_steps] * nb_ks_samples
    # noise
    observational_noise_sigma: torch.Tensor = torch.Tensor(
        (0.05, 0.1)
    )  # one error variance for each output(A1, A2)

    data = _generate_data(
        ks_tensor,
        list_time_steps,
        initial_conditions,
        observational_noise_sigma,
        nb_parameters,
        nb_outputs,
    )
    return data, nb_parameters, nb_outputs, time_steps, param_names, output_names


# %%
def _generate_data(
    ks_tensor: torch.Tensor,
    list_time_steps: List[torch.Tensor],
    initial_conditions: torch.Tensor,
    observational_noise_sigma: torch.Tensor,
    nb_parameters: int,
    nb_outputs: int,
):

    # solve numerically for each set of ks and store the solution in the final tensor data
    sol_list = []
    for i in range(0, ks_tensor.shape[0]):
        time_steps = list_time_steps[i]
        time_span = (time_steps[0], time_steps[-1])
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
    )  # data is a tensor with columns k_a k12 k21 k_el t A1 A2
    data = data.to(torch.float32)

    # Add noise to the data
    data[:, nb_parameters:] += torch.normal(
        torch.zeros(data.shape[0], nb_outputs), observational_noise_sigma
    )
    return data
