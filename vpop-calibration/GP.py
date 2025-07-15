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
# Train a simple Gaussian Process on several parameters of an ODE, for several outputs (not yet) et variational (not yet)

# %%
# imports
from matplotlib import pyplot as plt
import numpy as np
import math
import torch
import gpytorch
from tqdm import tqdm
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from torch.utils.data import TensorDataset, DataLoader

# %%
torch.set_default_dtype(torch.float32)


# %%
class IMVModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, nb_outputs, ard_num_dims):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.shape[0],
            batch_shape=torch.Size([nb_outputs]),
            mean_init_std=6e-3,
        )
        variational_strategy = (
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=True,
                ),
                num_tasks=nb_outputs,
            )
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([nb_outputs])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=torch.Size([nb_outputs]), ard_num_dims=ard_num_dims
            ),
            batch_shape=torch.Size([nb_outputs]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# %%
class LMCVModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, nb_outputs, nb_latents, ard_num_dims):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.shape[0],
            batch_shape=torch.Size([nb_latents]),
            mean_init_std=6e-3,
        )

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=nb_outputs,
            num_latents=nb_latents,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([nb_latents])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=torch.Size([nb_latents]), ard_num_dims=ard_num_dims
            ),
            batch_shape=torch.Size([nb_latents]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# %%
class GP:
    def __init__(
        self,
        nb_parameters,
        nb_outputs,
        data,
        time_steps,
        strategy="IMV",  # either IMV (Independent Multitask Variational) or LMCV (Linear Model of Coregionalization Variational)
        data_already_normalized=False,
        nb_training_iter=400,
        training_proportion=0.7,
        nb_inducing_points=200,
        nb_latents=3,
        mll="ELBO",
        learning_rate=None,
    ):
        self.nb_parameters = nb_parameters
        self.nb_outputs = nb_outputs
        self.original_data = data
        self.data_already_normalized = data_already_normalized
        self.time_steps = time_steps
        self.strategy = strategy
        self.nb_training_iter = nb_training_iter
        self.training_proportion = training_proportion
        self.nb_inducing_points = nb_inducing_points
        self.nb_latents = nb_latents  # optional, only in case of LMCV
        self.learning_rate = learning_rate
        self.mll = mll
        self.train_indices = None
        self.eval_indices = None
        self.train_params = None
        self.train_outputs = None
        self.inducing_points = None
        self.time_steps_eval = None
        self.mini_batch_size = None
        self.likelihood = None
        self.model = None
        self.nb_sets_parameters = None
        self.nb_sets_parameters_training = None
        self.data_eval = None
        self.mean = None
        self.lower = None
        self.upper = None
        self.losses = None

        # Process and normalize the data
        self.input_data = self.original_data[:, : self.nb_parameters]
        if self.data_already_normalized == True:
            self.normalized_output_data = self.original_data[:, self.nb_parameters :]
        else:
            self.normalizing_output_mean = self.original_data[
                :, self.nb_parameters :
            ].mean(dim=0)
            self.normalizing_output_std = self.original_data[
                :, self.nb_parameters :
            ].std(dim=0)

            self.normalized_output_data = (
                self.original_data[:, self.nb_parameters :]
                - self.normalizing_output_mean
            ) / self.normalizing_output_std

    def set_up(self):
        # create training set
        self.nb_sets_parameters = self.input_data.shape[0] // len(self.time_steps)
        self.nb_sets_parameters_training = math.floor(
            self.training_proportion * self.nb_sets_parameters
        )

        # detect if all sets are used but it was not intended (no 1 training proportion)
        if self.training_proportion != 1:
            if self.nb_sets_parameters_training == self.nb_sets_parameters:
                raise ValueError(
                    "Training proportion too high for the number of sets of parameters: all would be used for training. Set training_proportion as 1 if that is your intention."
                )

            mixed_indices_sets_parameters = torch.randperm(self.nb_sets_parameters)[
                :
            ]  # list of randomly mixed indices
            train_indices_sets_parameters = mixed_indices_sets_parameters[
                0 : self.nb_sets_parameters_training
            ]
            train_indices_list = []
            for ind in train_indices_sets_parameters:
                start = ind * len(self.time_steps)
                end = (ind + 1) * len(self.time_steps)
                for i in range(start, end):
                    train_indices_list.append(i)
            self.train_indices = torch.tensor(train_indices_list)
            # Determine evaluation indices
            eval_indices_sets_parameters = mixed_indices_sets_parameters[
                self.nb_sets_parameters_training :
            ]
            eval_indices_list = []
            for ind in eval_indices_sets_parameters:
                start = ind * len(self.time_steps)
                end = (ind + 1) * len(self.time_steps)
                for i in range(start, end):
                    eval_indices_list.append(i)
            self.eval_indices = torch.tensor(eval_indices_list)

        else:
            self.train_indices = torch.arange(
                0, self.nb_sets_parameters * len(self.time_steps)
            )

        self.train_params = torch.Tensor(
            self.input_data[self.train_indices, :]
        )  # select nb_sets_training lines randomly and the columns corresponding to the parameters

        self.train_outputs = torch.Tensor(
            self.normalized_output_data[self.train_indices, :]
        )  # select the lines according to sample_indices and the columns corresponding to the outputs

        # create inducing points
        self.inducing_points = self.train_params[
            torch.randperm(self.train_params.shape[0])[: self.nb_inducing_points],
            :,
        ].float()

        # initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.nb_outputs, has_global_noise=True, has_task_noise=False
        )
        if self.strategy == "IMV":
            self.model = IMVModel(
                self.inducing_points, self.nb_outputs, self.nb_parameters
            )  # Added ard_num_dims
        elif self.strategy == "LMCV":
            self.model = LMCVModel(
                self.inducing_points,
                self.nb_outputs,
                self.nb_latents,
                self.nb_parameters,
            )  # Added ard_num_dims

    def train(self, mini_batching=False, mini_batch_size=None):

        # Errors for missing parameters
        if self.train_params is None:
            raise ValueError(
                "Please set up the GP before training it using the set_up() method. train_params is None"
            )
        elif self.train_outputs is None:
            raise ValueError(
                "Please set up the GP before training it using the set_up() method. train_outputs is None"
            )
        elif self.inducing_points is None:
            raise ValueError(
                "Please set up the GP before training it using the set_up() method. inducing_points is None"
            )
        elif self.likelihood is None:
            raise ValueError(
                "Please set up the GP before training it using the set_up() method. likelihood is None"
            )
        elif self.model is None:
            raise ValueError(
                "Please set up the GP before training it using the set_up() method. model is None"
            )

        # set model and likelihood in training mode
        self.model.train()
        self.likelihood.train()

        # initialize the adam optimizer
        params_to_optim = [
            {"params": self.model.parameters()},
            {"params": self.likelihood.parameters()},
        ]
        if self.learning_rate is None:
            optimizer = torch.optim.Adam(params_to_optim)
        else:
            optimizer = torch.optim.Adam(params_to_optim, lr=self.learning_rate)

        # set the marginal log likelihood
        if self.mll == "ELBO":
            mll = VariationalELBO(
                self.likelihood, self.model, num_data=self.train_outputs.size(0)
            )
        elif self.mll == "PLL":
            mll = PredictiveLogLikelihood(
                self.likelihood, self.model, num_data=self.train_outputs.size(0)
            )
        else:
            raise ValueError("Invalid mll. Chose ELBO or PLL.")

        # keeping track of the loss
        losses_list = []
        epochs = tqdm(range(self.nb_training_iter))
        if mini_batching == True:
            # set the mini_batch_size to a power of two of the total size -4
            if mini_batch_size == None:
                power = (
                    math.floor(
                        math.log2(
                            self.nb_sets_parameters_training * len(self.time_steps)
                        )
                    )
                    - 4
                )
                mini_batch_size = 2**power
            self.mini_batch_size = mini_batch_size

            # prepare mini-batching
            train_dataset = TensorDataset(self.train_params, self.train_outputs)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.mini_batch_size,
                shuffle=True,
            )

            # main training loop
            for i in epochs:
                for batch_params, batch_outputs in train_loader:
                    optimizer.zero_grad()  # zero gradients from previous iteration
                    output = self.model(batch_params)  # recalculate the prediction
                    loss = -mll(output, batch_outputs)
                    loss.backward()  # compute the gradients of the parameters that can be changed
                    losses_list.append(loss.item())
                    optimizer.step()
                    epochs.set_postfix({"loss": loss.item()})
        else:
            for i in epochs:  # tqdm prints a progress bar
                optimizer.zero_grad()  # zero gradients from previous iteration
                output = self.model(
                    self.train_params
                )  # calculate the prediction with current parameters
                loss = -mll(output, self.train_outputs)
                loss.backward()  # compute the gradients of the parameters that can be changed
                losses_list.append(loss.item())
                optimizer.step()
                epochs.set_postfix({"loss": loss.item()})
        self.losses = torch.tensor(losses_list)

    def eval(self, given_data=None, time_steps_eval=None):
        if given_data is None:
            if self.training_proportion == 1:
                raise ValueError(
                    "The training proportion is 1 and you did not provide new data to evaluate the GP on."
                )
            self.data_eval = self.original_data[self.eval_indices, :]
            self.time_steps_eval = self.time_steps
            if self.time_steps_eval is None:
                raise ValueError(
                    "No time_steps_eval provided for the new data to be evaluated."
                )
        else:
            self.data_eval = given_data
            self.eval_indices = torch.arange(0, len(given_data))
            self.time_steps_eval = time_steps_eval

        # set model and likelihood in evalutation mode
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            predictions = self.likelihood(
                self.model(self.data_eval[:, 0 : self.nb_parameters].float())
            )
        self.mean = predictions.mean
        self.lower = predictions.confidence_region()[0]
        self.upper = predictions.confidence_region()[1]

        if self.data_already_normalized == False:

            def unnormalize(vec):
                return vec * self.normalizing_output_std + self.normalizing_output_mean

            self.mean = unnormalize(self.mean)
            self.lower = unnormalize(self.lower)
            self.upper = unnormalize(self.upper)

    # plot function
    def plot_solution(self, param_names, output_names, all_at_once=True):

        nb_time_steps = len(self.time_steps_eval)
        nb_sets_parameters_eval = math.floor(len(self.eval_indices) / nb_time_steps)
        if all_at_once == False:
            j = torch.randint(
                0, nb_sets_parameters_eval, (1,)
            )  # random set of parameters for which the timeseries will be plotted
            start = j * nb_time_steps
            stop = (j + 1) * nb_time_steps
            fig, axes = plt.subplots(
                1, self.nb_outputs, figsize=(9.0 * self.nb_outputs, 4.0)
            )
            if (
                self.nb_outputs == 1
            ):  # handle single output case where axes is not an array
                axes = [axes]

            for output_index in range(self.nb_outputs):  # time is not included
                output_data_index = output_index + self.nb_parameters
                ax = axes[output_index]
                ax.set_xlabel("Time")
                ax.plot(
                    self.time_steps_eval,
                    self.data_eval[start:stop, output_data_index],
                    "-",
                    color="C0",
                    linewidth=2,
                    alpha=0.6,
                    label=output_names[output_index],
                )  # true values

                # Plot GP prediction
                ax.plot(
                    self.time_steps_eval,
                    self.lower[start:stop, output_index],
                    "-",
                    color="C3",
                    linewidth=2,
                    alpha=0.5,
                    label="GP prediction for "
                    + output_names[output_index]
                    + " (lower)",
                )
                ax.plot(
                    self.time_steps_eval,
                    self.upper[start:stop, output_index],
                    "-",
                    color="C3",
                    linewidth=2,
                    alpha=0.5,
                    label="GP prediction for "
                    + output_names[output_index]
                    + " (upper)",
                )
                ax.plot(
                    self.time_steps_eval,
                    self.mean[start:stop, output_index],
                    "-",
                    color="C3",
                    linewidth=2,
                    alpha=0.5,
                    label="GP prediction for " + output_names[output_index] + " (mean)",
                )
                ax.legend(loc="upper right")
                title = f"{output_names[output_index]} for a random set of parameters"
                ax.set_title(title)

                # text box with the k values
                current_param_values = (
                    self.data_eval[start, 0 : self.nb_parameters - 1].flatten().tolist()
                )
                param_text = "Parameters:\n"
                for i, name in enumerate(param_names):
                    param_text += f"  {name}: {current_param_values[i]}\n"  # Format to 4 decimal places

                ax.text(
                    1.02,
                    0.98,
                    param_text,
                    transform=ax.transAxes,  # Coordinate system is relative to the axis
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5, ec="k"),
                )

        else:  # plot for all sets of parameters
            n_cols = self.nb_outputs
            n_rows = nb_sets_parameters_eval
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = axes.flatten()

            plot_index = 0
            for j in range(0, nb_sets_parameters_eval):
                start = j * nb_time_steps
                stop = (j + 1) * nb_time_steps

                for output_index in range(self.nb_outputs):
                    output_data_index = output_index + self.nb_parameters
                    ax = axes[plot_index]
                    ax.set_xlabel("Time")
                    ax.plot(
                        self.time_steps_eval,
                        self.data_eval[start:stop, output_data_index],
                        "-",
                        color="C0",
                        linewidth=2,
                        alpha=0.6,
                        label=output_names[output_index],
                    )

                    # Plot GP prediction
                    ax.plot(
                        self.time_steps_eval,
                        self.lower[start:stop, output_index],
                        "-",
                        color="C3",
                        linewidth=2,
                        alpha=0.5,
                        label="GP prediction for "
                        + output_names[output_index]
                        + " (lower)",
                    )
                    ax.plot(
                        self.time_steps_eval,
                        self.upper[start:stop, output_index],
                        "-",
                        color="C3",
                        linewidth=2,
                        alpha=0.5,
                        label="GP prediction for "
                        + output_names[output_index]
                        + " (upper)",
                    )
                    ax.plot(
                        self.time_steps_eval,
                        self.mean[start:stop, output_index],
                        "-",
                        color="C3",
                        linewidth=2,
                        alpha=0.5,
                        label="GP prediction for "
                        + output_names[output_index]
                        + " (mean)",
                    )
                    ax.legend(loc="upper right")
                    title = f"{output_names[output_index]} for parameters set {j+1}"  # More descriptive title
                    ax.set_title(title)

                    # text box with the k values
                    current_param_values = (
                        self.data_eval[start, 0 : self.nb_parameters].flatten().tolist()
                    )
                    param_text = "Parameters:\n"
                    for i, name in enumerate(param_names):
                        param_text += f"  {name}: {current_param_values[i]:.4f}\n"

                    ax.text(
                        1.02,
                        0.98,
                        param_text,
                        transform=ax.transAxes,
                        fontsize=9,
                        verticalalignment="top",
                        bbox=dict(
                            boxstyle="round,pad=0.5", fc="wheat", alpha=0.5, ec="k"
                        ),
                    )
                    plot_index += 1

        plt.tight_layout()
        plt.show()

    def plot_loss(self):
        # plot the loss over iterations
        iterations = torch.linspace(1, self.nb_training_iter, self.nb_training_iter)

        plt.plot(iterations, self.losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss over Iterations")
        plt.show()
