from matplotlib import pyplot as plt
import math
import torch
import gpytorch
from tqdm import tqdm
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from torch.utils.data import TensorDataset, DataLoader

torch.set_default_dtype(torch.float32)
gpytorch.settings.cholesky_jitter(1e-6)


class SVGP(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points,
        nb_params,
        nb_outputs,
        var_dist="Chol",
        var_strat="IMV",
        kernel="RBF",
        jitter=1e-6,
        nb_mixtures=4,  # only for the SMK kernel
    ):
        if var_dist == "Chol":
            variational_distribution = (
                gpytorch.variational.CholeskyVariationalDistribution(
                    inducing_points.shape[0],
                    batch_shape=torch.Size([nb_outputs]),
                    mean_init_std=6e-3,
                )
            )
        else:
            raise ValueError(f"Unsupported variational distribution: {var_dist}")

        if var_strat == "IMV":
            variational_strategy = (
                gpytorch.variational.IndependentMultitaskVariationalStrategy(
                    gpytorch.variational.VariationalStrategy(
                        self,
                        inducing_points,
                        variational_distribution,
                        learn_inducing_locations=True,
                        jitter_val=jitter,
                    ),
                    num_tasks=nb_outputs,
                )
            )
        elif var_strat == "LMCV":
            variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=True,
                    jitter_val=jitter,
                ),
                num_tasks=nb_outputs,
                num_latents=nb_outputs,
                latent_dim=-1,
            )
        else:
            raise ValueError(f"Unsupported variational strategy {var_strat}")

        super().__init__(variational_strategy)

        # Todo : allow for different mean choices
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([nb_outputs])
        )

        if kernel == "RBF":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    batch_shape=torch.Size([nb_outputs]),
                    ard_num_dims=nb_params,
                    jitter=jitter,
                ),
                batch_shape=torch.Size([nb_outputs]),
            )
        elif kernel == "SMK":
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
                batch_size=nb_outputs,
                num_mixtures=nb_mixtures,
                ard_num_dims=nb_params,
                jitter=jitter,
            )
        else:
            raise ValueError(f"Unsupported kernel {kernel}")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP:
    def __init__(
        self,
        nb_parameters,
        param_names,
        nb_outputs,
        output_names,
        data,
        var_dist="Chol",  # only Cholesky currently supported
        var_strat="IMV",  # either IMV (Independent Multitask Variational) or LMCV (Linear Model of Coregionalization Variational)
        kernel="RBF",  # RBF or SMK
        data_already_normalized=False,
        nb_training_iter=400,
        training_proportion=0.7,
        nb_inducing_points=200,
        nb_latents=None,  # optional: by default we will use nb_latents = nb_outputs
        mll="ELBO",  # ELBO or PLL
        learning_rate=None,  # optional
        num_mixtures=4,  # optional: only for the SMK kernel
        jitter=1e-6,
    ):
        self.nb_parameters = nb_parameters
        if (
            isinstance(param_names, tuple)
            and len(param_names) == 1
            and isinstance(param_names[0], list)
        ):
            self.param_names = param_names[0]
        else:
            self.param_names = param_names
        self.nb_outputs = nb_outputs
        if (
            isinstance(output_names, tuple)
            and len(output_names) == 1
            and isinstance(output_names[0], list)
        ):
            self.output_names = output_names[0]
        else:
            self.output_names = output_names
        self.data_already_normalized = data_already_normalized
        self.var_dist = var_dist
        self.var_strat = var_strat
        self.kernel = kernel
        self.nb_training_iter = nb_training_iter
        self.training_proportion = training_proportion
        self.nb_inducing_points = nb_inducing_points
        if nb_latents:
            self.nb_latents = nb_latents
        else:
            self.nb_latents = nb_outputs
        self.learning_rate = learning_rate
        self.mll = mll
        self.num_mixtures = num_mixtures
        self.jitter = jitter

        # Separate data into inputs and outputs
        self.input_data = data[:, : self.nb_parameters]
        self.output_data = data[:, self.nb_parameters :]

        # Normalize the inputs and the outputs (only if required)
        if self.data_already_normalized == True:
            self.normalized_input_data = self.input_data
            self.normalized_output_data = self.output_data
        else:
            (
                self.normalized_input_data,
                self.normalizing_input_mean,
                self.normalizing_input_std,
            ) = self.normalize_data(self.input_data)

            (
                self.normalized_output_data,
                self.normalizing_output_mean,
                self.normalizing_output_std,
            ) = self.normalize_data(self.output_data)

        # SET-UP
        # 1. Identify the timesteps in the provided data
        # self.time_steps = data[:, nb_parameters - 1].unique().tolist()
        self.time_steps = [0]
        self.nb_time_steps = len(self.time_steps)

        # 2. Separate training and validation data sets
        # Isolate time steps in a separate dimension
        X_full = self.normalized_input_data.reshape(
            [-1, len(self.time_steps), nb_parameters]
        )
        Y_full = self.normalized_output_data.reshape(
            [-1, len(self.time_steps), nb_outputs]
        )

        # Find the total number of patients in the input data set
        self.nb_patients = X_full.shape[0]
        # Compute the number of patients for training
        self.nb_patients_training = math.floor(
            self.training_proportion * self.nb_patients
        )

        if self.training_proportion != 1:  # non-empty validation data set
            if self.nb_patients_training == self.nb_patients:
                raise ValueError(
                    "Training proportion too high for the number of sets of parameters: all would be used for training. Set training_proportion as 1 if that is your intention."
                )

            # list of randomly mixed indices
            mixed_indices = torch.randperm(self.nb_patients)

            training_indices = mixed_indices[: self.nb_patients_training]
            validation_indices = mixed_indices[self.nb_patients_training :]

            self.X_training = X_full[training_indices, :, :].reshape(
                [-1, nb_parameters]
            )
            self.Y_training = Y_full[training_indices, :, :].reshape([-1, nb_outputs])

            self.X_validation = X_full[validation_indices, :, :].reshape(
                [-1, nb_parameters]
            )
            self.Y_validation = Y_full[validation_indices, :, :].reshape(
                [-1, nb_outputs]
            )

        else:  # no validation data set provided
            self.X_training = X_full.reshape([-1, nb_parameters])
            self.Y_training = Y_full.reshape([-1, nb_outputs])

            self.X_validation = None
            self.Y_validation = None

        # 3. Create inducing points
        self.inducing_points = self.X_training[
            torch.randperm(self.X_training.shape[0])[: self.nb_inducing_points],
            :,
        ]

        # 4. Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.nb_outputs, has_global_noise=True, has_task_noise=True
        )
        self.model = SVGP(
            self.inducing_points,
            self.nb_parameters,
            self.nb_outputs,
            self.var_dist,
            self.var_strat,
            self.kernel,
            self.jitter,
            self.num_mixtures,
        )

    def normalize_data(self, data_in):
        """Normalize a tensor with respect to its mean and std."""
        mean = data_in.mean(dim=0)
        std = data_in.std(dim=0)
        norm_data = (data_in - mean) / std
        return norm_data, mean, std

    def unnormalize_output(self, vec):
        """Unnormalize outputs from the GP."""
        return vec * self.normalizing_output_std + self.normalizing_output_mean

    def normalize_inputs(self, vec):
        """Normalize new inputs provided to the GP."""
        return (vec - self.normalizing_input_mean) / self.normalizing_input_std

    def train(self, mini_batching=False, mini_batch_size=None):
        # TRAINING

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
                self.likelihood, self.model, num_data=self.Y_training.size(0)
            )
        elif self.mll == "PLL":
            mll = PredictiveLogLikelihood(
                self.likelihood, self.model, num_data=self.Y_training.size(0)
            )
        else:
            raise ValueError(f"Invalid MLL choice ({mll}). Choose ELBO or PLL.")

        # keep track of the loss
        losses_list = []
        epochs = tqdm(range(self.nb_training_iter))

        # Batch training loop
        if mini_batching:
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
            train_dataset = TensorDataset(self.X_training, self.Y_training)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.mini_batch_size,
                shuffle=True,
            )

            # main training loop
            for _ in epochs:
                epoch_losses = []
                for batch_params, batch_outputs in train_loader:
                    optimizer.zero_grad()  # zero gradients from previous iteration
                    output = self.model(batch_params)  # recalculate the prediction
                    loss = -mll(output, batch_outputs)
                    loss.backward()  # compute the gradients of the parameters that can be changed
                    epoch_losses.append(loss.item())
                    optimizer.step()
                epoch_loss = sum(epoch_losses) / len(epoch_losses)
                epochs.set_postfix({"loss": epoch_loss})
                losses_list.append(epoch_loss)

        # Full data set training loop
        else:
            for _ in epochs:
                optimizer.zero_grad()  # zero gradients from previous iteration
                output = self.model(
                    self.X_training
                )  # calculate the prediction with current parameters
                loss = -mll(output, self.Y_training)
                loss.backward()  # compute the gradients of the parameters that can be changed
                losses_list.append(loss.item())
                optimizer.step()
                epochs.set_postfix({"loss": loss.item()})
        self.losses = torch.tensor(losses_list)

    def plot_loss(self):
        # plot the loss over iterations
        iterations = torch.linspace(1, self.nb_training_iter, self.nb_training_iter)

        plt.plot(iterations, self.losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss over Iterations")
        plt.show()

    def predict(self, X):
        """Predict mean and interval confidence values for a given input tensor (normalized inputs). This function outputs normalized values."""
        # set model and likelihood in evaluation mode
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            prediction = self.likelihood(self.model(X))
            return (
                prediction.mean,
                prediction.confidence_region()[0],
                prediction.confidence_region()[1],
            )

    def predict_scaled(self, X):
        """Predict mean and interval confidence values for a given input tensor (normalized inputs). This function outputs rescaled values."""
        pred = self.predict(X)
        if self.data_already_normalized:
            return pred
        else:
            return tuple(map(self.unnormalize_output, pred))

    def RMSE(self, y1, y2):
        """Given two tensors of same shape, compute the Root Mean Squared Error on each column (outputs)."""
        return torch.sqrt(torch.pow(y1 - y2, 2).sum(dim=0) / y1.shape[0])

    def eval_perf(self):
        """Evaluate the model performance on its training data set and validation data set."""
        (
            self.Y_training_predicted_mean,
            self.Y_training_predicted_lower,
            self.Y_training_predicted_upper,
        ) = self.predict(self.X_training)
        self.RMSE_training = self.RMSE(self.Y_training_predicted_mean, self.Y_training)
        print("Root mean squared error on training data set (for each output):")
        print(self.RMSE_training.tolist())
        if self.training_proportion != 1:
            (
                self.Y_validation_predicted_mean,
                self.Y_validation_predicted_lower,
                self.Y_validation_predicted_upper,
            ) = self.predict(self.X_validation)
            self.RMSE_validation = self.RMSE(
                self.Y_validation_predicted_mean, self.Y_validation
            )
            print("Root mean squared error on validation data set (for each output):")
            print(self.RMSE_validation.tolist())

    def plot_obs_vs_predicted(self, logScale=None, data_set="training"):
        """Plots the observed vs. predicted values on the training and validation data sets."""
        n_cols = self.nb_outputs
        n_rows = 1
        if n_cols == 1:
            _, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = [axes]
        else:
            _, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = axes.flatten()

        if data_set == "training":
            obs_values = self.unnormalize_output(self.Y_training).reshape(
                [-1, self.nb_time_steps, self.nb_outputs]
            )
            pred_values = self.unnormalize_output(
                self.Y_training_predicted_mean
            ).reshape([-1, self.nb_time_steps, self.nb_outputs])
        elif data_set == "validation":
            if self.training_proportion == 1:
                raise ValueError("No validation data set available.")

            obs_values = self.unnormalize_output(self.Y_validation).reshape(
                [-1, self.nb_time_steps, self.nb_outputs]
            )
            pred_values = self.unnormalize_output(
                self.Y_validation_predicted_mean
            ).reshape([-1, self.nb_time_steps, self.nb_outputs])

        if not logScale:
            logScale = [True] * self.nb_parameters

        for output_index in range(self.nb_outputs):
            log_viz = logScale[output_index]
            ax = axes[output_index]
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            ax.plot(
                obs_values[:, :, output_index],
                pred_values[:, :, output_index],
                "o",
                linewidth=1,
                alpha=0.6,
            )
            # Reset the color cycling to ensure proper correspondence between data and prediction
            plt.gca().set_prop_cycle(None)

            min_val = obs_values[:, :, output_index].min().min()
            max_val = obs_values[:, :, output_index].max().max()
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "-",
                linewidth=1,
                alpha=0.5,
                color="black",
            )
            ax.fill_between(
                [min_val, max_val],
                [min_val / 2, max_val / 2],
                [min_val * 2, max_val * 2],
                linewidth=1,
                alpha=0.25,
                color="black",
            )
            title = f"{self.output_names[output_index]}"  # More descriptive title
            ax.set_title(title)
            if log_viz:
                ax.set_xscale("log")
                ax.set_yscale("log")
        plt.tight_layout()
        plt.suptitle(f"Observed vs predicted values for the {data_set} data set")
        plt.show()

    # plot function
    def plot_individual_solution(self, patient_number):
        """Plot the model prediction (and confidence interval) vs. the input data for a single patient. Can be either training or validation patient."""

        # Reshape the data to put time in a separate dimension
        def expand_tensor(u):
            return u.reshape([-1, self.nb_time_steps, self.nb_outputs])

        obs_data_set = expand_tensor(self.output_data)

        pred_data_set_mean, pred_data_set_lower, pred_data_set_upper = map(
            expand_tensor, self.predict_scaled(self.normalized_input_data)
        )
        real_inputs = self.input_data

        fig, axes = plt.subplots(
            1, self.nb_outputs, figsize=(9.0 * self.nb_outputs, 4.0)
        )
        if self.nb_outputs == 1:  # handle single output case where axes is not an array
            axes = [axes]
        patient_data = obs_data_set[patient_number, :, :].view(-1, self.nb_outputs)
        patient_params = real_inputs[patient_number, :].flatten().tolist()
        patient_pred_mean = pred_data_set_mean[patient_number, :, :].view(
            -1, self.nb_outputs
        )
        patient_pred_lower = pred_data_set_lower[patient_number, :, :].view(
            -1, self.nb_outputs
        )
        patient_pred_upper = pred_data_set_upper[patient_number, :, :].view(
            -1, self.nb_outputs
        )
        for output_index in range(self.nb_outputs):
            ax = axes[output_index]
            ax.set_xlabel("Time")
            ax.plot(
                self.time_steps,
                patient_data[:, output_index],
                "+",
                color="C0",
                linewidth=2,
                alpha=0.6,
                label=self.output_names[output_index],
            )  # true values

            # Plot GP prediction
            ax.plot(
                self.time_steps,
                patient_pred_mean[:, output_index],
                "*",
                color="C3",
                linewidth=2,
                alpha=0.5,
                label="GP prediction for "
                + self.output_names[output_index]
                + " (mean)",
            )
            ax.fill_between(
                self.time_steps,
                patient_pred_upper[:, output_index],
                patient_pred_lower[:, output_index],
                alpha=0.5,
                color="C3",
                label="GP prediction for " + self.output_names[output_index] + " (CI)",
            )

            ax.legend(loc="upper right")
            title = f"{self.output_names[output_index]} for patient {patient_number}"
            ax.set_title(title)

            param_text = "Parameters:\n"
            for i, name in enumerate(self.param_names):
                param_text += (
                    f"  {name}: {patient_params[i]:.3f}\n"  # Format to 4 decimal places
                )

            ax.text(
                1.02,
                0.98,
                param_text,
                transform=ax.transAxes,  # Coordinate system is relative to the axis
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5, ec="k"),
            )

        plt.tight_layout()
        plt.show()

    def plot_all_solutions(self, data_set="training"):
        """Plot the overlapped observations and model predictions for all patients, both in the training and in the validation data set."""

        # Reshape the data to put time in a separate dimension
        def expand_tensor(u):
            return u.reshape([-1, self.nb_time_steps, self.nb_outputs])

        if data_set == "training":
            obs = expand_tensor(self.unnormalize_output(self.Y_training))
            pred = expand_tensor(
                self.unnormalize_output(self.Y_training_predicted_mean)
            )
        elif data_set == "validation":
            if self.training_proportion == 1:
                raise ValueError("No validation data set available.")
            obs = expand_tensor(self.unnormalize_output(self.Y_validation))
            pred = expand_tensor(
                self.unnormalize_output(self.Y_validation_predicted_mean)
            )
        else:
            raise ValueError(
                f"Incorrect data set choice ({data_set}), please select training or validation"
            )

        n_cols = self.nb_outputs
        n_rows = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_cols != 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for output_index in range(self.nb_outputs):
            ax = axes[output_index]
            ax.set_xlabel("Time")
            plt.gca().set_prop_cycle(None)
            ax.plot(
                self.time_steps,
                obs[:, :, output_index].transpose(0, 1),
                "+",
                linewidth=2,
                alpha=0.6,
            )
            # Reset the color cycling to ensure proper correspondence between data and prediction
            plt.gca().set_prop_cycle(None)

            # Plot GP prediction
            ax.plot(
                self.time_steps,
                pred[:, :, output_index].transpose(0, 1),
                "-",
                linewidth=2,
                alpha=0.5,
            )

            title = f"{self.output_names[output_index]}"  # More descriptive title
            ax.set_title(title)

        plt.suptitle(f"Observed vs predicted values for the {data_set} data set")
        plt.tight_layout()
        plt.show()
