from matplotlib import pyplot as plt
import math
import torch
import gpytorch
from tqdm import tqdm
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

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
        training_df: pd.DataFrame,
        descriptors,
        outputs,
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
        if not ("id" in training_df.columns.to_list()):
            raise ValueError("Training data should contain an `id` column.")
        if set(descriptors + outputs + ["id"]) != set(training_df.columns.to_list()):
            raise ValueError(
                "The provided inputs and outputs do not match the training data."
            )

        self.parameter_names = descriptors
        self.nb_parameters = len(self.parameter_names)
        self.output_names = outputs
        self.nb_outputs = len(self.output_names)
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
            self.nb_latents = self.nb_outputs
        self.learning_rate = learning_rate
        self.mll = mll
        self.num_mixtures = num_mixtures
        self.jitter = jitter

        # Find the total number of patients in the input data set
        self.patients = training_df["id"].unique()
        self.nb_patients = self.patients.shape[0]

        # Separate data into inputs and outputs
        self.raw_training_df = training_df
        self.input_data = training_df[descriptors]
        self.output_data = training_df[outputs]

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
            # Add the id column back in for filtering
            self.normalized_input_data["id"] = training_df["id"]

            (
                self.normalized_output_data,
                output_mean_series,
                output_std_series,
            ) = self.normalize_data(self.output_data)
            self.normalized_output_data["id"] = training_df["id"]
            # Convert the output normalizing params to tensors to integrated
            self.normalizing_output_mean = torch.Tensor(output_mean_series)
            self.normalizing_output_std = torch.Tensor(output_std_series)

        # SET-UP
        # Compute the number of patients for training
        self.nb_patients_training = math.floor(
            self.training_proportion * self.nb_patients
        )
        self.nb_patients_validation = self.nb_patients - self.nb_patients_training

        if self.training_proportion != 1:  # non-empty validation data set
            if self.nb_patients_training == self.nb_patients:
                raise ValueError(
                    "Training proportion too high for the number of sets of parameters: all would be used for training. Set training_proportion as 1 if that is your intention."
                )

            # Randomly mixing up patients
            mixed_patients = np.random.permutation(self.patients)

            self.training_patients = mixed_patients[: self.nb_patients_training]
            self.validation_patients = mixed_patients[self.nb_patients_training :]

            self.training_inputs_df = self.normalized_input_data.loc[
                self.normalized_input_data["id"].isin(self.training_patients)
            ]
            self.X_training = torch.Tensor(
                self.training_inputs_df.drop("id", axis=1).values
            )

            self.training_outputs_df = self.normalized_output_data.loc[
                self.normalized_output_data["id"].isin(self.training_patients)
            ]
            self.Y_training = torch.Tensor(
                self.training_outputs_df.drop("id", axis=1).values
            )

            self.validation_inputs_df = self.normalized_input_data.loc[
                self.normalized_input_data["id"].isin(self.validation_patients)
            ]
            self.X_validation = torch.Tensor(
                self.validation_inputs_df.drop("id", axis=1).values
            )

            self.validation_outputs_df = self.normalized_output_data.loc[
                self.normalized_output_data["id"].isin(self.validation_patients)
            ]
            self.Y_validation = torch.Tensor(
                self.validation_outputs_df.drop("id", axis=1).values
            )

        else:  # no validation data set provided
            self.training_inputs_df = self.normalized_input_data
            self.X_training = torch.Tensor(
                self.training_inputs_df.drop("id", axis=1).values
            )
            self.training_outputs_df = self.normalized_output_data
            self.Y_training = torch.Tensor(
                self.training_outputs_df.drop("id", axis=1).values
            )

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

    def normalize_data(
        self, data_in: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Normalize a data frame with respect to its mean and std."""
        mean = data_in.mean()
        std = data_in.std()
        norm_data = (data_in - mean) / std
        return norm_data, mean, std

    def unnormalize_output(self, vec: torch.Tensor) -> torch.Tensor:
        """Unnormalize outputs from the GP."""
        return vec * self.normalizing_output_std + self.normalizing_output_mean

    def normalize_inputs(self, inputs_df: pd.DataFrame) -> torch.Tensor:
        """Normalize new inputs provided to the GP."""
        normalized_df = (
            inputs_df - self.normalizing_input_mean
        ) / self.normalizing_input_std
        return torch.Tensor(normalized_df.values)

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
            raise ValueError(f"Invalid MLL choice ({self.mll}). Choose ELBO or PLL.")

        # keep track of the loss
        losses_list = []
        epochs = tqdm(range(self.nb_training_iter))

        # Batch training loop
        if mini_batching:
            # set the mini_batch_size to a power of two of the total size -4
            if mini_batch_size == None:
                power = math.floor(math.log2(self.raw_training_df.shape[0])) - 4
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

    def predict_to_df(self, data_set="training"):
        if data_set == "training":
            pred_values = pd.DataFrame(
                self.unnormalize_output(self.Y_training_predicted_mean).numpy()
            )
            pred_values.columns = self.output_names
            pred_values["id"] = self.training_inputs_df["id"].values
        elif data_set == "validation":
            if self.training_proportion == 1:
                raise ValueError("No validation data set available.")
            pred_values = pd.DataFrame(
                self.unnormalize_output(self.Y_validation_predicted_mean).numpy()
            )
            pred_values.columns = self.output_names
            pred_values["id"] = self.validation_inputs_df["id"].values
        else:
            raise ValueError(
                f"Unsupported data set choic {data_set}: choose `training` or `validation`"
            )
        return pred_values

    def observed_data_to_df(self, data_set="training"):
        if data_set == "training":
            obs_values = pd.DataFrame(self.unnormalize_output(self.Y_training).numpy())
            obs_values.columns = self.output_names
            obs_values["id"] = self.training_inputs_df["id"].values
        elif data_set == "validation":
            if self.Y_validation is not None:
                obs_values = pd.DataFrame(
                    self.unnormalize_output(self.Y_validation).numpy()
                )
                obs_values.columns = self.output_names
                obs_values["id"] = self.validation_inputs_df["id"].values
            else:
                raise ValueError("No validation data set available.")
        else:
            raise ValueError(
                f"Unsupported data set choic {data_set}: choose `training` or `validation`"
            )
        return obs_values

    def patient_list(self, data_set="training"):
        if data_set == "training":
            patients = self.training_patients
        elif data_set == "validation":
            if self.training_proportion == 1:
                raise ValueError("No validation data set available.")
            patients = self.validation_patients
        else:
            raise ValueError(
                f"Unsupported data set choic {data_set}: choose `training` or `validation`"
            )
        return patients

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

        obs_values = self.observed_data_to_df(data_set)
        pred_values = self.predict_to_df(data_set)
        patients = self.patient_list(data_set)

        if not logScale:
            logScale = [True] * self.nb_parameters

        for output_index, output_name in enumerate(self.output_names):
            log_viz = logScale[output_index]
            ax = axes[output_index]
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            for _, ind in enumerate(patients):
                obs_vec = obs_values.loc[obs_values["id"] == ind][output_name]
                pred_vec = pred_values.loc[pred_values["id"] == ind][output_name]
                ax.plot(
                    obs_vec,
                    pred_vec,
                    "o",
                    linewidth=1,
                    alpha=0.6,
                )

            min_val = obs_values[output_name].min().min()
            max_val = obs_values[output_name].max().max()
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
        patient_index = self.patients[patient_number]
        patient_raw_df = self.raw_training_df.loc[
            self.raw_training_df["id"] == patient_index
        ]
        patient_normalized_inputs = torch.Tensor(
            self.normalized_input_data.loc[
                self.normalized_input_data["id"] == patient_index
            ]
            .drop("id", axis=1)
            .values
        )

        patient_pred_mean, patient_pred_lower, patient_pred_upper = self.predict_scaled(
            patient_normalized_inputs
        )

        fig, axes = plt.subplots(
            1, self.nb_outputs, figsize=(9.0 * self.nb_outputs, 4.0)
        )
        if self.nb_outputs == 1:  # handle single output case where axes is not an array
            axes = [axes]
        patient_data = patient_raw_df[self.output_names]
        patient_params = patient_raw_df[self.parameter_names]
        time_steps = patient_params["time"].values
        sorted_indices = np.argsort(time_steps)
        sorted_time_steps = time_steps[sorted_indices]
        for output_index, output_name in enumerate(self.output_names):
            ax = axes[output_index]
            ax.set_xlabel("Time")
            ax.plot(
                sorted_time_steps,
                patient_data[output_name].values[sorted_indices],
                ".-",
                color="C0",
                linewidth=2,
                alpha=0.6,
                label=output_name,
            )  # true values

            # Plot GP prediction
            ax.plot(
                sorted_time_steps,
                patient_pred_mean.numpy()[sorted_indices, output_index],
                "-",
                color="C3",
                linewidth=2,
                alpha=0.5,
                label="GP prediction for " + output_name + " (mean)",
            )
            ax.fill_between(
                sorted_time_steps,
                patient_pred_upper.numpy()[sorted_indices, output_index],
                patient_pred_lower.numpy()[sorted_indices, output_index],
                alpha=0.5,
                color="C3",
                label="GP prediction for " + output_name + " (CI)",
            )

            ax.legend(loc="upper right")
            title = f"{output_name} for patient {patient_number}"
            ax.set_title(title)

            param_text = "Parameters:\n"
            for name in self.parameter_names:
                param_text += f"  {name}: {patient_params[name].values[0]:.3f}\n"  # Format to 4 decimal places

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

        obs = self.observed_data_to_df(data_set)
        pred = self.predict_to_df(data_set)
        patients = self.patient_list(data_set)
        n_cols = self.nb_outputs
        n_rows = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        cmap = plt.cm.get_cmap("Spectral")
        colors = cmap(np.linspace(0, 1, len(patients)))
        if n_cols != 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for output_index, output_name in enumerate(self.output_names):
            ax = axes[output_index]
            ax.set_xlabel("Time")
            for patient_num, patient_ind in enumerate(patients):
                time_vec = self.raw_training_df.loc[
                    self.raw_training_df["id"] == patient_ind
                ]["time"].values
                sorted_indices = np.argsort(time_vec)
                sorted_times = time_vec[sorted_indices]
                obs_vec = obs.loc[obs["id"] == patient_ind][output_name].values[
                    sorted_indices
                ]
                pred_vec = pred.loc[pred["id"] == patient_ind][output_name].values[
                    sorted_indices
                ]
                ax.plot(
                    sorted_times,
                    obs_vec,
                    "+",
                    color=colors[patient_num],
                    linewidth=2,
                    alpha=0.6,
                )
                ax.plot(
                    sorted_times,
                    pred_vec,
                    "-",
                    color=colors[patient_num],
                    linewidth=2,
                    alpha=0.5,
                )

            title = f"{output_name}"  # More descriptive title
            ax.set_title(title)

        plt.suptitle(f"Observed vs predicted values for the {data_set} data set")
        plt.tight_layout()
        plt.show()
