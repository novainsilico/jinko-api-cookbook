import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
from typing import List, Dict, Union, Callable, Optional, Tuple
from pandas import DataFrame
import pandas as pd
import GP
import os
from datetime import datetime
import jinko_helpers as jinko
import zipfile, io, os


torch.set_default_dtype(torch.float32)

with torch.no_grad():
    # --- 1. NLME Model Definition (No change required here, it's independent of the data storage) ---
    class NLMEModel:
        """
        This class encompasses the info on the model from which we want to infer population parameters' distribution.
        It includes the methods individual_parameters to go from population parameters + individual effects to individual parameters, structural_model to compute outputs from sets of parameters (this method is given by the user, as well as the initial conditions fed to it), calculate_residuals and log_likelihood_observation.
        MI refer to Model Intrinsic parameters, i.e. parameters with no InterIndividual Variance (IIV).
        PDU refer to Patient Descriptor Unknown, i.e. parameters with InterIndividual Variance (IIV). The population means for the PDUs are denoted mus and the individual effects etas.
        cov_coeffs refer to the coefficients between the covariates and the PDUs that they influence.
        We assume log-normal distribution for individual parameters and covariate effects: theta_i[PDU] = mu_pop * exp(eta_i) * exp(covariates_i * cov_coeffs) where eta_i is from N(0, Omega) and theta_i[MI]=MI.
        We set betas as [log(μ1), coeff_cov1_having_effect_on_μ1, coeff_cov2_having_effect_on_μ1, ...log(μ2), ...] so that θ_i = concat(MI, exp(design_matrix_X_i @ population_β + individual_η).
        """

        def __init__(
            self,
            MI_names: List[str],
            PDU_names: List[str],
            cov_coeffs_names: List[str],
            outputs_names: List[str],
            error_model_type: str = "additive",  # the only one implemented for now
            covariate_map: Optional[Dict[str, List[str]]] = None,
            nb_decimal_rounding_time: int = 3,
        ):
            self.MI_names: List[str] = MI_names
            self.nb_MI: int = len(MI_names)
            self.PDU_names: List[str] = PDU_names
            self.nb_PDU: int = len(PDU_names)
            self.nb_parameters = self.nb_PDU + self.nb_MI
            self.cov_coeffs_names: List[str] = cov_coeffs_names
            self.nb_cov_coeffs: int = len(cov_coeffs_names)
            self.nb_betas: int = self.nb_PDU + self.nb_cov_coeffs
            self.outputs_names: List[str] = outputs_names
            self.nb_outputs: int = len(outputs_names)
            self.error_model_type: str = error_model_type
            self.covariate_map: Optional[Dict[str, List[str]]] = covariate_map

            self.population_betas_names: List[str] = []
            idx = 0
            for PDU_name in self.PDU_names:
                self.population_betas_names.append(PDU_name)
                if self.covariate_map and PDU_name in self.covariate_map:
                    for _ in range(len(self.covariate_map[PDU_name])):
                        self.population_betas_names.append(cov_coeffs_names[idx])
                        idx += 1
            self.nb_decimal_rounding_time = nb_decimal_rounding_time

        def individual_parameters(
            self,
            log_MI: torch.Tensor,
            population_betas: torch.Tensor,
            individual_etas: torch.Tensor,
            design_matrices_X_i: Union[torch.Tensor, List[torch.Tensor]],
        ) -> List[torch.Tensor]:
            """
            Transforms log(MI) (Model intrinsic), betas: log(mu)s & coeffs for covariates and individual random effects (etas) into individual parameters (theta_i), for each set of etas of the list and corresponding design matrix.
            Assumes log-normal distribution for individual parameters and covariate effects: theta_i[PDU] = mu_pop * exp(eta_i) * exp(covariates_i * cov_coeffs) where eta_i is from N(0, Omega) and theta_i[MI]=MI.
            log_MI: torch.Tensor of the logs of the MI model parameters (i.e. parameters without InterIndividual Variability)
            population_betas: torch.Tensor of the log(mu1), coeff_for_covariate_mu1_1, coeff_for_covariate_mu1_2 ... log(mu2), coeff_for_covariate_mu2_1 ... (dim: [nb_betas])
            individual_etas: torch.Tensor with the eta value for each PDU for each individual (dim: [nb_eta_sets x nb_PDU])
            design matrices X_i: either a list (of length nb_eta_sets) of design matrices, i.e. torch.Tensors [nb_PDU x nb_betas], each row containing [0 , 0 ... 1, cov1, cov2, 0, 0 ...] or one design matrix that will be replicated nb_eta_sets times.
            """
            # errors
            if isinstance(design_matrices_X_i, torch.Tensor):
                design_matrices_X_i = [design_matrices_X_i] * len(
                    individual_etas
                )  # if there is only one design matrix (i.e., we want to compute for the same individual), create a list with the same design matrix repeated
            elif len(design_matrices_X_i) != len(individual_etas):
                raise ValueError(
                    "design_matrices_X_i must be a single tensor or a list of tensors matching the length of list_individual_etas."
                )
            if design_matrices_X_i[0].shape[1] != len(population_betas):
                raise ValueError(
                    "Dimension mismatch between design_matrix_X_i and beta."
                )
            if individual_etas.shape[1] != self.nb_PDU:
                raise ValueError(
                    "Dimension mismatch between individual_etas and the number of population parameters."
                )
            if self.covariate_map is None:
                raise ValueError(
                    "Covariate map must be defined in order to compute individual parameters."
                )
            # check betas dim
            if population_betas.dim() > 1:
                population_betas = population_betas.squeeze()

            # compute individual parameters
            stacked_X = torch.cat(
                design_matrices_X_i, dim=0
            )  # stack all design matrices into a single large tensor
            eta_flat = individual_etas.flatten().unsqueeze(
                -1
            )  # reshape individual_etas to be a single column tensor
            log_thetas_PDU = (stacked_X @ population_betas.unsqueeze(-1)) + eta_flat
            individual_log_thetas_PDU_list = torch.split(
                log_thetas_PDU, self.nb_PDU
            )  # split the result

            individual_params_list = [
                torch.exp(
                    torch.concat((log_MI.flatten(), ind_log_thetas_PDU.squeeze()))
                )
                for ind_log_thetas_PDU in individual_log_thetas_PDU_list
            ]  # combine MI and PDU parameters for each individual

            return individual_params_list

        def calculate_residuals(
            self, observed_data: torch.Tensor, predictions: torch.Tensor
        ) -> torch.Tensor:
            """
            Calculates residuals based on the error model.
            observed_data: torch.Tensor of observations for one individual. dim: [nb_outputs x nb_time_points]
            predictions: torch.Tensor of predictions for one individual. Must be organized like observed_data to compare both by subtraction, dim: [nb_outputs x time_steps]
            """
            # print(
            #     f"DEBUG observations \n{observed_data} \n vs predictions \n {predictions}"
            # )
            if observed_data.dim() == 1:
                observed_data = observed_data.unsqueeze(0)
            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(0)
            if self.error_model_type == "additive":
                return observed_data - predictions
            elif self.error_model_type == "proportional":
                return (observed_data - predictions) / predictions
            else:
                raise ValueError("Unsupported error model type.")

        def log_likelihood_observation(
            self,
            observed_data: torch.Tensor,
            predictions: torch.Tensor,
            residual_error_var: torch.Tensor,
        ) -> float:
            """
            Calculates the log-likelihood of observations given predictions and additive error model, assuming errors follow N(0,sqrt(residual_error_var))
            observed_data: torch.Tensor of observations for one individual, dim: [nb_outputs x nb_time_points]
            predictions: torch.Tensor of predictions for one individual organized in the same way as observed_data, dim: [nb_outputs x nb_time_points]
            residual_error_var: torch.Tensor of the error for each output, dim: [nb_outputs]
            """
            if torch.any(torch.isinf(predictions)) or torch.any(
                torch.isnan(predictions)
            ):
                return -torch.inf  # invalid predictions
            residuals: torch.Tensor = self.calculate_residuals(
                observed_data, predictions
            )
            # ensure error_std is positive
            res_error_var = torch.max(
                torch.full_like(residual_error_var, 1e-6), residual_error_var.clone()
            )
            # Log-likelihood of normal distribution
            if self.error_model_type == "additive":
                log_lik: float = (
                    -0.5
                    * torch.sum(
                        torch.log(2 * torch.pi * res_error_var)
                        + (residuals**2 / res_error_var)
                    ).item()
                )  # each row of res_error_var is the variance of the residual error corresponding to the same row of residuals (one row = one output)
            elif self.error_model_type == "proportional":
                log_lik: float = (
                    -0.5
                    * torch.sum(
                        torch.log(2 * torch.pi * res_error_var * predictions)
                        + (residuals**2 / res_error_var)
                    ).item()
                )  # each row of res_error_var is the variance of the residual error corresponding to the same row of residuals (one row = one output)
            else:
                raise ValueError("Non supported error type.")
            return log_lik

    class NLMEModel_from_struct_model(NLMEModel):
        """
        This class inherits NLMEModel. The structural model must be given when instantiating the class. It must be a method of signature:
        structural_model(
                list_time_steps: List[Dict[str, torch.Tensor]],
                list_params: List[torch.Tensor],
            ) -> List[List[torch.Tensor]]
        with inputs: list_time_steps: list of dictionaries associating an output name with a time_steps torch.Tensor
                     list_params: list of individual parameters torch.Tensors for which to predict. Each parameter set correspond to one dictionary of list_time_steps.
        """

        def __init__(
            self,
            MI_names: List[str],
            PDU_names: List[str],
            cov_coeffs_names: List[str],
            outputs_names: List[str],
            structural_model: Callable[
                [List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]
            ],
            error_model_type: str = "additive",
            covariate_map: Optional[Dict[str, List[str]]] = None,
            nb_decimal_rounding_time: int = 3,
        ):
            super().__init__(
                MI_names,
                PDU_names,
                cov_coeffs_names,
                outputs_names,
                error_model_type,
                covariate_map,
                nb_decimal_rounding_time,
            )
            self.structural_model: Callable[
                [List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]
            ] = structural_model

    class NLMEModel_from_GP(NLMEModel):
        """
        This class inherits NLMEModel. A GP must be given when instantiating the class. The structural model will be constructed from it.
        """

        def __init__(
            self,
            MI_names: List[str],
            PDU_names: List[str],
            cov_coeffs_names: List[str],
            outputs_names: List[str],
            GP_model: GP,
            GP_error_threshold: float = 0.75,
            error_model_type: str = "additive",
            covariate_map: Optional[Dict[str, List[str]]] = None,
            nb_decimal_rounding_time: int = 3,
        ):
            super().__init__(
                MI_names,
                PDU_names,
                cov_coeffs_names,
                outputs_names,
                error_model_type,
                covariate_map,
                nb_decimal_rounding_time,
            )
            self.GP_model: GP = GP_model
            self.GP_error_threshold = GP_error_threshold

        def structural_model(
            self,
            input_data: DataFrame,
        ) -> DataFrame:
            """
            Predicts outputs using the GP model based on a single input DataFrame.
            This DataFrame must contain columns for all individual parameters and time points.
            input_data (DataFrame): DataFrame with columns corresponding to individual parameters, time, and output_name.
            Returns a DataFrame with the same columns plus a predicted_value column.
            """
            # ensure the input DataFrame has the necessary columns
            required_params_cols = self.MI_names + self.PDU_names
            required_cols = ["ind_id"] + required_params_cols + ["time", "output_name"]
            if not all(col in input_data.columns for col in required_cols):
                raise ValueError(
                    f"Input DataFrame to structural_model is missing required columns. "
                    f"Required: {required_cols}"
                )

            # get the unique parameter/time combinations to feed to the GP
            # we need ind_id for merging later, but the GP input only needs parameters and time.
            unique_input_cols = ["ind_id"] + required_params_cols + ["time"]
            unique_inputs_df = (
                input_data[unique_input_cols].drop_duplicates().reset_index(drop=True)
            )
            input_cols = required_params_cols + ["time"]
            # reorder the columns to match the GP (parameters then time)
            right_columns_input = torch.tensor(
                unique_inputs_df[input_cols].values, dtype=torch.float32
            )
            # Predict outputs with the GP model
            # The GP predicts for all outputs at once
            normalized_input = self.GP_model.normalize_inputs(right_columns_input)
            mean, _, _ = self.GP_model.predict_scaled(
                normalized_input
            )  # of dim [nb_unique_inputs, nb_outputs]

            # create a DataFrame from the predictions for easy merging
            predicted_values_df = unique_inputs_df.copy()
            for i, output_name in enumerate(self.outputs_names):
                predicted_values_df[output_name] = mean[:, i]

            # transform in the format for SAEM where each line corresponds to one output
            predicted_reshaped_df = pd.melt(
                predicted_values_df,
                id_vars=unique_input_cols,
                value_vars=self.outputs_names,
                var_name="output_name",
                value_name="predicted_value",
            )
            # merge the original input data with the predictions
            output_df = pd.merge(
                input_data,
                predicted_reshaped_df,
                on=unique_input_cols + ["output_name"],
                how="left",
            )
            output_df["time"] = output_df["time"].round(self.nb_decimal_rounding_time)
            return output_df

    # --- 3. MCMC Sampler for Individual Random Effects (eta_i), used in the E-step of SAEM ---
    class MCMC_Eta_Sampler:  # one independent markov chain per individual
        """
        This class is used in the E-step of SAEM to sample the individual effects etas according to their log-posterior distribution.
        A Metropolis-Hastings MCMC is used. The methods are all helpers to the sample method that performs the sampling.
        There are as many chains as individuals, they all advance in parallel until all of them reach the desired number of samples.
        The chains start at the values of the etas at the last iteration. The proposal_var_eta determines how far we jump from sample to sample in the chain.
        To compute the likelihoods necessary to sample from the log-posterior distribution, predictions are made with the NLME Model and compared to observations.
        The predictions are made with the individual parameters computed using the fixed effects after the M-step update and the proposed etas.
        """

        def __init__(
            self,
            model: NLMEModel,
            log_MI: torch.Tensor,
            population_betas: torch.Tensor,
            population_omega: torch.Tensor,
            residual_error_var: torch.Tensor,
            proposal_var_eta: torch.Tensor,
            verbose: bool,
            observations_df: DataFrame,
            design_matrices: Dict[Union[str, int], torch.Tensor],
        ):
            self.model: NLMEModel = model
            self.population_betas: torch.Tensor = population_betas.clone()
            self.log_MI: torch.Tensor = log_MI.clone()
            self.population_omega: torch.Tensor = population_omega.clone()
            self.residual_error_var: torch.Tensor = residual_error_var
            self.proposal_var_eta: torch.Tensor = proposal_var_eta.clone()
            self.verbose = verbose
            self.observations_df = observations_df
            self.design_matrices = design_matrices
            self.unique_ind_ids = observations_df["ind_id"].unique().tolist()

        def _log_prior_etas(self, etas: torch.Tensor) -> torch.Tensor:
            """
            Calculates the log-prior for all the eta_i, i.e. assuming eta_i ~ N(0, Omega), what is the log-probability of sampling this eta? Considers each eta_i independently of the others.
            P(eta) = (1/sqrt((2pi)^k * |Omega|)) * exp(-0.5 * eta.T * omega.inv * eta)
            log P(eta) = -0.5 * (k * log(2pi) + log|Omega| + eta.T * omega.inv * eta)
            etas: torch.Tensor of dim [nb_eta_i x nb_PDU]
            """
            etas_dist = torch.distributions.MultivariateNormal(
                loc=torch.zeros(self.model.nb_PDU),
                covariance_matrix=self.population_omega,
            )
            log_priors: torch.Tensor = etas_dist.log_prob(etas)
            return log_priors

        def _log_posterior_etas(
            self, etas: torch.Tensor, ind_ids_for_etas: List[Union[str, int]]
        ) -> Tuple[torch.Tensor, List[torch.Tensor], DataFrame]:
            """
            Calculates the log-posterior for MCMC for a subset of individuals.
            The function now returns a DataFrame of predictions.
            """
            list_design_matrices = [
                self.design_matrices[ind_id] for ind_id in ind_ids_for_etas
            ]

            # Get individual parameters in a list of tensors
            list_individual_params: List[torch.Tensor] = (
                self.model.individual_parameters(
                    self.log_MI,
                    self.population_betas,
                    individual_etas=etas,
                    design_matrices_X_i=list_design_matrices,
                )
            )
            # create a single input DataFrame for the structural model from observations_df and the individual parameters
            # because we want to predict for the same time points and outputs that we have observations for
            input_data_for_model = self.observations_df[
                self.observations_df["ind_id"].isin(ind_ids_for_etas)
            ].copy()
            if "value" in input_data_for_model.columns:
                input_data_for_model = input_data_for_model.drop(
                    columns=["value"]
                )  # drop "value" which is the true observation
            # adding the individual parameters
            param_cols = self.model.MI_names + self.model.PDU_names
            param_dict = {
                ind_id: params.detach().cpu().numpy()
                for ind_id, params in zip(ind_ids_for_etas, list_individual_params)
            }
            for p_name_idx, p_name in enumerate(param_cols):
                input_data_for_model[p_name] = input_data_for_model["ind_id"].map(
                    lambda x: param_dict[x][p_name_idx]
                )

            # call the structural model with the DataFrame input
            predictions_df = self.model.structural_model(input_data_for_model)

            # calculate log-likelihoods of observations
            log_priors: torch.Tensor = self._log_prior_etas(etas)

            # merge observations with predictions to calculate residuals
            merged_df = pd.merge(
                self.observations_df,
                predictions_df,
                on=["ind_id", "output_name", "time"],
                how="left",
            )
            # group by individual and calculate log-likelihood for each
            list_log_lik_obs: List[float] = []
            for ind_id in ind_ids_for_etas:
                ind_df = merged_df[merged_df["ind_id"] == ind_id]
                total_log_lik_for_ind = 0.0

                for output_name, output_df in ind_df.groupby("output_name"):
                    if output_name in self.model.outputs_names:
                        output_idx = self.model.outputs_names.index(output_name)
                        observed_data = torch.tensor(
                            output_df["value"].values, dtype=torch.float32
                        )
                        predictions = torch.tensor(
                            output_df["predicted_value"].values, dtype=torch.float32
                        )

                        total_log_lik_for_ind += self.model.log_likelihood_observation(
                            observed_data,
                            predictions,
                            self.residual_error_var[output_idx],
                        )
                list_log_lik_obs.append(total_log_lik_for_ind)

            log_lik_obs = torch.tensor(list_log_lik_obs)

            return (log_lik_obs + log_priors), list_individual_params, predictions_df

        def sample(
            self,
            current_eta_for_all_ind: torch.Tensor,
            nb_samples: int,
            nb_burn_in: int,
        ) -> Tuple[torch.Tensor, List[List[torch.Tensor]], List[List[torch.Tensor]]]:
            """
            Performs Metropolis-Hastings sampling for the individuals'etas. The MCMC chains (one per individual) advance in parallel.
            The acceptance criterion for each sample is log(random_uniform) < proposed_log_posterior − current_log_posterior.
            Returns the mean of sampled etas per individual, the mean of the log_thetas_PDU (individual PDU parameters associated with the sampled etas) and the mean predictions associated with these individual parameters/etas.
            current_eta_for_all_ind: torch.Tensor of dim [nb_individuals x nb_PDUs]
            nb_samples: int, how many samples will be kept from each chain
            nb_burn_in: int, how many accepted samples are disgarded before we consider that the chain has converged enough
            """
            nb_individuals = len(self.unique_ind_ids)

            all_states_history: List[torch.Tensor] = []

            sum_etas = torch.zeros_like(current_eta_for_all_ind)
            sum_log_thetas_PDU = torch.zeros(nb_individuals, self.model.nb_PDU)

            predictions_history = []

            current_log_posteriors, _, _ = self._log_posterior_etas(
                current_eta_for_all_ind, self.unique_ind_ids
            )

            sample_counts = torch.zeros(nb_individuals)
            accepted_counts = torch.zeros(nb_individuals)
            total_proposals = torch.zeros(nb_individuals)
            done = torch.full((nb_individuals, 1), False)

            while not torch.all(done).item():
                active_indices = torch.where(~done)[0]
                active_ind_ids = [self.unique_ind_ids[i] for i in active_indices]

                total_proposals[active_indices] += 1
                proposal_dist = torch.distributions.MultivariateNormal(
                    current_eta_for_all_ind[active_indices], self.proposal_var_eta
                )
                proposed_etas: torch.Tensor = proposal_dist.sample()

                proposed_log_posteriors, list_thetas, proposed_predictions_df = (
                    self._log_posterior_etas(proposed_etas, active_ind_ids)
                )

                deltas: torch.Tensor = (
                    proposed_log_posteriors - current_log_posteriors[active_indices]
                )
                accept_mask: torch.Tensor = (
                    torch.log(torch.rand(len(active_indices))) < deltas
                )

                # update only the accepted etas and log-posteriors
                current_eta_for_all_ind[active_indices[accept_mask]] = proposed_etas[
                    accept_mask
                ]
                current_log_posteriors[active_indices[accept_mask]] = (
                    proposed_log_posteriors[accept_mask]
                )
                accepted_counts[active_indices] += accept_mask.int()

                for j, idx in enumerate(active_indices):
                    if accept_mask[j] and accepted_counts[idx] >= nb_burn_in:
                        sum_etas[idx] += current_eta_for_all_ind[idx].clone()
                        sum_log_thetas_PDU[idx] += torch.log(
                            list_thetas[j][self.model.nb_MI :]
                        )

                        # store predictions for averaging later and then use in the M-step
                        accepted_ind_id = self.unique_ind_ids[idx]
                        accepted_preds_for_ind = proposed_predictions_df[
                            proposed_predictions_df["ind_id"] == accepted_ind_id
                        ]
                        predictions_history.append(accepted_preds_for_ind.copy())

                        sample_counts[idx] += 1
                        if sample_counts[idx] == nb_samples:
                            done[idx] = True

                all_states_history.append(current_eta_for_all_ind.clone())

            # calculate mean etas and log of individual parameters PDU
            mean_etas = sum_etas / nb_samples
            mean_log_thetas_PDU = sum_log_thetas_PDU / nb_samples

            # calculate mean predictions from the collected history
            if predictions_history:
                all_predictions_df = pd.concat(predictions_history, ignore_index=True)
                mean_predictions_df = all_predictions_df.groupby(
                    ["ind_id", "output_name", "time"], as_index=False
                )["predicted_value"].mean()
            else:
                mean_predictions_df = DataFrame(
                    columns=["ind_id", "output_name", "time", "predicted_value"]
                )

            if self.verbose:
                acceptance_rate = accepted_counts / total_proposals
                for i in range(nb_individuals):
                    print(
                        f"  MCMC Acceptance Rate for individual {i}: {acceptance_rate[i]:.2f}"
                    )
            # plot the convergence of the MCMC chains
            # all_states_history = torch.stack(all_states_history)
            # nb_individuals = all_states_history.shape[
            #     1
            # ]  # This is the number of MCMC chains
            # nb_PDU = all_states_history.shape[2]
            # # Loop through each individual (MCMC chain) to create a separate plot
            # for individual_idx in range(nb_individuals):
            #     plt.figure(
            #         figsize=(10, 6)
            #     )  # Create a new figure for each individual's chain
            #     # Loop through each parameter (mu) for the current individual's chain
            #     for param_idx in range(nb_PDU):
            #         plt.plot(
            #             all_states_history[:, individual_idx, param_idx].cpu().numpy(),
            #             label=f"Parameter {param_idx+1}",
            #         )
            #     plt.title(f"MCMC Chain for Individual {individual_idx+1} Convergence")
            #     plt.xlabel("Iteration")
            #     plt.ylabel("Parameter Value")
            #     plt.legend()
            #     plt.grid(True)
            #     plt.show()
            return (mean_etas, mean_log_thetas_PDU, mean_predictions_df)

    # Main SAEM Algorithm Class
    class PySAEM:
        """
        The method run of this class handles the whole SAEM iterations, alternating between the E-step (sampling individual effects) and the M-step (updating the fixed effects).
        The E-step is mainly performed through the MCMC_Eta_Sampler class, and the M-step directly in this run method.
        The learning rate is updated though the method update_learning_rate.
        SAEM needs initial guesses for all fixed effects that are given by the user.
        They can also set some parameters for the sampling of the etas in the MCMC, and change the simulated annealing factor, applied on omega (covariance of individual effects etas) and the residual error update.
        """

        def __init__(
            self,
            model: NLMEModel,
            observations_df: DataFrame,
            covariates_df: Optional[DataFrame],
            initial_pop_MI: torch.Tensor,
            initial_pop_betas: torch.Tensor,
            initial_pop_omega: torch.Tensor,
            initial_res_var: torch.Tensor,
            # MCMC parameters for the E-step
            mcmc_burn_in: int = 3,
            mcmc_first_burn_in: int = 30,
            mcmc_nb_samples: int = 10,
            mcmc_proposal_var_scaling_factor: float = 0.2,
            nb_phase1_iterations: int = 100,
            nb_phase2_iterations: Union[int, None] = None,
            convergence_threshold: float = 1e-4,
            patience: int = 5,
            learning_rate_power: float = 0.8,
            annealing_factor: float = 0.95,
            verbose: bool = False,
        ):
            self.model: NLMEModel = model
            if isinstance(self.model, NLMEModel_from_GP):
                observations_df["time"] = observations_df["time"].round(
                    self.model.nb_decimal_rounding_time
                )
            self.observations_df = observations_df
            self.covariates_df = covariates_df
            self.unique_ind_ids = self.observations_df["ind_id"].unique().tolist()
            self.nb_individuals: int = len(self.unique_ind_ids)

            self.log_MI: torch.Tensor = torch.log(
                initial_pop_MI.clone()  # dim: [nb_MI]
            )  #
            self.population_betas: torch.Tensor = (
                initial_pop_betas.clone()  # dim: [nb_betas]
            )  # log of mus and coefficients of the influence of the covariates on the parameters
            # self.estimated_MI_mus: Optional[List[float]] = (
            #     None  # TO DO: try and remove it from here
            # )
            self.population_omega: torch.Tensor = (
                initial_pop_omega.clone()  # [nb_PDU x nb_PDU]
            )  # covariance matrix of the random individual effects
            self.residual_error_var: torch.Tensor = (
                initial_res_var  # variance of the additive error on the observations
            )
            # MCMC sampling in the E-step parameters
            self.mcmc_first_burn_in: int = mcmc_first_burn_in
            self.mcmc_burn_in: int = mcmc_burn_in
            self.mcmc_nb_samples: int = mcmc_nb_samples
            self.mcmc_proposal_var_scaling_factor: float = (
                mcmc_proposal_var_scaling_factor  # the variance of the multivariate normal distribution that the next eta from the Markov Chain is sampled from is scaling_factor * omega
            )
            # SAEM iteration parameters
            # phase 1 = exploratory: learning rate = 0 and simulated annealing on
            # phase 2 = smoothing: learning rate 1/phase2_iter^factor
            self.nb_phase1_iterations: int = nb_phase1_iterations
            self.current_phase1_iteration = 0
            self.nb_phase2_iterations: int = (
                nb_phase2_iterations
                if nb_phase2_iterations is not None
                else nb_phase1_iterations
            )
            self.current_phase2_iteration = 0
            # convergence parameters
            self.convergence_threshold: float = convergence_threshold
            self.patience: int = patience
            self.consecutive_converged_iters: int = 0
            # learning rate and simulated annealing (both depending on the phase exploratory/smoothing)
            self.learning_rate_power: float = learning_rate_power
            self.annealing_factor: float = annealing_factor
            # meta
            self.verbose = verbose
            # pySAEM initialization
            self.history: Dict[str, List[torch.Tensor]] = {
                "log_MI": [],
                "population_betas": [],
                "population_omega": [],
                "residual_error_var": [],
            }
            # pre-compute design matrices once
            self.design_matrices = self._create_all_design_matrices()
            self.X_bar = (
                1
                / self.nb_individuals
                * torch.stack(
                    [self.design_matrices[ind_id] for ind_id in self.unique_ind_ids],
                    dim=2,
                ).sum(dim=2)
            )

        def _create_all_design_matrices(self) -> Dict[Union[str, int], torch.Tensor]:
            """Creates a design matrix for each unique individual based on their covariates."""
            design_matrices = {}
            if self.covariates_df is None:
                for ind_id in self.unique_ind_ids:
                    design_matrices[ind_id] = self._create_design_matrix(
                        covariates={}, model=self.model
                    )
            else:
                for ind_id in self.unique_ind_ids:
                    individual_covariates = self.covariates_df[
                        self.covariates_df["ind_id"] == ind_id
                    ].iloc[0]
                    covariates_dict = {
                        col: individual_covariates[col]
                        for col in self.covariates_df.columns
                        if col != "ind_id"
                    }
                    design_matrices[ind_id] = self._create_design_matrix(
                        covariates_dict, model=self.model
                    )
            return design_matrices

        def _create_design_matrix(
            self, covariates: Dict[str, float], model: NLMEModel
        ) -> torch.Tensor:
            """
            Creates the design matrix X_i for a single individual based on the model's covariate map.
            This matrix will be multiplied with population betas so that log(theta_i[PDU]) = X_i @ betas + eta_i.
            """
            design_matrix_X_i = torch.zeros(
                (model.nb_PDU, model.nb_betas), dtype=torch.float32
            )
            col_idx = 0
            for i, PDU_name in enumerate(model.PDU_names):
                design_matrix_X_i[i, col_idx] = 1.0
                col_idx += 1
                if model.covariate_map is not None:
                    for cov_name in model.covariate_map.get(PDU_name, []):
                        design_matrix_X_i[i, col_idx] = float(
                            covariates.get(cov_name, 0.0)
                        )
                        col_idx += 1
            return design_matrix_X_i

        def _check_convergence(self) -> bool:
            """Checks for convergence based on the relative change in parameters."""
            current_params = {
                "log_MI": self.log_MI,
                "population_betas": self.population_betas,
                "population_omega": self.population_omega,
                "residual_error_var": self.residual_error_var,
            }
            all_converged = True
            for name, current_val in current_params.items():
                prev_val = self.prev_params[name]
                abs_diff = torch.abs(current_val - prev_val)
                abs_sum = torch.abs(current_val) + torch.abs(prev_val) + 1e-9
                relative_change = abs_diff / abs_sum
                if torch.any(relative_change > self.convergence_threshold):
                    all_converged = False
                    break
            return all_converged

        def _update_learning_rate(self, iteration: int) -> float:
            """
            Calculates the SAEM learning rate (alpha_k).
            Phase 1: alpha_k = 1 (exploration)
            Phase 2: alpha_k = 1 / k_prime, with k_prime = (iteration - phase1_iterations + 1) (the iteration index in phase 2)
            """
            if iteration < self.nb_phase1_iterations:
                return 1.0
            else:
                k_prime: int = (
                    iteration - self.nb_phase1_iterations + 1
                )  # iteration index in phase 2
                return 1.0 / (k_prime**self.learning_rate_power)

        def iterate(self) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict[str, List[torch.Tensor]],
        ]:
            """
            This method handles the main E-step M-step loop.
            """
            current_iter = self.current_phase1_iteration + self.current_phase2_iteration
            total_iter_remaining = (
                self.nb_phase2_iterations + self.nb_phase1_iterations - current_iter
            )

            for i in tqdm(range(current_iter, current_iter + total_iter_remaining)):
                current_alpha_k: float = self._update_learning_rate(i)
                if self.verbose:
                    print(
                        f"\n--- SAEM iteration {i+1}/{(self.nb_phase1_iterations+self.nb_phase2_iterations)} (Alpha_k: {current_alpha_k:.3f}) ---"
                    )

                # --- E-Step: sample individual random effects (eta_i) ---
                scaling_factor: float = self.mcmc_proposal_var_scaling_factor
                current_mcmc_proposal_var_eta: torch.Tensor = (
                    scaling_factor * self.population_omega
                    + torch.eye(self.model.nb_PDU) * 1e-7
                )

                mcmc_sampler = MCMC_Eta_Sampler(
                    model=self.model,
                    log_MI=self.log_MI,
                    population_betas=self.population_betas,
                    population_omega=self.population_omega,
                    residual_error_var=self.residual_error_var,
                    proposal_var_eta=current_mcmc_proposal_var_eta,
                    verbose=self.verbose,
                    observations_df=self.observations_df,
                    design_matrices=self.design_matrices,
                )

                if i == 0:
                    self.mean_etas, mean_log_thetas_PDU, mean_predictions_df = (
                        mcmc_sampler.sample(
                            current_eta_for_all_ind=self.mean_etas,
                            nb_samples=self.mcmc_nb_samples,
                            nb_burn_in=self.mcmc_first_burn_in,
                        )
                    )
                else:
                    self.mean_etas, mean_log_thetas_PDU, mean_predictions_df = (
                        mcmc_sampler.sample(
                            current_eta_for_all_ind=self.mean_etas,
                            nb_samples=self.mcmc_nb_samples,
                            nb_burn_in=self.mcmc_burn_in,
                        )
                    )

                # --- M-Step: Update Population Means, Omega and Residual Error ---

                # 1. Update residual error variances
                sum_squared_residuals = torch.zeros(self.model.nb_outputs).unsqueeze(-1)
                total_observations = torch.zeros(self.model.nb_outputs)

                # Merge observations with mean predictions to calculate residuals
                merged_df = pd.merge(
                    self.observations_df,
                    mean_predictions_df,
                    on=["ind_id", "output_name", "time"],
                    how="left",
                )

                for output_name, output_df in merged_df.groupby("output_name"):
                    if output_name in self.model.outputs_names:
                        output_idx = self.model.outputs_names.index(output_name)
                        observed_data = torch.tensor(
                            output_df["value"].values, dtype=torch.float32
                        )
                        predictions = torch.tensor(
                            output_df["predicted_value"].values, dtype=torch.float32
                        )
                        if predictions is not None:
                            residuals = self.model.calculate_residuals(
                                observed_data, predictions
                            )
                            sum_squared_residuals[output_idx] += torch.sum(
                                torch.square(residuals)
                            )
                            total_observations[output_idx] += len(observed_data)

                total_observations[total_observations == 0] = 1e-9

                if (
                    self.model.error_model_type == "additive"
                    or self.model.error_model_type == "proportional"
                ):
                    target_res_var: torch.Tensor = (
                        sum_squared_residuals / total_observations.unsqueeze(1)
                    )
                    current_res_var: torch.Tensor = self.residual_error_var
                    if i < self.nb_phase1_iterations:
                        target_res_var = torch.max(
                            current_res_var * self.annealing_factor, target_res_var
                        )
                    self.residual_error_var = (
                        1 - current_alpha_k
                    ) * current_res_var + current_alpha_k * target_res_var

                # 2. Update Omega (covariance matrix of etas)
                sum_outer_product_etas: torch.Tensor = (
                    self.mean_etas.transpose(0, 1) @ self.mean_etas
                )
                target_omega: torch.Tensor = (
                    sum_outer_product_etas / self.nb_individuals
                )
                if i < self.nb_phase1_iterations:
                    updated_omega_diag = torch.diag(self.population_omega).clone()
                    target_omega_diag = torch.diag(target_omega)
                    for j in range(self.model.nb_PDU):
                        old_variance: float = updated_omega_diag[j].item()
                        new_target_variance: float = target_omega_diag[j].item()
                        annealed_target_variance: float = max(
                            old_variance * self.annealing_factor,
                            new_target_variance,
                        )
                        updated_omega_diag[j] = (
                            (1 - current_alpha_k) * old_variance
                            + current_alpha_k * annealed_target_variance
                        )
                    self.population_omega = torch.diag(updated_omega_diag)
                else:
                    self.population_omega = (
                        1 - current_alpha_k
                    ) * self.population_omega + current_alpha_k * target_omega
                self.population_omega = (
                    self.population_omega + self.population_omega.T
                ) / 2
                self.population_omega += (
                    torch.eye(self.population_omega.shape[0]) * 1e-6
                )
                self.population_omega_chol: torch.Tensor = torch.linalg.cholesky(
                    self.population_omega
                )

                # 3. Update fixed effects MIs
                def MI_objective_function(log_MI):
                    # Create a temporary input DataFrame with the new log_MI values
                    input_to_model = self.observations_df.copy()
                    input_to_model = input_to_model.drop(columns=["value"])

                    param_cols = self.model.MI_names + self.model.PDU_names

                    # Combine population MI parameters with sampled PDU parameters
                    params_dict_for_all_inds = {}
                    for i, ind_id in enumerate(self.unique_ind_ids):
                        new_params = torch.exp(
                            torch.cat(
                                (
                                    torch.tensor(log_MI).flatten(),
                                    mean_log_thetas_PDU[i].squeeze(),
                                )
                            )
                        )
                        params_dict_for_all_inds[ind_id] = (
                            new_params.detach().cpu().numpy()
                        )

                    for p_name_idx, p_name in enumerate(param_cols):
                        input_to_model[p_name] = input_to_model["ind_id"].map(
                            lambda x: params_dict_for_all_inds[x][p_name_idx]
                        )

                    predictions_df_obj = self.model.structural_model(input_to_model)
                    merged_df_obj = pd.merge(
                        self.observations_df,
                        predictions_df_obj,
                        on=["ind_id", "output_name", "time"],
                        how="left",
                    )

                    total_log_lik = 0.0
                    for output_name, output_df in merged_df_obj.groupby("output_name"):
                        if output_name in self.model.outputs_names:
                            output_idx = self.model.outputs_names.index(output_name)
                            observed_data = torch.tensor(
                                output_df["value"].values, dtype=torch.float32
                            )
                            predictions = torch.tensor(
                                output_df["predicted_value"].values, dtype=torch.float32
                            )
                            total_log_lik += self.model.log_likelihood_observation(
                                observed_data,
                                predictions,
                                self.residual_error_var[output_idx],
                            )
                    return -total_log_lik

                target_log_MI_np = minimize(
                    fun=MI_objective_function,
                    x0=self.log_MI.squeeze().numpy(),
                    method="L-BFGS-B",
                ).x
                target_log_MI = torch.from_numpy(target_log_MI_np).to(torch.float32)
                self.log_MI = (
                    1 - current_alpha_k
                ) * self.log_MI + current_alpha_k * target_log_MI

                # 4. Update fixed effects betas
                log_theta_bar = (
                    1
                    / self.nb_individuals
                    * torch.stack(
                        [
                            mean_log_thetas_PDU[ind]
                            for ind in range(self.nb_individuals)
                        ],
                        dim=1,
                    )
                    .sum(dim=1)
                    .unsqueeze(-1)
                )
                lhs_matrix: torch.Tensor = self.X_bar.T @ torch.cholesky_solve(
                    self.X_bar, self.population_omega_chol
                ) + 1e-6 * torch.eye(self.model.nb_betas)
                rhs_vector: torch.Tensor = self.X_bar.T @ torch.cholesky_solve(
                    log_theta_bar, self.population_omega_chol
                )

                target_beta: torch.Tensor = torch.linalg.solve(
                    lhs_matrix + 1e-6 * torch.eye(self.model.nb_betas),
                    rhs_vector,
                )
                self.population_betas = (
                    1 - current_alpha_k
                ) * self.population_betas + current_alpha_k * target_beta

                if self.verbose:
                    print(
                        f"  Updated MIs: {', '.join([f'{torch.exp(logMI).item():.4f}' for logMI in self.log_MI])}"
                    )
                    print(
                        f"  Updated Betas: {', '.join([f'{beta:.4f}' for beta in self.population_betas.detach().cpu().numpy().flatten()])}"
                    )
                    print(
                        f"  Updated Omega (diag): {', '.join([f'{val.item():.4f}' for val in torch.diag(self.population_omega)])}"
                    )
                    print(
                        f"  Updated Residual Var: {', '.join([f'{res_var:.4f}' for res_var in self.residual_error_var.detach().cpu().numpy().flatten()])}"
                    )

                # store history
                self.history["log_MI"].append(self.log_MI.clone())
                self.history["population_betas"].append(self.population_betas.clone())
                self.history["population_omega"].append(self.population_omega.clone())
                self.history["residual_error_var"].append(
                    self.residual_error_var.clone()
                )

                if i < self.nb_phase1_iterations:
                    self.current_phase1_iteration += 1
                else:
                    self.current_phase2_iteration += 1

                if i > 0:
                    is_converged = self._check_convergence()
                    if is_converged:
                        self.consecutive_converged_iters += 1
                        if self.verbose:
                            print(
                                f"Convergence met. Consecutive iterations: {self.consecutive_converged_iters}/{self.patience}"
                            )
                        if self.consecutive_converged_iters >= self.patience:
                            if self.verbose:
                                print(
                                    f"\nConvergence reached after {i + 1} iterations. Stopping early."
                                )
                            break
                    else:
                        self.consecutive_converged_iters = 0

                # update prev_params for the next iteration's convergence check
                self.prev_params: Dict[str, torch.Tensor] = {
                    "log_MI": self.log_MI.clone(),
                    "population_betas": self.population_betas.clone(),
                    "population_omega": self.population_omega.clone(),
                    "residual_error_var": self.residual_error_var.clone(),
                }

            print("\nEstimation Finished.")
            idx: int = 0
            self.estimated_MI_mus = []
            for j in range(self.model.nb_MI):
                self.estimated_MI_mus.append(torch.exp(self.log_MI[j]).item())
            for PDU_name in self.model.PDU_names:
                self.estimated_MI_mus.append(
                    torch.exp(self.population_betas[idx]).item()
                )
                idx += 1
                if self.model.covariate_map and PDU_name in self.model.covariate_map:
                    for i in range(len(self.model.covariate_map[PDU_name])):
                        idx += 1

            if self.verbose:
                print(
                    f"Estimated MI: {', '.join([f'{MI:.4f}' for MI in self.estimated_MI_mus[:self.model.nb_MI]])}"
                )
                print(
                    f"Estimated mus: {', '.join([f'{mu:.4f}' for mu in self.estimated_MI_mus[self.model.nb_MI:]])}"
                )
                print(
                    f"Estimated population betas: {', '.join([f'{estimated_beta.item():.4f}' for estimated_beta in self.population_betas])}"
                )
                print(
                    f"Estimated omega (diagonal): {', '.join([f'{val.item():.4f}' for val in torch.diag(self.population_omega)])}"
                )
                print(
                    f"Estimated residual var: {', '.join([f'{val.item():.4f}' for val in self.residual_error_var.flatten()])}"
                )

            return (
                self.estimated_MI_mus,
                self.population_betas,
                self.population_omega,
                self.residual_error_var,
                self.history,
            )

        def run(
            self,
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict[str, List[torch.Tensor]],
        ]:
            """
            This method starts the SAEM estimation by initiating some class attributes then calling the iterate method.
            returns self.population_betas, self.estimated_population_mus, self.population_omega, self.residual_error_var, self.history
            stores the current state of the estimation so that the iterations can continue later with the continue_iterating method.
            """
            if self.verbose:
                print("Starting SAEM Estimation...")
                print(
                    f"Initial Population Betas: {', '.join([f'{beta.item():.2f}' for beta in self.population_betas])}"
                )
                print(
                    f"Initial Population MIs: {', '.join([f'{torch.exp(logMI).item():.2f}' for logMI in self.log_MI])}"
                )
                print(f"Initial Omega:\n{self.population_omega}")
                print(f"Initial Residual Variance: {self.residual_error_var}")

            self.history["log_MI"].append(self.log_MI.clone())
            self.history["population_betas"].append(self.population_betas.clone())
            self.history["population_omega"].append(self.population_omega.clone())
            self.history["residual_error_var"].append(self.residual_error_var.clone())

            self.prev_params: Dict[str, torch.Tensor] = {
                "log_MI": self.log_MI.clone(),
                "population_betas": self.population_betas.clone(),
                "population_omega": self.population_omega.clone(),
                "residual_error_var": self.residual_error_var.clone(),
            }
            self.mean_etas: torch.Tensor = torch.zeros(
                (self.nb_individuals, self.model.nb_PDU),
                dtype=torch.float32,
            )
            return self.iterate()

        def continue_iterating(
            self, nb_phase1_further_iterations=0, nb_phase2_further_iterations=0
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict[str, List[torch.Tensor]],
        ]:
            """
            This method is to be used when the run method has already run and the user wants to further iterate.
            It updates the iterations necessary and calls the iterate method.
            returns self.population_betas, self.estimated_population_mus, self.population_omega, self.residual_error_var, self.history
            """
            if nb_phase1_further_iterations > 0 and self.current_phase2_iteration > 0:
                raise ValueError(
                    "Phase 2 (smoothing) has already started. No further phase1 iterations can be conducted."
                )
            self.nb_phase1_iterations += nb_phase1_further_iterations
            self.nb_phase2_iterations += nb_phase2_further_iterations
            return self.iterate()

        def plot_convergence_history(
            self,
            true_MI: Optional[torch.Tensor] = None,
            true_betas: Optional[torch.Tensor] = None,
            true_omega: Optional[torch.Tensor] = None,
            true_residual_var: Optional[torch.Tensor] = None,
        ):
            """
            This method plots the evolution of the estimated parameters (MI, betas, omega, residual error variances) across iterations
            """
            history: Dict[str, List[Union[torch.Tensor, float]]] = self.history
            nb_MI: int = self.model.nb_MI
            nb_betas: int = self.model.nb_betas
            nb_omega_diag_params: int = self.model.nb_PDU
            nb_var_res_params: int = self.model.nb_outputs
            fig, axs = plt.subplots(
                nb_MI + nb_betas + nb_omega_diag_params + nb_var_res_params,
                1,
                figsize=(
                    10,
                    4 * (nb_betas + nb_omega_diag_params + nb_var_res_params),
                ),
            )
            plot_idx: int = 0
            colors = plt.cm.get_cmap("tab10", nb_MI)
            for j, MI_name in enumerate(self.model.MI_names):
                MI_history = [torch.exp(h[j]).item() for h in history["log_MI"]]
                axs[plot_idx].plot(
                    MI_history,
                    label=f"Estimated MI for {MI_name} ",
                    color=colors(j),
                )
                if true_MI is not None and true_MI.shape[0] > j:
                    axs[plot_idx].axhline(
                        y=true_MI[j].item(),
                        color=colors(j),
                        linestyle="--",
                        label=f"True MI for {MI_name}",
                    )
                axs[plot_idx].set_title(f"Convergence of MI ${{{MI_name}}}$")
                axs[plot_idx].set_xlabel("SAEM Iteration")
                axs[plot_idx].set_ylabel("Parameter Value")
                axs[plot_idx].legend()
                axs[plot_idx].grid(True)
                plot_idx += 1
            for j, beta_name in enumerate(self.model.population_betas_names):
                beta_history = [h[j].item() for h in history["population_betas"]]
                axs[plot_idx].plot(
                    beta_history,
                    label=f"Estimated beta for {beta_name} ",
                    color=colors(j),
                )
                if true_betas is not None and true_betas.shape[0] > j:
                    axs[plot_idx].axhline(
                        y=true_betas[j].item(),
                        color=colors(j),
                        linestyle="--",
                        label=f"True beta for {beta_name}",
                    )
                axs[plot_idx].set_title(f"Convergence of beta_${{{beta_name}}}$")
                axs[plot_idx].set_xlabel("SAEM Iteration")
                axs[plot_idx].set_ylabel("Parameter Value")
                axs[plot_idx].legend()
                axs[plot_idx].grid(True)
                plot_idx += 1
            for j, PDU_name in enumerate(self.model.PDU_names):
                omega_diag_history = [
                    h[j, j].item() for h in history["population_omega"]
                ]
                axs[plot_idx].plot(
                    omega_diag_history,
                    label=f"Estimated Omega for {PDU_name}",
                    color=colors(j),
                )
                if true_omega is not None and true_omega.shape[0] > j:
                    axs[plot_idx].axhline(
                        y=true_omega[j, j].item(),
                        color=colors(j),
                        linestyle="--",
                        label=f"True Omega for {PDU_name}",
                    )
                axs[plot_idx].set_title(f"Convergence of Omega for {PDU_name}")
                axs[plot_idx].set_xlabel("SAEM Iteration")
                axs[plot_idx].set_ylabel("Variance")
                axs[plot_idx].legend()
                axs[plot_idx].grid(True)
                plot_idx += 1
            for j, res_name in enumerate(self.model.outputs_names):
                var_res_history = [h[j].item() for h in history["residual_error_var"]]
                axs[plot_idx].plot(
                    var_res_history,
                    label=f"Estimated residual error variance for {res_name}",
                    color=colors(j),
                )
                if true_residual_var is not None and true_residual_var.shape[0] > j:
                    axs[plot_idx].axhline(
                        y=true_residual_var[j].item(),
                        color=colors(j),
                        linestyle="--",
                        label=f"True residual variance for {res_name}",
                    )
                axs[plot_idx].set_title("Residual Error var Convergence")
                axs[plot_idx].set_xlabel("SAEM Iteration")
                axs[plot_idx].set_ylabel("var Value")
                axs[plot_idx].legend()
                axs[plot_idx].grid(True)
                plot_idx += 1
            plt.tight_layout()
            plt.show()
