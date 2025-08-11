import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Dict, Union, Callable, Optional, Tuple
import GP

torch.set_default_dtype(torch.float32)
with torch.no_grad():
    # --- 1. NLME Model Definition ---
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

        def individual_parameters(
            self,
            population_MI: torch.Tensor,
            population_betas: torch.Tensor,
            individual_etas: torch.Tensor,
            design_matrices_X_i: Union[torch.Tensor, List[torch.Tensor]],
        ) -> List[torch.Tensor]:
            """
            Transforms MIs (Model intrinsic), betas: log(mu)s & coeffs for covariates and individual random effects (etas) into individual parameters (theta_i), for each set of etas of the list and corresponding design matrix.
            Assumes log-normal distribution for individual parameters and covariate effects: theta_i[PDU] = mu_pop * exp(eta_i) * exp(covariates_i * cov_coeffs) where eta_i is from N(0, Omega) and theta_i[MI]=MI.
            population_MI: torch.Tensor of MI model parameters (i.e. parameters without InterIndividual Variability)
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
            if len(individual_etas[0]) != self.nb_PDU:
                raise ValueError(
                    "Dimension mismatch between individual_etas and the number of population parameters."
                )
            if self.covariate_map is None:
                raise ValueError(
                    "Covariate map must be defined in order to compute individual parameters."
                )

            if population_betas.dim() > 1:
                population_betas = population_betas.squeeze()
            # compute the individual parameters one set after the other
            individual_params_list: List[torch.Tensor] = []
            for i in range(len(individual_etas)):

                eta = individual_etas[i].unsqueeze(-1)
                log_individual_thetas_PDU: torch.Tensor = (
                    design_matrices_X_i[i] @ population_betas.unsqueeze(-1)
                ) + eta
                individual_thetas_PDU: torch.Tensor = torch.exp(
                    log_individual_thetas_PDU
                )  # dim: [nb_population_params]

                individual_parameters: torch.Tensor = torch.concat(
                    (population_MI.squeeze(-1), individual_thetas_PDU.squeeze())
                )
                individual_params_list.append(individual_parameters)
            return individual_params_list

        def calculate_residuals(
            self, observed_data: torch.Tensor, predictions: torch.Tensor
        ) -> torch.Tensor:
            """
            Calculates residuals based on the error model.
            observed_data: torch.Tensor of observations for one individual. dim: [nb_outputs x nb_time_points]
            predictions: torch.Tensor of predictions for one individual. Must be organized like observed_data to compare both by subtraction, dim: [nb_outputs x time_steps]
            """
            if (
                self.error_model_type == "additive"
            ):  # right now the only one implemented, but may evolve in the future
                return observed_data - predictions
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
            log_lik: float = (
                -0.5
                * torch.sum(
                    torch.log(2 * torch.pi * res_error_var)
                    + (residuals**2 / res_error_var)
                ).item()
            )  # each row of res_error_var is the variance of the residual error corresponding to the same row of residuals (one row = one output)
            return log_lik

    class NLMEModel_from_struct_model(NLMEModel):
        def __init__(
            self,
            MI_names: List[str],
            PDU_names: List[str],
            cov_coeffs_names: List[str],
            outputs_names: List[str],
            structural_model: Callable[
                [List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]
            ],
            error_model_type: str = "additive",  # the only one implemented for now
            covariate_map: Optional[Dict[str, List[str]]] = None,
        ):
            super().__init__(
                MI_names,
                PDU_names,
                cov_coeffs_names,
                outputs_names,
                error_model_type,
                covariate_map,
            )
            self.structural_model: Callable[
                [List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]
            ] = structural_model

    class NLMEModel_from_GP(NLMEModel):
        def __init__(
            self,
            MI_names: List[str],
            PDU_names: List[str],
            cov_coeffs_names: List[str],
            outputs_names: List[str],
            GP_model: GP,
            error_model_type: str = "additive",  # the only one implemented for now
            covariate_map: Optional[Dict[str, List[str]]] = None,
        ):
            super().__init__(
                MI_names,
                PDU_names,
                cov_coeffs_names,
                outputs_names,
                error_model_type,
                covariate_map,
            )
            self.GP_model: GP = GP_model

            def structural_model(
                list_time_steps: List[torch.Tensor], list_params: List[torch.Tensor]
            ) -> List[torch.Tensor]:
                # extend time_steps_list if there is only one time_steps by repeating it
                if len(list_time_steps) == 1:
                    list_time_steps = list_time_steps * len(list_params)

                # Step 1: create a torch.Tensor handled by GP, i.e. columns are parameters and time (each set of parameters is repeated with all the different time of its time_steps)
                list_tensors: List[torch.Tensor] = []
                for j, params in enumerate(list_params):
                    repeated_params: torch.Tensor = params.repeat(
                        len(list_time_steps[j]), 1
                    )
                    reshaped_time_steps: torch.Tensor = list_time_steps[j].reshape(
                        -1, 1
                    )
                    combined_tensor: torch.Tensor = torch.cat(
                        [repeated_params, reshaped_time_steps], dim=1
                    )
                    list_tensors.append(combined_tensor)
                final_input: torch.Tensor = self.GP_model.normalize_inputs(
                    torch.cat(list_tensors, dim=0)
                )

                # Step 2: predict outputs with the GP
                mean, lower, upper = self.GP_model.predict_scaled(final_input)
                # if confidence region too big, retrain model here (to be implemented)
                # reshape results horizontally
                mean = torch.Tensor(mean).T

                # Step 3: transform back into a list of torch.Tensor solutions
                lengths = [len(t) for t in list_time_steps]
                list_solutions: List[torch.Tensor] = list(
                    torch.split(mean, lengths, dim=1)
                )
                return list_solutions

            self.structural_model: Callable[
                [List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]
            ] = structural_model

    # --- 2. Data Handling ---
    class IndividualData:
        """
        This class is mainly a data structure for each individual's observations with time_points and covariates.
        The design matrix of the individual can be created and stored with create_design_matrix. Its structure depends on the NLMEModel to allow correct matrix multiplication with betas, and the only info it contains are the covariates of the individual.
        """

        def __init__(
            self,
            id: Union[str, int],
            time: torch.Tensor,
            observations: torch.Tensor,
            covariates: Optional[Dict[str, float]] = None,
        ):
            self.id = id
            self.time: torch.Tensor = time  # dim: [nb_time_points]
            self.observations: Optional[torch.Tensor] = (
                observations  # recommended dim: [nb_outputs x nb_time_points]
            )
            self.covariates: Dict[str, float] = (
                covariates if covariates is not None else {}
            )
            self.design_matrix_X_i: Optional[torch.Tensor] = (
                None  # dim: [nb_PDU x nb_betas]
            )

        def create_design_matrix(
            self,
            model: NLMEModel,
        ):
            """
            Creates the design matrix X_i for this individual based on the model's covariate map. It will be multiplied with the population betas so that theta_i[PDU] = X @ betas + etas_i
            Each row corresponds to one PDU and contains [0, 0... ,1, cov1_val, cov2_val, 0...],
            i.e.: 1 in the column corresponding to log(mu) in betas. For each covariate influencing the PDU, the value of this covariate for this individual in the column corresponding to the coeff of this covariate in betas. Else zeros.
            model: the NLMEModel determining the structure of the matrix
            """

            design_matrix_X_i: torch.Tensor = torch.zeros(
                (model.nb_PDU, model.nb_betas), dtype=torch.float32
            )  # dim: [nb_population_params x nb_betas]

            col_idx: int = 0
            for i, PDU_name in enumerate(model.PDU_names):
                # Intercept term for log(mu)
                design_matrix_X_i[i, col_idx] = 1.0
                col_idx += 1

                # Covariate effects
                if model.covariate_map is not None:
                    for cov_name in model.covariate_map.get(PDU_name, []):
                        if cov_name in self.covariates:
                            design_matrix_X_i[i, col_idx] = float(
                                self.covariates[cov_name]
                            )
                        else:
                            design_matrix_X_i[i, col_idx] = 0.0
                        col_idx += 1
            self.design_matrix_X_i = design_matrix_X_i

    # --- 3. MCMC Sampler for Individual Random Effects (eta_i), used in the E-step of SAEM ---
    class MCMC_Eta_Sampler:  # one independant markov chain per individual
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
            population_MI: torch.Tensor,
            population_betas: torch.Tensor,
            population_omega: torch.Tensor,
            residual_error_var: torch.Tensor,
            proposal_var_eta: torch.Tensor,
            verbose: bool,
        ):
            self.model: NLMEModel = model
            self.population_betas: torch.Tensor = (
                population_betas.clone()
            )  # dim: [nb_betas]
            self.population_MI: torch.Tensor = population_MI.clone()  # dim: [nb_MI]
            self.population_omega: torch.Tensor = (
                population_omega.clone()
            )  # dim: [nb_PDU x nb_PDU], nb_PDU = nb_etas, omega is the covariance matrix for eta
            self.residual_error_var: torch.Tensor = (
                residual_error_var  # dim: [nb_outptus]
            )
            self.proposal_var_eta: torch.Tensor = (
                proposal_var_eta.clone()
            )  # dim: [nb_PDU x nb_PDU], covariance used when sampling a new eta from a multivariate normal distribution from the current position in the Markov Chain
            self.verbose = verbose

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
            self, etas: torch.Tensor, ind_data_list: List[IndividualData]
        ) -> torch.Tensor:
            """
            Calculates the log-posterior for MCMC for all the given eta_i.
            log(P(eta_i | y_i, mu, Omega, var_res)) = log(P(y_i | theta_i(eta_i))) + log(P(eta_i | Omega)), (notice the second term is the prior of eta, the first one is the log_likelihood of observations given predictions)
            etas: torch.Tensor of dim [nb_eta_i x nb_PDU]
            ind_data_list: a list of IndividualData with the associated design matrices. Must be of length nb_eta_i.
            """
            # errors
            for ind in ind_data_list:
                if ind.observations is None:
                    raise TypeError(
                        "Observations for an individual cannot be None for PysAEM to work."
                    )
                if ind.design_matrix_X_i is None:
                    raise TypeError(
                        "Design Matrix could not be created for an individual."
                    )

            # for each eta_i, transform eta_i into individual parameters theta_i[PDU]
            list_individual_params: List[torch.Tensor] = (
                self.model.individual_parameters(
                    self.population_MI,
                    self.population_betas,
                    individual_etas=etas,
                    design_matrices_X_i=[
                        ind_data.design_matrix_X_i for ind_data in ind_data_list
                    ],
                )
            )  # of dim List[torch.Tensor[nb_PDU]]

            # predict observations using the structural model and the previously computed individual parameters
            times = [ind_data_list[i].time for i in range(len(ind_data_list))]
            predictions: List[torch.Tensor] = self.model.structural_model(
                times,
                list_individual_params,
            )
            # calculate log_likelihoods of observations given predictions
            list_log_lik_obs: List[float] = [
                self.model.log_likelihood_observation(
                    ind_data.observations, predictions[i], self.residual_error_var
                )
                for i, ind_data in enumerate(ind_data_list)
            ]
            log_lik_obs = torch.tensor(list_log_lik_obs)

            # calculate log_priors
            log_priors: torch.Tensor = self._log_prior_etas(etas)

            # log_posterior is the sum of both
            return log_lik_obs + log_priors

        def sample(
            self,
            list_ind_data: List[IndividualData],
            current_eta_for_all_ind: torch.Tensor,
            nb_samples: int,
            nb_burn_in: int,
        ) -> torch.Tensor:
            """
            Performs Metropolis-Hastings sampling for the individuals'etas. The MCMC chains (one per individual) advance in parallel.
            The acceptance criterion for each sample is log(random_uniform) < proposed_log_posterior − current_log_posterior.
            Returns samples for all individuals after burn-in.
            list_ind_data: List[IndividualData]
            current_eta_for_all_ind: torch.Tensor of dim [nb_individuals x nb_parameters]
            nb_samples: int, how many samples will be kept from each chain
            nb_burn_in: int, how many accepted samples are disgarded before we consider that the chain has converged enough
            """
            nb_individuals = len(list_ind_data)
            all_states_history: List[torch.Tensor] = []

            samples: List[List[torch.Tensor]] = [[] for _ in range(nb_individuals)]

            current_log_posteriors: torch.Tensor = self._log_posterior_etas(
                current_eta_for_all_ind, list_ind_data
            )

            accepted_counts = torch.zeros(nb_individuals)
            total_proposals = torch.zeros(nb_individuals)

            done = torch.full((nb_individuals, 1), False)
            while not torch.all(done).item():
                active_indices = torch.where(~done)[0]
                total_proposals[active_indices] += 1
                proposal_dist = torch.distributions.MultivariateNormal(
                    current_eta_for_all_ind[active_indices], self.proposal_var_eta
                )
                proposed_etas: torch.Tensor = (
                    proposal_dist.sample()
                )  # dim [nb_active x nb_PDU]
                proposed_log_posteriors: torch.Tensor = self._log_posterior_etas(
                    proposed_etas, [list_ind_data[i] for i in active_indices]
                )
                deltas: torch.Tensor = (
                    proposed_log_posteriors - current_log_posteriors[active_indices]
                )
                accept_mask: torch.Tensor = (
                    torch.log(torch.rand(len(active_indices))) < deltas
                )  # Dim: [num_active_individuals]
                current_eta_for_all_ind[active_indices] = torch.where(
                    accept_mask.unsqueeze(-1),
                    proposed_etas,
                    current_eta_for_all_ind[active_indices],
                )
                current_log_posteriors[active_indices] = torch.where(
                    accept_mask,
                    proposed_log_posteriors,
                    current_log_posteriors[active_indices],
                )
                # try this instead of torch where
                # current_eta_for_all_ind[active_indices[accept_mask]] = proposed_etas[accept_mask]
                # current_log_posteriors[active_indices[accept_mask]] = proposed_log_posteriors[accept_mask]

                accepted_counts[active_indices] += accept_mask.int()

                for idx in active_indices[accept_mask]:
                    if accepted_counts[idx] >= nb_burn_in:
                        samples[idx].append(current_eta_for_all_ind[idx].clone())
                        if len(samples[idx]) == nb_samples:
                            done[idx] = True
                all_states_history.append(current_eta_for_all_ind.clone())

            if self.verbose == True:
                acceptance_rate = accepted_counts / total_proposals
                for i in range(nb_individuals):
                    print(
                        f"  MCMC Acceptance Rate for individual {i}: {acceptance_rate[i]:.2f}"
                    )

            stacked_samples = [torch.stack(s) for s in samples]

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

            return torch.stack(stacked_samples)

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
            list_individual_data: List[IndividualData],
            initial_pop_MI: torch.Tensor,
            initial_pop_betas: torch.Tensor,
            initial_pop_omega: torch.Tensor,
            initial_res_var: torch.Tensor,
            # MCMC parameters for the E-step
            mcmc_burn_in: int = 3,
            mcmc_first_burn_in: int = 30,
            mcmc_nb_samples: int = 10,
            mcmc_proposal_var_scaling_factor: float = 0.2,
            nb_saem_iterations: int = 100,
            saem_phase1_iterations: Union[
                int, None
            ] = None,  # initial exploration (alpha_k = 1)
            saem_learning_rate_power: float = 0.8,
            saem_annealing_factor: float = 0.95,
            verbose: bool = False,
        ):

            self.model: NLMEModel = model
            self.list_individual_data: List[IndividualData] = list_individual_data
            self.nb_individuals: int = len(list_individual_data)
            # population parameters betas to be updated during the SAEM iterations
            self.population_MI: torch.Tensor = initial_pop_MI.clone()  # dim: [nb_MI]
            self.population_betas: torch.Tensor = (
                initial_pop_betas.clone()
            )  # dim: [nb_betas]
            # for printing to the user, a Tensor with the values for the PDUs and Mis (no log), completed at the end of the iterations
            self.estimated_population_MI_mus: Optional[List[float]] = None
            # population parameters MI (model intrinsic) to be updated during the SAEM iterations, without IIV/individual effects
            # covariance matrix of the individual random effects etas
            self.population_omega: torch.Tensor = (
                initial_pop_omega.clone()  # [nb_PDU x nb_PDU], nb_PDU = nb_etas, omega is the covariance matrix for etas
            )
            # residual error variance
            self.residual_error_var: torch.Tensor = (
                initial_res_var  # variance of the additive error on the observations
            )

            # MCMC parameters for the E-step
            self.mcmc_first_burn_in: int = (
                mcmc_first_burn_in  # number of disgarded samples per chain in the first iteration, where the chain starts at zero.
            )
            self.mcmc_burn_in: int = (
                mcmc_burn_in  # number of disgarded samples per chain
            )
            self.mcmc_nb_samples: int = mcmc_nb_samples
            self.mcmc_proposal_var_scaling_factor: float = (
                mcmc_proposal_var_scaling_factor  # the variance of the multivariate normal distribution that the next eta from the Markov Chain is sampled from is scaling_factor * omega
            )

            # SAEM Iteration Parameters
            if nb_saem_iterations < 2:
                print(
                    "There must be at least two iterations, more are recommended. Switching to 2."
                )
                self.nb_saem_iterations = 2
            else:
                self.nb_saem_iterations: int = nb_saem_iterations
            self.saem_phase1_iterations: int = (
                saem_phase1_iterations
                if saem_phase1_iterations is not None
                else nb_saem_iterations // 2
            )
            self.saem_phase2_iterations: int = (
                self.nb_saem_iterations - self.saem_phase1_iterations
            )

            self.saem_learning_rate_power: float = saem_learning_rate_power
            self.saem_annealing_factor: float = (
                saem_annealing_factor  # for the updates of omega and residual_var in phase 2
            )

            self.verbose = verbose
            self.history: Dict[str, List[torch.Tensor]] = {
                "population_MI": [],
                "population_betas": [],
                "population_omega": [],
                "residual_error_var": [],
            }

            # to switch from individual id to an array index
            self.ind_id_to_idx: Dict[Union[str, int], int] = {
                ind_data.id: i for i, ind_data in enumerate(self.list_individual_data)
            }

        def _update_learning_rate(self, iteration: int) -> float:
            """
            Calculates the SAEM learning rate (alpha_k) (stochastic part of SAEM)
            Phase 1: alpha_k = 1 (exploration)
            Phase 2: alpha_k = 1 / k_prime, with k_prime = (iteration - phase1_iterations + 1) (the iteration index in phase 2)
            iteration: int
            """
            if iteration < self.saem_phase1_iterations:
                return 1.0
            else:
                k_prime: int = (
                    iteration - self.saem_phase1_iterations + 1
                )  # iteration index in phase 2
                return 1.0 / (k_prime**self.saem_learning_rate_power)

        def run(
            self,
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict[str, List[torch.Tensor]],
        ]:
            # this method handles the main SAEM loop with E-step and M-step
            # returns self.population_betas, self.estimated_population_mus, self.population_omega, self.residual_error_var, self.history
            print("Starting SAEM Estimation...")
            print(
                f"Initial Population Betas: {", ".join([f"{beta.item():.2f}" for beta in self.population_betas])}"
            )
            print(
                f"Initial Population MIs: {", ".join([f"{MI.item():.2f}" for MI in self.population_MI])}"
            )
            print(f"Initial Omega:\n{self.population_omega}")
            print(f"Initial Residual Variance: {self.residual_error_var}")

            # store initial parameters
            self.history["population_MI"].append(self.population_MI.clone())
            self.history["population_betas"].append(self.population_betas.clone())
            self.history["population_omega"].append(self.population_omega.clone())
            self.history["residual_error_var"].append(self.residual_error_var.clone())

            # create design matrices for each individual
            for ind_data in self.list_individual_data:
                ind_data.create_design_matrix(self.model)

            # main loop
            for i in range(self.nb_saem_iterations):
                current_alpha_k: float = self._update_learning_rate(i)
                print(
                    f"\n--- SAEM Iteration {i+1}/{self.nb_saem_iterations} (Alpha_k: {current_alpha_k:.3f}) ---"
                )

                # --- E-Step: sample individual random effects (eta_i) ---
                # Run MCMC for each individual, sampling from the log-posterior distribution of etas knowing the observations for this individual (due to their true etas)

                # compute the covariance of the multivariate normal distribution the next eta will be sampled from in the chain
                scaling_factor: float = self.mcmc_proposal_var_scaling_factor
                current_mcmc_proposal_var_eta: torch.Tensor = (
                    scaling_factor * self.population_omega
                    + torch.eye(self.model.nb_PDU)
                    * 1e-7  # add small jitter for numerical stability
                )  # dim: [nb_PDU x nb_PDU]

                # initialize storage
                mean_etas: torch.Tensor = torch.zeros(
                    (self.nb_individuals, self.model.nb_PDU),
                    dtype=torch.float32,
                )  # dim: [nb_individuals x nb_PDU]
                individual_predictions: Dict[int, torch.Tensor] = (
                    {}
                )  # associates the individual index with the mean of the predictions made with the sampled etas for this individual (stores torch.Tensor of predictions for each individual)
                mean_log_thetas_for_all_ind: Dict[int, torch.Tensor] = (
                    {}
                )  # associates the individual index with the mean of the log of thetas calculated with the sampled etas for this individual (stores torch.Tensor [nb_PDU] of mean_log_thetas for each individual)

                mcmc_sampler = MCMC_Eta_Sampler(
                    model=self.model,
                    population_MI=self.population_MI,
                    population_betas=self.population_betas,
                    population_omega=self.population_omega,
                    residual_error_var=self.residual_error_var,
                    proposal_var_eta=current_mcmc_proposal_var_eta,
                    verbose=self.verbose,
                )

                if (
                    i == 0
                ):  # different (higher recommended) burn in for first iter, because the chain starts with zero etas
                    eta_samples: torch.Tensor = mcmc_sampler.sample(
                        list_ind_data=self.list_individual_data,
                        current_eta_for_all_ind=mean_etas,
                        nb_samples=self.mcmc_nb_samples,
                        nb_burn_in=self.mcmc_first_burn_in,
                    )  # dim: [nb_ind x nb_samples x nb_PDU]
                else:  # different (recommended reduced) burn-in for the rest of the iterations, because the chain starts at the eta of the previous iter
                    eta_samples: torch.Tensor = mcmc_sampler.sample(
                        list_ind_data=self.list_individual_data,
                        current_eta_for_all_ind=mean_etas,
                        nb_samples=self.mcmc_nb_samples,
                        nb_burn_in=self.mcmc_burn_in,
                    )  # dim: [nb_ind x nb_samples x nb_PDU]

                # store the mean of the sampled etas for each individual
                mean_etas: torch.Tensor = torch.mean(
                    eta_samples, dim=1
                )  # dim: [nb_ind x nb_PDU]

                # for the residual error update in the M-step, for each individual, calculate predictions based on each sampled eta_j, average them and store them in individual_predictions
                for ind_data in self.list_individual_data:
                    ind_idx = self.ind_id_to_idx[ind_data.id]
                    eta_samples_for_ind = eta_samples[ind_idx, :, :]
                    thetas_for_ind: List[torch.Tensor] = (
                        self.model.individual_parameters(
                            self.population_MI,
                            self.population_betas,
                            eta_samples_for_ind,
                            ind_data.design_matrix_X_i,
                        )
                    )  # dim: List[nb_betas] of length nb_samples
                    log_thetas_for_ind = torch.log(
                        torch.stack(thetas_for_ind)
                    )  # dim: [nb_samples, nb_betas]
                    mean_log_thetas_for_all_ind[ind_idx] = torch.mean(
                        log_thetas_for_ind, dim=0
                    )
                    # Calculate predictions with the computed thetas
                    preds_for_ind: List[torch.Tensor] = self.model.structural_model(
                        [ind_data.time], thetas_for_ind
                    )  # Expected dim: List[nb_outputs, nb_time_steps] of length nb_samples
                    individual_predictions[ind_idx] = torch.mean(
                        torch.stack(preds_for_ind), dim=0
                    )

                # --- M-Step: Update Population Means, Omega and Residual Error ---

                # 1. Update fixed effects MIs (Model Intrinsic)
                def MI_objective_function(MI):
                    all_times: List[torch.Tensor] = [
                        ind.time for ind in self.list_individual_data
                    ]
                    all_obs: List[torch.Tensor] = [
                        ind.observations for ind in self.list_individual_data
                    ]

                    # Combine MI with betas to form a single parameter vector for structural model
                    list_individual_params_for_minimization = [
                        torch.cat(
                            (
                                torch.Tensor(MI),
                                torch.exp(
                                    mean_log_thetas_for_all_ind[i][self.model.nb_MI :]
                                ),
                            ),
                            dim=0,
                        )
                        for i in range(self.nb_individuals)
                    ]
                    predictions = self.model.structural_model(
                        all_times,
                        list_individual_params_for_minimization,
                    )

                    total_log_lik = 0.0
                    for i in range(self.nb_individuals):
                        total_log_lik += self.model.log_likelihood_observation(
                            all_obs[i], predictions[i], self.residual_error_var
                        )
                    return -total_log_lik

                target_MI_np = minimize(
                    fun=MI_objective_function,
                    x0=self.population_MI.squeeze(),
                    method="L-BFGS-B",
                    # options={"maxiter": 5, "maxls": 3},
                ).x
                target_MI = torch.from_numpy(target_MI_np).to(torch.float32)

                # update fixed effects MI (Model Intrinsic) using stochastic approximation
                self.population_MI = (
                    1 - current_alpha_k
                ) * self.population_MI + current_alpha_k * target_MI  # dim: [nb_MI]

                # 2. Update fixed effects betas ; i.e. logs of populations means mus and population coefficients for covariates (covariates = PDKs that influence certain individual parameters according to population coeffs)
                # theta_i = mu * exp(eta_i) * exp(coeffs * covariates_i)
                # So we can write log(theta_i) = log(mu) + eta_i + coeffs * covariates_i = X_i @ betas + etas_i
                # from which we derive the normal equations (sum_i X_i.T @ inv_population_omega @ X_i) @ population_betas = sum_i X_i.T @ inv_population_omega @ mean_log_theta_i
                # we work with the sums over all individuals to update population_betas

                sum_lhs_matrix: torch.Tensor = torch.zeros(
                    (self.model.nb_betas, self.model.nb_betas), dtype=torch.float32
                )  # dim [nb_betas x nb_betas]
                sum_rhs_vector: torch.Tensor = torch.zeros(
                    (self.model.nb_betas, 1), dtype=torch.float32
                )  # dim [nb_betas]

                for ind_idx in range(self.nb_individuals):
                    X_i: torch.Tensor = self.list_individual_data[
                        ind_idx
                    ].design_matrix_X_i  # dim [nb_PDU x nb_betas]
                    sum_lhs_matrix += X_i.T @ torch.linalg.solve(
                        self.population_omega, X_i
                    )
                    sum_rhs_vector += X_i.T @ torch.linalg.solve(
                        self.population_omega,
                        mean_log_thetas_for_all_ind[ind_idx][
                            self.model.nb_MI :
                        ].unsqueeze(-1),
                    )

                target_beta: torch.Tensor = torch.linalg.solve(
                    sum_lhs_matrix
                    + 1e-6
                    * torch.eye(
                        self.model.nb_betas
                    ),  # Add small diagonal for numerical stability
                    sum_rhs_vector,
                )  # dim [nb_betas]

                # update fixed effects (betas) using stochastic approximation
                self.population_betas = (
                    1 - current_alpha_k
                ) * self.population_betas + current_alpha_k * target_beta  # dim: [nb_betas]

                # 3. Update Omega (covariance matrix of eta)
                # E[eta_i * eta_i.T] is approximated by the average of (eta_i_sampled * eta_i_sampled.T) for each individual and then averaged over individuals
                sum_outer_product_etas: torch.Tensor = torch.zeros_like(
                    self.population_omega
                )  # dim: [nb_PDU x nb_PDU]
                for eta_i_mean in mean_etas:  # eta_i_mean dim: [nb_PDU]
                    sum_outer_product_etas += torch.outer(
                        eta_i_mean, eta_i_mean
                    )  # outer product dim: [nb_PDU x nb_PDU]
                target_omega: torch.Tensor = (
                    sum_outer_product_etas / self.nb_individuals
                )  # dim: [nb_PDU x nb_PDU]

                # simulated annealing for each diagonal element of omega in Phase 1
                if i < self.saem_phase1_iterations:
                    updated_omega_diag = torch.diag(self.population_omega).clone()
                    target_omega_diag = torch.diag(target_omega)
                    for j in range(self.model.nb_PDU):
                        old_variance: float = updated_omega_diag[j].item()
                        new_target_variance: float = target_omega_diag[j].item()

                        annealed_target_variance: float = max(
                            old_variance * self.saem_annealing_factor,
                            new_target_variance,
                        )

                        updated_omega_diag[j] = (
                            (1 - current_alpha_k) * old_variance
                            + current_alpha_k * annealed_target_variance
                        )
                    self.population_omega = torch.diag(
                        updated_omega_diag
                    )  # Reconstruct Omega from its diagonal
                else:
                    # direct stochastic approximation in Phase 2
                    self.population_omega = (
                        1 - current_alpha_k
                    ) * self.population_omega + current_alpha_k * target_omega

                # ensure Omega remains symmetric and positive semi-definite
                self.population_omega = (
                    self.population_omega + self.population_omega.T
                ) / 2
                self.population_omega += (
                    torch.eye(self.population_omega.shape[0]) * 1e-6
                )  # Add jitter to diagonal

                # 4. Update residual error variances (var_res)
                # E[sum((y_ij - f(theta_ij))^2] approximated by total sum_squared_residuals/nb_observations, each row of sum_squared_residuals corresponds to an output and a residual error variance

                sum_squared_residuals: torch.Tensor = torch.zeros(
                    self.model.nb_outputs
                ).unsqueeze(-1)
                total_observations: torch.Tensor = torch.zeros(self.model.nb_outputs)

                for ind_data in self.list_individual_data:
                    # get the mean predictions for this iteration individual from the sampled etas in the E-step
                    predictions_for_res_update: torch.Tensor = individual_predictions[
                        self.ind_id_to_idx[ind_data.id]
                    ]  # dim: [nb_outputs x nb_time_steps]

                    residuals: torch.Tensor = self.model.calculate_residuals(
                        ind_data.observations,
                        predictions_for_res_update,
                    )  # dim: [nb_outputs x nb_time_steps]

                    sum_squared_residuals += torch.sum(
                        torch.square(residuals), dim=1
                    ).unsqueeze(-1)
                    total_observations += torch.Tensor(
                        [
                            len(ind_data.observations[i])
                            for i in range(len(ind_data.observations))
                        ]
                    )

                # update
                if (
                    self.model.error_model_type == "additive"
                ):  # sanity check, it is the only one implemented yet

                    target_res_var: torch.Tensor = (
                        sum_squared_residuals / total_observations.unsqueeze(1)
                    )
                    current_res_var: torch.Tensor = self.residual_error_var
                    # simulated annealing for residual error var in phase 1
                    if i < self.saem_phase1_iterations:
                        target_res_var: torch.Tensor = torch.max(
                            current_res_var * self.saem_annealing_factor, target_res_var
                        )

                    self.residual_error_var = (
                        1 - current_alpha_k
                    ) * current_res_var + current_alpha_k * target_res_var

                if self.verbose == True:
                    print(
                        f"  Updated MIs: {", ".join([f"{MI.item():.4f}" for MI in self.population_MI])}"
                    )
                    print(
                        f"  Updated Betas: {", ".join([f"{beta:.4f}" for beta in self.population_betas.detach().cpu().numpy().flatten()])}"
                    )
                    print(
                        f"  Updated Omega (diag): {", ".join([f"{val.item():.4f}" for val in torch.diag(self.population_omega)])}"
                    )
                    print(
                        f"  Updated Residual Var: {", ".join([f"{res_var:.4f}" for res_var in self.residual_error_var.detach().cpu().numpy().flatten()])}"
                    )

                # Store history
                self.history["population_MI"].append(self.population_MI.clone())
                self.history["population_betas"].append(self.population_betas.clone())
                self.history["population_omega"].append(self.population_omega.clone())
                self.history["residual_error_var"].append(
                    self.residual_error_var.clone()
                )

            print("\nSAEM Estimation Finished.")
            idx: int = 0
            self.estimated_population_MI_mus = []
            for j in range(self.model.nb_MI):
                self.estimated_population_MI_mus.append(self.population_MI[j].item())
            for PDU_name in self.model.PDU_names:
                self.estimated_population_MI_mus.append(
                    torch.exp(self.population_betas[idx]).item()
                )
                idx += 1
                if self.model.covariate_map and PDU_name in self.model.covariate_map:
                    for i in range(len(self.model.covariate_map[PDU_name])):
                        idx += 1

            if self.verbose:
                print(
                    f"Estimated MI: {", ".join([f"{MI:.4f}" for MI in self.estimated_population_MI_mus[:self.model.nb_MI]])}"
                )
                print(
                    f"Estimated mus: {", ".join([f"{mu:.4f}" for mu in self.estimated_population_MI_mus[self.model.nb_MI:]])}"
                )
                print(
                    f"Estimated population betas: {", ".join([f"{estimated_beta.item():.4f}" for estimated_beta in self.population_betas])}"
                )
                print(
                    f"Estimated omega (diagonal): {", ".join([f"{val.item():.4f}" for val in torch.diag(self.population_omega)])}"
                )
                print(
                    f"Estimated residual var: {", ".join([f"{val.item():.4f}" for val in self.residual_error_var.flatten()])}"
                )

            return (
                self.estimated_population_MI_mus,
                self.population_betas,
                self.population_omega,
                self.residual_error_var,
                self.history,
            )

        def plot_convergence_history(
            self,
            true_MI: Optional[torch.Tensor] = None,
            true_betas: Optional[torch.Tensor] = None,
            true_omega: Optional[torch.Tensor] = None,
            true_residual_var: Optional[torch.Tensor] = None,
        ):
            """
            Displays convergence plots for population parameters (MIs, betas, Omega diagonal) and the residual error variance. When given, plots the true values for comparison.
            true_MI: torch.Tensor of true Model Intrinsic
            true_betas : torch.Tensor of true population mean parameters and coeffs for covariates, dim: [nb_betas]
            true_omega: [nb_PDU x nb_PDU], true Omega matrix, diagonal elements are plotted only, dim: [nb_PDU x nb_PDU]
            true_residual_var: torch.Tensor [nb_outputs], true residual variances per output

            """
            history: Dict[str, List[Union[torch.Tensor, float]]] = self.history

            # determine the number of subplots needed
            # nb_MI + nb_betas + nb_PDU for omega diag + nb_outputs for var_res
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

            # plot MIs
            colors = plt.cm.get_cmap("tab10", nb_betas)

            for j, MI_name in enumerate(self.model.MI_names):
                MI_history = [h[j].item() for h in history["population_MI"]]
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

            # plot the diagonal elements of Omega (variances of random effects etas)
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

            # plot residual variances
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
