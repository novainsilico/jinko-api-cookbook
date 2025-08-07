import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Callable, Optional, Tuple
from multiprocessing import Pool


# --- 1. NLME Model Definition ---
class NLMEModel:
    def __init__(
        self,
<<<<<<< HEAD
        population_params_names: List[str],
        population_coeffs_names: List[str],
        structural_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        structural_model_several_params: Callable[
            [torch.Tensor, List[torch.Tensor]], List[torch.Tensor]
        ],
        error_model_type: str = "additive",  # the only one implemented for now
        covariate_map: Optional[Dict[str, List[str]]] = None,
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        population_param_names,
        structural_model,
        error_model_type="additive",  # the only one implemented for now
=======
        population_params_names,
        population_coeffs_names,
        structural_model,
        error_model_type="additive",  # the only one implemented for now
        covariate_map=None,
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
    ):
        self.population_params_names: List[str] = population_params_names
        self.nb_population_params: int = len(population_params_names)
        self.population_coeffs_names: List[str] = population_coeffs_names
        self.nb_coeffs: int = len(population_coeffs_names)
        self.nb_betas: int = self.nb_population_params + self.nb_coeffs
        self.error_model_type: str = error_model_type
        self.structural_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            structural_model
        )
        self.structural_model_several_params: Callable[
            [torch.Tensor, List[torch.Tensor]], List[torch.Tensor]
        ] = structural_model_several_params
        self.covariate_map: Optional[Dict[str, List[str]]] = covariate_map

<<<<<<< HEAD
        self.population_betas_names: List[str] = []
        idx = 0
        for param_name in self.population_params_names:
            self.population_betas_names.append(param_name)
            if self.covariate_map and param_name in self.covariate_map:
                for _ in range(len(self.covariate_map[param_name])):
                    self.population_betas_names.append(population_coeffs_names[idx])
                    idx += 1
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        self.population_param_names = population_param_names
        self.nb_population_params = len(population_param_names)
        self.error_model_type = error_model_type
        self.structural_model = structural_model
=======
        self.population_params_names = population_params_names
        self.nb_population_params = len(population_params_names)
        self.population_coeffs_names = population_coeffs_names
        self.nb_coeffs = len(population_coeffs_names)
        self.nb_betas = self.nb_population_params + self.nb_coeffs
        self.error_model_type = error_model_type
        self.structural_model = structural_model
        self.covariate_map = covariate_map
        self.nb_betas = 0
        for param_name in self.population_params_names:
            self.nb_betas += 1
            if param_name in self.covariate_map:
                for cov_name in self.covariate_map[param_name]:
                    self.nb_betas += 1
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)

<<<<<<< HEAD
    def individual_parameters(
        self,
        population_betas: torch.Tensor,
        individual_etas: torch.Tensor,
        design_matrix_X_i: torch.Tensor,
    ) -> torch.Tensor:
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
    def individual_parameters(self, population_mus, individual_etas, covariates=None):
=======
    def individual_parameters(
        self, population_mus_coeffs, individual_etas, design_matrix_X_i
    ):
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        """
<<<<<<< HEAD
        Transforms fixed effects (betas: log(mu)s and coeffs for covariates) and individual random effects (eta) into individual parameters (theta_i).
        Assumes log-normal distribution for individual parameters and covariate effects: theta_i = mu_pop * exp(eta_i) * exp(covariates_i * coeffs) where eta_i is from N(0, Omega).
        population_betas: torch.Tensor of the log(mu1), coeff_for_covariate_mu1_1, coeff_for_covariate_mu1_2 ... log(mu2), coeff_for_covariate_mu2_1 ... (dim: [nb_betas])
        individual_etas: torch.Tensor of the current eta value for each parameter for this individual (dim: [nb_population_params])
        design matrix X_i: torch.Tensor [nb_population_params x nb_betas], each row contains [0 , 0 ... 1, cov1, cov2, 0, 0 ...]
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        Transforms population parameters (mu) and individual random effects (eta) into individual parameters (theta_i).
        Assumes log-normal distribution for individual parameters: theta_i = mu_pop * exp(eta_i) where eta_i is from N(0, Omega).
        population_mus: dictionary of population parameters
        individual_etas: np.array of the eta value for each parameter for this individual
        covariates: dict of covariate values for this individual, not implemented yet
=======
        Transforms fixed effects (betas: log(mu)s and coeffs for covariates) and individual random effects (eta) into individual parameters (theta_i).
        Assumes log-normal distribution for individual parameters: theta_i = mu_pop * exp(eta_i) * exp(covariates_i * coeffs) where eta_i is from N(0, Omega).
        population_mus_coeffs: np.array of the mu1, coeff_for_covariate_mu1_1, coeff_for_covariate_mu1_2 ... mu2, coeff_for_covariate_mu2_1 ...
        individual_etas: np.array of the eta value for each parameter for this individual
        design matrix X_i: np.array [nb_population_params x nb_betas], each row contains [0 , 0 ... 1, cov1_val, cov2_val, 0, 0 ...]
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        """
<<<<<<< HEAD
        if design_matrix_X_i.shape[1] != len(population_betas):
            raise ValueError(
                "Dimension mismatch between design_matrix_X_i and population_betas."
            )
        if len(individual_etas) != self.nb_population_params:
            raise ValueError(
                "Dimension mismatch between individual_etas and the number of population parameters."
            )
        if self.covariate_map is None:
            raise ValueError(
                "Covariate map must be defined in order to compute individual parameters."
            )

        # compute the individual parameters
        log_individual_thetas: torch.Tensor = (
            design_matrix_X_i @ population_betas
        ) + individual_etas
        individual_params: torch.Tensor = torch.exp(
            log_individual_thetas
        )  # dim: [nb_population_params]
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        individual_params = {}
        for i, param_name in enumerate(self.population_param_names):
            if param_name in population_mus:
                # log-normal transformation: param_i = exp(log(mu_param) + eta_param_i)
                individual_params[param_name] = population_mus[param_name] * np.exp(
                    individual_etas[i]
                )
            else:
                raise ValueError(f"Population mean for '{param_name}' not found.")
=======

        if design_matrix_X_i.shape[1] != len(population_mus_coeffs):
            raise ValueError("Dimension mismatch between design_matrix_X_i and beta.")
        if len(individual_etas) != self.nb_population_params:
            raise ValueError(
                "Dimension mismatch between individual_etas and the number of population parameters."
            )
        if self.covariate_map is None:
            raise ValueError(
                "Covariate map must be defined in order to compute individual parameters."
            )

        # create population_betas by applying a log transformation only on the mus (not the coeffs) of population_mus_coeffs
        population_betas = np.zeros(len(population_mus_coeffs))
        idx = 0
        for i, param_name in enumerate(self.population_params_names):
            population_betas[idx] = np.log(population_mus_coeffs[idx])
            idx += 1
            for cov_name in self.covariate_map.get(param_name, []):
                idx += 1

        # compute the individual parameters
        log_individual_thetas = (
            design_matrix_X_i @ population_betas
        ).flatten() + individual_etas
        individual_params = np.zeros(self.nb_population_params)
        for i in range(self.nb_population_params):
            individual_params[i] = np.exp(log_individual_thetas[i])
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        return individual_params

    def several_individual_parameters(
        self,
        population_betas: torch.Tensor,
        individual_etas: torch.Tensor,
        design_matrices_X_i: Union[torch.Tensor, List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        """
        Transforms fixed effects (betas: log(mu)s and coeffs for covariates) and individual random effects (eta) into individual parameters (theta_i), for each set of etas of the list.
        Assumes log-normal distribution for individual parameters and covariate effects: theta_i = mu_pop * exp(eta_i) * exp(covariates_i * coeffs) where eta_i is from N(0, Omega).
        population_betas: torch.Tensor of the log(mu1), coeff_for_covariate_mu1_1, coeff_for_covariate_mu1_2 ... log(mu2), coeff_for_covariate_mu2_1 ... (dim: [nb_betas])
        individual_etas: torch.Tensor with the eta value for each parameter for each individual (dim: [nb_eta_sets x nb_population_params])
        design matrices X_i: either a list (of length nb_eta_sets) of design matrices, i.e. torch.Tensors [nb_population_params x nb_betas], each row containing [0 , 0 ... 1, cov1, cov2, 0, 0 ...] or one design matrix that will be replicated nb_eta_ests times.
        """
        # errors
        if isinstance(design_matrices_X_i, torch.Tensor):
            design_matrices_X_i = [design_matrices_X_i] * len(
                individual_etas
            )  # if there is only one design matrix (i.e., we want to compute for the same individual), create a list with the same design matrix repeated
        elif not isinstance(design_matrices_X_i, list) or len(
            design_matrices_X_i
        ) != len(individual_etas):
            raise ValueError(
                "design_matrices_X_i must be a single tensor or a list of tensors matching the length of list_individual_etas."
            )
        if design_matrices_X_i[0].shape[1] != len(population_betas):
            raise ValueError("Dimension mismatch between design_matrix_X_i and beta.")
        if len(individual_etas[0]) != self.nb_population_params:
            raise ValueError(
                "Dimension mismatch between individual_etas and the number of population parameters."
            )
        if self.covariate_map is None:
            raise ValueError(
                "Covariate map must be defined in order to compute individual parameters."
            )

        # compute the individual parameters one set after the other
        individual_params_list = []
        for i in range(len(individual_etas)):
            log_individual_thetas: torch.Tensor = (
                design_matrices_X_i[i] @ population_betas
            ) + individual_etas[i]
            individual_params: torch.Tensor = torch.exp(
                log_individual_thetas
            )  # dim: [nb_population_params]
            individual_params_list.append(individual_params)

        return individual_params_list

    def calculate_residuals(
        self, observed_data: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates residuals based on the error model.
        observed_data: torch.Tensor of observations for one individual. (dim: e.g., [nb_observations, 1])
        predictions: torch.Tensor of predictions for one individual. Must be organized like observed_data to compare both by subtraction. Depends on the structural model given to the Model. (recommended dim: [nb_outputs x time_steps])
        """
        if self.error_model_type == "additive":
            return observed_data - predictions
        else:
            raise ValueError("Unsupported error model type.")

    def log_likelihood_observation(
        self,
        observed_data: torch.Tensor,
        predictions: torch.Tensor,
        residual_error_var: float,
    ) -> float:
        """
        Calculates the log-likelihood of observations given predictions and additive error model, assuming errors follow N(0,sqrt(residual_error_var))
        observed_data: torch.Tensor of observations for one individual
        predictions: torch.Tensor of predictions for one individual organized in the same way as observed_data
        residual_error_var: float
        """
        if torch.any(torch.isinf(predictions)) or torch.any(torch.isnan(predictions)):
            return -torch.inf  # invalid predictions

        residuals: torch.Tensor = self.calculate_residuals(observed_data, predictions)
        # ensure error_std is positive
        residual_error_var = max(1e-6, residual_error_var)

        # Log-likelihood of normal distribution
        log_lik: float = (
            -0.5
            * torch.sum(
                torch.log(torch.tensor(2 * torch.pi * residual_error_var))
                + (residuals**2 / residual_error_var)
            ).item()
        )
        return log_lik


# --- 2. Data Handling ---
class IndividualData:
    def __init__(
        self,
        id: Union[str, int],
        time: List[float],
        observations: List[float],
        covariates: Optional[Dict[str, float]] = None,
    ):
        self.id = id
<<<<<<< HEAD
        self.time: torch.Tensor = torch.tensor(
            time, dtype=torch.float32
        )  # dim: [nb_time_points]
        self.observations: Optional[torch.Tensor] = (
            torch.tensor(observations, dtype=torch.float32)
            if observations is not None
            else None
        )  # dim depends on the structural model given to Model. recommended: [nb_outputs x nb_time_points]
        self.covariates: Dict[str, float] = covariates if covariates is not None else {}
        self.design_matrix_X_i: Optional[torch.Tensor] = (
            None  # dim: [nb_population_params x nb_betas]
        )

    def create_design_matrix(
        self,
        model_covariate_map: Dict[str, List[str]],
        population_params_names: List[str],
    ):
        """
        Creates the design matrix X_i for this individual based on the model's covariate map.
        The rows of X_i correspond to the population parameters.
        The columns of X_i correspond to the elements of the beta vector.
        Each row contains [1, cov1_val, cov2_val, ...].
        model_covariate_map: dict[str,list[str]] with mus as keys and arrays of the covariates that have an effect on this mu as values
        population_params_names: list of the names of population parameters
        """
        nb_population_params: int = len(population_params_names)
        nb_betas: int = 0
        for param_name in population_params_names:
            nb_betas += 1
            if param_name in model_covariate_map:
                nb_betas += len(model_covariate_map[param_name])

        design_matrix_X_i: torch.Tensor = torch.zeros(
            (nb_population_params, nb_betas), dtype=torch.float32
        )  # dim: [nb_population_params x nb_betas]

        col_idx: int = 0
        for i, param_name in enumerate(population_params_names):
            # Intercept term for log(mu)
            design_matrix_X_i[i, col_idx] = 1.0
            col_idx += 1

            # Covariate effects
            for cov_name in model_covariate_map.get(param_name, []):
                if cov_name in self.covariates:
                    design_matrix_X_i[i, col_idx] = float(self.covariates[cov_name])
                else:
                    design_matrix_X_i[i, col_idx] = 0.0
                col_idx += 1
        self.design_matrix_X_i = design_matrix_X_i
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        self.time = np.array(time)
        self.observations = np.array(observations)
        self.covariates = (
            covariates if covariates is not None else {}
        )  # not implemented yet
=======
        self.time = np.array(time)
        self.observations = np.array(observations)
        self.covariates = (
            covariates if covariates is not None else {}  # replace by just covariates?
        )
        self.design_matrix_X_i = None

    def create_design_matrix(self, model_covariate_map, population_params_names):
        """
        Creates the design matrix X_i for this individual based on the model's covariate map.
        The rows of X_i correspond to the population parameters.
        The columns of X_i correspond to the elements of the beta vector.
        Each row contains [1, cov1_val, cov2_val, ...].
        """
        nb_population_params = len(population_params_names)
        nb_betas = 0  # use a parameter to not recompute it? populations mus and
        for param_name in population_params_names:
            nb_betas += 1
            if param_name in model_covariate_map:
                for cov_name in model_covariate_map[param_name]:
                    nb_betas += 1

        design_matrix_X_i = np.zeros((nb_population_params, nb_betas))

        col_idx = 0
        for i, param_name in enumerate(population_params_names):
            # Intercept term for log(mu)
            design_matrix_X_i[i, col_idx] = 1
            col_idx += 1

            # Covariate effects
            for cov_name in model_covariate_map.get(param_name, []):
                if cov_name in self.covariates:
                    design_matrix_X_i[i, col_idx] = self.covariates[cov_name]
                else:
                    design_matrix_X_i[i, col_idx] = 0
                col_idx += 1
        self.design_matrix_X_i = design_matrix_X_i
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)


# --- 3. MCMC Sampler for Individual Random Effects (eta_i), used in the E-step of SAEM ---
class MCMC_Eta_Sampler:  # one independant markov chain per individual
    def __init__(
        self,
<<<<<<< HEAD
        model: NLMEModel,
        population_betas: torch.Tensor,
        population_omega: torch.Tensor,
        residual_error_var: float,
        proposal_var_eta: torch.Tensor,
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        model,
        individual_data,
        initial_eta,
        population_mus,
        population_omega,
        residual_error_sigma,
        proposal_sigma_eta,  # covariance used when sampling a new eta from a multivariate normal distribution from the current position in the Markov Chain
=======
        model,
        individual_data,
        initial_eta,
        population_betas,
        population_omega,
        residual_error_sigma,
        proposal_sigma_eta,  # covariance used when sampling a new eta from a multivariate normal distribution from the current position in the Markov Chain
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
    ):
<<<<<<< HEAD
        self.model: NLMEModel = model
        self.population_betas: torch.Tensor = (
            population_betas.clone().detach()
        )  # dim: [nb_betas]
        self.population_omega: torch.Tensor = (
            population_omega.clone().detach()
        )  # dim: [nb_mus x nb_mus], nb_mus = nb_etas, omega is the covariance matrix for eta
        self.residual_error_var: float = residual_error_var
        self.proposal_var_eta: torch.Tensor = (
            proposal_var_eta.clone().detach()
        )  # dim: [nb_mus x nb_mus], covariance used when sampling a new eta from a multivariate normal distribution from the current position in the Markov Chain
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        self.model = model
        self.individual_data = individual_data
        self.current_eta = np.array(initial_eta, dtype=float)
        self.population_mus = population_mus
        self.population_omega = (
            population_omega  # omega is the covariance matrix for eta
        )
        self.residual_error_sigma = residual_error_sigma
        self.proposal_sigma_eta = proposal_sigma_eta
=======
        self.model = model
        self.individual_data = individual_data
        self.current_eta = np.array(initial_eta, dtype=float)
        self.population_betas = population_betas
        self.population_omega = (
            population_omega  # omega is the covariance matrix for eta
        )
        self.residual_error_sigma = residual_error_sigma
        self.proposal_sigma_eta = proposal_sigma_eta
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)

        self.inv_population_omega: torch.Tensor = torch.linalg.inv(
            self.population_omega
        )  # dim: [nb_mus x nb_mus]
        self.log_det_population_omega: float = torch.logdet(
            self.population_omega
        ).item()

    def _log_prior_eta(self, eta: torch.Tensor) -> float:
        """
        Calculates the log-prior for eta_i, i.e. assuming eta_i ~ N(0, Omega), what is the log-probability of sampling this eta?
        P(eta) = (1/sqrt((2pi)^k * |omega|)) * exp(-0.5 * eta.T * omega.inv * eta)
        log P(eta) = -0.5 * (k * log(2pi) + log|omega| + eta.T * omega.inv * eta)
        We could use the log_prob function of torch MultivariateNormal distribution as below, but like this it avoids recomputation of the determinant and of the inverse of omega, speeding up the process.
        eta: torch.Tensor of dim [nb_mus] with an eta value for each population parameter
        """
        term1: float = -0.5 * (
            len(eta) * torch.log(torch.tensor(2 * torch.pi))
            + self.log_det_population_omega
        )
        term2: float = (
            -0.5 * (eta.unsqueeze(0) @ self.inv_population_omega @ eta).item()
        )
        return term1 + term2

    def _log_prior_several_etas(self, etas: torch.Tensor) -> float:
        """
        Calculates the log-prior for all the eta_i, i.e. assuming eta_i ~ N(0, Omega), what is the log-probability of sampling this eta? Considers each eta_i independently of the others.
        P(eta) = (1/sqrt((2pi)^k * |Omega|)) * exp(-0.5 * eta.T * omega.inv * eta)
        log P(eta) = -0.5 * (k * log(2pi) + log|Omega| + eta.T * omega.inv * eta)
        etas: torch.Tensors of dim [nb_eta_i x nb_mus]
        """
        etas_dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.model.nb_population_params),
            covariance_matrix=self.population_omega,
        )
        log_priors: torch.Tensor = etas_dist.log_prob(etas)

        return log_priors

    def _log_posterior_eta(self, eta: torch.Tensor, ind_data: IndividualData) -> float:
        """
        Calculates the log-posterior for MCMC for eta_i.
        log(P(eta_i | y_i, mu, Omega, var_res)) = log(P(y_i | theta_i(eta_i))) + log(P(eta_i | Omega)), (notice the second term is the prior of eta)
        from bayes formula, disgarding the normalizing term p(y) which does not depend on theta
        eta: torch.Tensor of dim [nb_mus] with an eta value for each population parameter
        ind_data: IndividualData object, of the individual with the associated eta_i individual effects.
        """
<<<<<<< HEAD
        # transform eta_i into individual parameters theta_i
        individual_params: torch.Tensor = self.model.individual_parameters(
            self.population_betas,
            individual_etas=eta,
            design_matrix_X_i=ind_data.design_matrix_X_i,
        )  # of dim [nb_mus]
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        # transform eta into individual parameters theta_i
        individual_params = self.model.individual_parameters(
            self.population_mus, eta, self.individual_data.covariates
        )
=======
        # transform eta into individual parameters theta_i
        individual_params = self.model.individual_parameters(
            self.population_betas, eta, self.individual_data.design_matrix_X_i
        )
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)

        # predict observations using the structural model
        predictions: torch.Tensor = self.model.structural_model(
            ind_data.time, individual_params
        )

        # calculate log-likelihood of observations given predictions
        log_lik_obs: float = self.model.log_likelihood_observation(
            ind_data.observations, predictions, self.residual_error_var
        )

        # calculate log-prior for eta
        log_prior_eta_val: float = self._log_prior_eta(eta)

        return log_lik_obs + log_prior_eta_val

    def _log_posterior_several_etas(
        self, etas: torch.Tensor, ind_data_list: List[IndividualData]
    ) -> torch.Tensor:
        """
        Calculates the log-posterior for MCMC for all the given eta_i.
        log(P(eta_i | y_i, mu, Omega, var_res)) = log(P(y_i | theta_i(eta_i))) + log(P(eta_i | Omega)), (notice the second term is the prior of eta, the first one is the log_likelihood of observations given predictions)
        from bayes formula, disgarding the normalizing term p(y) which does not depend on theta
        etas: torch.Tensor of dim [nb_eta_i x nb_mus]
        ind_data_list: a list of IndividualData with the associated design matrices. Must be of length nb_eta_i.
        """
        # for each eta_i, transform eta_i into individual parameters theta_i
        list_individual_params: List[torch.Tensor] = (
            self.model.several_individual_parameters(
                self.population_betas,
                individual_etas=etas,
                design_matrices_X_i=[
                    ind_data.design_matrix_X_i for ind_data in ind_data_list
                ],
            )
        )  # of dim List[torch.Tensor[nb_mus]]

        # predict observations using the structural model
        predictions: List[torch.Tensor] = self.model.structural_model_several_params(
            [ind_data_list[i].time for i in range(len(ind_data_list))],
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
        log_priors: torch.Tensor = self._log_prior_several_etas(etas)

        # log_posterior is the sum of both
        return log_lik_obs + log_priors

    def sample_from_one_chain(
        self,
        ind_data: IndividualData,
        current_eta: torch.Tensor,
        nb_samples,
        nb_burn_in,
    ) -> torch.Tensor:
        """
        Performs Metropolis-Hastings sampling for a single individual's eta.
        ind_data: IndividualData
        current_eta: torch.Tensor
        Returns samples after burn-in.
        nb_samples: int
        nb_burn_in: int
        """

        all_states_history: List[torch.Tensor] = (
            []
        )  # used for plotting convergence of the MCMC chain to determine burn in

        samples: List[torch.Tensor] = []

        current_log_posterior: float = self._log_posterior_eta(current_eta, ind_data)
        accepted_count = 0
        total_proposals = 0

        while accepted_count < (nb_samples + nb_burn_in):
            total_proposals += 1
            # propose new eta values using a multivariate normal distribution
            proposal_dist = torch.distributions.MultivariateNormal(
                current_eta, self.proposal_var_eta
            )
            proposed_eta: torch.Tensor = proposal_dist.sample()  # dim: [nb_mus]

            # calculate the delta of the log posteriors
            proposed_log_posterior: float = self._log_posterior_eta(
                proposed_eta, ind_data
            )
            delta: float = proposed_log_posterior - current_log_posterior

            if torch.log(torch.rand(1)).item() < delta:
                accepted_count += 1
                current_eta = proposed_eta
                current_log_posterior = proposed_log_posterior
                if (
                    accepted_count >= nb_burn_in
                ):  # during burn-in, the MCMC chain converges and we do not take the sampled etas into account
                    samples.append(current_eta.clone())

            all_states_history.append(current_eta.clone())

        acceptance_rate: float = accepted_count / total_proposals
        print(f"  MCMC Acceptance Rate: {acceptance_rate:.2f}")

        # all_states_history = torch.stack(all_states_history)
        # nb_mus = all_states_history.shape[
        #     1
        # ]

        # plt.figure(figsize=(10, 6))
        # for j in range(nb_mus):
        #     plt.plot(all_states_history[:, j], label=f"Parameter {j+1}")

        # plt.title("MCMC Chain Convergence (All States History)")
        # plt.xlabel("Iteration")
        # plt.ylabel("Parameter Value")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        return torch.stack(samples)  # dim: [nb_samples x nb_mus]

    def sample_from_several_chains(
        self,
        list_ind_data: List[IndividualData],
        current_eta_for_all_ind: torch.Tensor,
        nb_samples: int,
        nb_burn_in: int,
    ) -> torch.Tensor:
        """
        Performs Metropolis-Hastings sampling for the individuals'etas. The MCMC chains (one per individual) advance in parallel.
        Returns samples after burn-in.
        list_ind_data: List[IndividualData]
        current_eta_for_all_ind: torch.Tensor of dim [nb_individuals x nb_parameters]
        nb_samples: int
        nb_burn_in: int
        """
        nb_individuals = len(list_ind_data)
        all_states_history: List[torch.Tensor] = []

        samples: List[List[torch.Tensor]] = [[] for i in range(nb_individuals)]

        current_log_posteriors: torch.Tensor = self._log_posterior_several_etas(
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
            )  # dim [nb_active x nb_mus]
            proposed_log_posteriors: torch.Tensor = self._log_posterior_several_etas(
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
            accepted_counts[active_indices] += accept_mask

            for idx in active_indices[accept_mask]:
                if accepted_counts[idx] >= nb_burn_in:
                    samples[idx.item()].append(current_eta_for_all_ind[idx].clone())
                    if len(samples[idx.item()]) == nb_samples:
                        done[idx] = True
            all_states_history.append(current_eta_for_all_ind.clone())

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
        # nb_mus = all_states_history.shape[2]
        # # Loop through each individual (MCMC chain) to create a separate plot
        # for individual_idx in range(nb_individuals):
        #     plt.figure(
        #         figsize=(10, 6)
        #     )  # Create a new figure for each individual's chain
        #     # Loop through each parameter (mu) for the current individual's chain
        #     for param_idx in range(nb_mus):
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
    def __init__(
        self,
<<<<<<< HEAD
        model: NLMEModel,
        individual_data_list: List[IndividualData],
        initial_pop_betas: torch.Tensor,
        initial_pop_omega: torch.Tensor,
        initial_res_var: float,
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        model: NLMEMModel,
        individual_data_list,
        initial_pop_mus,
        initial_pop_omega,
        initial_sigma_res,
=======
        model: NLMEMModel,
        individual_data_list,
        initial_pop_betas,
        initial_pop_omega,
        initial_sigma_res,
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
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
        parallel_computing=True,
    ):

        self.model: NLMEModel = model
        self.individual_data_list: List[IndividualData] = individual_data_list

<<<<<<< HEAD
        # population parameters betas to be updated during the SAEM iterations
        self.population_betas: torch.Tensor = (
            initial_pop_betas.clone().detach()
        )  # dim: [nb_betas]
        self.estimated_population_mus: Optional[List[float]] = (
            None  # completed after the SAEM iterations from the estimated betas
        )
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        # population parameters mus to be updated during the SAEM iterations
        self.population_mus = initial_pop_mus.copy()
=======
        # population parameters mus to be updated during the SAEM iterations
        self.population_betas = initial_pop_betas
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        # omega: covariance matrix of the individual random effects etas
<<<<<<< HEAD
        self.population_omega: torch.Tensor = (
            initial_pop_omega.clone().detach()  # [nb_mus x nb_mus], nb_mus = nb_etas, omega is the covariance matrix for etas
        )
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        self.population_omega = initial_pop_omega.copy()
=======
        self.population_omega = initial_pop_omega
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        # residual error variance
        self.residual_error_var: float = (
            initial_res_var  # variance of the additive error on the observations
        )

        # MCMC parameters for the E-step
        self.mcmc_first_burn_in: int = (
            mcmc_first_burn_in  # number of disgarded samples per chain in the first iteration, where the chain starts at zero.
        )
        self.mcmc_burn_in: int = mcmc_burn_in  # number of disgarded samples per chain
        self.mcmc_nb_samples: int = mcmc_nb_samples
        self.mcmc_proposal_var_scaling_factor: float = (
            mcmc_proposal_var_scaling_factor  # the variance of the multivariate normal distribution that the next eta from the Markov Chain is sampled from is scaling_factor * omega
        )

        # SAEM Iteration Parameters
        self.nb_saem_iterations: int = nb_saem_iterations
        self.saem_phase1_iterations: int = saem_phase1_iterations
        self.saem_phase2_iterations: int = (
            (nb_saem_iterations - saem_phase1_iterations)
            if saem_phase1_iterations is not None
            else nb_saem_iterations // 2
        )
        self.saem_learning_rate_power: float = saem_learning_rate_power
        self.saem_annealing_factor: float = (
            saem_annealing_factor  # for the updates of omega and residual_var in phase 2
        )

<<<<<<< HEAD
        # determines if the Markov Chains are run in parallel
        self.parallel_computing = parallel_computing

        self.history: Dict[str, List[Union[torch.Tensor, float]]] = {
            "population_betas": [],
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        self.history = {
            "population_mus": [],
=======
        self.history = {
            "population_betas": [],
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
            "population_omega": [],
            "residual_error_var": [],
        }

        # to switch from individual id to an array index
        self.ind_id_to_idx: Dict[Union[str, int], int] = {
            ind_data.id: i for i, ind_data in enumerate(self.individual_data_list)
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
        List[float],
        torch.Tensor,
        float,
        Dict[str, List[Union[torch.Tensor, float]]],
    ]:
        # this method handles the main SAEM loop with E-step and M-step
        # returns self.population_betas, self.estimated_population_mus, self.population_omega, self.residual_error_var, self.history
        print("Starting SAEM Estimation...")
<<<<<<< HEAD
        print(
            f"Initial Population Betas: {", ".join([f"{beta.item():.2f}" for beta in self.population_betas])}"
        )
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        print(f"Initial Population Mus: {self.population_mus}")
=======
        print(f"Initial Population Mus: {self.population_betas}")
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        print(f"Initial Omega:\n{self.population_omega}")
        print(f"Initial Residual Variance: {self.residual_error_var}")

        # store initial parameters
<<<<<<< HEAD
        self.history["population_betas"].append(self.population_betas.clone())
        self.history["population_omega"].append(self.population_omega.clone())
        self.history["residual_error_var"].append(self.residual_error_var)
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        self.history["population_mus"].append(self.population_mus.copy())
        self.history["population_omega"].append(self.population_omega.copy())
        self.history["residual_error_sigma"].append(self.residual_error_sigma.copy())
=======
        self.history["population_betas"].append(self.population_betas.copy())
        self.history["population_omega"].append(self.population_omega.copy())
        self.history["residual_error_sigma"].append(self.residual_error_sigma.copy())
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)

<<<<<<< HEAD
        # create design matrices for each individual
        for ind_data in self.individual_data_list:
            ind_data.create_design_matrix(
                self.model.covariate_map, self.model.population_params_names
            )
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        # initialize storage
        etas_prev_iter = np.zeros(
            (len(self.individual_data_list), self.model.nb_population_params)
        )
=======
        # create design matrices for each individual
        for ind_data in self.individual_data_list:
            ind_data.create_design_matrix(
                self.model.covariate_map, self.model.population_params_names
            )

        # initialize storage
        etas_prev_iter = np.zeros(
            (len(self.individual_data_list), self.model.nb_population_params)
        )
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)

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
                + torch.eye(self.model.nb_population_params)
                * 1e-7  # add small jitter for numerical stability
            )  # dim: [nb_mus x nb_mus]

            # initialize storage
            mean_etas: torch.Tensor = torch.zeros(
                (len(self.individual_data_list), self.model.nb_population_params),
                dtype=torch.float32,
            )  # dim: [nb_individuals x nb_mus]
            individual_predictions: Dict[int, torch.Tensor] = (
                {}
            )  # associates the individual index with the mean of the predictions made with the sampled etas for this individual (stores torch.Tensor of predictions for each individual)
            mean_log_thetas_for_all_ind: Dict[int, torch.Tensor] = (
                {}
            )  # associates the individual index with the mean of the log of thetas calculated with the sampled etas for this individual (stores torch.Tensor [nb_mus] of mean_log_thetas for each individual)

            mcmc_sampler = MCMC_Eta_Sampler(
                model=self.model,
                population_betas=self.population_betas,
                population_omega=self.population_omega,
                residual_error_var=self.residual_error_var,
                proposal_var_eta=current_mcmc_proposal_var_eta,
            )
<<<<<<< HEAD
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
            all_individual_predictions = {}
=======
            all_individual_predictions = {}
            all_sampled_log_thetas = np.empty(
                (
                    len(self.individual_data_list),
                    self.model.nb_population_params,
                    self.mcmc_nb_samples,
                )
            )  # for betas update
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)

<<<<<<< HEAD
            if self.parallel_computing == False:
                # loop through individuals to run an MCMC chain each
                for ind_data in self.individual_data_list:
                    ind_idx: int = self.ind_id_to_idx[ind_data.id]
                    # the eta_samples are the samples from the log-posterior distribution of etas (approximated by MCMC) knowing the observations for this individual
                    if (
                        i == 0
                    ):  # different (higher recommended) burn in for first iter, because the chain starts with zero etas
                        eta_samples: torch.Tensor = mcmc_sampler.sample_from_one_chain(
                            ind_data,
                            mean_etas[
                                ind_idx
                            ],  # start at the mean eta of the previous iteration (zero for this first iter)
                            nb_samples=self.mcmc_nb_samples,
                            nb_burn_in=self.mcmc_first_burn_in,
                        )  # dim: [nb_samples x nb_mus]
                    else:  # different (recommended reduced) burn-in for the rest of the iterations, because the chain starts at the eta of the previous iter
                        eta_samples: torch.Tensor = mcmc_sampler.sample_from_one_chain(
                            ind_data,
                            mean_etas[
                                ind_idx
                            ],  # start at the mean eta of the previous iteration
                            nb_samples=self.mcmc_nb_samples,
                            nb_burn_in=self.mcmc_burn_in,
                        )  # dim: [nb_samples x nb_mus]
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
            # loop through individuals to run an MCMC chain each
            for ind_data in self.individual_data_list:
                ind_idx = self.ind_id_to_idx[ind_data.id]
=======
            # loop through individuals to run an MCMC chain each
            for ind_data in self.individual_data_list:
                ind_idx = self.ind_id_to_idx[ind_data.id]
                design_matrix_X_i = ind_data.design_matrix_X_i
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)

<<<<<<< HEAD
                    # store the mean of the sampled etas for this individual in the corresponding slot of mean_etas
                    mean_eta_i: torch.Tensor = torch.mean(
                        eta_samples, dim=0
                    )  # dim: [nb_mus]
                    mean_etas[ind_idx] = mean_eta_i
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
                mcmc_sampler = MCMC_Eta_Sampler(
                    model=self.model,
                    individual_data=ind_data,
                    initial_eta=etas_prev_iter[
                        ind_idx
                    ],  # start at the mean of the sampled etas of the previous iteration for this individual
                    population_mus=self.population_mus,
                    population_omega=self.population_omega,
                    residual_error_sigma=self.residual_error_sigma,
                    proposal_sigma_eta=current_mcmc_proposal_sigma_eta,
                )
=======
                mcmc_sampler = MCMC_Eta_Sampler(
                    model=self.model,
                    individual_data=ind_data,
                    initial_eta=etas_prev_iter[
                        ind_idx
                    ],  # start at the mean of the sampled etas of the previous iteration for this individual
                    population_betas=self.population_betas,
                    population_omega=self.population_omega,
                    residual_error_sigma=self.residual_error_sigma,
                    proposal_sigma_eta=current_mcmc_proposal_sigma_eta,
                )
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)

<<<<<<< HEAD
                    # for the residual error update in the M-step, for each individual, calculate predictions based on each sampled eta_j, average them and store them in individual_predictions
                    all_pred_for_ind: List[torch.Tensor] = []
                    all_log_thetas_for_ind: List[torch.Tensor] = []
                    for eta_s in eta_samples:
                        theta_s: torch.Tensor = self.model.individual_parameters(
                            self.population_betas, eta_s, ind_data.design_matrix_X_i
                        )  # dim: [nb_mus]
                        all_log_thetas_for_ind.append(
                            torch.log(theta_s)
                        )  # dim: [nb_mus]
                        pred_s: torch.Tensor = self.model.structural_model(
                            ind_data.time, theta_s
                        )  # dim: [nb_outputs x nb_time_steps]
                        all_pred_for_ind.append(pred_s)
                    # store the average of these predictions for this individual
                    individual_predictions[ind_idx] = torch.mean(
                        torch.stack(all_pred_for_ind), dim=0
                    )  # dim: [nb_outputs x nb_time_steps]
                    mean_log_thetas_for_all_ind[ind_idx] = torch.mean(
                        torch.stack(all_log_thetas_for_ind), dim=0
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
                # these are the samples from the log-posterior distribution of etas (approximated by MCMC) knowing the observations for this individual
                eta_samples = mcmc_sampler.sample(
                    nb_samples=self.mcmc_nb_samples,
                    nb_burn_in=self.mcmc_burn_in,
                )

                # store the mean of the sampled etas for this individual in the corresponding slot of all_etas_sampled_this_iter
                mean_eta_i = np.mean(eta_samples, axis=0)
                all_etas_sampled_this_iter[ind_idx] = mean_eta_i

                etas_prev_iter[ind_idx] = mean_eta_i

                # for the residual error update in the M-step, for each individual, calculate predictions based on each sampled eta_j, average them and store them in all_individual_predictions
                all_pred_for_ind = []
                for eta_s in eta_samples:
                    theta_s = self.model.individual_parameters(
                        self.population_mus, eta_s, ind_data.covariates
=======
                # these are the samples from the log-posterior distribution of etas (approximated by MCMC) knowing the observations for this individual
                eta_samples = mcmc_sampler.sample(
                    nb_samples=self.mcmc_nb_samples,
                    nb_burn_in=self.mcmc_burn_in,
                )

                # store the mean of the sampled etas for this individual in the corresponding slot of all_etas_sampled_this_iter
                mean_eta_i = np.mean(eta_samples, axis=0)
                all_etas_sampled_this_iter[ind_idx] = mean_eta_i

                etas_prev_iter[ind_idx] = mean_eta_i

                # for the residual error update in the M-step, for each individual, calculate predictions based on each sampled eta_j, average them and store them in all_individual_predictions
                all_pred_for_ind = []
                all_log_thetas_for_ind = []
                for eta_s in eta_samples:
                    theta_s = self.model.individual_parameters(
                        self.population_betas, eta_s, ind_data.design_matrix_X_i
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
                    )
<<<<<<< HEAD

            elif self.parallel_computing == True:
                # these are the samples from the log-posterior distribution of etas (approximated by MCMC) knowing the observations for each individual
                if (
                    i == 0
                ):  # different (higher recommended) burn in for first iter, because the chain starts with zero etas
                    eta_samples: torch.Tensor = mcmc_sampler.sample_from_several_chains(
                        list_ind_data=self.individual_data_list,
                        current_eta_for_all_ind=mean_etas,
                        nb_samples=self.mcmc_nb_samples,
                        nb_burn_in=self.mcmc_first_burn_in,
                    )  # dim: [nb_ind x nb_samples x nb_mus]
                else:  # different (recommended reduced) burn-in for the rest of the iterations, because the chain starts at the eta of the previous iter
                    eta_samples: torch.Tensor = mcmc_sampler.sample_from_several_chains(
                        list_ind_data=self.individual_data_list,
                        current_eta_for_all_ind=mean_etas,
                        nb_samples=self.mcmc_nb_samples,
                        nb_burn_in=self.mcmc_burn_in,
                    )  # dim: [nb_ind x nb_samples x nb_mus]

                # store the mean of the sampled etas for each individual
                mean_etas: torch.Tensor = torch.mean(
                    eta_samples, dim=1
                )  # dim: [nb_ind x nb_mus]

                # for the residual error update in the M-step, for each individual, calculate predictions based on each sampled eta_j, average them and store them in individual_predictions
                for ind_data in self.individual_data_list:
                    ind_idx = self.ind_id_to_idx[ind_data.id]
                    eta_samples_for_ind = eta_samples[ind_idx, :, :]
                    thetas_for_ind: List[torch.Tensor] = (
                        self.model.several_individual_parameters(
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
                    preds_for_ind: List[torch.Tensor] = (
                        self.model.structural_model_several_params(
                            ind_data.time, thetas_for_ind
                        )
                    )  # Expected dim: List[nb_outputs, nb_time_steps] of length nb_samples
                    individual_predictions[ind_idx] = torch.mean(
                        torch.stack(preds_for_ind), dim=0
                    )

            # --- M-Step: Update Population Means, Omega and Residual Error ---

            nb_individuals: int = len(self.individual_data_list)
            inv_population_omega: torch.Tensor = torch.linalg.inv(
                self.population_omega
            )  # dim: [nb_mus x nb_mus]

            # 1. Update fixed effects betas ; i.e. logs of populations means mus and population coefficients for covariates (covariates = PDKs that influence certain individual parameters according to population coeffs)
            # theta_i = mu * exp(eta_i) * exp(coeffs * covariates_i)
            # So we can write log(theta_i) = log(mu) + eta_i + coeffs * covariates_i = X_i @ betas + etas_i
            # from which we derive the normal equations (sum_i X_i.T @ inv_population_omega @ X_i) @ population_betas = sum_i X_i.T @ inv_population_omega @ mean_log_theta_i
            # we work with the sums over all individuals to update population_betas

            sum_lhs_matrix: torch.Tensor = torch.zeros(
                (self.model.nb_betas, self.model.nb_betas), dtype=torch.float32
            )  # dim [nb_betas x nb_betas]
            sum_rhs_vector: torch.Tensor = torch.zeros(
                self.model.nb_betas, dtype=torch.float32
            )  # dim [nb_betas]

            for ind_idx in range(nb_individuals):
                X_i: torch.Tensor = self.individual_data_list[
                    ind_idx
                ].design_matrix_X_i  # dim [nb_mus x nb_betas]
                sum_lhs_matrix += X_i.T @ inv_population_omega @ X_i
                sum_rhs_vector += (
                    X_i.T @ inv_population_omega @ mean_log_thetas_for_all_ind[ind_idx]
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
                    pred_s = self.model.structural_model(ind_data.time, theta_s)
                    all_pred_for_ind.append(pred_s)
                # store the average of these predictions for this individual
                all_individual_predictions[ind_data.id] = np.mean(
                    np.array(all_pred_for_ind), axis=0
=======
                    all_log_thetas_for_ind.append(np.log(theta_s))
                    pred_s = self.model.structural_model(ind_data.time, theta_s)
                    all_pred_for_ind.append(pred_s)
                # store the average of these predictions for this individual
                all_individual_predictions[ind_data.id] = np.mean(
                    np.array(all_pred_for_ind), axis=0
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
                )
                all_sampled_log_thetas[ind_idx, :, :] = np.array(
                    all_log_thetas_for_ind
                ).T

            target_beta: torch.Tensor = torch.linalg.solve(
                sum_lhs_matrix
                + 1e-6
                * torch.eye(
                    self.model.nb_betas
                ),  # Add small diagonal for numerical stability
                sum_rhs_vector,
            )  # dim [nb_betas]

<<<<<<< HEAD
            # update fixed effects (betas) using stochastic approximation
            self.population_betas = (
                1 - current_alpha_k
            ) * self.population_betas + current_alpha_k * target_beta  # dim: [nb_betas]
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
            nb_individuals = len(self.individual_data_list)

            # 1. Update population means (mu)
            # For log-normal parameters: theta_i = mu * exp(eta_i)
            # The M-step updates mu_j based on the average of sampled eta_j across individuals.
            # update: log(mu_j,k+1) = log(mu_j,k) + alpha_k * mean_over_individuals(eta_j_sampled_for_individual_i)

            mean_eta_across_individuals = np.mean(all_etas_sampled_this_iter, axis=0)

            # iterate through all of the mus to update them
            for j, param_name in enumerate(self.model.population_param_names):
                if param_name in self.population_mus:
                    current_log_mu = np.log(self.population_mus[param_name])
                    # target log_mu update term: mean_eta_j across individuals
                    target_log_mu_increment = mean_eta_across_individuals[j]

                    # stochastic approximation update for log(mu)
                    updated_log_mu = (
                        current_log_mu + current_alpha_k * target_log_mu_increment
                    )
                    self.population_mus[param_name] = np.exp(updated_log_mu)
=======
            nb_individuals = len(self.individual_data_list)
            inv_population_omega = np.linalg.inv(self.population_omega)
            # 1. Update fixed effects betas ; i.e. logs of populations means mus and population coefficients for covariates (covariates = PDKs that influence certain individual parameters according to population coeffs)
            # theta_i = mu * exp(eta_i) * exp(coeffs * covariates_i)
            # So we can write theta_i = X_i @ betas + etas_i
            # from which we derive the normal equations X_i.T @ inv_population_omega @ X_i) @ population_betas = X_i.T @ inv_population_omega @ theta_i
            # we work with the sums over all individuals to update population_betas

            sum_lhs_matrix = np.zeros((self.model.nb_betas, self.model.nb_betas))
            sum_rhs_vector = np.zeros(self.model.nb_betas)

            for ind_idx in range(nb_individuals):
                X_i = self.individual_data_list[ind_idx].design_matrix_X_i
                mean_log_theta_i = np.mean(
                    all_sampled_log_thetas[ind_idx, :, :], axis=1
                )

                sum_lhs_matrix += X_i.T @ inv_population_omega @ X_i
                sum_rhs_vector += X_i.T @ inv_population_omega @ mean_log_theta_i

            target_beta = np.linalg.solve(
                sum_lhs_matrix + 1e-6 * np.eye(self.model.nb_betas),
                sum_rhs_vector,
            )

            # update
            self.population_betas = np.array(
                (1 - current_alpha_k) * np.array(self.population_betas)
                + current_alpha_k * np.array(target_beta)
            )
            # TAKE THE EXP AGAIN
            idx = 0
            for i, param_name in enumerate(self.model.population_params_names):
                self.population_betas[idx] = np.exp(self.population_betas[idx])
                idx += 1
                for cov_name in self.model.covariate_map.get(param_name, []):
                    idx += 1
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)

            # 2. Update Omega (covariance matrix of eta)
            # E[eta_i * eta_i.T] is approximated by the average of (eta_i_sampled * eta_i_sampled.T) for each individual and then averaged over individuals
            sum_outer_product_etas: torch.Tensor = torch.zeros_like(
                self.population_omega
            )  # dim: [nb_mus x nb_mus]
            for eta_i_mean in mean_etas:  # eta_i_mean dim: [nb_mus]
                sum_outer_product_etas += torch.outer(
                    eta_i_mean, eta_i_mean
                )  # outer product dim: [nb_mus x nb_mus]
            target_omega: torch.Tensor = (
                sum_outer_product_etas / nb_individuals
            )  # dim: [nb_mus x nb_mus]

            # simulated annealing for each diagonal element of omega in Phase 1
            if i < self.saem_phase1_iterations:
                updated_omega_diag = torch.diag(self.population_omega).clone()
                target_omega_diag = torch.diag(target_omega)
                for j in range(self.model.nb_population_params):
                    old_variance: float = updated_omega_diag[j].item()
                    new_target_variance: float = target_omega_diag[j].item()

                    annealed_target_variance: float = max(
                        old_variance * self.saem_annealing_factor, new_target_variance
                    )

                    updated_omega_diag[j] = (
                        1 - current_alpha_k
                    ) * old_variance + current_alpha_k * annealed_target_variance
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

            # 3. Update residual error variance (var_res)
            # E[sum((y_ij - f(theta_ij))^2] approximated by total sum_squared_residuals/nb_observations

            sum_squared_residuals: float = 0.0
            total_observations: int = 0

            for ind_data in self.individual_data_list:
                # get the mean predictions for this iteration individual from the sampled etas in the E-step
                predictions_for_res_update: torch.Tensor = individual_predictions[
                    self.ind_id_to_idx[ind_data.id]
                ]  # dim: [nb_outputs x nb_time_steps]

                residuals: torch.Tensor = self.model.calculate_residuals(
                    ind_data.observations,
                    predictions_for_res_update,
                )  # dim: [nb_outputs x nb_time_steps]

                sum_squared_residuals += torch.sum(torch.square(residuals)).item()
                total_observations += ind_data.observations.numel()

            # update
            if (
                self.model.error_model_type == "additive"
            ):  # sanity check, it is the only one implemented yet
                target_res_var: float = sum_squared_residuals / total_observations
                current_res_var: float = self.residual_error_var

                # simulated annealing for residual error var in phase 1
                if i < self.saem_phase1_iterations:
                    target_res_var: float = max(
                        current_res_var * self.saem_annealing_factor, target_res_var
                    )

<<<<<<< HEAD
                self.residual_error_var = (
                    1 - current_alpha_k
                ) * current_res_var + current_alpha_k * target_res_var

            print(
                f"  Updated Betas: {", ".join([f"{beta.item():.4f}" for beta in self.population_betas])}"
            )
            print(
                f"  Updated Omega (diag): {", ".join([f"{val.item():.4f}" for val in torch.diag(self.population_omega)])}"
            )
            print(f"  Updated Residual Var: {self.residual_error_var:.4f}")
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
            print(f"  Updated Mus: {self.population_mus}")
            print(f"  Updated Omega (diag): {np.diag(self.population_omega)}")
            print(f"  Updated Residual Sigma: {self.residual_error_sigma}")
=======
            print(f"  Updated Betas: {self.population_betas}")
            print(f"  Updated Omega (diag): {np.diag(self.population_omega)}")
            print(f"  Updated Residual Sigma: {self.residual_error_sigma}")
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)

            # Store history
<<<<<<< HEAD
            self.history["population_betas"].append(self.population_betas.clone())
            self.history["population_omega"].append(self.population_omega.clone())
            self.history["residual_error_var"].append(self.residual_error_var)
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
            self.history["population_mus"].append(self.population_mus.copy())
            self.history["population_omega"].append(self.population_omega.copy())
            self.history["residual_error_sigma"].append(
                self.residual_error_sigma.copy()
            )
=======
            self.history["population_betas"].append(self.population_betas.copy())
            self.history["population_omega"].append(self.population_omega.copy())
            self.history["residual_error_sigma"].append(
                self.residual_error_sigma.copy()
            )
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)

        print("\nSAEM Estimation Finished.")
        idx: int = 0
        self.estimated_population_mus = []
        for param_name in self.model.population_params_names:
            self.estimated_population_mus.append(
                torch.exp(self.population_betas[idx]).item()
            )
            idx += 1
            if self.model.covariate_map and param_name in self.model.covariate_map:
                for cov_name in self.model.covariate_map[param_name]:
                    idx += 1
        return (
<<<<<<< HEAD
            self.population_betas,
            self.estimated_population_mus,
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
            self.population_mus,
=======
            self.population_betas,
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
            self.population_omega,
            self.residual_error_var,
            self.history,
        )

    def plot_convergence_history(
        self,
        true_betas: Optional[torch.Tensor] = None,
        true_omega: Optional[torch.Tensor] = None,
        true_residual_var: Optional[float] = None,
    ):
        """
        Displays convergence plots for population parameters (betas, Omega diagonal) and residual error var.
        true_betas : torch.Tensor of true population mean parameters and coeffs for covariates (dim: [nb_betas])
        true_omega: [nb_mus x nb_mus], true Omega matrix, diagonal elements are plotted only
        true_residual_var: float, true residual varvalue

        """
        history: Dict[str, List[Union[torch.Tensor, float]]] = self.history

        # determine the number of subplots needed
<<<<<<< HEAD
        # (nb_betas) for betas, (nb_eta_params) for Omega diag, 1 for var_res
        nb_betas: int = (
            len(true_betas) if true_betas is not None else len(self.population_betas)
        )  # dynamically determine based on provided true_betas or estimated ones
        nb_omega_diag_params: int = self.model.nb_population_params
        nb_var_res_params: int = 1
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        # (nb_mu_params) for mu, (nb_eta_params) for Omega diag, 1 for sigma_res
        nb_mu_params = len(self.model.population_param_names)
        nb_omega_diag_params = (
            self.model.nb_population_params
        )  # Same as nb_mu_params here
        nb_sigma_res_params = 1
=======
        # (nb_mu_params) for mu, (nb_eta_params) for Omega diag, 1 for sigma_res
        nb_mu_params = len(self.model.population_params_names)
        nb_omega_diag_params = (
            self.model.nb_population_params
        )  # Same as nb_mu_params here
        nb_sigma_res_params = 1
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)

        fig, axs = plt.subplots(
            nb_betas + nb_omega_diag_params + nb_var_res_params,
            1,
            figsize=(
                10,
                4 * (nb_betas + nb_omega_diag_params + nb_var_res_params),
            ),
        )

        plot_idx: int = 0

        # plot betas
        colors = plt.cm.get_cmap("tab10", nb_betas)

<<<<<<< HEAD
        for j, beta_name in enumerate(self.model.population_betas_names):
            beta_history = [
                h[j].item() for h in history["population_betas"]
            ]  # .item() to extract scalar
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
        for j, param_name in enumerate(self.model.population_param_names):
            mu_history = [h[param_name] for h in history["population_mus"]]
=======
        for j, param_name in enumerate(self.model.population_params_names):
            beta_history = [h[param_name] for h in history["population_betas"]]
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
            axs[plot_idx].plot(
<<<<<<< HEAD
                beta_history, label=f"Estimated beta for {beta_name} ", color=colors(j)
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
                mu_history, label=f"Estimated mu for {param_name} ", color=colors(j)
=======
                betas_history, label=f"Estimated mu for {param_name} ", color=colors(j)
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
            )
            if true_betas is not None and true_betas.shape[0] > j:
                axs[plot_idx].axhline(
                    y=true_betas[j].item(),  # .item() to extract scalar
                    color=colors(j),
                    linestyle="--",
                    label=f"True beta for {beta_name}",
                )

<<<<<<< HEAD
            axs[plot_idx].set_title(f"Convergence of beta_${{{beta_name}}}$")
||||||| parent of f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
            axs[plot_idx].set_title(f"Convergence of mu_{{{param_name}}}$")
=======
            axs[plot_idx].set_title(f"Convergence of beta_{{{param_name}}}$")
>>>>>>> f8b3ea5 (feat-vpop-calib): add covariates to PySAEM)
            axs[plot_idx].set_xlabel("SAEM Iteration")
            axs[plot_idx].set_ylabel("Parameter Value")
            axs[plot_idx].legend()
            axs[plot_idx].grid(True)
            plot_idx += 1

        # plot the diagonal elements of Omega (variances of random effects etas)
        for j, param_name in enumerate(
            self.model.population_params_names
        ):  # Assuming 1:1 mapping with eta indices
            omega_diag_history = [
                h[j, j].item() for h in history["population_omega"]
            ]  # .item() to extract scalar
            axs[plot_idx].plot(
                omega_diag_history,
                label=f"Estimated Omega for {param_name}",
                color=colors(j),
            )
            if true_omega is not None and true_omega.shape[0] > j:
                axs[plot_idx].axhline(
                    y=true_omega[j, j].item(),  # .item() to extract scalar
                    color=colors(j),
                    linestyle="--",
                    label=f"True Omega for {param_name}",
                )

            axs[plot_idx].set_title(f"Convergence of Omega for {param_name}")
            axs[plot_idx].set_xlabel("SAEM Iteration")
            axs[plot_idx].set_ylabel("Variance")
            axs[plot_idx].legend()
            axs[plot_idx].grid(True)
            plot_idx += 1

        # plot Residual var
        var_res_history = [h for h in history["residual_error_var"]]
        axs[plot_idx].plot(
            var_res_history, label="Estimated residual var", color="black"
        )
        if true_residual_var is not None:
            axs[plot_idx].axhline(
                y=true_residual_var,
                color="black",
                linestyle="--",
                label="True Residual var",
            )

        axs[plot_idx].set_title("Residual Error var Convergence")
        axs[plot_idx].set_xlabel("SAEM Iteration")
        axs[plot_idx].set_ylabel("var Value")
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        plot_idx += 1

        plt.tight_layout()
        plt.show()
