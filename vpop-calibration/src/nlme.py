import torch
from typing import List, Dict, Union, Tuple, Optional
import pandas as pd
import numpy as np

from .gp_surrogate import GP
from .ode_model.ode_model import OdeModel


class StructuralModel:
    def __init__(self, parameter_names, output_names):
        self.parameter_names = parameter_names
        self.nb_parameters = len(parameter_names)
        self.output_names = output_names
        self.nb_outputs = len(output_names)
        print(
            f"Successfully initiated a structural model, that expects the following parameters to be supplied: {self.parameter_names}"
        )

    def simulate(self, data_in: pd.DataFrame) -> pd.DataFrame:
        """Simulate the structural model on new data

        Args:
            data_in (pd.DataFrame): An input data frame containing the following columns
            - `id`
            - `protocol_arm`
            - `output_name`
            - `time`
            - one column per patient descriptor (PDU or MI, discard the protocol overrides)

        Returns:
            pd.DataFrame: Same structure as the input data, with an additional `predicted_value` column
        """
        return data_in


class StructuralGp(StructuralModel):
    """Wrapper class to integrate GP surrogate models with NLME models"""

    def __init__(self, gp_model: GP, time_steps: np.ndarray):
        """Create a structural model from a GP

        Args:
            gp_model (GP): The trained GP
            time_steps (np.ndarray): The total list of time steps. Each observation time for each patient should be contained in this array
        """
        # list the GP parameters, except time, as it will be handled differently in the NLME model
        parameter_names = [p for p in gp_model.parameter_names if p != "time"]
        output_names = gp_model.output_names
        super().__init__(parameter_names, output_names)
        self.gp_model = gp_model
        self.time_steps_df = pd.DataFrame({"time": time_steps})

    def simulate(self, data_in: pd.DataFrame) -> pd.DataFrame:
        input_for_gp = data_in.merge(self.time_steps_df, how="cross")
        if "value" in data_in.columns.to_list():
            input_for_gp = input_for_gp.drop(columns=["value"])
        out = self.gp_model.predict_new_data(input_for_gp)
        out_filt = out[
            ["id", "protocol_arm", "output_name", "time"]
            + self.parameter_names
            + ["pred_mean"]
        ].rename(columns={"pred_mean": "predicted_value"})
        return out_filt


class StructuralOdeModel(StructuralModel):
    """Wrapper class to define an NLME model from an ODE system"""

    def __init__(
        self,
        ode_model: OdeModel,
        protocol_design: pd.DataFrame,
        init_conditions: np.ndarray,
        time_steps: np.ndarray,
    ):

        protocol_overrides = set(
            protocol_design.drop("protocol_arm", axis=1).columns.to_list()
        )
        parameter_names = set(ode_model.param_names) - protocol_overrides

        output_names = ode_model.variable_names
        super().__init__(list(parameter_names), output_names)
        self.ode_model = ode_model
        self.protocol_design = protocol_design
        self.init_conditions = init_conditions
        self.time_steps = time_steps

    def simulate(self, data_in: pd.DataFrame) -> pd.DataFrame:
        out = self.ode_model.run_trial(
            data_in, self.init_conditions, self.protocol_design, self.time_steps
        )
        return out


# Note: cannot give a covariate map without knowing which patients we want to simulate
# In any case, the patient set is required in order to initialize the nlme model
# (covariates or not)


class NLMEModel:
    def __init__(
        self,
        structural_model: StructuralModel,
        patients_df: pd.DataFrame,
        init_log_MI: Dict[str, float],
        init_PDU: Dict[str, Dict[str, float]],
        init_res_var: List[float],
        covariate_map: Optional[dict[str, dict[str, dict[str, str | float]]]] = None,
        error_model_type: str = "additive",
    ):
        """Create a non-linear mixed effects model

        Using a structural model (simulation model) and a covariate structure, create a non-linear mixed effects model, to be used in PySAEM, or to predict data using a covariance structure.

        Args:
            structural_model (StructuralModel): A simulation model defined via the convenience class StructuralModel
            patients_df (DataFrame): the list of patients to be considered, with potential covariate values listed, and the name of the protocol arm on which the patient was evaluated (optional - if not supplied, `identity` will be used)
            init_log_MI: for each model intrinsic parameter, provide an initial value (log)
            init_PDU: for each patient descriptor unknown parameter, provide an initial mean and sd of the log
            init_res_var: for each model output, provide an initial residual variance The `id` column is expected, any additional column will be handled as a covariate
            covariate_map (optional[dict]): for each PDU, the list of covariates that affect it - each associated with a covariation coefficient (to be calibrated)
            Exampel
                {"pdu_name":
                    {"covariate_name":
                        {"coef": "coef_name", "value": initial_value}
                    }
                }
            error_model_type (str): either `additive` or `proportional` error model
        """
        self.structural_model: StructuralModel = structural_model

        self.MI_names: List[str] = list(init_log_MI.keys())
        self.nb_MI: int = len(self.MI_names)
        self.initial_log_MI = torch.Tensor([val for _, val in init_log_MI.items()])
        self.PDU_names: List[str] = list(init_PDU.keys())
        self.nb_PDU: int = len(self.PDU_names)
        self.descriptors: List[str] = self.MI_names + self.PDU_names
        self.nb_descriptors: int = len(self.descriptors)
        self.patients_df: pd.DataFrame = patients_df
        self.patients: List[str | int] = self.patients_df["id"].unique().tolist()
        self.nb_patients: int = len(self.patients)
        # Analyze the provided patients data file
        covariate_columns = self.patients_df.columns.to_list()
        if "protocol_arm" not in covariate_columns:
            self.patients_df["protocol_arm"] = "identity"
        self.protocols = self.patients_df["protocol_arm"].unique().tolist()
        self.nb_protocols = len(self.protocols)

        additional_columns: List[str] = self.patients_df.drop(
            ["id", "protocol_arm"], axis=1
        ).columns.tolist()

        init_betas_list: List = []
        if covariate_map is None:
            print(
                f"No covariate map provided. All additional columns in `patients_df` will be handled as known descriptors: {additional_columns}"
            )
            self.covariate_map = None
            self.covariate_names = []
            self.nb_covariates = 0
            self.population_betas_names = self.PDU_names
            init_betas_list = [val["mean"] for _, val in init_PDU.items()]
            self.PDK_names = additional_columns
            self.nb_PDK = len(self.PDK_names)
        else:
            self.covariate_map = covariate_map
            self.population_betas_names: List = []
            covariate_set = set()
            pdk_names = set(additional_columns)
            for PDU_name in self.PDU_names:
                self.population_betas_names.append(PDU_name)
                init_betas_list.append(init_PDU[PDU_name]["mean"])
                if PDU_name not in covariate_map:
                    raise ValueError(
                        f"No covariate map listed for {PDU_name}. Add an empty set if it has no covariate."
                    )
                for covariate, coef in self.covariate_map[PDU_name].items():
                    if covariate not in additional_columns:
                        raise ValueError(
                            f"Covariate appears in the map but not in the patient set: {covariate}"
                        )
                    if covariate is not None:
                        covariate_set.add(covariate)
                        pdk_names.remove(covariate)
                        coef_name = coef["coef"]
                        coef_val = coef["value"]
                        self.population_betas_names.append(coef_name)
                        init_betas_list.append(coef_val)
            self.covariate_names = list(covariate_set)
            self.nb_covariates = len(self.covariate_names)
            self.PDK_names = list(pdk_names)
            self.nb_PDK = len(self.PDK_names)

        print(f"Successfully loaded {self.nb_covariates} covariates:")
        print(self.covariate_names)
        if self.nb_PDK > 0:
            print(f"Successfully loaded {self.nb_PDK} known descriptors:")
            print(self.PDK_names)

        if set(self.PDK_names + self.PDU_names + self.MI_names) != set(
            self.structural_model.parameter_names
        ):
            raise ValueError(
                f"Non-matching descriptor set and structural model parameter set:\n{set(self.PDK_names + self.PDU_names + self.MI_names)}\n{set(self.structural_model.parameter_names)}"
            )
        self.initial_betas = torch.Tensor(init_betas_list)
        self.nb_betas: int = len(self.population_betas_names)
        self.outputs_names: List[str] = self.structural_model.output_names
        self.nb_outputs: int = self.structural_model.nb_outputs
        self.error_model_type: str = error_model_type
        self.init_res_var = torch.Tensor(init_res_var)
        self.init_omega = torch.diag(
            torch.tensor([float(pdu["sd"] ** 2) for pdu in init_PDU.values()])
        )

        # Assemble the list of design matrices from the covariance structure
        self.design_matrices = self._create_all_design_matrices()

        # Initiate the nlme parameters
        self.log_MI = self.initial_log_MI
        self.population_betas = self.initial_betas
        self.omega_pop = self.init_omega
        self.omega_pop_lower_chol = torch.linalg.cholesky(self.omega_pop)
        self.residual_var = self.init_res_var
        self.eta_distribution = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.nb_PDU),
            covariance_matrix=self.omega_pop,
        )

    def _create_design_matrix(self, covariates: Dict[str, float]) -> torch.Tensor:
        """
        Creates the design matrix X_i for a single individual based on the model's covariate map.
        This matrix will be multiplied with population betas so that log(theta_i[PDU]) = X_i @ betas + eta_i.
        """
        design_matrix_X_i = torch.zeros((self.nb_PDU, self.nb_betas))
        col_idx = 0
        for i, PDU_name in enumerate(self.PDU_names):
            design_matrix_X_i[i, col_idx] = 1.0
            col_idx += 1
            if self.covariate_map is not None:
                for covariate, coef in self.covariate_map[PDU_name].items():
                    design_matrix_X_i[i, col_idx] = float(covariates[covariate])
                    col_idx += 1
        return design_matrix_X_i

    def _create_all_design_matrices(self) -> Dict[Union[str, int], torch.Tensor]:
        """Creates a design matrix for each unique individual based on their covariates, given the in the covariates_df."""
        design_matrices = {}
        if self.nb_covariates == 0:
            for ind_id in self.patients:
                design_matrices[ind_id] = self._create_design_matrix({})
        else:
            for ind_id in self.patients:
                individual_covariates = (
                    self.patients_df[self.patients_df["id"] == ind_id]
                    .iloc[0]
                    .drop("id")
                )
                covariates_dict = individual_covariates.to_dict()
                design_matrices[ind_id] = self._create_design_matrix(covariates_dict)
        return design_matrices

    def add_observations(self, observations_df: pd.DataFrame) -> None:
        """Associate the NLME model with a data frame of observations

        Args:
            observations_df (pd.DataFrame): A data frame of observations, with columns
            - `id`: the patient id. Should be consistent with self.patients_df
            - `time`: the observation time
            - `output_name`
            - `value`
        """
        # Data validation
        input_columns = observations_df.columns.tolist()
        unique_outputs = observations_df["output_name"].unique().tolist()
        if "id" not in input_columns:
            raise ValueError(
                "Provided observation data frame should contain `id` column."
            )
        input_patients = observations_df["id"].unique()
        if set(input_patients) != set(self.patients):
            # Note this check might be unnecessary
            raise ValueError(
                f"Missing observations for the following patients: {set(self.patients) - set(input_patients)}"
            )
        if "time" not in input_columns:
            raise ValueError(
                "Provided observation data frame should contain `time` column."
            )
        if not (set(unique_outputs) <= set(self.outputs_names)):
            raise ValueError(
                f"Unknown model output: {set(unique_outputs) - set(self.outputs_names)}"
            )
        if hasattr(self, "observations_df"):
            print(
                "Warning: overriding existing observation data frame for the NLME model"
            )
        if "value" not in input_columns:
            raise ValueError(
                "The provided observations data frame does not contain a `value` column."
            )
        self.observations_df = observations_df[["id", "output_name", "time", "value"]]

    def update_omega(self, omega: torch.Tensor) -> None:
        self.omega_pop = omega
        self.omega_pop_lower_chol = torch.linalg.cholesky(self.omega_pop)
        self.eta_distribution = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.nb_PDU),
            covariance_matrix=self.omega_pop,
        )

    def update_res_var(self, residual_var: torch.Tensor) -> None:
        self.residual_var = residual_var

    def update_betas(self, betas: torch.Tensor) -> None:
        self.population_betas = betas

    def update_log_mi(self, log_MI: torch.Tensor) -> None:
        self.log_MI = log_MI

    def sample_individual_etas(self) -> torch.Tensor:
        """Sample individual random effects from the current estimate of Omega

        Returns:
            torch.Tensor (size nb_patients x nb_PDUs): individual random effects for all patients in the population
        """
        etas_dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.nb_PDU),
            covariance_matrix=self.omega_pop,
        ).expand([self.nb_patients])
        etas = etas_dist.sample()
        return etas

    def individual_parameters(
        self,
        individual_etas: torch.Tensor,
        ind_ids_for_etas: List[Union[str, int]],
    ) -> torch.Tensor:
        """Compute individual patient parameters

        Transforms log(MI) (Model intrinsic), betas: log(mu)s & coeffs for covariates and individual random effects (etas) into individual parameters (theta_i), for each set of etas of the list and corresponding design matrix.
        Assumes log-normal distribution for individual parameters and covariate effects: theta_i[PDU] = mu_pop * exp(eta_i) * exp(covariates_i * cov_coeffs) where eta_i is from N(0, Omega) and theta_i[MI]=MI.

        Args:
            individual_etas (torch.Tensor): one set of sampled random effects for each patient
            ind_ids_for_etas (List[Union[str, int]]): List of individual ids corresponding to the sampled etas, used to fetch the design matrices
        Returns:
            torch.Tensor [nb_patients x nb_parameters]: One parameter set for each patient. Dim 0 corresponds to the patients, dim 1 is the parameters
        """
        # Gather the necessary design matrices
        list_design_matrices = [
            self.design_matrices[ind_id] for ind_id in ind_ids_for_etas
        ]
        nb_patients_for_etas = len(ind_ids_for_etas)
        # compute individual parameters
        stacked_X = torch.stack(
            list_design_matrices
        )  # stack all design matrices into a single large tensor
        log_thetas_PDU = stacked_X @ self.population_betas + individual_etas
        log_MI_expanded = self.log_MI.unsqueeze(0).repeat(nb_patients_for_etas, 1)
        thetas = torch.exp(torch.cat((log_MI_expanded, log_thetas_PDU), dim=1))

        return thetas

    def param_tensor_to_df(
        self, thetas: torch.Tensor, ind_ids: List[str | int]
    ) -> pd.DataFrame:
        """Transform individual parameter tensor to data frame

        Args:
            thetas (torch.Tensor): Tensor of individual parameter values
            ind_ids (List[str  |  int]): List of patient ids correponding to the thetas

        Returns:
            pd.DataFrame: Data frame with one column per descriptor + `id` column
        """
        out_df = pd.DataFrame(data=thetas.numpy(), columns=self.descriptors)
        out_df.insert(0, "id", ind_ids)
        return out_df

    def sample_patients(self) -> pd.DataFrame:
        """Generate patients using the current population parameters

        Returns:
            pd.DataFrame: parameter values for each generated patient
        """
        etas = self.sample_individual_etas()
        thetas = self.individual_parameters(etas, self.patients)
        output_df = self.param_tensor_to_df(thetas, self.patients)
        return output_df

    def predict_outputs_from_theta(self, thetas: torch.Tensor) -> pd.DataFrame:
        """Return model predictions for all patients

        Args:
            individual_parameters (torch.Tensor): Parameter values per patient (one by row)

        Returns:
            pd.DataFrame: predicted values for all patients
        """
        input_df = self.param_tensor_to_df(thetas, self.patients)
        full_df = input_df.merge(self.patients_df, on="id")
        outputs = pd.DataFrame({"output_name": self.structural_model.output_names})
        full_df = full_df.merge(outputs, how="cross")
        out = self.structural_model.simulate(full_df)

        return out

    def generate_dataset_from_omega(self) -> pd.DataFrame:
        """Generate a synthetic data set from the current estimates of the pop distribution

        Returns:
            pd.DataFrame: A data frame with the following columns
            - `id`
            - one column per `self.descriptor`
            - `time`
            - `output_name`
            - `value`
        """
        input_df = self.sample_patients()
        full_df = input_df.merge(self.patients_df, on="id")
        outputs_df = pd.DataFrame({"output_name": self.structural_model.output_names})
        full_df = full_df.merge(outputs_df, how="cross")
        out = self.structural_model.simulate(full_df)
        out = out[
            ["id"]
            + self.descriptors
            + ["time", "protocol_arm", "output_name", "predicted_value"]
        ].rename(columns={"predicted_value": "value"})
        return out

    def _log_prior_etas(self, etas: torch.Tensor) -> torch.Tensor:
        """Compute log-prior of random effect samples (etas)

        Args:
            etas (torch.Tensor): Individual samples, assuming eta_i ~ N(0, Omega)

        Returns:
            torch.Tensor [nb_eta_i x nb_PDU]: Values of log-prior, computed according to:

            P(eta) = (1/sqrt((2pi)^k * |Omega|)) * exp(-0.5 * eta.T * omega.inv * eta)
            log P(eta) = -0.5 * (k * log(2pi) + log|Omega| + eta.T * omega.inv * eta)

        """

        log_priors: torch.Tensor = self.eta_distribution.log_prob(etas)
        return log_priors

    def _log_posterior_etas(
        self, etas: torch.Tensor, ind_ids_for_etas: List[Union[str, int]]
    ) -> Tuple[torch.Tensor, pd.DataFrame, pd.DataFrame]:
        """Compute the log-posterior of a list of random effects

        Args:
            etas (torch.Tensor): Random effects samples
            ind_ids_for_etas (List[Union[str, int]]): Patient ids corresponding to each eta

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], DataFrame]:
            - log-posterior likelihood of etas
            - data frame of individual parameter values
            - data frame of observations from the simulation model

        """
        if not hasattr(self, "observations_df"):
            raise ValueError(
                "Cannot compute log-posterior without an associated observations data frame."
            )

        # Get individual parameters in a tensor
        individual_params: torch.Tensor = self.individual_parameters(
            individual_etas=etas,
            ind_ids_for_etas=ind_ids_for_etas,
        )
        input_data_for_model = self.param_tensor_to_df(
            individual_params, ind_ids_for_etas
        )
        full_df = input_data_for_model.merge(self.patients_df, on="id")
        outputs = pd.DataFrame({"output_name": self.structural_model.output_names})
        full_df = full_df.merge(outputs, how="cross")

        # call the structural model with the DataFrame input
        # Note - this is the computationally intensive step
        predictions_df = self.structural_model.simulate(full_df)

        # calculate log-prior of the random samples
        log_priors: torch.Tensor = self._log_prior_etas(etas)

        # merge observations with predictions to calculate residuals
        merged_df = pd.merge(
            self.observations_df,
            predictions_df,
            on=["id", "output_name", "time"],
            how="left",
        )
        # group by individual and calculate log-likelihood for each
        list_log_lik_obs: List[float] = []
        for ind_id in ind_ids_for_etas:
            ind_df = merged_df.loc[merged_df["id"] == ind_id]
            total_log_lik_for_ind = 0.0
            for output_name, output_df in ind_df.groupby("output_name"):
                # Safe conversion to str
                output_name_str = str(output_name)
                output_idx = self.outputs_names.index(output_name_str)
                observed_data = torch.tensor(output_df["value"].values)
                predictions = torch.tensor(output_df["predicted_value"].values)

                total_log_lik_for_ind += self.log_likelihood_observation(
                    observed_data,
                    predictions,
                    self.residual_var[output_idx],
                )
            list_log_lik_obs.append(total_log_lik_for_ind)

        log_lik_obs = torch.tensor(list_log_lik_obs)
        log_posterior = log_lik_obs + log_priors

        return log_posterior, input_data_for_model, predictions_df

    def calculate_residuals(
        self, observed_data: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Calculates residuals based on the error model for a single patient

        Args:
            observed_data: torch.Tensor of observations for one individual. dim: [nb_outputs x nb_time_points]
            predictions: torch.Tensor of predictions for one individual. Must be organized like observed_data to compare both by subtraction, dim: [nb_outputs x time_steps]

        Returns:
            torch.Tensor: a tensor of residual values
        """
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

    def error_model_per_row(self, row: pd.DataFrame) -> pd.Series:
        if self.error_model_type == "additive":
            return row["value"] - row["predicted_value"]
        elif self.error_model_type == "proportional":
            return (row["value"] - row["predicted_value"]) / row["predicted_value"]
        else:
            raise ValueError(f"Incorrect error model choice {self.error_model_type}")

    def sum_sq_residuals(self, predictions_df: pd.DataFrame) -> torch.Tensor:
        obs_vs_predicted = pd.merge(
            self.observations_df,
            predictions_df,
            on=["id", "output_name", "time"],
            how="left",
        )
        obs_vs_predicted["residual"] = obs_vs_predicted.apply(
            lambda row: np.pow(self.error_model_per_row(row), 2), axis=1
        )
        summary = (
            obs_vs_predicted.groupby(["output_name"])
            .agg({"residual": "sum", "value": "count"})
            .reset_index()
        ).rename(columns={"residual": "sum_of_sq_residuals", "value": "n_obs"})
        summary["res_var"] = summary.apply(
            lambda row: row["sum_of_sq_residuals"] / row["n_obs"], axis=1
        )
        return torch.Tensor(summary["res_var"].values)

    def log_likelihood_observation(
        self,
        observed_data: torch.Tensor,
        predictions: torch.Tensor,
        residual_error_var: torch.Tensor,
    ) -> float:
        """
        Calculates the log-likelihood of observations given predictions and error model, assuming errors follow N(0,sqrt(residual_error_var))
        observed_data: torch.Tensor of observations for one individual
        predictions: torch.Tensor of predictions for one individual organized in the same way as observed_data
        residual_error_var: torch.Tensor of the error for each output, dim: [nb_outputs]
        """
        if torch.any(torch.isinf(predictions)) or torch.any(torch.isnan(predictions)):
            return -torch.inf  # invalid predictions
        residuals: torch.Tensor = self.calculate_residuals(observed_data, predictions)
        # ensure error_std is positive
        res_error_var = torch.maximum(
            torch.full_like(residual_error_var, 1e-6), residual_error_var
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

    def init_mcmc_sampler(
        self,
        observations_df: pd.DataFrame,
        verbose: bool,
    ) -> None:
        self.add_observations(observations_df)
        self.verbose = verbose

    def mcmc_sample(
        self,
        init_eta_for_all_ind: torch.Tensor,
        proposal_var_eta: torch.Tensor,
        nb_samples: int,
        nb_burn_in: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """Metropolis-Hastings sampling of individual MCMC

        Performs Metropolis-Hastings sampling for the individuals' etas. The MCMC chains (one per individual) advance in parallel.
        The acceptance criterion for each sample is
                log(random_uniform) < proposed_log_posterior - current_log_posterior

        Returns the mean of sampled etas per individual, the mean of the log_thetas_PDU (individual PDU parameters associated with the sampled etas) and the mean predictions associated with these individual parameters/etas.

        Args:
            init_eta_for_all_ind: torch.Tensor of dim [nb_individuals x nb_PDUs]: the current samples, i.e. starting points for each chain
            nb_samples: int, how many samples will be kept from each chain
            nb_burn_in: int, how many accepted samples are disgarded before we consider that the chain has converged enough

        Returns:

        """
        if not hasattr(self, "observations_df"):
            raise ValueError("Please run `init_mcmc_sampler` first.")

        all_states_history: List[torch.Tensor] = []

        sum_etas = torch.zeros_like(init_eta_for_all_ind)
        sum_log_thetas_PDU = torch.zeros(self.nb_patients, self.nb_PDU)

        predictions_history = []
        current_eta_for_all_ind = init_eta_for_all_ind
        current_log_posteriors, _, _ = self._log_posterior_etas(
            init_eta_for_all_ind, self.patients
        )

        sample_counts = torch.zeros(self.nb_patients)
        accepted_counts = torch.zeros(self.nb_patients)
        total_proposals = torch.zeros(self.nb_patients)
        done = torch.full((self.nb_patients, 1), False)

        while not torch.all(done).item():
            active_indices = torch.where(~done)[0]
            active_ind_ids = [self.patients[i] for i in active_indices]

            total_proposals[active_indices] += 1
            proposal_dist = torch.distributions.MultivariateNormal(
                current_eta_for_all_ind[active_indices], proposal_var_eta
            )
            proposed_etas: torch.Tensor = proposal_dist.sample()

            proposed_log_posteriors, patients_df, proposed_predictions_df = (
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
                    accepted_ind_id = self.patients[idx]
                    sum_etas[idx] += current_eta_for_all_ind[idx]
                    patient_descriptors = patients_df.loc[
                        patients_df["id"] == accepted_ind_id
                    ][self.PDU_names]
                    theta = torch.Tensor(patient_descriptors.to_numpy()).squeeze(0)
                    sum_log_thetas_PDU[idx] += torch.log(theta)

                    # store predictions for averaging later and then use in the M-step
                    accepted_preds_for_ind = proposed_predictions_df[
                        proposed_predictions_df["id"] == accepted_ind_id
                    ]
                    predictions_history.append(accepted_preds_for_ind)

                    sample_counts[idx] += 1
                    if sample_counts[idx] == nb_samples:
                        done[idx] = True

            all_states_history.append(current_eta_for_all_ind)
        # calculate mean etas and log of individual parameters PDU
        mean_etas = sum_etas / nb_samples
        mean_log_thetas_PDU = sum_log_thetas_PDU / nb_samples

        # calculate mean predictions from the collected history
        if predictions_history:
            mean_predictions_df = (
                pd.concat(predictions_history, ignore_index=True)
                .groupby(["id", "output_name", "protocol_arm", "time"])
                .agg({"predicted_value": "mean"})
            )
        else:
            mean_predictions_df = pd.DataFrame(
                columns=["id", "output_name", "protocol_arm", "time", "predicted_value"]
            )

        if self.verbose:
            acceptance_rate = accepted_counts / total_proposals
            print(f"Average acceptance: {acceptance_rate.mean():.2f}")
        return (mean_etas, mean_log_thetas_PDU, mean_predictions_df)
