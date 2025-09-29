import pandas as pd
import torch
import numpy as np
import uuid
from math import sqrt
from scipy.integrate import solve_ivp
from scipy.stats.qmc import Sobol, scale
from multiprocessing import Pool
from typing import List, Dict, Union, Tuple
from PySAEM import NLMEModel
import pandas as pd


torch.set_default_dtype(torch.float32)


class DataGenerator:
    """
    The base class for generating data from an ODE solver. This is designed to integrate with the GP surrogate modeling module.

    Todo: integrate with the API to generate data using a trial on jinko?
    """

    def __init__(
        self,
        equations: callable,
        variable_names: List[str],
        param_names: List[str],
    ):
        """Data generator base class

        Create a data generator given a set of equations.

        Args:
            equations (callable): A function describing the right hand side of the ODE system
            variable_names (List[str]): The names of the outputs of the system
            param_names (List[str]): The name of the parameters of the system
        """
        self.equations = equations
        if (
            isinstance(variable_names, tuple)
            and len(variable_names) == 1
            and isinstance(variable_names[0], list)
        ):
            self.variable_names = variable_names[0]
        else:
            self.variable_names = variable_names
        self.nb_outputs = len(variable_names)

        if (
            isinstance(param_names, tuple)
            and len(param_names) == 1
            and isinstance(param_names[0], list)
        ):
            self.param_names = param_names[0]
        else:
            self.param_names = param_names
        self.nb_parameters = len(param_names)
        self.initial_cond_names = [v + "_0" for v in self.variable_names]

    # this needs to be a static method for the multiprocessing to work (it cannot handle the self object)
    @staticmethod
    def _structural_model_worker(args: dict):
        """Worker function to simulate model on a single patient

        Args:
            args (dict): describes the simulation to be performed. Requires
                ind_id: the id of the patient to be simulated
                time_steps_dict: the time steps to be evaluated at, one np.array for each output of the model
                initial_conditions: the initial conditions, np.array of length nb outputs
                params_np: the parameters of the current patient
                output_names: output names
                equations: system right-hand side
                tol: solver tolerance

        Returns:
            List(dict): A list of model result entries
        """

        # extract args
        ind_id = args["ind_id"]
        time_steps_dict = args["time_steps_dict"]
        initial_conditions = args["initial_conditions"]
        params_np = args["params_np"]
        output_names = args["output_names"]
        equations = args["equations"]
        tol = args["tol"]

        all_time_steps = sorted(
            list(
                set(
                    ts.item()
                    for ts in [
                        t for sublist in time_steps_dict.values() for t in sublist
                    ]
                )
            )
        )
        if not all_time_steps:
            return [np.array([])] * len(output_names)

        time_span = (all_time_steps[0], all_time_steps[-1])

        sol = solve_ivp(
            equations,
            time_span,
            initial_conditions,
            method="LSODA",
            t_eval=all_time_steps,
            rtol=tol,
            atol=tol,
            args=params_np,
        )
        if not sol.success:
            raise ValueError(f"ODE integration failed: {sol.message}")

        # Filter the solver output to keep only the requested time steps for each output
        result = []
        for output_name in output_names:
            if output_name in time_steps_dict:
                output_index = output_names.index(output_name)
                requested_time_steps = [
                    ts.item() for ts in time_steps_dict[output_name]
                ]
                requested_time_steps_indices = [
                    all_time_steps.index(ts) for ts in requested_time_steps
                ]
                predicted_values = sol.y[output_index, requested_time_steps_indices]
                individual_result = pd.DataFrame.from_dict(
                    {
                        "ind_id": ind_id,
                        "output_name": output_name,
                        "time": requested_time_steps,
                        "predicted_value": predicted_values,
                    }
                )
                result.append(individual_result)
        return pd.concat(result)

    def structural_model(
        self,
        input_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Solve the structural model using formatted input data

        Args:
            input_data (pd.DataFrame): a DataFrame containing 'ind_id', 'output_name', 'time', and columns for all individual parameters and initial conditions ('var_0', for var in variable_names) for each patient

        Returns:
            pd.DataFrame: a DataFrame with the same inputs and a new 'predicted_value' column
        """

        # group the data by individual to create tasks for each process
        tasks = []
        for ind_id, ind_df in input_data.groupby("ind_id"):
            time_steps_dict = {
                output_name: output_df["time"].values
                for output_name, output_df in ind_df.groupby("output_name")
            }

            params_np = ind_df.iloc[0].loc[self.param_names].values
            initial_conditions_np = ind_df.iloc[0].loc[self.initial_cond_names].values
            indiv_task = {
                "ind_id": ind_id,
                "time_steps_dict": time_steps_dict,
                "initial_conditions": initial_conditions_np,
                "params_np": params_np,
                "output_names": self.variable_names,
                "equations": self.equations,
                "tol": 1e-7,
            }
            tasks.append(indiv_task)
        with Pool() as pool:
            all_solutions = pool.map(DataGenerator._structural_model_worker, tasks)
        output_data = pd.concat(all_solutions)
        return output_data

    def simulate_wide_dataset_from_ranges(
        self,
        log_nb_individuals: int,
        param_ranges: list[dict],
        initial_conditions: np.array,
        residual_error_variance: np.array,
        error_model: str,  # "additive" or "proportional"
        time_steps: np.array,
    ) -> pd.DataFrame:
        """Generate a simulated data set with an ODE model

        Simulates a dataset for training a surrogate model. Timesteps can be different for each output.
        The parameter space is explored with Sobol sequences.

        Args:
            log_nb_individuals (int): The number of simulated patients will be 2^this parameter
            param_ranges (list[dict]): For each parameter in the model, a dict describing the search space 'low': low bound, 'high': high bound, and 'log': True if the search space is log-scaled
            initial_conditions (np.array): set of initial conditions
            residual_error_variance (np.array): A 1D array of residual error variances for each output.
            error_model (str): the type of error model ("additive" or "proportional").
            time_steps (np.array): an array with the time points
        Returns:
            pd.DataFrame: A DataFrame with columns 'ind_id', parameter names, 'time', 'output_name', and 'value'.
        """
        nb_individuals = np.power(2, log_nb_individuals)

        # Validate input data
        if not (len(param_ranges) == self.nb_parameters):
            raise ValueError(
                "Parameter bounds are not the same length as the number of parameters."
            )
        if initial_conditions.size != self.nb_outputs:
            raise ValueError("Invalid count of initial conditions supplied.")

        # Create a sobol sampler to generate parameter values
        sobol_engine = Sobol(d=self.nb_parameters, scramble=True)
        sobol_sequence = sobol_engine.random_base2(log_nb_individuals)
        samples = scale(
            sobol_sequence,
            [param_ranges[param_name]["low"] for param_name in self.param_names],
            [param_ranges[param_name]["high"] for param_name in self.param_names],
        )

        # Handle log-scaled parameters
        for j, param_name in enumerate(self.param_names):
            if param_ranges[param_name]["log"] == True:
                samples[:, j] = np.exp(np.log(10) * samples[:, j])
        ids = [str(uuid.uuid4()) for _ in range(nb_individuals)]
        # Create the full data frame of patient descriptors
        patients_df = pd.DataFrame(data=samples, columns=self.param_names)
        patients_df.insert(0, "ind_id", ids)

        time_steps_df = pd.concat(
            [
                pd.DataFrame({"output_name": output_name, "time": time_steps})
                for output_name in self.variable_names
            ]
        )

        init_cond_df = pd.DataFrame(
            data=[initial_conditions], columns=self.initial_cond_names
        )

        all_data = patients_df.merge(time_steps_df, how="cross")
        all_data = all_data.merge(init_cond_df, how="cross")

        output = self.structural_model(all_data)
        merged_df = pd.merge(
            all_data, output, on=["ind_id", "output_name", "time"], how="left"
        )
        wide_output = merged_df.pivot_table(
            index=["ind_id", *self.param_names, "time"],
            columns="output_name",
            values="predicted_value",
        ).reset_index()
        # Add noise to the data
        noise = np.random.normal(
            np.zeros_like(residual_error_variance),
            np.sqrt(residual_error_variance),
            (nb_individuals * time_steps.size, self.nb_outputs),
        )
        if error_model == "additive":
            wide_output[self.variable_names] += noise
        elif error_model == "proportional":
            wide_output[self.variable_names] += noise * wide_output[self.variable_names]
        return wide_output

    def simulate_dataset_from_omega(
        self,
        nb_individuals: int,
        true_MI: torch.Tensor,
        true_betas: torch.Tensor,
        true_omega: torch.Tensor,
        true_residual_var: torch.Tensor,
        list_covariates_dict: Union[None, List[Dict[str, float]]],
        list_time_steps: List[List[torch.Tensor]],
        pk_model: NLMEModel,
        initial_conditions: np.array,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates a DataFrame for observations and a DataFrame for covariates
        for the new PySAEM class, adapted to work with DataGenerator's structural_model.

        Returns:
        - observations_df: A pandas DataFrame with columns ['ind_id', 'output_name', 'time', 'value'].
        - list_covariates_dict: a list of dictionaries (each corresponding to one covariate to simulate), with the name, mean and std (only the normal distribution is supported for now)
        """
        all_covariates = []
        input_data_for_structural_model = []

        # Sample random effects (etas) and covariates
        distrib_etas = torch.distributions.MultivariateNormal(
            loc=torch.zeros(true_omega.shape[0]), covariance_matrix=true_omega
        )
        etas = distrib_etas.sample((nb_individuals,))

        if list_covariates_dict is not None and len(list_covariates_dict) > 0:
            for covariate in list_covariates_dict:
                if (
                    "name" not in covariate.keys()
                    or "mean" not in covariate.keys()
                    or "var" not in covariate.keys()
                ):
                    raise ValueError(
                        "covariates_list_dict dictionaries must contain keys name, mean and var."
                    )

                values = torch.normal(
                    mean=torch.full((nb_individuals,), float(covariate["mean"])),
                    std=torch.full((nb_individuals,), sqrt(covariate["var"])),
                )
                covariate.update({"values": values})

        for i in range(nb_individuals):
            ind_id = str(uuid.uuid4())
            if list_covariates_dict is not None and len(list_covariates_dict) > 0:
                covariates_ind_dict = {"ind_id": ind_id}
                for covariate in list_covariates_dict:
                    covariates_ind_dict.update(
                        {covariate["name"]: covariate["values"][i].item()}
                    )
                all_covariates.append(covariates_ind_dict)

            # Create a design matrix for parameter calculation
            design_matrix_X_i = torch.zeros((pk_model.nb_PDU, pk_model.nb_betas))
            col_idx = 0
            for pdu_name in pk_model.PDU_names:
                design_matrix_X_i[pk_model.PDU_names.index(pdu_name), col_idx] = 1.0
                col_idx += 1
                if pk_model.covariate_map and pdu_name in pk_model.covariate_map:
                    for cov_name in pk_model.covariate_map[pdu_name]:
                        design_matrix_X_i[
                            pk_model.PDU_names.index(pdu_name), col_idx
                        ] = covariates_ind_dict[cov_name]
                        col_idx += 1
            # Compute individual parameters
            true_individual_params = pk_model.individual_parameters(
                torch.log(true_MI).unsqueeze(-1),
                true_betas,
                etas[i].unsqueeze(0),
                design_matrix_X_i,
            )[0]
            params = pk_model.MI_names + pk_model.PDU_names
            params_dict = {
                name: val.item() for name, val in zip(params, true_individual_params)
            }
            # Prepare the input DataFrame for the structural model
            for j, output_name in enumerate(pk_model.outputs_names):
                time_points = list_time_steps[i][j]
                for t in time_points:
                    row = {
                        "ind_id": ind_id,
                        "output_name": output_name,
                        "time": t.item(),
                    }
                    row.update(params_dict)
                    input_data_for_structural_model.append(row)

        # Convert the list of dictionaries to a DataFrame
        input_df = pd.DataFrame(input_data_for_structural_model)
        init_cond_df = pd.DataFrame(
            data=[initial_conditions], columns=self.initial_cond_names
        )

        input_df = input_df.merge(init_cond_df, how="cross")
        # Use the DataGenerator's structural_model to get predictions
        predicted_df = self.structural_model(input_df)

        # Merge the predictions back with the input data
        merged_df = pd.merge(
            input_df, predicted_df, on=["ind_id", "output_name", "time"], how="left"
        )

        all_observations = []
        # Add residual error and format the final observations DataFrame
        for ind_id in merged_df["ind_id"].unique():
            ind_df = merged_df[merged_df["ind_id"] == ind_id]

            for j, output_name in enumerate(pk_model.outputs_names):
                output_df = ind_df[ind_df["output_name"] == output_name].copy()
                if not output_df.empty:
                    true_concentration = torch.tensor(
                        output_df["predicted_value"].values, dtype=torch.float32
                    )

                    if pk_model.error_model_type == "additive":
                        noise = torch.normal(
                            torch.zeros_like(true_concentration),
                            (torch.sqrt(true_residual_var[j])).expand(
                                true_concentration.shape
                            ),
                        )
                        observed_concentration = true_concentration + noise
                    elif pk_model.error_model_type == "proportional":
                        proportional_stdev = torch.sqrt(
                            true_residual_var[j]
                        ) * torch.abs(true_concentration)
                        noise = torch.normal(
                            torch.zeros_like(true_concentration), proportional_stdev
                        )
                        observed_concentration = true_concentration + noise
                    else:
                        raise ValueError("Unsupported error model type.")

                    output_df["value"] = observed_concentration.detach().cpu().numpy()
                    all_observations.extend(
                        output_df[["ind_id", "output_name", "time", "value"]].to_dict(
                            "records"
                        )
                    )

        observations_df = pd.DataFrame(all_observations)
        covariates_df = pd.DataFrame(all_covariates)

        return observations_df, covariates_df
