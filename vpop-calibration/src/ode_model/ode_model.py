import pandas as pd
import numpy as np
import uuid
from scipy.integrate import solve_ivp
from scipy.stats.qmc import Sobol, scale
from multiprocessing import Pool
from typing import List, Any, Callable, Optional
import pandas as pd
import itertools


class OdeModel:
    def __init__(
        self,
        equations: Callable,
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
        self.variable_names = variable_names
        self.nb_outputs = len(variable_names)

        self.param_names = param_names
        self.nb_parameters = len(param_names)
        # Define the name of initial conditions as `<variable>_0`
        self.initial_cond_names = [v + "_0" for v in self.variable_names]

    # this needs to be a static method for the multiprocessing to work (it cannot handle the self object)
    @staticmethod
    def _structural_model_worker(args: dict) -> pd.DataFrame:
        """Worker function to simulate model on a single patient

        Args:
            args (dict): describes the simulation to be performed. Requires
                id (str|int): the id of the patient to be simulated
                protocol_arm (str): the name of the scenario on which it is simulated
                time_steps_dict (dict[str, np.array]): the time steps to be evaluated at, one np.array for each output of the model
                initial_conditions (dict[str, float]): the initial conditions, one for each variable
                params (dict[str, float]): the parameters of the current patient
                output_names (List[str]): output names
                equations (Callable): system right-hand side
                tol (float): solver tolerance

        Returns:
            List(dict): A list of model result entries
        """

        # extract args
        ind_id = args["id"]
        protocol_arm = args["protocol_arm"]
        time_steps_dict = args["time_steps_dict"]
        initial_conditions = args["initial_conditions"]
        params = args["params"]
        output_names = args["output_names"]
        equations = args["equations"]
        tol = args["tol"]

        all_time_steps = sorted(
            list(set(itertools.chain.from_iterable(time_steps_dict.values())))
        )
        if not all_time_steps:
            # no requested time steps
            # return an empty data frame with the right column names
            return pd.DataFrame(
                columns=["id", "protocol_arm", "output_name", "time", "predicted_value"]
            )

        time_span = (all_time_steps[0], all_time_steps[-1])

        sol = solve_ivp(
            equations,
            time_span,
            initial_conditions,
            method="LSODA",
            t_eval=all_time_steps,
            rtol=tol,
            atol=tol,
            args=params,
        )
        if not sol.success:
            raise ValueError(f"ODE integration failed: {sol.message}")

        # Filter the solver output to keep only the requested time steps for each output
        result: list[pd.DataFrame] = []
        for output_name in time_steps_dict:
            output_index = output_names.index(output_name)
            requested_time_steps = time_steps_dict[output_name]
            requested_time_steps_indices = [
                all_time_steps.index(ts) for ts in requested_time_steps
            ]
            predicted_values = sol.y[output_index, requested_time_steps_indices]
            individual_result = pd.DataFrame.from_dict(
                {
                    "id": ind_id,
                    "protocol_arm": protocol_arm,
                    "output_name": output_name,
                    "time": requested_time_steps,
                    "predicted_value": predicted_values,
                }
            )
            result.append(individual_result)
        full_output: pd.DataFrame = pd.concat(result)
        return full_output

    def simulate_model(
        self,
        input_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Solve the ODE model using formatted input data

        Args:
            input_data (pd.DataFrame): a DataFrame containing 'id', 'output_name', 'time', and columns for all individual parameters and initial conditions ('var_0', for var in variable_names) for each patient

        Returns:
            pd.DataFrame: a DataFrame with the same inputs and a new 'predicted_value' column
        """

        # group the data by individual to create tasks for each process
        tasks: list[Any] = []
        for ind_id, ind_df in input_data.groupby("id"):
            for arm_id, filtered_df in ind_df.groupby("protocol_arm"):
                # Construct the dictionary of requested time steps for this patient, on this protocol arm
                time_steps_dict = {
                    output_name: output_df["time"].values
                    for output_name, output_df in filtered_df.groupby("output_name")
                }

                params = filtered_df[self.param_names].iloc[0].values
                initial_conditions = filtered_df[self.initial_cond_names].iloc[0].values
                indiv_task = {
                    "id": ind_id,
                    "protocol_arm": arm_id,
                    "time_steps_dict": time_steps_dict,
                    "initial_conditions": initial_conditions,
                    "params": params,
                    "output_names": self.variable_names,
                    "equations": self.equations,
                    "tol": 1e-6,
                }
                tasks.append(indiv_task)
        with Pool() as pool:
            all_solutions: list[pd.DataFrame] = pool.map(
                OdeModel._structural_model_worker, tasks
            )
        output_data = pd.concat(all_solutions)
        return output_data

    def generate_vpop_from_ranges(
        self, log_nb_individuals: int, param_ranges: dict[str, dict[str, float | bool]]
    ) -> pd.DataFrame:
        """Generate a vpop of patients from parameter ranges

        Args:
            log_nb_individuals (int): The vpop size will be 2^log_nb_individuals
            param_ranges (dict[str, dict[str, float  |  bool]]): One entry for each parameter to be explored
            - `param_name`: {`low`: float, `high`: float, `log`: bool}. Turn `log` to true to define log-scaled ranges

        Returns:
            pd.DataFrame: A set of patients with a generated `id`, and a column per descriptor

        Note:
            This method may be called with an empty dict, to return a list of patient ids.
        """

        nb_individuals = np.power(2, log_nb_individuals)
        params_to_explore = list(param_ranges.keys())
        nb_parameters = len(params_to_explore)
        if nb_parameters != 0:

            # Create a sobol sampler to generate parameter values
            sobol_engine = Sobol(d=nb_parameters, scramble=True)
            sobol_sequence = sobol_engine.random_base2(log_nb_individuals)
            samples = scale(
                sobol_sequence,
                [param_ranges[param_name]["low"] for param_name in params_to_explore],
                [param_ranges[param_name]["high"] for param_name in params_to_explore],
            )

            # Handle log-scaled parameters
            for j, param_name in enumerate(params_to_explore):
                if param_ranges[param_name]["log"] == True:
                    samples[:, j] = np.exp(np.log(10) * samples[:, j])
            # Create the full data frame of patient descriptors
            patients_df = pd.DataFrame(data=samples, columns=params_to_explore)
        else:
            # No parameter requested, create empty data frame
            patients_df = pd.DataFrame()

        ids = [str(uuid.uuid4()) for _ in range(nb_individuals)]
        patients_df.insert(0, "id", ids)
        return patients_df

    def run_trial(
        self,
        vpop: pd.DataFrame,
        initial_conditions: np.ndarray,
        protocol_design: Optional[pd.DataFrame],
        time_steps: np.ndarray,
    ) -> pd.DataFrame:
        """Run a trial given a vpop, protocol and solving times

        Args:
            vpop (pd.DataFrame): The patient descriptors. Should contain the following columns
            - `id`
            - `protocol_arm`
            - `output_name`
            - one column per patient descriptor
            initial_conditions (np.ndarray): one set of initial conditions (same for all patients)
            protocol_design (Optional[pd.DataFrame]): Protocol design linking `protocol_arm` to actual parameter overrides
            time_steps (np.ndarray): The requested observation times. Same for all outputs

        Returns:
            pd.DataFrame: A merged output containing the following columns
            - `id`
            - one column per patient descriptor
            - `protocol_arm`
            - `output_name`
            - `predicted_value`: the simulated value

        Notes:
            Each patient will be run on each protocol arm, and all outputs will be included
        """

        # List the requested time steps for each output (here we use same solving times for all outputs)
        time_steps_df = pd.DataFrame({"time": time_steps})
        # Assemble the initial conditions in a dataframe
        init_cond_df = pd.DataFrame(
            data=[initial_conditions], columns=self.initial_cond_names
        )
        if protocol_design is None:
            protocol_design_to_use = pd.DataFrame({"protocol_arm": "identity"})
        else:
            protocol_design_to_use = protocol_design

        # Merge the data frames together
        # Add time steps and output names for all patients
        full_input_data = vpop.merge(time_steps_df, how="cross")
        # Add initial conditions for all patients
        full_input_data = full_input_data.merge(init_cond_df, how="cross")
        # Add protocol arm info by merging the protocol design
        full_input_data = full_input_data.merge(
            protocol_design_to_use, how="left", on="protocol_arm"
        )
        # Run the model
        output = self.simulate_model(full_input_data)

        merged_df = pd.merge(
            full_input_data,
            output,
            on=["id", "output_name", "time", "protocol_arm"],
            how="left",
        )
        return merged_df

    def simulate_wide_dataset_from_ranges(
        self,
        log_nb_individuals: int,
        param_ranges: dict[str, dict[str, float | bool]],
        initial_conditions: np.ndarray,
        protocol_design: Optional[pd.DataFrame],
        residual_error_variance: Optional[np.ndarray],
        error_model: Optional[str],  # "additive" or "proportional"
        time_steps: np.ndarray,
    ) -> pd.DataFrame:
        """Generate a simulated data set with an ODE model

        Simulates a dataset for training a surrogate model. Timesteps can be different for each output.
        The parameter space is explored with Sobol sequences.

        Args:
            log_nb_individuals (int): The number of simulated patients will be 2^this parameter
            param_ranges (list[dict]): For each parameter in the model, a dict describing the search space 'low': low bound, 'high': high bound, and 'log': True if the search space is log-scaled
            initial_conditions (array): set of initial conditions, one for each variable
            protocol_design (optional): a DataFrame with a `protocol_arm` column, and one column per parameter override
            residual_error_variance (np.array): A 1D array of residual error variances for each output.
            error_model (str): the type of error model ("additive" or "proportional").
            time_steps (np.array): an array with the time points
        Returns:
            pd.DataFrame: A DataFrame with columns 'id', parameter names, 'time', 'output_name', and 'value'.

        Notes:
            If a parameter appears both in the ranges and in the protocol design, the ranges take precedence.
        """

        # Validate input data
        params_to_explore = list(param_ranges.keys())

        if protocol_design is None:
            print("No protocol")
            params = params_to_explore
            params_in_protocol = []
            protocol_design_filt = pd.DataFrame({"protocol_arm": ["identity"]})
        else:
            params_in_protocol = protocol_design.drop(
                "protocol_arm", axis=1
            ).columns.tolist()
            # Find the paramaters that appear both in the ranges and the protocol
            overlap = set(params_to_explore) & set(params_in_protocol)
            if overlap != set():
                protocol_design_filt = protocol_design.drop(list(overlap), axis=1)
                print(
                    f"Warning: ignoring entries {overlap} from the protocol design (already defined in the ranges)."
                )
            else:
                protocol_design_filt = protocol_design

            params = params_to_explore + params_in_protocol
        if set(params) != set(self.param_names):
            raise ValueError(
                f"Under-defined system: missing {set(self.param_names) - set(params)}"
            )
        # Generate the vpop using sobol sequences
        patients_df = self.generate_vpop_from_ranges(log_nb_individuals, param_ranges)

        # Add a choice of protocol arm for each patient
        protocol_arms = pd.DataFrame(
            protocol_design_filt["protocol_arm"].drop_duplicates()
        )
        patients_df = patients_df.merge(protocol_arms, how="cross")
        # Add the outputs for each patient
        outputs = pd.DataFrame({"output_name": self.variable_names})
        patients_df = patients_df.merge(outputs, how="cross")
        # Simulate the ODE model
        output_df = self.run_trial(
            patients_df, initial_conditions, protocol_design_filt, time_steps
        )
        # Pivot to wide to add noise per model output
        wide_output = output_df.pivot_table(
            index=["id", *self.param_names, "time", "protocol_arm"],
            columns="output_name",
            values="predicted_value",
        ).reset_index()

        if error_model is None:
            pass
        else:
            if residual_error_variance is None:
                raise ValueError("Undefined residual error variance.")
            else:
                # Add noise to the data
                noise = np.random.normal(
                    np.zeros_like(residual_error_variance),
                    np.sqrt(residual_error_variance),
                    (wide_output.shape[0], self.nb_outputs),
                )
                if error_model == "additive":
                    wide_output[self.variable_names] += noise
                elif error_model == "proportional":
                    wide_output[self.variable_names] += (
                        noise * wide_output[self.variable_names]
                    )
                else:
                    raise ValueError(f"Incorrect error_model choice: {error_model}")
        # Pivot back to long format
        long_output = wide_output.melt(
            id_vars=[
                "id",
                "protocol_arm",
                "time",
                *self.param_names,
            ],
            value_vars=self.variable_names,
            var_name="output_name",
            value_name="value",
        )
        # Remove the protocol arm overrides from the data set, they described by the protocol_arm column now
        long_output = long_output.drop(params_in_protocol, axis=1)
        return long_output
