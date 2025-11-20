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
# Train a Gaussian Process given a trial
#

# %%
import pandas as pd
import torch
import os
import zipfile
import time
import io
import jinko_helpers as jinko
import sys
from IPython.display import display

from vpop_calibration import GP

# %%
# jinko set-up
jinko.initialize()

# %%
# REPLACE WITH YOUR INFO
trial_sid = "tr-OJvV-CPhT"

output_names = ["A1", "A2"]  # these are the outputs you want to train the GP on

# %% [markdown]
# From now on, no modification is necessary. The notebook will get the patient descriptors of the Vpop, the timeseries of the outputs and train a GP on them.
#

# %%
trial_project_item = jinko.get_project_item(sid=trial_sid)
trial_core_item_id = trial_project_item["coreId"]["id"]
trial_snapshot_id = trial_project_item["coreId"]["snapshotId"]

# %%
trial_info = jinko.make_request(
    path=f"/core/v2/trial_manager/trial/{trial_core_item_id}/snapshots/{trial_snapshot_id}",
).json()
model_core_item_id = trial_info["computationalModelId"]["coreItemId"]
model_snapshot_id = trial_info["computationalModelId"]["snapshotId"]
vpop_core_item_id = trial_info["vpopId"]["coreItemId"]
vpop_snapshot_id = trial_info["vpopId"]["snapshotId"]
protocol_design_core_id = trial_info["protocolDesignId"]["coreItemId"]
protocol_design_snapshot_id = trial_info["protocolDesignId"]["snapshotId"]

# %%
# get virtual patients attributes into a dataframe
response = jinko.make_request(
    path=f"/core/v2/vpop_manager/vpop/{vpop_core_item_id}",
    method="GET",
    json={
        "Accept": "application/json;charset=utf-8, text/csv",
    },
)
vpop_data = response.json()
patient_attributes_list = (
    []
)  # list of dictionaries (one dictionary corresponds to one patient, with its id and parameters)
for patient in vpop_data["patients"]:
    patient_index = patient["patientIndex"]
    attributes = {"id": patient_index}
    for attr in patient["patientAttributes"]:
        attributes[attr["id"]] = attr["val"]
    patient_attributes_list.append(attributes)

pd_names = [attr["id"] for attr in vpop_data["patients"][0]["patientAttributes"]]
display(pd_names)
nb_pds = len(pd_names)
df_patient_attributes = pd.DataFrame(patient_attributes_list)

descriptors = pd_names + ["Time"]
display(descriptors)

# %%
# retrieve protocol arms from the design
response = jinko.make_request(
    path=f"/core/v2/scenario_manager/protocol_design/{protocol_design_core_id}/snapshots/{protocol_design_snapshot_id}",
    method="GET",
    json={
        "Accept": "application/json;charset=utf-8, text/csv",
    },
)
protocol_design = response.json()
protocol_arms = [arm["armName"] for arm in protocol_design["scenarioArms"]]
display(protocol_arms)

selected_protocol_arms = protocol_arms[:2]
display(selected_protocol_arms)

# %%
# retrieve results
timeseries_json = {
    "timeseries": {output: selected_protocol_arms for output in output_names}
}
csvTimeSeries = ""
try:
    response = jinko.make_request(
        path=f"/core/v2/result_manager/trial/{trial_core_item_id}/snapshots/{trial_snapshot_id}/timeseries/download",
        method="POST",
        json=timeseries_json,
    )
    if response.status_code == 200:
        print("Time series data retrieved successfully.")
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        filename = archive.namelist()[0]
        print(f"Extracted time series file: {filename}")

        csvTimeSeries = archive.read(filename).decode("utf-8")

    else:
        print(
            f"Failed to retrieve time series data: {response.status_code} - {response.reason}"
        )
        response.raise_for_status()
except Exception as e:
    print(f"Error during time series retrieval or processing: {e}")
    raise

# %%
# Merge time series with patient descriptors together in a single data frame
df_time_series = pd.read_csv(io.StringIO(csvTimeSeries))
df_time_series = df_time_series.rename(columns={"Patient Id": "id"})
merged_df = pd.merge(df_time_series, df_patient_attributes, on="id").rename(
    columns={
        "Arm": "protocol_arm",
        "Value": "value",
        "Descriptor": "output_name",
        "Time": "time",
    }
)
display(merged_df)

# %%
# initiate our GP class
myGP = GP(
    merged_df,
    pd_names + ["time"],
    var_dist="Chol",
    var_strat="IMV",
    kernel="RBF",
    nb_training_iter=200,
    training_proportion=0.7,
    learning_rate=0.1,
    lr_decay=0.99,
    jitter=1e-6,
)

# %%
myGP.train()

# %%
myGP.eval_perf()
myGP.plot_loss()

# %%
myGP.plot_obs_vs_predicted(data_set="training")

# %%
myGP.plot_all_solutions(data_set="training")
