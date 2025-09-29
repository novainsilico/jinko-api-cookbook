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

sys.path.append("../src")
from GP import *

torch.set_default_dtype(torch.float32)

# %%
# jinko set-up
jinko.initialize()
resources_dir = os.path.normpath("resources/download_models")
if not os.path.exists(resources_dir):
    os.makedirs(resources_dir)

# %%
# REPLACE WITH YOUR INFO
trial_sid = "tr-QE4V-BMVk"

output_names = [
    "Line1:timeToProgression"
]  # these are the outputs you want to train the GP on
arm_name = "Gefitinib"  # default if you have not named an arm
nb_outputs = len(output_names)

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
    attributes = {
        "PatientId": patient_index
    }  # Renaming directly to PatientId for easier merge
    for attr in patient["patientAttributes"]:
        attributes[attr["id"]] = attr["val"]
    patient_attributes_list.append(attributes)

pd_names = [attr["id"] for attr in vpop_data["patients"][0]["patientAttributes"]]
nb_pds = len(pd_names)
df_patient_attributes = pd.DataFrame(patient_attributes_list)

# %%
# retrieve results
time.sleep(0.5)
timeseries_json = {"timeseries": {output: [arm_name] for output in output_names}}
try:
    response = jinko.make_request(
        path=f"/core/v2/result_manager/trial/{trial_core_item_id}/snapshots/{trial_snapshot_id}/timeseries/download",
        method="POST",
        json=timeseries_json,
        options={
            "X-jinko-project-id": "",
            "Content-Type": "application/json;charset=utf-8",
            "Accept": "application/zip",
        },
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
# data processing
df_time_series = pd.read_csv(io.StringIO(csvTimeSeries))
df_time_series = df_time_series.rename(columns={"Patient Id": "PatientId"})
merged_df = pd.merge(df_time_series, df_patient_attributes, on="PatientId")
pivot_df = merged_df.pivot_table(
    index=["PatientId", "Arm", "Time"] + list(pd_names),
    columns="Descriptor",
    values="Value",
).reset_index()
final_df = pivot_df[list(pd_names) + ["Time"] + list(output_names)]
data = torch.tensor(final_df.values, dtype=torch.float32)
time_steps = pivot_df[pivot_df["PatientId"] == pivot_df["PatientId"][0]]["Time"]

# %%
data

# %%
final_df

# %%
# initiate our GP class
myGP = GP(
    nb_pds + 1,
    pd_names,
    nb_outputs,
    output_names,
    data,
    var_strat="IMV",  # either IMV (Independent Multitask Variational) or LMCV (Linear Model of Coregionalization Variational)
    kernel="RBF",  # Either RBF or SMK
    data_already_normalized=False,  # default
    nb_inducing_points=100,
    mll="ELBO",  # default, otherwise PLL
    nb_training_iter=100,
    training_proportion=0.7,
    learning_rate=0.01,
    num_mixtures=3,
    jitter=1e-4,
)

# %%
myGP.train(mini_batching=False, mini_batch_size=None)

# %%
myGP.eval_perf()
myGP.plot_loss()

# %%
myGP.plot_all_solutions("training")
myGP.plot_all_solutions("validation")
