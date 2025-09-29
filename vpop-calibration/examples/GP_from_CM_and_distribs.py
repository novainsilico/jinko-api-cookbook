# %% [markdown]
# Train a Gaussian Process given a CM, Measure Design and parameters' distribution

# %%
import pandas as pd
import torch
import os
import zipfile
import io
from datetime import datetime
import uuid
from math import log2, floor
from scipy.stats import qmc
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
folder_id = "c59b071c-9bdc-41e8-8b26-2762490f0bb6"
model_sid = "cm-Tvhs-kbhn"
model_rev = None  # if you want a revision other than the last one
protocol_sid = None
protocol_rev = None  # if you want a revision other than the last one

# %%
# REPLACE WITH YOUR INFO
nb_patients = 40
# make sure to include all patient descriptors so as to be able to create a virtual population. You can set identical min and max to fix a parameter value.
patient_descriptors_space = [
    {
        "id": "k12",
        "min": 0.2,
        "max": 1,
    },
    {
        "id": "k21",
        "min": 0,
        "max": 0.5,
    },
    {
        "id": "k_el",
        "min": 0,
        "max": 0.5,
    },
]
solving_times = [
    {
        "tMax": "PT24S",
        "tMin": "PT0S",
        "tStep": "PT0.125S",
    }
]
outputs = [
    {"timeseriesId": "A1"},
    {"timeseriesId": "A2"},
]  # this is what you want the GP to model

# %% [markdown]
# From now on, no modification is necessary. The notebook will create a measure design, a virtual population, run a trial, and train a GP on the results.

# %%
# retrieve core item and snapshot ids
model_project_item = jinko.get_project_item(sid=model_sid, revision=model_rev)
model_core_item_id = model_project_item["coreId"]["id"]
model_snapshot_id = model_project_item["coreId"]["snapshotId"]
if protocol_sid is not None:
    protocol_project_item = jinko.get_project_item(
        sid=protocol_sid, revision=protocol_rev
    )
    protocol_core_item_id = protocol_project_item["coreId"]["id"]
    protocol_snapshot_id = protocol_project_item["coreId"]["snapshotId"]
else:
    protocol_core_item_id = None
    protocol_snapshot_id = None

# %%
# retrieve model name and current time to later name vpop generator, vpop and trial
model_name = jinko.make_request(
    path=f"/core/v2/model_manager/jinko_model/{model_core_item_id}/snapshots/{model_snapshot_id}/model",
).json()["modelName"]
date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# %%
# get patient descriptors names from the wanted distributions
pd_names = [pd["id"] for pd in patient_descriptors_space]
nb_pds = len(pd_names)
# get the number of outputs and their names
output_names = [output["timeseriesId"] for output in outputs]
nb_outputs = len(output_names)

# %%
# create the measure design
response = jinko.make_request(
    path="/core/v2/scorings_manager/measure_design",
    method="POST",
    json={
        "computationalModelId": {
            "coreItemId": model_core_item_id,
            "snapshotId": model_snapshot_id,
        },
        "measures": outputs,
    },
    options={
        "name": "measure design for two compartments PK model",
        "folder_id": folder_id,
    },
)

measure_design_info = jinko.get_project_item_info_from_response(response)
measure_design_core_item_id = measure_design_info["coreItemId"]["id"]
measure_design_snapshot_id = measure_design_info["coreItemId"]["snapshotId"]

print(
    f"Measure Design Resource link: {jinko.get_project_item_url_from_response(response)}"
)


# %%
# generating a virtual population exploring the given parameter space with sobol sequences
def to_patient(x):
    return {
        "patientIndex": str(uuid.uuid4()),
        "patientCategoricalAttributes": [],
        "patientAttributes": [{"id": p, "val": x[j]} for j, p in enumerate(pd_names)],
    }


m = floor(
    log2(nb_patients)
)  # we use the power of 2 for the sobol sequences to be equilibrated
sampler = qmc.Sobol(nb_pds, scramble=False)
samples_unscaled = sampler.random_base2(m)
samples = qmc.scale(
    samples_unscaled,
    [pd["min"] for pd in patient_descriptors_space],
    [pd["max"] for pd in patient_descriptors_space],
)
vpop = {"patients": [to_patient(x) for x in samples]}

# storing virtual patients attributes into a dataframe
patient_attributes_list = (
    []
)  # list of dictionaries (one dictionary corresponds to one patient, with its id and parameters)
for patient in vpop["patients"]:
    patient_index = patient["patientIndex"]
    attributes = {
        "PatientId": patient_index
    }  # renaming directly to PatientId for easier merge
    for attr in patient["patientAttributes"]:
        attributes[attr["id"]] = attr["val"]
    patient_attributes_list.append(attributes)

df_patient_attributes = pd.DataFrame(patient_attributes_list)

# %%
# posting the vpop
response = jinko.make_request(
    path="/core/v2/vpop_manager/vpop",
    method="POST",
    json=vpop,
    options={
        "folder_id": folder_id,
        "name": "Vpop for two compartments PK model",
    },
)

project_item_info = jinko.get_project_item_info_from_response(response)
vpop_core_item_id = project_item_info["coreItemId"]["id"]
vpop_snapshot_id = project_item_info["coreItemId"]["snapshotId"]

print(
    f"Virtual Population Resource link: {jinko.get_project_item_url_from_response(response)}"
)

# %%
# upload the trial
protocol_dict = (
    None
    if ((protocol_core_item_id is None) or (protocol_snapshot_id is None))
    else {"coreItemId": protocol_core_item_id, "snapshotId": protocol_snapshot_id}
)

response = jinko.make_request(
    path="/core/v2/trial_manager/trial",
    method="POST",
    json={
        "computationalModelId": {
            "coreItemId": model_core_item_id,
            "snapshotId": model_snapshot_id,
        },
        "protocolDesignId": protocol_dict,
        "vpopId": {
            "coreItemId": vpop_core_item_id,
            "snapshotId": vpop_snapshot_id,
        },
        "measureDesignId": {
            "coreItemId": measure_design_core_item_id,
            "snapshotId": measure_design_snapshot_id,
        },
        "solvingOptions": {
            "solvingTimes": solving_times,
        },
    },
    options={
        "folder_id": folder_id,
    },
)

trial_info = jinko.get_project_item_info_from_response(response)
trial_core_item_id = trial_info["coreItemId"]["id"]
trial_snapshot_id = trial_info["coreItemId"]["snapshotId"]

print(f"Trial Resource link: {jinko.get_project_item_url_from_response(response)}")

# %%
# run the trial
response = jinko.make_request(
    path=f"/core/v2/trial_manager/trial/{trial_core_item_id}/snapshots/{trial_snapshot_id}/run",
    method="POST",
)
jinko.monitor_trial_until_completion(trial_core_item_id, trial_snapshot_id)

# %%
timeseries_json = {"timeseries": {output: ["identity"] for output in output_names}}
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

# %%
j = torch.randint(nb_patients, (1,))[0]
myGP.plot_individual_solution(j)

# %%
myGP.plot_obs_vs_predicted(data_set="training")
myGP.plot_obs_vs_predicted(data_set="validation")
