# Jinko specifics imports & initialization
# Please fold this section and do not change
import jinko_helpers as jinko
import json


def get_trial_summary(trial_viz_sid: str) -> dict:
    """
    Returns the trial summary in JSON format

    Parameters
    ----------
    trial_sid : str
        The short id of the trial viz

    Returns
    -------
    dict
        A dictionary containing the trial summary
    """
    # get the trial viz id
    trial_viz_id = jinko.get_core_item_id(trial_viz_sid, 1)
    # get the trial viz
    trial_viz = jinko.make_request(
        f"/core/v2/result_manager/trial_visualization/{trial_viz_id['id']}/snapshots/{trial_viz_id['snapshotId']}"
    ).json()
    # get the trial id
    trial_id = trial_viz.get("trialId")

    if trial_id is None:
        return {}

    # get the trial
    trial = jinko.make_request(
        f"/core/v2/trial_manager/trial/{trial_id['coreItemId']}/snapshots/{trial_id['snapshotId']}"
    ).json()
    # get the protocol id
    protocol_id = trial.get("protocolDesignId")
    # get the protocol
    protocol = (
        jinko.make_request(
            f"/core/v2/scenario_manager/protocol_design/{protocol_id['coreItemId']}/snapshots/{protocol_id['snapshotId']}"
        ).json()
        if protocol_id
        else {}
    )
    # get the measures id
    measures_id = trial.get("measureDesignId")
    # get the measures
    measures = (
        jinko.make_request(
            f"/core/v2/scorings_manager/measure_design/{measures_id['coreItemId']}/snapshots/{measures_id['snapshotId']}"
        ).json()
        if measures_id
        else {}
    )
    measures_content = measures.get("measures", [])

    # get the trial duration
    # it's defined as the max of all the solving options tMax
    trial_duration = max(
        [t["tMax"] for t in trial.get("solvingOptions", {}).get("solvingTimes", [])],
        default=0,
    )

    # get the patient count
    patient_count = trial.get("metadata", {}).get("public", {}).get("patientCount", 0)

    scenario_arms = protocol.get("scenarioArms", [])

    # return the summary
    summary = {
        "trial_duration": trial_duration,
        "patient_count": patient_count,
        "scenario_arms": scenario_arms,
        "measures": measures_content,
    }

    return summary


def print_and_save_stream(response):
    """
    Iterate over a cohere response stream and print/save the content to a file and return as a string.

    :param response: a cohere response stream
    :return: a string of the content
    """
    streamed_response = []
    for event in response:
        if event.type == "content-delta":
            streamed_response.append(event.delta.message.content.text)
            print(event.delta.message.content.text, end="")
    return "".join(streamed_response)
