# Jinko specifics imports & initialization
# Please fold this section and do not change
import jinko_helpers as jinko
import json
from typing import Dict, List, Optional


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

def run_chatbot(
    co,
    message: str,
    chat_history: List[Dict[str, str]],
    model: str = "command-r-plus-08-2024",
    seed: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Interact with the chatbot using a given message and chat history, then return the updated chat history.

    Parameters
    ----------
    co : object
        The cohere client object used to interact with the chat model.
    message : str
        The user's message to be appended to the chat history and sent to the chatbot.
    chat_history : list
        A list of dictionaries representing the chat history, where each dictionary contains a role ('user' or 'assistant') and content.
    model : str, optional
        The model name to be used for generating responses (default is "command-r-plus-08-2024").
    seed : Optional[int], optional
        Seed value for deterministic response generation (default is None).

    Returns
    -------
    List[Dict[str, str]]
        The updated chat history including the user's message and the chatbot's response.
    """
    # Append the user message
    chat_history.append({"role": "user", "content": message})

    # Generate the response with the current chat history
    response = co.chat_stream(model=model, messages=chat_history, seed=seed)

    # Print the chatbot response
    print("\nChatbot:")
    chatbot_response = process_cohere_stream(response)

    # Append the chatbot's response
    chat_history.append(chatbot_response)

    return chat_history

def process_cohere_stream(response):
    """
    Iterate over a cohere response stream, save the content to a list, and return it as a string.

    :param response: a cohere response stream
    :return: a string of the content
    """
    content_list = []
    for event in response:
        if event.type == "content-delta":
            content_list.append(event.delta.message.content.text)
            print(event.delta.message.content.text, end="")
    return {"role": "assistant", "content": "".join(content_list)}
