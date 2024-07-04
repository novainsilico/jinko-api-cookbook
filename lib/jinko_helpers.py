"""This modules provides a set of helper functions for Jinko API

- configure authentication (jinko.initialize)
- check authentication (jinko.checkAuthentication)
- retrieve a ProjectItem (jinko.getProjectItem)
- retrieve the CoreItemId of a ProjectItem (jinko.getCoreItemId)
- make HTTP requests (jinko.makeRequest)
"""

import base64 as _base64
import requests as _requests
import getpass as _getpass
import json as _json
import os as _os
from typing import TypedDict as _TypedDict

_projectId: str | None = None
_apiKey: str | None = None
_baseUrl: str = "https://api.jinko.ai"


class CoreItemId(_TypedDict):
    id: str
    snapshotId: str


class CustomHeadersRaw(_TypedDict):
    name: str
    description: str
    folder_id: str
    version_name: str


_headers_map = {
    "name": "X-jinko-project-item-name",
    "description": "X-jinko-project-item-description",
    "folder_id": "X-jinko-project-item-folder-ids",
    "version_name": "X-jinko-project-item-version-name",
}


def _getHeaders() -> dict[str, str]:
    apiKey = _apiKey
    if apiKey is None:
        apiKey = ""
    return {"X-jinko-project-id": _projectId, "Authorization": "ApiKey " + apiKey}


def encodeCustomHeaders(custom_headers_raw: CustomHeadersRaw) -> dict:
    """Encodes and prepares custom headers for the Jinko API.

    Args:
        custom_data (dict): Dictionary containing 'description', 'folder_id', 'name', 'version_name'

    Returns:
        dict: Dictionary containing encoded and formatted headers.
    """
    headers = {}
    for key, header_name in _headers_map.items():
        if key in custom_headers_raw:
            value = custom_headers_raw[key]
            if key == "folder_id":
                value = _json.dumps([{"id": value, "action": "add"}])
            headers[header_name] = _base64.b64encode(value.encode("utf-8")).decode(
                "utf-8"
            )
    return headers


def makeUrl(path: str):
    return _baseUrl + path


def makeRequest(
    path: str,
    method: str = "GET",
    json=None,
    options: CustomHeadersRaw = None,
):
    """Makes an HTTP request to the Jinko API.

    Args:
        path (str): HTTP path
        method (str, optional): HTTP method. Defaults to 'GET'
        json (Any, optional): JSON payload. Defaults to None
        options (dict, optional): Additional headers to include in the request. Defaults to None

    Returns:
        Response: HTTP response object

    Raises:
        Exception: if HTTP status code is not 200

    Examples:
        response = jinko.makeRequest('/app/v1/auth/check')

        projectItem = jinko.makeRequest(
            '/app/v1/project-item/tr-EUsp-WjjI',
            method='GET',
        ).json()
    """
    # Get the default headers from _getHeaders()
    headers = _getHeaders()

    # Encode custom headers as base64 and update the default headers
    if options:
        encoded_custom_headers = encodeCustomHeaders(options)
        headers.update(encoded_custom_headers)

    response = _requests.request(method, _baseUrl + path, headers=headers, json=json)
    if response.status_code != 200:
        if response.headers["content-type"] == "application/json":
            print(response.json())
        else:
            print("%s: %s" % (response.status_code, response.text))
        response.raise_for_status()
    return response


def checkAuthentication() -> bool:
    """Checks authentication

    Returns:
        bool: whether or not authentication was successful

    Raises:
        Exception: if HTTP status code is not one of [200, 401]

    Examples:
        if not jinko.checkAuthentication():
            print('Authentication failed')
    """
    response = _requests.get(makeUrl("/app/v1/auth/check"), headers=_getHeaders())
    if response.status_code == 401:
        return False
    if response.status_code != 200:
        print(response.json())
        response.raise_for_status()
    return True


# Ask user for API key/projectId and check authentication


def initialize(
    projectId: str | None = None, apiKey: str | None = None, baseUrl: str | None = None
):
    """Configures the connection to Jinko API and checks authentication

    Args:
        projectId (str | None, optional): project Id. Defaults to None
            If None, fallbacks to JINKO_PROJECT_ID environment variable
            If environment variable is not set, you will be asked for it interactively
        apiKey (str | None, optional): API key value. Defaults to None
            If None, fallbacks to JINKO_API_KEY environment variable
            If environment variable is not set, you will be asked for it interactively
        baseUrl (str | None, optional): root url to reach Jinko API. Defaults to None
            If None, fallbacks to JINKO_BASE_URL environment variable
            If environment variable is not set, fallbacks to 'https://api.jinko.ai'

    Raises:
        Exception: if API key is empty
        Exception: if Project Id is empty
        Exception: if authentication is invalid

    Examples:
        jinko.initialize()

        jinko.initialize(
            '016140de-1753-4133-8cbf-e67d9a399ec1',
            apiKey='50b5085e-3675-40c9-b65b-2aa8d0af101c'
        )

        jinko.initialize(
            baseUrl='http://localhost:8000'
        )
    """
    global _projectId, _apiKey, _baseUrl
    if baseUrl is not None:
        _baseUrl = baseUrl
    else:
        baseUrlFromEnv = _os.environ.get("JINKO_BASE_URL")
        if baseUrlFromEnv is not None and baseUrlFromEnv.strip() != "":
            _baseUrl = baseUrlFromEnv.strip()
    if apiKey is not None:
        _apiKey = apiKey
    else:
        _apiKey = _os.environ.get("JINKO_API_KEY")
    if projectId is not None:
        _projectId = projectId
    else:
        _projectId = _os.environ.get("JINKO_PROJECT_ID")

    if _apiKey is None or _apiKey.strip() == "":
        _apiKey = _getpass.getpass("Please enter your API key")
    if _apiKey.strip() == "":
        message = "API key cannot be empty"
        print(message)
        raise Exception(message)

    if _projectId is None or _projectId.strip() == "":
        _projectId = _getpass.getpass("Please enter your Project Id")
    if _projectId.strip() == "":
        message = "Project Id cannot be empty"
        print(message)
        raise Exception(message)

    if not checkAuthentication():
        message = 'Authentication failed for Project "%s"' % (_projectId)
        print(message)
        raise Exception(message)
    print("Authentication successful")


def getProjectItem(shortId: str, revision: int | None = None) -> dict:
    """Retrieves a single ProjectItem from its short Id
    and optionally its revision number

    Args:
        shortId (str): short Id of the ProjectItem
        revision (int | None, optional): revision number. Defaults to None

    Returns:
        dict: ProjectItem

    Raises:
        Exception: if HTTP status code is not 200

    Examples:
        projectItem = jinko.getProjectItem('tr-EUsp-WjjI')

        projectItem = jinko.getProjectItem('tr-EUsp-WjjI', 1)
    """
    if revision is None:
        return makeRequest("/app/v1/project-item/%s" % (shortId)).json()
    else:
        return makeRequest(
            "/app/v1/project-item/%s?revision=%s" % (shortId, revision)
        ).json()


def getCoreItemId(shortId: str, revision: int | None = None) -> CoreItemId:
    """Retrieves the CoreItemId corresponding to a ProjectItem

    Args:
        shortId (str): short Id of the ProjectItem
        revision (int | None, optional): revision number. Defaults to None

    Returns:
        CoreItemId: corresponding CoreItemId

    Raises:
        Exception: if HTTP status code is not 200
        Exception: if this type of ProjectItem has no CoreItemId

    Examples:
        id = jinko.getCoreItemId('tr-EUsp-WjjI')

        id = jinko.getCoreItemId('tr-EUsp-WjjI', 1)
    """
    item = getProjectItem(shortId, revision)
    if "coreId" not in item or item["coreId"] is None:
        message = 'ProjectItem "%s" has no CoreItemId' % (shortId)
        print(message)
        raise Exception(message)
    return item["coreId"]
