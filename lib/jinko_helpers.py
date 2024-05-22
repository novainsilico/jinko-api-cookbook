# Mandatory imports
import requests
import getpass
import os
from typing import TypedDict

# Helper functions (this cell could be inserted in every notebook)

_projectId: str | None = None
_apiKey: str | None = None
_baseUrl: str = 'https://api.jinko.ai'

class CoreItemId(TypedDict):
  id: str
  snapshotId: str

def getHeaders() -> dict[str, str]:
  return {
  'X-jinko-project-id': _projectId,
  'Authorization': 'ApiKey ' + _apiKey
}

def makeUrl(path: str):
  return _baseUrl + path

def makeRequest(path: str, method: str ='GET', json=None):
  response = requests.request(method, _baseUrl + path, headers=getHeaders(), json=json)
  if response.status_code != 200:
    print(response.json())
    response.raise_for_status()
  return response

def checkAuthentication() -> bool:
  response = requests.get(makeUrl('/app/v1/auth/check'), headers=getHeaders())
  if response.status_code == 401:
    return False
  if response.status_code != 200:
    print(response.json())
    response.raise_for_status()
  return True

# Ask user for API key/projectId and check authentication
def initialize(projectId: str | None = None, apiKey: str | None = None, baseUrl: str | None = None):
   global _projectId, _apiKey, _baseUrl
   if baseUrl is not None:
     _baseUrl = baseUrl     
   if apiKey is not None:
     _apiKey = apiKey
   else:
     _apiKey = os.environ.get('JINKO_API_KEY')
   if projectId is not None:
     _projectId = projectId
   else:
     _projectId = os.environ.get('JINKO_PROJECT_ID')

   if _apiKey is None or _apiKey.strip() == '':
     _apiKey = getpass.getpass('Please enter your API key')
   if _apiKey.strip() == '':
     message = 'API key cannot be empty'
     print(message)
     raise Exception(message)

   if _projectId is None or _projectId.strip() == '':
     _projectId = getpass.getpass('Please enter your Project Id')
   if _projectId.strip() == '':
     message = 'Project Id cannot be empty'
     print(message)
     raise Exception(message)

   if not checkAuthentication():
     message = 'Authentication failed for Project "%s"' % (_projectId)
     print(message)
     raise Exception(message)
   print('Authentication successful')

def getProjectItem(shortId: str, revision: int | None = None):
    return makeRequest('/app/v1/project-item/%s' % (shortId)).json()

def getCoreItemId(shortId: str, revision: int | None = None) -> CoreItemId:
    item = getProjectItem(shortId, revision)
    if 'coreId' not in item or item['coreId'] is None:
       message = 'ProjectItem "%s" has no CoreItemId' % (shortId)
       print(message)
       raise Exception(message)
    return item['coreId']