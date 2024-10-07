MASTER_GROUP_NAME="Master Group (e2e)"
FOLDER_NAME="Folder"

_get_group_name() {
  echo "Cookbooks $(date -u +'%Y-%m-%d')"
}

# $1: cookbook name
_get_project_name() {
  echo "${1} - $(date -u +'%H:%M:%S.%3N')"
}

# ensure test group exists and create it if needed
# return group id
ensure_group() {
  jk-path-check --silent -g "${MASTER_GROUP_NAME}" \
    -g "$(_get_group_name)" \
    --create | jq . || return $?
}

# ensure test project exists and create it if needed
# $1: cookbook name
# return project id
ensure_project() {
  if [ -z "${1}" ]; then
    echo "Missing cookbook name when calling 'ensure_project'" 1>&2
    return 1
  fi
  jk-path-check --silent -g "${MASTER_GROUP_NAME}" \
    -g "$(_get_group_name)" \
    -p "$(_get_project_name ${1})" \
    --create | jq . || return $?
}

# ensure test folder exists and create it if needed
# $1: cookbook name
# return folder id
ensure_folder() {
  if [ -z "${1}" ]; then
    echo "Missing cookbook name when calling 'ensure_folder'" 1>&2
    return 1
  fi
  jk-path-check --silent -g "${MASTER_GROUP_NAME}" \
    -g "$(_get_group_name)" \
    -p "$(_get_project_name ${1})" \
    -f ${FOLDER_NAME} \
    --create | jq . || return $?
}

# $1: json object returned by one of the ensure_x functions
# return group identifier
get_group_id() {
  value=$(echo "${1}" | jq -r .groupId) || return $?
  [ ${value} == "null" ] && value=""
  echo "${value}"
}

# $1: json object returned by one of the ensure_x functions
# return project identifier
get_project_id() {
  value=$(echo "${1}" | jq -r .projectId) || return $?
  [ ${value} == "null" ] && value=""
  echo "${value}"
}

# $1: json object returned by one of the ensure_x functions
# return project identifier
get_folder_id() {
  value=$(echo "${1}" | jq -r .folderId) || return $?
  [ ${value} == "null" ] && value=""
  echo "${value}"
}
