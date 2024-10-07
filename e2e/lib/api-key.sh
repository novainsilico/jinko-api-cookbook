# $1: project id
create_api_key() {
  jk-key-create --silent -p "${1}" \
    -d "20m" --value || return $?
}
