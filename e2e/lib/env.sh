get_email() {
  if [ -z "${JK_EMAIL}" ]; then
    echo "Missing 'JK_EMAIL' environment variable" 1>&2
    return 1
  fi
  echo "${JK_EMAIL}"
}

# $1: project id
# $2: folder id (optional)
# return frontend url
get_frontend_url() {
  if [ -z "${JK_BASE_URL}" ]; then
    echo "Missing 'JK_BASE_URL' environment variable" 1>&2
    return 1
  fi
  url="${JK_BASE_URL%/}/project/${1}"
  [ -n "${2}" ] && url="${url}?labels=${2}"
  echo "${url}"
}

get_api_url() {
  if [ -z "${JK_CORE_REQ_ROOT_URL}" ]; then
    echo "Missing 'JK_CORE_REQ_ROOT_URL' environment variable" 1>&2
    return 1
  fi
  echo "${JK_CORE_REQ_ROOT_URL}"
}
