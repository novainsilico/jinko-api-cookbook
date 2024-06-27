#!/bin/sh
which jq >/dev/null
if [ $? -ne 0 ]; then
  cat
  exit 0
fi
jq --indent 1 \
  '
    (.cells[] | select(has("outputs")) | .outputs) = []
    | (.cells[] | select(has("execution_count")) | .execution_count) = null
  '