#!/bin/sh
which jq >/dev/null || cat
jq --indent 1 \
  '
    (.cells[] | select(has("outputs")) | .outputs) = []
    | (.cells[] | select(has("execution_count")) | .execution_count) = null
  '