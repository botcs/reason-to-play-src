#!/usr/bin/env bash
# Fetch one subject's MRI files from the exact ds004323 snapshot.
set -euo pipefail
if [[ $# != 2 || ! "$2" =~ ^sub-(0[1-9]|[12][0-9]|3[0-2])$ ]]; then
    echo "Usage: bash reconstruction/tomov23/download_openneuro.sh DESTINATION sub-13" >&2
    exit 2
fi
dataset_dir="$1"
subject="$2"
snapshot=17d00770c7b7885ed51dd42d9dee44909bd26c6c
if [[ ! -d "$dataset_dir/.git" ]]; then
    datalad clone https://github.com/OpenNeuroDatasets/ds004323.git "$dataset_dir"
fi
git -C "$dataset_dir" checkout --detach "$snapshot"
test "$(git -C "$dataset_dir" rev-parse HEAD)" = "$snapshot"
datalad -C "$dataset_dir" get "$subject"
