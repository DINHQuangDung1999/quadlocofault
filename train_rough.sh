#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec bash ./train_rough_equivgcnmlp_v84_complete_fault.sh "$@"
