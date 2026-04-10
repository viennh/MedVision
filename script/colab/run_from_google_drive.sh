#!/usr/bin/env bash
# Run MedVision benchmarks when the repo (and Data) live under Google Drive or any path.
# Usage:
#   export MEDVISION_HOME="/content/drive/MyDrive/UCF/MedVision"
#   bash script/colab/run_from_google_drive.sh --model medgemma --suite AD
#
# Or pass the root as the first argument:
#   bash script/colab/run_from_google_drive.sh /content/drive/MyDrive/UCF/MedVision --model all --suite all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -gt 0 ]] && [[ "${1:-}" != -* ]]; then
  export MEDVISION_HOME="$(cd "$1" && pwd)"
  shift
fi

if [[ -z "${MEDVISION_HOME:-}" ]]; then
  export MEDVISION_HOME="${REPO_ROOT}"
fi

cd "${MEDVISION_HOME}"
exec python "${SCRIPT_DIR}/run_medvision_benchmark.py" --medvision-home "${MEDVISION_HOME}" "$@"
