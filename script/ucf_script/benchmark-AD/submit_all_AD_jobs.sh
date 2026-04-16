#!/bin/bash
# Submit AD benchmark evaluation jobs to SLURM — UCF Newton
# Jobs are chained with --dependency=afterany to avoid shared-env conflicts.
#
# Usage:
#   cd script/ucf_newton/benchmark-AD && bash submit_all_AD_jobs.sh
#   bash submit_all_AD_jobs.sh MODEL_SUBSTRING   # submit only job scripts whose name contains MODEL_SUBSTRING
#
# Examples:
#   bash submit_all_AD_jobs.sh medgemma          # -> eval__medgemma__AD_job.sh
#   bash submit_all_AD_jobs.sh Qwen2.5-VL
#   bash submit_all_AD_jobs.sh eval__MedDr__AD_job.sh
#
# Override defaults:
#   TIME=48:00:00 GRES=gpu:2 MEM=128G bash submit_all_AD_jobs.sh medgemma

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TIME="${TIME:-24:00:00}"
GRES="${GRES:-gpu:2}"
MEM="${MEM:-128G}"

MODEL_FILTER="${1:-}"

if [ "${MODEL_FILTER}" = "-h" ] || [ "${MODEL_FILTER}" = "--help" ]; then
    grep '^#' "$0" | head -20 | sed 's/^# \{0,1\}//'
    exit 0
fi

mkdir -p "${SCRIPT_DIR}/logs"

JOBS=(
    eval__medgemma__AD_job.sh
    eval__LLaVA_Med__AD_job.sh
    #eval__llava-onevision__AD_job.sh
    eval__HealthGPT-L14__AD_job.sh
    eval__MedDr__AD_job.sh
    #eval__lingshu__AD_job.sh
    #eval__InternVL3-38B__AD_job.sh
    eval__Llama3.2-Vision__AD_job.sh
    #eval__Gemma3_27B_it__AD_job.sh
    eval__Qwen2.5-VL__AD_job.sh
    #eval__HuatuoGPT-Vision-34B__AD_job.sh
)

if [ -n "${MODEL_FILTER}" ]; then
    FILTERED=()
    for job in "${JOBS[@]}"; do
        case "${job}" in
            *"${MODEL_FILTER}"*) FILTERED+=("${job}") ;;
        esac
    done
    if [ "${#FILTERED[@]}" -eq 0 ]; then
        echo "Error: no job script matches '${MODEL_FILTER}'. Known scripts in this submitter:" >&2
        printf '  %s\n' "${JOBS[@]}" >&2
        exit 1
    fi
    if [ "${#FILTERED[@]}" -gt 1 ]; then
        echo "Warning: '${MODEL_FILTER}' matches multiple jobs; submitting all of them:" >&2
        printf '  %s\n' "${FILTERED[@]}" >&2
    fi
    JOBS=("${FILTERED[@]}")
fi

echo "============================================"
echo "  UCF Newton — Submitting ${#JOBS[@]} AD benchmark job(s)"
echo "  --time=${TIME}  --gres=${GRES}  --mem=${MEM}"
echo "  Jobs chained with --dependency=afterany"
echo "============================================"

PREV_JOB=""

for job in "${JOBS[@]}"; do
    script="${SCRIPT_DIR}/${job}"
    if [ ! -f "${script}" ]; then
        echo "[SKIP] ${job} — file not found"
        continue
    fi
    echo -n "[SUBMIT] ${job} "
    if [ -n "${PREV_JOB}" ]; then
        echo -n "(after ${PREV_JOB}) "
        PREV_JOB=$(sbatch --parsable --dependency=afterany:${PREV_JOB} --time="${TIME}" --gres="${GRES}" --mem="${MEM}" "${script}")
    else
        PREV_JOB=$(sbatch --parsable --time="${TIME}" --gres="${GRES}" --mem="${MEM}" "${script}")
    fi
    echo "... JobID=${PREV_JOB}"
done

echo ""
echo "All ${#JOBS[@]} job(s) submitted as a chain. Check status with: squeue -u \$USER"
