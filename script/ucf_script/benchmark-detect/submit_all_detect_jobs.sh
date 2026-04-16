#!/bin/bash
# Submit Detection benchmark evaluation jobs to SLURM — UCF Newton
# Jobs are chained with --dependency=afterany to avoid shared-env conflicts.
#
# Usage:
#   cd script/ucf_newton/benchmark-detect && bash submit_all_detect_jobs.sh
#   bash submit_all_detect_jobs.sh MODEL_SUBSTRING   # submit only GPU/API job scripts whose name contains MODEL_SUBSTRING
#
# Examples:
#   bash submit_all_detect_jobs.sh medgemma          # -> eval__medgemma__detect_job.sh
#   bash submit_all_detect_jobs.sh Qwen2.5-VL
#   bash submit_all_detect_jobs.sh eval__MedDr__detect_job.sh
#
# Override defaults:
#   TIME=48:00:00 GRES=gpu:2 MEM=128G bash submit_all_detect_jobs.sh medgemma

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TIME="${TIME:-24:00:00}"
GRES="${GRES:-gpu:2}"
MEM="${MEM:-128G}"

MODEL_FILTER="${1:-}"

if [ "${MODEL_FILTER}" = "-h" ] || [ "${MODEL_FILTER}" = "--help" ]; then
    grep '^#' "$0" | head -24 | sed 's/^# \{0,1\}//'
    exit 0
fi

mkdir -p "${SCRIPT_DIR}/logs"

# GPU models (need --gres)
GPU_JOBS=(
    eval__medgemma__detect_job.sh
    eval__LLaVA_Med__detect_job.sh
    #eval__llava-onevision__detect_job.sh
    eval__HealthGPT-L14__detect_job.sh
    eval__MedDr__detect_job.sh
    #eval__lingshu__detect_job.sh
    #eval__InternVL3-38B__detect_job.sh
    eval__Llama3.2-Vision__detect_job.sh
    #eval__Gemma3_27B_it__detect_job.sh
    eval__Qwen2.5-VL__detect_job.sh
    #eval__HuatuoGPT-Vision-34B__detect_job.sh
)

# API models (no GPU)
API_JOBS=(
    #eval__gemini2_5_flash_w_tool__detect_job.sh
    #eval__gemini2_5_flash_wo_tool__detect_job.sh
    #eval__gemini2_5_pro_w_tool__detect_job.sh
    #eval__gemini2_5_pro_wo_tool__detect_job.sh
)

if [ -n "${MODEL_FILTER}" ]; then
    FILTERED_GPU=()
    for job in "${GPU_JOBS[@]}"; do
        case "${job}" in
            *"${MODEL_FILTER}"*) FILTERED_GPU+=("${job}") ;;
        esac
    done
    FILTERED_API=()
    for job in "${API_JOBS[@]}"; do
        case "${job}" in
            *"${MODEL_FILTER}"*) FILTERED_API+=("${job}") ;;
        esac
    done
    TOTAL_MATCH=$((${#FILTERED_GPU[@]} + ${#FILTERED_API[@]}))
    if [ "${TOTAL_MATCH}" -eq 0 ]; then
        echo "Error: no job script matches '${MODEL_FILTER}'. Known scripts in this submitter:" >&2
        echo "  GPU:" >&2
        printf '    %s\n' "${GPU_JOBS[@]}" >&2
        echo "  API:" >&2
        printf '    %s\n' "${API_JOBS[@]}" >&2
        exit 1
    fi
    if [ "${TOTAL_MATCH}" -gt 1 ]; then
        echo "Warning: '${MODEL_FILTER}' matches multiple jobs; submitting all of them:" >&2
        printf '  %s\n' "${FILTERED_GPU[@]}" "${FILTERED_API[@]}" >&2
    fi
    GPU_JOBS=("${FILTERED_GPU[@]}")
    API_JOBS=("${FILTERED_API[@]}")
fi

GPU_COUNT="${#GPU_JOBS[@]}"
API_COUNT="${#API_JOBS[@]}"

echo "============================================"
echo "  UCF Newton — Submitting Detection benchmark jobs"
echo "  GPU jobs: ${GPU_COUNT}  |  API jobs: ${API_COUNT}"
echo "  GPU: --time=${TIME} --gres=${GRES} --mem=${MEM}"
echo "  API: --time=${TIME} --mem=32G (no GPU)"
echo "  Jobs chained with --dependency=afterany"
echo "============================================"

PREV_JOB=""

for job in "${GPU_JOBS[@]}"; do
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

for job in "${API_JOBS[@]}"; do
    script="${SCRIPT_DIR}/${job}"
    if [ ! -f "${script}" ]; then
        echo "[SKIP] ${job} — file not found"
        continue
    fi
    echo -n "[SUBMIT] ${job} "
    if [ -n "${PREV_JOB}" ]; then
        echo -n "(after ${PREV_JOB}) "
        PREV_JOB=$(sbatch --parsable --dependency=afterany:${PREV_JOB} --time="${TIME}" --mem=32G "${script}")
    else
        PREV_JOB=$(sbatch --parsable --time="${TIME}" --mem=32G "${script}")
    fi
    echo "... JobID=${PREV_JOB}"
done

TOTAL_SUBMITTED=$((GPU_COUNT + API_COUNT))
echo ""
echo "All ${TOTAL_SUBMITTED} job(s) submitted as a chain. Check status with: squeue -u \$USER"
