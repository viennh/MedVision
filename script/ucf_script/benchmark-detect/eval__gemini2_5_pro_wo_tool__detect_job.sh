#!/bin/bash
#SBATCH --job-name=gemini-pro-wot-detect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gemini-pro-wot-detect_%j.out
#SBATCH --error=logs/gemini-pro-wot-detect_%j.err

# ============================================================
# Gemini 2.5 Pro w/o Tool — Detection Evaluation — UCF Newton
# (API model — no GPU required)
# ============================================================

source ~/.env

set -euo pipefail

module load anaconda

ENV_NAME="eval-gemini25"
MEDVISION_HOME="${medvision_home_dir:-/home/ha813608/MedVision}"
CONDA_HOME="${HOME}/.conda"
eval "$(conda shell.bash hook)"
if [ ! -d "${CONDA_HOME}/envs/${ENV_NAME}" ]; then
    conda create --prefix "${CONDA_HOME}/envs/${ENV_NAME}" python==3.11 -y
fi
conda activate "${CONDA_HOME}/envs/${ENV_NAME}"
PYTHON="${CONDA_HOME}/envs/${ENV_NAME}/bin/python"

$PYTHON -m pip install --upgrade pip setuptools wheel
$PYTHON -m pip install "datasets>=3.6.0,<4.0.0"

export PYTHONPATH="${MEDVISION_HOME}/Data/src:${PYTHONPATH:-}"
rm -rf "${MEDVISION_HOME}/build" "${MEDVISION_HOME}/src/medvision_bm.egg-info"
$PYTHON -m pip install "${MEDVISION_HOME}"

benchmark_dir="${MEDVISION_HOME}"
data_dir="${benchmark_dir}/Data"
model_name="gemini-2.5-pro-woTool"
batch_size=1
google_model_code="gemini-2.5-pro"

task_tag="MedVision-detect"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-detect.json"
task_status_json_path="${benchmark_dir}/completed_tasks/completed_tasks_${task_tag}.json"
sample_limit=100

rm -rf "${benchmark_dir}/build" "${benchmark_dir}/src/medvision_bm.egg-info"
$PYTHON -m pip install "${benchmark_dir}"

$PYTHON -m medvision_bm.benchmark.eval__gemini2_5_wo_tool \
    --google_model_code $google_model_code \
    --model_name $model_name \
    --results_dir $result_dir \
    --data_dir $data_dir \
    --tasks_list_json_path $tasks_list_json_path \
    --task_status_json_path $task_status_json_path \
    --batch_size $batch_size \
    --sample_limit $sample_limit

conda deactivate
