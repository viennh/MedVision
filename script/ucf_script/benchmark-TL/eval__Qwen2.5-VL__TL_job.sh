#!/bin/bash
#SBATCH --job-name=qwen25vl-TL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qwen25vl-TL_%j.out
#SBATCH --error=logs/qwen25vl-TL_%j.err

# ============================================================
# Qwen2.5-VL TL Evaluation — UCF Newton
# ============================================================

source ~/.env

set -euo pipefail

module load cuda
module load anaconda

ENV_NAME="eval-qwen25vl"
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
model_hf_id="Qwen/Qwen2.5-VL-7B-Instruct"
model_name="Qwen2.5-VL-7B-Instruct"
batch_size_per_gpu=20
gpu_memory_utilization=0.99

task_tag="MedVision-TL"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-TL.json"
task_status_json_path="${benchmark_dir}/completed_tasks/completed_tasks_${task_tag}.json"
sample_limit=100

rm -rf "${benchmark_dir}/build" "${benchmark_dir}/src/medvision_bm.egg-info"
$PYTHON -m pip install "${benchmark_dir}"

CUDA_VISIBLE_DEVICES=0 \
$PYTHON -m medvision_bm.benchmark.eval__qwen2_5_vl \
    --model_hf_id $model_hf_id \
    --model_name $model_name \
    --results_dir $result_dir \
    --data_dir $data_dir \
    --tasks_list_json_path $tasks_list_json_path \
    --task_status_json_path $task_status_json_path \
    --batch_size_per_gpu $batch_size_per_gpu \
    --gpu_memory_utilization $gpu_memory_utilization \
    --sample_limit $sample_limit

conda deactivate
