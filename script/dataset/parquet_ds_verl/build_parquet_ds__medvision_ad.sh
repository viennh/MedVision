ENV_NAME="medvision-prep-ds"

# Only create the env if it doesn't already exist
source activate base
eval "$(conda shell.bash hook)"
if [ -d "$(conda info --base)/envs/${ENV_NAME}" ]; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    conda create -n "${ENV_NAME}" python==3.11 -y
fi
conda activate "${ENV_NAME}"


# Set paths
benchmark_dir="/root/Documents/MedVision"
data_dir="${benchmark_dir}/Data"
export MedVision_DATA_DIR=${data_dir}
parquet_ds_dir="${data_dir}/raw_parquet/medvision_ad"

model_family_name="qwen2_5_vl" # NOTE: model_family_name must be in src/medvision_bm/rft/config/model_info.yaml
num_workers_concat_datasets=4
num_workers_format_dataset=64

# Data configs
# ----------------------------------------------------------------------------------
# ------
# NOTE: At least one of the following 3 task JSON paths must be provided. Set multiple task JSON paths for multi-task training
# ------
tasks_list_json_path_AD="${benchmark_dir}/tasks_list/tasks_MedVision-AD__train_SFT.json" # Total samples: 5545
# tasks_list_json_path_detect="${benchmark_dir}/tasks_list/tasks_MedVision-detect__train_SFT.json" # Total samples: 2695205
# tasks_list_json_path_TL="${benchmark_dir}/tasks_list/tasks_MedVision-TL__train_SFT.json" # Total samples: 5551

# ------
# NOTE: If the sample limit is larger than the dataset size, the full dataset will be used.
# ------
# [Required] Sample limits in total
train_sample_limit=5000
val_sample_limit=500

# # [Option 1] For approximately balanced sampling across 3 tasks
# train_sample_limit_per_task=333333 
# val_sample_limit_per_task=166

# # [Option 2] For task-specific sampling across 3 tasks (these numbers are the maximum samples per task)
train_sample_limit_task_AD=5000
val_sample_limit_task_AD=500
# train_sample_limit_task_Detection=11000
# val_sample_limit_task_Detection=105
# train_sample_limit_task_TL=5000
# val_sample_limit_task_TL=500

# Limit the test samples per subset (the default setting of the MedVision benchmark)
test_sample_limit_per_subset=1000


# # (Optional) Resize shape for images during dataset preparation
# new_shape_hw=(256 256)  # explicitly reshape images to size (height, width)
# ----------------------------------------------------------------------------------


# Install medvision_bm (locked shared build)
set -euo pipefail
lockfile="${benchmark_dir}/.medvision_build.lock"
wheelhouse="${benchmark_dir}/.wheelhouse"
mkdir -p "${wheelhouse}"
flock "${lockfile}" bash -c '
    set -euo pipefail
    benchmark_dir="'"${benchmark_dir}"'"
    wheelhouse="'"${wheelhouse}"'"
    rm -rf "${benchmark_dir}/build" "${benchmark_dir}/src/medvision_bm.egg-info"
    python -m pip wheel "${benchmark_dir}" -w "${wheelhouse}" --no-deps
    latest_wheel="$(ls -t "${wheelhouse}"/medvision_bm-*.whl | head -n1)"
    python -m pip install --force-reinstall "${latest_wheel}"
'

# Setup environment for SFT since we import SFT-related modules
python -m medvision_bm.sft.env_setup --data_dir ${data_dir}


# Build Verl datasets
# ------
# Add optional argument below:
# To resize all images to a new shape during dataset preparation:
# --new_shape_hw ${new_shape_hw[0]} ${new_shape_hw[1]} \
# ------
python -m medvision_bm.dataset.build_parquet_ds \
--parquet_ds_dir ${parquet_ds_dir} \
--tasks_list_json_path_AD ${tasks_list_json_path_AD} \
--num_workers_concat_datasets ${num_workers_concat_datasets} \
--test_sample_limit_per_subset ${test_sample_limit_per_subset} \
--train_sample_limit ${train_sample_limit} \
--val_sample_limit ${val_sample_limit} \
--train_sample_limit_task_AD ${train_sample_limit_task_AD} \
--val_sample_limit_task_AD ${val_sample_limit_task_AD} \

conda deactivate
# conda remove -n $ENV_NAME --all -y
