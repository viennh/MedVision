import argparse
import os
import subprocess

from medvision_bm.utils import (
    ensure_hf_hub_installed,
    install_medvision_ds,
    install_torch_cu124,
    install_vendored_lmms_eval,
    install_vllm,
    load_tasks,
    load_tasks_status,
    set_cuda_num_processes,
    setup_env_hf_medvision_ds,
    setup_env_vllm,
    update_task_status,
)


def run_evaluation_for_task_vllm_proxy(
    lmmseval_module: str,
    model_args: str,
    task: str,
    batch_size: int,
    sample_limit: int,
    output_path: str,
):
    print(f"\nRunning task: {task}\n")
    subprocess.run("conda env list", check=True, shell=True)
    cmd = [
        "python3",
        "-m",
        "lmms_eval",
        "--model",
        lmmseval_module,
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--batch_size",
        f"{batch_size}",
        "--limit",
        f"{sample_limit}",
        "--log_samples",
        "--output_path",
        output_path,
    ]
    cmd_result = subprocess.run(cmd, check=False)
    print(f"Command executed with return code: {cmd_result.returncode}")
    return cmd_result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Run MedVision benchmarking.")
    # model-specific arguments
    parser.add_argument(
        "--model_hf_id",
        default="llava-hf/llava-onevision-qwen2-72b-ov-hf",
        type=str,
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--model_name",
        default="llava-onevision-qwen2-72b-ov-hf",
        type=str,
        help="Name of the model to evaluate.",
    )
    # set max_new_tokens
    parser.add_argument(
        "--max_new_tokens",
        default=4096,
        type=int,
        help="Maximum number of new tokens to generate per sample.",
    )
    # resource-specific arguments
    parser.add_argument(
        "--batch_size_per_gpu",
        default=1,
        type=int,
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        default=0.99,
        type=float,
        help="GPU memory utilization fraction, used in vllm",
    )
    # task-specific arguments
    parser.add_argument(
        "--tasks_list_json_path",
        type=str,
        help="Path to the tasks list JSON file.",
    )
    # data, output and status paths
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to the results directory.",
    )
    parser.add_argument(
        "--task_status_json_path",
        type=str,
        help="Path to the task status JSON file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the MedVision data directory.",
    )
    # evaluation-specific arguments
    parser.add_argument(
        "--sample_limit",
        default=1000,
        type=int,
        help="Maximum number of samples to evaluate per task.",
    )
    # debugging and control arguments
    parser.add_argument(
        "--skip_env_setup",
        action="store_true",
        help="Skip environment setup steps.",
    )
    parser.add_argument(
        "--skip_update_status",
        action="store_true",
        help="Skip updating task status after completion -- useful for debugging.",
    )
    parser.add_argument(
        "--env_setup_only",
        action="store_true",
        help="Only perform environment setup and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configuration
    model_hf = args.model_hf_id
    model_name = args.model_name
    tasks_list_json_path = args.tasks_list_json_path
    result_dir = args.results_dir
    task_status_json_path = args.task_status_json_path
    data_dir = args.data_dir
    gpu_memory_utilization = args.gpu_memory_utilization
    sample_limit = args.sample_limit
    max_new_tokens = args.max_new_tokens

    num_processes = set_cuda_num_processes()

    # NOTE: DO NOT change the order of these calls
    # ------
    setup_env_hf_medvision_ds(data_dir)
    if not args.skip_env_setup:
        # NOTE: Install huggingface-hub, required version may vary for different models, check requirements 
        ensure_hf_hub_installed(hf_hub_version="0.35.3")
        install_vendored_lmms_eval()
        install_medvision_ds(data_dir)
        install_torch_cu124()
        # NOTE: vllm version may need to be adjusted based on compatibility of model and transformers version
        install_vllm(data_dir, version="0.10.0")
        if args.env_setup_only:
            print(
                "\nEnvironment setup completed as per argument --env_setup_only. Exiting now.\n"
            )
            return
    else:
        print(
            "\n[Warning] Skipping environment setup as per argument --skip_env_setup. This should only be used for debugging.\n"
        )
        setup_env_vllm(data_dir)
    # ------

    tasks = load_tasks(tasks_list_json_path)

    for task in tasks:
        completed_tasks = load_tasks_status(task_status_json_path, model_name)
        if task in completed_tasks:
            print(f"Task {task} already completed. Skipping...")
            continue

        batch_size = args.batch_size_per_gpu * num_processes
        # NOTE: adjust max_new_tokens according to your task
        vllm_model_args = (
            f"model_version={model_hf},"
            f"gpu_memory_utilization={gpu_memory_utilization},"
            f"tensor_parallel_size={num_processes},"
            f"max_num_seqs={batch_size},"  # maximum batch size
            "dtype=float16,"
            f"max_new_tokens={max_new_tokens}"
        )

        rc = run_evaluation_for_task_vllm_proxy(
            lmmseval_module="vllm_llava_onevision",
            model_args=vllm_model_args,
            task=task,
            batch_size=batch_size,
            sample_limit=sample_limit,
            output_path=os.path.join(result_dir, model_name),
        )

        if rc == 0 and not args.skip_update_status:
            update_task_status(task_status_json_path, model_name, task)
        else:
            print(f"Warning: Task {task} failed (return code {rc})")


if __name__ == "__main__":
    main()
