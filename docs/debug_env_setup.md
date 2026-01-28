# Debugging env setup



## 📊 Benchmark

- **[Method 1]** Locate the script you are using in `src/medvision_bm/benchmark/` and check the `main()` function for the environment setup section. An example from `eval__qwen2_5_vl.py` is shown below. Note that dependencies are installed in a specific order; some have specified versions, while others do not. This can be a source of potential issues—errors may arise if a package without a specified version updates to an incompatible latest version. We provide requirement lists in `requirements/` that should work out of the box, though you can modify them if needed. In general, you can choose to let the script handle environment setup or skip it using the `--skip_env_setup` argument.
  
  ```python
  # NOTE: DO NOT change the order of these calls
  # ------
  setup_env_hf_medvision_ds(data_dir)
  if not args.skip_env_setup:
      ensure_hf_hub_installed()
      install_vendored_lmms_eval(proj_dependency="qwen2_5_vl")
      install_medvision_ds(data_dir)
      install_torch_cu124()

      # NOTE: vllm version may need to be adjusted based on compatibility of model and transformers version
      install_vllm(data_dir, version="0.14.0")

      # NOTE: Reinstall packages to overwrite potentially incompatible versions
      install_transformers_accelerate_for_qwen25vl(transformers_version="5.0.0.rc2", accelerate_version="1.9.0")

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
  ```
  
  - When using `--skip_env_setup`, you must install packages manually, for example, using the provided requirements list. Note that `medvision_ds` and the vendored `lmms_eval` must be installed separately.
    ```bash
    # Install from requirements list
    python -m medvision_bm.benchmark.install_medvision_ds --data_dir "${data_dir}"
    python -m medvision_bm.benchmark.install_vendored_lmms_eval
    pip install -r "${benchmark_dir}/requirements/requirements_eval_qwen25vl.txt" --no-deps

    # Run with --skip_env_setup 
    CUDA_VISIBLE_DEVICES=0,1 \
    python -m  medvision_bm.benchmark.eval__qwen2_5_vl \
    --skip_env_setup \
    --model_hf_id $model_hf_id \
    --model_name $model_name \
    --results_dir $result_dir \
    --data_dir $data_dir \
    --tasks_list_json_path $tasks_list_json_path \
    --task_status_json_path $task_status_json_path \
    --batch_size_per_gpu $batch_size_per_gpu \
    --gpu_memory_utilization $gpu_memory_utilization \
    --sample_limit $sample_limit \
    ```

  

- **[Method 2]** Use the dependencies list in `requirements` for debugging packages conflict

  1. Install the benchmark codebase `medvision_bm`

  2. Modify dependencies list, such as `requirements/requirements_eval_qwen25vl.txt`

  3. Setup env

     📝 Match `--lmms_eval_opt_deps` with model:
     - Choose from [`meddr`, `lingshu`, `huatuogpt_vision`, `llava_med`, `qwen2_5_vl`, `gemini`] – defined [here](https://github.com/YongchengYAO/MedVision/blob/master/src/medvision_bm/medvision_lmms-eval/pyproject.toml)
     - If model is not one of these, ignore `--lmms_eval_opt_deps` 

     ```bash
     # NOTE: replace <local-data-folder>
     python -m medvision_bm.benchmark.env_setup -r requirements/requirements_eval_qwen25vl.txt --lmms_eval_opt_deps qwen2_5_vl --data_dir <local-data-folder>
     ```

  4. Skip env setup with `--skip_env_setup` in benchmarking scripts `script/medvision-*/eval__*`

      ```bash
      # Run
      # Add these arguments for debugging:
      # --skip_env_setup \
      # --skip_update_status \
      CUDA_VISIBLE_DEVICES=0 \
      python -m  medvision_bm.benchmark.eval__qwen2_5_vl \
      --skip_env_setup \
      --model_hf_id $model_hf_id \
      --model_name $model_name \
      --results_dir $result_dir \
      --data_dir $data_dir \
      --tasks_list_json_path $tasks_list_json_path \
      --task_status_json_path $task_status_json_path \
      --batch_size_per_gpu $batch_size_per_gpu \
      --gpu_memory_utilization $gpu_memory_utilization \
      --sample_limit $sample_limit \
      2>&1 | tee eval__Qwen2.5-VL__AD.log
      ```

<br/>

## 🎯 Training: SFT

- **[Debug]** Use the dependencies list in `requirements` for debugging packages conflict

  1. Modify dependencies list, such as `requirements/requirements_sft_qwen25vl.txt`

  2. Use the alternative setup command in `script/medvision-*/train__SFT__*` 
      ```bash
      # [Alternative] Setup training env: use a specific requirements file
      python -m medvision_bm.sft.env_setup --data_dir ${data_dir} --requirement "${benchmark_dir}/requirements/requirements_sft_qwen25vl.txt"
      ```
      

<br/>

