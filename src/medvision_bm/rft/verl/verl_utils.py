from medvision_bm.sft.sft_utils import (
    _doc_to_target_TumorLesionTask,
    _doc_to_text_TumorLesionTask_CoT,
    format_dataset,
    img_proccessor_nii2png_save2dataset,
    load_split_limit_dataset,
)


def _format_data_TumorLesionTask_CoT_verl(
    example,
    model_name,
):
    """
    NOTE: The function is tailored for Verl framework.

    Format data for TumorLesionTask with CoT reasoning.

    Feilds required by Verl:
        - prompt: List of messages with roles and content.
        - ground_truth: Target string.
        - data_source: Data source identifier.
        - ability: Ability identifier.
        - reward_model: Reward model information.
        - extra_info: Additional information.

    Reference:
    RLHFDataset class in Verl
    (https://github.com/YongchengYAO/verl/blob/670aeea7cd6af2de0ce7da9ae8d3fd0c522d0f0e/verl/utils/dataset/rl_dataset.py#L69)

    """
    from medvision_bm.rft.verl.rft_prompts import SYSTEM_PROMPT
    
    # Reuse existing function for SFT with CoT for TumorLesionTask
    # We can extract GT landmark coordinates from value_dict
    prompt, values_dict = _doc_to_text_TumorLesionTask_CoT(example, model_name)
    target = _doc_to_target_TumorLesionTask(example)
    target_str = ", ".join([f"{value:.3f}" for value in target])

    # Build: "prompt"
    example["prompt"] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    # Build: "images", embedded processed image list
    example["images"] = img_proccessor_nii2png_save2dataset(example)

    # Build: "extra_info", used for medvision-tl reward
    # ---
    # Reference: https://github.com/YongchengYAO/verl/blob/medvision-rl/verl/utils/reward_score/medvision_rewards/medvision_tl.py
    # Required fields:
    #   - landmark_P1_wh
    #   - landmark_P2_wh
    #   - landmark_P3_wh
    #   - landmark_P4_wh
    # Dimensions definition:
    #   - landmark_P*_wh: [relative width, relative height]
    # Note: the origin of coordinate depends on _doc_to_text_TumorLesionTask_CoT()
    # ---
    extra_info = {
        "landmark_P1_wh": [
            float(values_dict["<x1_major>"]),
            float(values_dict["<y1_major>"]),
        ],
        "landmark_P2_wh": [
            float(values_dict["<x2_major>"]),
            float(values_dict["<y2_major>"]),
        ],
        "landmark_P3_wh": [
            float(values_dict["<x1_minor>"]),
            float(values_dict["<y1_minor>"]),
        ],
        "landmark_P4_wh": [
            float(values_dict["<x2_minor>"]),
            float(values_dict["<y2_minor>"]),
        ],
    }

    # Other fields required by Verl
    example["ground_truth"] = target_str
    example["data_source"] = "medvision-tl"
    example["ability"] = "medvision-tl"
    example["reward_model"] = {"style": "rule", "ground_truth": target_str}
    example["extra_info"] = extra_info

    return example


def _format_data_AngleDistanceTask_CoT_verl():
    raise NotImplementedError(
        "Mapping function for the angle distance task not implemented yet."
    )


def _format_data_DetectionTask_CoT_verl():
    raise NotImplementedError(
        "Mapping function for the detection task not implemented yet."
    )


def prepare_dataset_for_verl(
    *,
    tasks_list_json_path,
    limit_train_sample,
    limit_val_sample,
    mapping_func,
    model_family_name,
    num_workers_concat_datasets=4,
    num_workers_format_dataset=32,
    tag_ds=None,
):
    # Load and split dataset
    dataset = load_split_limit_dataset(
        tasks_list_json_path=tasks_list_json_path,
        limit_train_sample=limit_train_sample,
        limit_val_sample=limit_val_sample,
        num_workers_concat_datasets=num_workers_concat_datasets,
        tag_ds=tag_ds,
    )

    # Format dataset
    mapping_func_args = {
        "model_name": model_family_name,
    }
    dataset = format_dataset(
        dataset=dataset,
        mapping_func=mapping_func,
        mapping_func_args=mapping_func_args,
        num_workers_format_dataset=num_workers_format_dataset,
    )
    return dataset
