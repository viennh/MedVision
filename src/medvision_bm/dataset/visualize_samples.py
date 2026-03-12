import argparse
import os
import sys

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (
    doc_to_visual_wBox,
    doc_to_visual_wVisualPrompt_angleTask,
    doc_to_visual_wVisualPrompt_distanceTask,
    doc_to_visual_wVisualPrompt_TLTask,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize dataset samples using doc_to_visual_wBox"
    )
    parser.add_argument(
        "--parquet_ds_path",
        type=str,
        required=True,
        help="Path to the prepared parquet dataset directory",
    )
    parser.add_argument(
        "--fig_dir", type=str, required=True, help="Output directory for figures"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="Detection",
        choices=["Detection", "Distance", "Angle", "TL"],
        help="Task type to visualize (Detection, Distance, Angle, TL)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check if dataset path exists
    if not os.path.exists(args.parquet_ds_path):
        print(f"Error: Dataset path not found at {args.parquet_ds_path}")
        sys.exit(1)

    print(f"Loading dataset from {args.parquet_ds_path}...")
    try:
        if os.path.isfile(args.parquet_ds_path) and args.parquet_ds_path.endswith(
            ".parquet"
        ):
            # Load parquet file directly
            print("Detected parquet file. Loading using load_dataset('parquet')...")
            # For parquet files, split="train" ensures we get a Dataset directly if it's a single split
            # However, if there are multiple splits in the parquet file logic, it might still return DatasetDict
            # But normally load_dataset('parquet', data_files=..., split='train') forces it into a Dataset
            dataset = load_dataset(
                "parquet", data_files=args.parquet_ds_path, split="train"
            )
        else:
            dataset = load_from_disk(args.parquet_ds_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Validate that we have a Dataset, not a DatasetDict
    from datasets import Dataset

    if not isinstance(dataset, Dataset):
        print(
            f"Error: Expected a Dataset object, but got {type(dataset)}. "
            "Please ensure the parquet file or directory contains a single split or specify split='train' when loading."
        )
        sys.exit(1)

    data_split = dataset

    print(f"Total samples in split: {len(data_split)}")

    # Select samples
    num_samples = min(len(data_split), args.num_samples)
    samples = data_split.select(range(num_samples))

    os.makedirs(args.fig_dir, exist_ok=True)
    print(f"Saving figures to {args.fig_dir}...")

    # Determine visualization function based on task type
    if args.task_type == "Detection":
        visual_func = doc_to_visual_wBox
    elif args.task_type == "Distance":
        visual_func = doc_to_visual_wVisualPrompt_distanceTask
    elif args.task_type == "Angle":
        visual_func = doc_to_visual_wVisualPrompt_angleTask
    elif args.task_type == "TL":
        visual_func = doc_to_visual_wVisualPrompt_TLTask
    else:
        # Should be caught by argparse choices, but for safety
        print(f"Unknown task type: {args.task_type}")
        sys.exit(1)

    success_count = 0
    for i, sample in tqdm(enumerate(samples), total=num_samples):
        try:
            # Prepare kwargs if needed, here passing None as per default usage or check what's needed
            # doc_to_visual_* signature: (doc, lmms_eval_specific_kwargs=None)
            images = visual_func(sample)

            # Extract info for filename
            dataset_name = sample.get("dataset_name", "unknown")
            image_file = sample.get("image_file", "unknown")
            file_name = (
                os.path.basename(image_file).replace(".nii.gz", "").replace(".nii", "")
            )
            slice_dim = sample.get("slice_dim", "NA")
            slice_idx = sample.get("slice_idx", "NA")

            if isinstance(images, list):
                for j, img in enumerate(images):
                    suffix = f"_{j}" if len(images) > 1 else ""
                    save_name = f"{dataset_name}__{file_name}__dim{slice_dim}__idx{slice_idx}{suffix}.png"
                    save_path = os.path.join(args.fig_dir, save_name)
                    img.save(save_path)
            else:
                # Assume single image if not list
                save_name = (
                    f"{dataset_name}__{file_name}__dim{slice_dim}__idx{slice_idx}.png"
                )
                save_path = os.path.join(args.fig_dir, save_name)
                images.save(save_path)
            success_count += 1

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback

            traceback.print_exc()

    print(f"Finished. Successfully visualized {success_count}/{num_samples} samples.")


if __name__ == "__main__":
    main()
