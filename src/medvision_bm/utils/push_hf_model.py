import argparse

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--folder_path", type=str, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--message", type=str, required=False, help="Commit message for the upload"
    )
    args = parser.parse_args()

    api = HfApi()

    api.create_repo(args.repo_id, exist_ok=True)
    if args.message:
        api.upload_folder(
            folder_path=args.folder_path,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.message,
        )
    else:
        api.upload_folder(
            folder_path=args.folder_path,
            repo_id=args.repo_id,
            repo_type="model",
        )
    print(f"Successfully uploaded {args.folder_path} to {args.repo_id}")


if __name__ == "__main__":
    main()
