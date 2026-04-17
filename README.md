# MedVision — Course Project (Fork & Colab)

This repository is a **fork** of [MedVision](https://github.com/YongchengYAO/MedVision): a dataset and benchmark for quantitative medical image analysis ([paper](https://arxiv.org/abs/2511.18676), [dataset on Hugging Face](https://huggingface.co/datasets/YongchengYAO/MedVision)). This README covers how we **fork upstream**, **set up data**, and **run benchmarks on Google Colab**.

For full upstream documentation (Docker, SLURM, SFT, etc.), see [`README-MedVision.md`](README-MedVision.md).

---

## 1) Fork from MedVision

1. On GitHub, open [YongchengYAO/MedVision](https://github.com/YongchengYAO/MedVision) and use **Fork** to create our copy.

2. Clone **our** fork locally:

   ```bash
   git clone https://github.com/viennh/MedVision.git
   cd MedVision
   ```

3. Add the upstream remote to pull official updates when needed:

   ```bash
   git remote add upstream https://github.com/YongchengYAO/MedVision.git
   git fetch upstream
   ```

4. Install the benchmark package in editable mode from the repo root:

   ```bash
   pip install -e .
   ```

5. (Optional) Merge upstream changes later:

   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   ```

---

## 2) Set up MedVision data

The benchmark expects a **`Data/`** directory at the **repository root**, alongside `src/`, `requirements/`, and `tasks_list/`.

### Layout (simplified)

- **`Data/`** — local imaging data, Hugging Face caches, and (after install) the `medvision_ds` dataset package source from Hub.
- Paths under **`Data/Datasets/<dataset_name>/...`** must match what evaluation scripts resolve (see environment variables below).

### Install dataset tooling and Hub snapshot

From the repo root, with `Data` as your data root (adjust if you use another path):

```bash
python -m medvision_bm.benchmark.install_medvision_ds --data_dir ./Data
```

This downloads dataset **code** from the [MedVision dataset repo](https://huggingface.co/datasets/YongchengYAO/MedVision) and configures Hugging Face cache locations under `Data/.cache/`.

### Local NIfTI / large files

- Download or sync the **local** dataset volumes you need into `Data/Datasets/...` per the upstream [file structure](docs/file-structure.md) (or your course’s subset).
- If Parquet/Arrow rows still contain **absolute paths from another machine** (e.g. macOS `/Volumes/...`), evaluation remaps them using:
  - **`MedVision_DATA_DIR`** — should point to the same directory you pass as `--data_dir` (e.g. `.../MedVision/Data`).
  - Optionally **`MEDVISION_HOME`** — repo root; used with `Data` for path resolution.

Benchmark entry scripts call `setup_env_hf_medvision_ds` and pass these into child processes so `lmms_eval` sees the correct roots.

### Gated models (e.g. MedGemma)

Some checkpoints require accepting terms on Hugging Face and a token:

```bash
export HF_TOKEN="hf_..."   # read token from https://huggingface.co/settings/tokens
```

---

## 3) Customize scripts to run on Google Colab

### Prerequisites

- **GPU runtime** (e.g. **A100** + **High-RAM** if you load large volumes from Drive).
- **Google Drive** mounted in Colab (`/content/drive/...`).
- Repo + `Data/` available under Drive (example path used in our scripts:  
  **`/content/drive/MyDrive/UCF/MedVision`** — change the folder name to match your Drive layout).

### Environment variables

In a Colab cell **before** running benchmarks:

```python
import os
os.environ["MEDVISION_HOME"] = "/content/drive/MyDrive/UCF/MedVision"
os.environ["MedVision_DATA_DIR"] = "/content/drive/MyDrive/UCF/MedVision/Data"
os.environ["HF_TOKEN"] = "hf_..."  # if using gated models
```

### Recommended entrypoints

| Script | Role |
|--------|------|
| [`script/colab/run_medvision_benchmark.py`](script/colab/run_medvision_benchmark.py) | Installs `medvision_bm`, vendored `lmms-eval`, model-specific requirements, then runs `eval__*` modules for chosen **model** and **suite** (`AD`, `TL`, `detect`). |
| [`script/colab/run_from_google_drive.sh`](script/colab/run_from_google_drive.sh) | Shell wrapper: sets `MEDVISION_HOME` and invokes the Python runner. |

### Example (Colab)

```bash
cd /content/drive/MyDrive/UCF/MedVision

python script/colab/run_medvision_benchmark.py \
  --medvision-home /content/drive/MyDrive/UCF/MedVision \
  --model medgemma \
  --suite AD \
  --install-only   # optional: dependencies only, no eval

python script/colab/run_medvision_benchmark.py \
  --medvision-home /content/drive/MyDrive/UCF/MedVision \
  --model medgemma \
  --suite AD
```

- **`--model`**: see [`script/colab/colab_model_tasks.py`](script/colab/colab_model_tasks.py) for keys (e.g. `medgemma`, `llava_med`, `qwen25vl`) or use `all`.
- **`--suite`**: `AD` (angle/distance), `TL` (tumor/lesion), `detect`, or `all`.

Model definitions, requirements files, and flags such as `--skip_env_setup` after manual installs are centralized in **`colab_model_tasks.py`**.

### Tips

- **First run**: use `--install-only` once to verify `pip` and vendored `lmms_eval` install; full installs on Colab can conflict with preinstalled packages—expect resolver warnings; a clean path is to use a **minimal** notebook or fix versions only when something breaks.
- **Paths**: keep **`--data_dir`** / `MedVision_DATA_DIR` consistent with where `Data/Datasets/` actually lives on Drive.
- **Upstream scripts**: SLURM job scripts under `script/benchmark-*` and `script/ucf_newton/` target clusters; Colab uses the **`script/colab/`** flow above.

---

## Citation

See [README-MedVision.md](README-MedVision.md) for the BibTeX entry.

---

## License

This project follows the upstream **MedVision** license (see repository `LICENSE` and [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) as referenced in `pyproject.toml`).
