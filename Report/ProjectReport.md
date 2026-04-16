# Can Vision-Language Models Measure Accurately from Medical Images?

**Course Project Report — Multimodal and Cross-Modal NLP (Group 3)**

---

## Abstract

Vision-Language Models (VLMs) are increasingly deployed in multimodal NLP systems, yet most evaluations emphasize qualitative reasoning rather than quantitative precision. Aligned with the course topic "Multimodal Reasoning with Text and Vision," this project investigates whether VLMs can **produce accurate numeric measurements from medical images** when prompted in natural language. Using the **MedVision** benchmark [1] — a large-scale dataset with 30.8 million annotated samples spanning 19 public medical imaging datasets — I evaluate 6 open-weight VLMs across three task families: (i) anatomical/abnormality detection (28 tasks), (ii) tumor/lesion size estimation (10 tasks), and (iii) angle/distance measurement (5 tasks). Each model is assessed under measurement-centric metrics including mean absolute error (MAE), mean relative error (MRE), IoU, success rate, and invalid response rate. This report documents the full experimental design, the end-to-end evaluation pipeline deployed on **Google Colab** with an **NVIDIA A100** GPU and **High-RAM** runtime (expanded system memory for large dataset I/O), and the engineering challenges encountered in deploying diverse VLM architectures at scale. This work is further motivated by my recent research assistant experience on a Cardio Vision-Language Model for cardiac cine-MRI, where measurement reliability is essential for safe clinical interpretation.

---

## 1. Introduction and Problem Statement

Multimodal large language models and Vision-Language Models (VLMs) are designed to answer questions grounded in both text and images. Common benchmarks — including VQA-v2, OK-VQA, and medical variants such as PathVQA and VQA-RAD — primarily test **descriptive** or **semantic** understanding: identifying objects, describing scenes, or answering classification-style questions. In medical imaging, however, clinical decisions often depend on **quantitative** measurements: the diameter of a tumor in millimeters, the angle of a cephalometric landmark, or the bounding-box coordinates of an abnormality. Even small numeric errors or hallucinated measurements can be harmful in clinical decision support.

Despite the rapid proliferation of medical VLMs, there is a notable gap in rigorous, large-scale evaluation of their **measurement fidelity** — the ability to produce numerically accurate outputs rather than merely plausible-sounding text. MedVision [1] directly addresses this gap by curating measurement-centric annotations across diverse anatomies and imaging modalities and demonstrating that off-the-shelf VLMs perform poorly on quantitative tasks.

**Research Question:**

> Can a VLM measure accurately from medical images when prompted in natural language?

This question is directly relevant to clinical deployment: if a VLM reports a tumor diameter of 12 mm when the true value is 22 mm, or if it returns free-text commentary instead of a numeric answer, the output is clinically useless or dangerous. I am motivated to study this question through my recent work as a research assistant on a Cardio Vision-Language Model project for cardiac cine-MRI, where measurement reliability is paramount for safe interpretation of ventricular volumes, wall thickness, and ejection fraction.

---

## 2. Related Work

### 2.1 Medical VLMs and Instruction Tuning

Biomedical adaptations of general-purpose VLMs, such as LLaVA-Med [2], apply visual instruction tuning on biomedical image-text pairs to improve conversational and VQA-style performance. While these models demonstrate improved engagement with medical imagery, their evaluations focus largely on qualitative understanding (e.g., describing pathology) rather than measurement fidelity. The gap between descriptive competence and numeric precision motivates the quantitative evaluation in this project.

### 2.2 Generalist Multimodal Medical AI

Med-PaLM M and the associated MultiMedBench [3] explore generalist multimodal medical reasoning across heterogeneous tasks including radiology report generation, visual question answering, and medical image classification. Although these efforts demonstrate breadth, measurement-focused evaluation — where the model must output a specific numeric value — remains comparatively underemphasized relative to descriptive and classification-based QA.

### 2.3 Clinical Radiology Foundations and Evaluation

Foundation-model efforts for chest radiography, such as CheXagent [4], demonstrate clinically relevant use cases with clinician-facing evaluation protocols. These works highlight the importance of rigorous, domain-specific assessment before deployment, reinforcing the need for measurement-centric benchmarks that go beyond classification accuracy.

### 2.4 Quantitative Medical Image Analysis Benchmark

MedVision [1] directly targets the quantitative gap by curating large-scale measurement annotations (30.8M samples across 19 datasets) and benchmarking VLMs on detection, tumor/lesion size estimation, and angle/distance measurement. The authors demonstrate that off-the-shelf VLMs perform poorly on these tasks and that supervised fine-tuning (SFT) substantially improves precision, establishing both the benchmark and the baseline results that this project builds upon.

---

## 3. Methodology

### 3.1 Benchmark and Task Families

I evaluate quantitative measurement capability using the **MedVision** benchmark, which provides precision-focused annotations for medical vision-language evaluation. The benchmark spans 19 public medical imaging datasets covering multiple anatomies (brain, abdomen, thorax, cardiac, musculoskeletal, dental) and modalities (CT, MRI, echocardiography, X-ray). Following MedVision, I consider three task families:

1. **Anatomical/Abnormality Detection (28 tasks, ~1.16M total samples):** Given a medical image with an annotated structure, the model must predict bounding-box coordinates. Datasets include BraTS24, AMOS22, AbdomenAtlas1.0Mini, CAMUS, MSD, and others.

2. **Tumor/Lesion (T/L) Size Estimation (10 tasks, ~5,456 total samples):** The model must estimate the major- and minor-axis dimensions of a tumor or lesion visible in the image. Datasets include BraTS24, autoPET-III, KiTS23, and MSD sub-tasks.

3. **Angle/Distance (A/D) Measurement (5 tasks, ~1,300 total samples):** The model must measure angles (in degrees) or distances (in mm) between anatomical landmarks. Datasets include Ceph-Biometrics-400 (cephalometric analysis) and FeTA24 (fetal brain biometry).

### 3.2 Datasets

The MedVision benchmark draws on 19 public medical imaging datasets. Of these, 13 are downloaded locally (totaling ~334 GB of raw imaging data) and used across the three task families. The remaining 6 datasets (AMOS22, AbdomenAtlas1.0Mini, AbdomenCT-1K, BCV15, CAMUS, FLARE22) participate only in the detection task family and are downloaded on-demand by the evaluation pipeline from HuggingFace Hub.

| Dataset | Modality | Anatomy | Local Size | Detection | T/L | A/D |
|---|---|---|---|---|---|---|
| BraTS24 | MRI | Brain (glioma, meningioma, metastasis, pediatric) | 93 GB | 3 tasks (28,286) | 3 tasks (1,302) | — |
| MSD | CT/MRI | Multi-organ (brain, colon, heart, liver, lung, pancreas, prostate, spleen, hepatic vessel) | 76 GB | 9 tasks (13,023) | 3 tasks (1,194) | — |
| autoPET-III | PET/CT | Whole-body (tumor/lesion) | 82 GB | 1 task (2,763) | 1 task (87) | — |
| KiTS23 | CT | Kidney (tumor) | 44 GB | 1 task (6,590) | 1 task (1,859) | — |
| KiPA22 | CT | Kidney (parenchyma/artery) | 1.1 GB | 1 task (4,540) | 1 task (866) | — |
| HNTSMRG24 | MRI | Head & neck (tumor) | 358 MB | 1 task (1,064) | 1 task (148) | — |
| TopCoW24 | CT/MRI | Brain (Circle of Willis vasculature) | 16 GB | 1 task (5,251) | — | — |
| OAIZIB-CM | MRI | Knee (cartilage/meniscus) | 12 GB | 1 task (59,621) | — | — |
| CrossMoDA | MRI | Brain (vestibular schwannoma) | 6.4 GB | 1 task (288) | — | — |
| Ceph-Biometrics-400 | X-ray | Dental (cephalometric) | 2.3 GB | — | — | 2 tasks (1,200) |
| FeTA24 | MRI | Fetal brain | 1.4 GB | 1 task (4,713) | — | 3 tasks (100) |
| SKM-TEA | MRI | Knee (musculoskeletal) | 27 MB | 1 task (6,847) | — | — |
| ISLES24 | MRI | Brain (ischemic stroke lesion) | 7.2 MB | 1 task (247) | — | — |
| AMOS22 | CT | Abdomen (multi-organ) | on-demand | 1 task (33,606) | — | — |
| AbdomenAtlas1.0Mini | CT | Abdomen (multi-organ) | on-demand | 1 task (916,437) | — | — |
| AbdomenCT-1K | CT | Abdomen (multi-organ) | on-demand | 1 task (55,342) | — | — |
| BCV15 | CT | Abdomen (multi-organ) | on-demand | 1 task (3,042) | — | — |
| CAMUS | Echocardiography | Cardiac | on-demand | 1 task (17,315) | — | — |
| FLARE22 | CT | Abdomen (multi-organ) | on-demand | 1 task (7,295) | — | — |
| | | | **~334 GB** | **28 tasks (1,166,270)** | **10 tasks (5,456)** | **5 tasks (1,300)** |

*Table 1. Datasets used in the MedVision benchmark. Sample counts (in parentheses) indicate the number of annotated 2D slices per task. "on-demand" datasets are fetched from HuggingFace Hub during evaluation and are not stored locally.*

The datasets span 6 imaging modalities (CT, MRI, PET/CT, X-ray, echocardiography, ultrasound) and cover diverse anatomical regions. The detection task family dominates in sample count (~1.17M) due to large abdominal CT datasets, while T/L and A/D tasks are smaller but clinically focused. Each evaluation is capped at 1,000 samples per task to manage compute costs.

### 3.3 Models Under Evaluation

I evaluate 6 open-weight VLMs spanning medical-domain and general-purpose models. Seven models from the original MedVision benchmark are excluded: five large models (Gemma-3-27B-it, lingshu-32B, HuatuoGPT-Vision-34B, InternVL3-38B, LLaVA-OneVision-72B) due to their multi-GPU memory requirements exceeding the available cluster allocation, and two proprietary API-based models (Gemini-2.5-Flash, Gemini-2.5-Pro) to focus exclusively on reproducible open-weight evaluation:

| Model | Parameters | Type | Backend | Python |
|---|---|---|---|---|
| medgemma-4b-it | 4B | Medical, open | accelerate | 3.11 |
| LLaVA-Med v1.5 | 7B | Medical, open | accelerate | 3.10 |
| MedDr | ~7B | Medical, open | accelerate | 3.9 |
| Qwen2.5-VL-7B | 7B | General, open | vLLM | 3.11 |
| Llama-3.2-Vision | 11B | General, open | vLLM | 3.11 |
| HealthGPT-L14 | ~14B | Medical, open | accelerate | 3.11 |

*Table 2. Models evaluated in this project. "Medical" denotes models specifically fine-tuned on biomedical data; "General" denotes general-purpose vision-language models. All models fit on a single GPU (≤14B parameters).*

### 3.4 Evaluation Pipeline

The end-to-end evaluation pipeline consists of three sequential stages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MedVision Evaluation Pipeline                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────┐ │
│  │ MedVision│    │  Model       │    │  Raw Model    │    │ Parsed   │ │
│  │ Dataset  │───>│  Inference   │───>│  Outputs      │───>│ Outputs  │ │
│  │ (HF Hub) │    │  (lmms-eval) │    │  (*.jsonl)    │    │ (*.jsonl)│ │
│  └──────────┘    └──────────────┘    └───────────────┘    └──────────┘ │
│       │                │                     │                  │       │
│       │          ┌─────┴─────┐               │           ┌─────┴─────┐ │
│       │          │ Per-model │               │           │ Metric    │ │
│       │          │ conda env │               │           │ Summary   │ │
│       │          │ + deps    │               │           │ (JSON)    │ │
│       │          └───────────┘               │           └───────────┘ │
│       │                                      │                         │
│  Stage 1: Data Loading              Stage 2: Parsing          Stage 3: │
│  & Model Inference                  & Validation              Summary  │
│                                                              & Analysis│
└─────────────────────────────────────────────────────────────────────────┘
```

*Figure 1. Three-stage evaluation pipeline: (1) data loading from HuggingFace Hub with model inference via the vendored lmms-eval framework, producing raw JSONL outputs per task; (2) output parsing with numeric extraction and validity checks; (3) metric computation and cross-model summary.*

**Stage 1 — Data Loading and Model Inference.** Each model runs within its own isolated conda environment with model-specific dependencies (e.g., specific versions of `transformers`, `vLLM`, `flash-attention`). The vendored `lmms-eval` framework handles prompt construction, image preprocessing, batched inference, and logging of raw model outputs as JSONL files. Each JSONL entry contains the model's free-text response, the ground-truth annotation, and metadata (dataset name, task ID, slice orientation).

**Stage 2 — Output Parsing and Validation.** The `parse_outputs` module extracts numeric predictions from raw model responses. For detection tasks, bounding-box coordinates are parsed; for T/L tasks, major/minor axis lengths; for A/D tasks, angle or distance values. A response is classified as *invalid* if: no numeric value is present, a required unit is missing or incorrect, multiple conflicting values are produced, or the response is otherwise unparseable.

**Stage 3 — Metric Computation and Summary.** Task-specific metrics are computed per model and aggregated across datasets:

- **Detection:** IoU (Intersection over Union), success rate (IoU > threshold), and invalid response rate.
- **T/L Size Estimation:** Mean Absolute Error (MAE), Mean Relative Error (MRE), and invalid response rate.
- **A/D Measurement:** MAE (degrees for angles, mm for distances), MRE, and invalid response rate.

### 3.5 Compute Infrastructure

All evaluations run on **Google Colab** using a paid runtime that provides an **NVIDIA A100** GPU (40 GB VRAM) and a **High-RAM** VM option so the host has enough system memory to mount data (e.g., from Google Drive), decompress medical volumes, and run Python dependencies without excessive swapping. Each model run uses a **single A100** in one Colab session. Hugging Face model weights and dataset caches use Colab ephemeral disk and/or a mounted Drive checkout of the MedVision repository and `Data/` tree.

Separately, the repository includes **SLURM** job scripts for the **UCF Newton HPC cluster** (mixed T4 / V100 / A100 nodes) for batch execution at scale; that path is optional relative to the Colab-based workflow described here.

| Resource | Specification |
|---|---|
| Platform | Google Colab (paid tier: A100 + High-RAM) |
| GPU | NVIDIA A100 (40 GB VRAM) |
| GPUs per session | 1 |
| System memory | High-RAM runtime (expanded host RAM) |
| Data / workspace | Google Drive mount + Colab VM disk |

*Table 3. Primary compute environment for this project (Colab A100 High-RAM).*

### 3.6 Reliability-Focused Error Analysis

To characterize failure modes beyond aggregate metrics, I plan to manually inspect a targeted subset of incorrect predictions (largest-error cases plus a random sample per task family) and assign each error to one or more categories:

- **Unit/scale errors:** wrong unit (mm vs. cm) or order-of-magnitude mistakes.
- **Localization errors:** measuring the wrong structure, endpoints, or axis.
- **Systematic bias:** consistent over- or under-estimation across cases.
- **Unsupported numeric claims:** seemingly precise values not grounded in the image.
- **Formatting/validity failures:** non-numeric output, multiple values, missing units.

---

## 4. Evaluation and Results

### 4.1 Model Resource Requirements

Before running the full benchmark, I analyzed the resource requirements for each model to ensure all 6 fit within a **single A100 GPU** on Colab (and remain feasible on one GPU in cluster settings where applicable).

| Model | Params | Backend | FP16 Memory | GPUs Required |
|---|---|---|---|---|
| medgemma-4b-it | 4B | accelerate | ~8 GB | 1 |
| LLaVA-Med v1.5 | 7B | accelerate | ~14 GB | 1 |
| MedDr | ~7B | accelerate | ~14 GB | 1 |
| Qwen2.5-VL-7B | 7B | vLLM | ~14 GB | 1 |
| Llama-3.2-Vision | 11B | vLLM | ~22 GB | 1 |
| HealthGPT-L14 | ~14B | accelerate | ~28 GB | 1 |

*Table 4. Model resource requirements for single-GPU evaluation. On Colab A100 (40 GB), all six models fit in VRAM; the largest (HealthGPT-L14, ~28 GB FP16) would not fit on a 16 GB T4 without further quantization or sharding.*

All 6 models are confirmed feasible on a single A100. The `dtype` is configured per GPU architecture: `float16` for V100 (compute capability 7.0) and `bfloat16` for A100 (compute capability 8.0+), including Colab’s A100.

### 4.2 Pipeline Deployment (Colab and UCF Newton)

A significant engineering contribution of this project is the deployment of the MedVision evaluation pipeline for **Google Colab** (see `script/colab/`) and, separately, on the **UCF Newton** SLURM cluster. Both required resolving numerous dependency, compatibility, and infrastructure issues:

**Dependency and build fixes:**
- Python version-specific dependency pinning (e.g., `cryptography==43.0.3` for Python 3.9 vs. `46.0.3` for Python 3.10+).
- `aiohttp<3.11` constraint for Python 3.9 environments (MedDr) to avoid `TypeError: unhashable type: 'list'` from incompatible typing features.
- Platform guards added for GPU-specific packages (`flash-attn`, `bitsandbytes`) in requirements files.
- `pyproject.toml` fixes for both `medvision_bm` and `medvision_ds` packages: deprecated license table format updated to SPDX string, missing README reference removed.

**SLURM cluster adaptation:**
- Created SLURM job scripts (`*_job.sh`) for all model-task combinations across all three task families, with appropriate `#SBATCH` directives for GPU allocation, memory, and walltime.
- Implemented sequential job chaining via `--dependency=afterany` to prevent concurrent builds on the shared Lustre filesystem.
- Environment variables (`HF_TOKEN`, `SYNAPSE_TOKEN`, `MEDVISION_HOME`) loaded from `~/.env` rather than hardcoded in scripts.
- Created master submission scripts (`submit_all_AD_jobs.sh`, `submit_all_TL_jobs.sh`, `submit_all_detect_jobs.sh`) and analysis/parsing job scripts.

### 4.3 Current Status

| Milestone | Status |
|---|---|
| Pipeline runnable on Google Colab (A100 + High-RAM) | Complete |
| Pipeline deployed to UCF Newton (SLURM scripts) | Complete |
| Dependency and build issues resolved (Colab / local) | Complete |
| Full benchmark (all 6 models, all task families) | Pending |
| Output parsing and metric computation | Pending |
| Error analysis | Pending |

*Table 5. Project milestone status.*

The evaluation pipeline is operational on **Google Colab (A100, High-RAM)** and on the **UCF Newton** cluster. On Newton, benchmark results were initially blocked by build-artifact collisions on the shared Lustre filesystem (stale `build/` directories causing `pip install` failures with `set -euo pipefail` aborting scripts before inference); those issues are being addressed through build-directory cleanup and sequential job isolation. Colab runs avoid the shared Lustre path but depend on **HF_TOKEN** for gated models, Drive mount bandwidth, and session time limits.

### 4.4 Expected Results

Based on the findings reported in the original MedVision paper [1], I anticipate the following trends once the full benchmark completes:

- **Detection tasks** will show moderate IoU and success rates for larger, well-defined structures (e.g., kidneys, liver) but poor performance on small or diffuse abnormalities (e.g., lesions < 20 pixels).
- **T/L size estimation** will exhibit high MRE across all models, reflecting the difficulty of extracting precise millimeter-scale measurements from visual prompts alone.
- **A/D measurement** will show relatively better performance on angular tasks (bounded output range) than distance tasks (unbounded, scale-dependent).
- **Invalid response rates** are expected to vary significantly across models, with smaller or less instruction-tuned models producing more unparseable outputs.
- **Medical-domain models** (medgemma, LLaVA-Med, MedDr) may show improved formatting compliance but not necessarily better numeric accuracy compared to larger general-purpose models.

---

## 5. Discussion and Challenges

### 5.1 Engineering Complexity

The most significant practical challenge in this project has been the **engineering complexity of running 6 diverse VLMs** with incompatible dependency trees in **Colab** and on the **UCF Newton** cluster. Each model requires a specific combination of Python version, `transformers` version, inference backend (`accelerate` vs. `vLLM`), and GPU-specific configurations (`float16` vs. `bfloat16`). The MedVision benchmark addresses this by using per-model conda environments (cluster) or isolated install steps (Colab); managing multiple isolated environments with shared build artifacts on a cluster filesystem introduces its own failure modes.

### 5.2 Build-Artifact Collisions on Shared Filesystems

A recurring issue on the UCF Newton cluster was **stale build artifacts** in shared directories (`build/`, `*.egg-info`) causing `pip install` failures during wheel construction. On the Lustre parallel filesystem, `rm -rf` followed immediately by `pip install` can encounter metadata caching effects where the build system sees ghost entries from deleted directories. This was the primary reason no `Results/MedVision-*` directories were created in initial cluster runs — every job aborted during the package installation phase before reaching model inference.

### 5.3 GPU Resource Constraints

All evaluations depend on access to a **CUDA GPU with sufficient VRAM**. Two of the 6 models (Qwen2.5-VL-7B and Llama-3.2-Vision) require the vLLM inference engine, which is Linux+CUDA only, making local macOS/CPU iteration impossible for those models—**Colab’s Linux + A100** satisfies this. HealthGPT-L14 requires up to ~28 GB of GPU memory in FP16, which fits **Colab A100 (40 GB)** and V100/A100 cluster nodes but not a 16 GB T4. Session limits, Drive I/O, and (on the cluster) queue wait times affect iteration speed.

### 5.4 The Quantitative Gap

Even from a design perspective, the benchmark reveals a fundamental tension: VLMs are trained primarily on descriptive text-image pairs, yet quantitative medical tasks demand **pixel-level spatial reasoning** and **scale-aware numeric output**. Current VLM architectures lack explicit mechanisms for spatial calibration (relating pixel coordinates to physical units), which likely explains the high error rates reported in [1].

### 5.5 Limitations

- **No results yet:** Quantitative results are not yet available end-to-end. Cluster runs were initially blocked by shared-filesystem build issues; Colab A100 runs depend on completing the full model suite within session and storage constraints. The pipeline is validated in principle but full benchmark execution is pending.
- **Single prompting condition:** The current MedVision benchmark uses a single prompting style per task. Investigating the effect of constrained-output prompting (e.g., "Return a single number in mm") is left for future work.
- **Sample limit:** To manage compute costs, all evaluations use a sample limit of 1,000 per task, which may not capture the full distribution of model behavior on tasks with larger pools.

---

## 6. Concluding Remarks

This project investigates whether Vision-Language Models can produce accurate quantitative measurements from medical images, a question with direct implications for clinical decision support. Using the MedVision benchmark — the largest measurement-centric medical VLM benchmark to date (30.8M annotations, 19 datasets, 3 task families) — I have designed and implemented an end-to-end evaluation pipeline for 6 open-weight VLMs on **Google Colab (NVIDIA A100, High-RAM)**, with optional **UCF Newton** SLURM scripts for large-scale batch runs.

**Key contributions of this work:**

1. **Systematic resource analysis** of 6 open-weight VLMs (≤14B parameters), confirming each fits within a single **Colab A100** GPU allocation.
2. **End-to-end pipeline deployment** on **Google Colab** (A100 + High-RAM) and on **UCF Newton** (SLURM job scripts, sequential job chaining, per-model conda environments, and automated submission workflows where used).
3. **Identification and diagnosis of deployment challenges** including build-artifact collisions on shared cluster filesystems, Python version incompatibilities, and GPU-specific dtype requirements.

**Future work:**

- Complete the full benchmark for all 6 models across all three task families on Colab A100 (and/or on UCF Newton once cluster build-artifact issues are resolved).
- Conduct the reliability-focused error analysis to categorize failure modes (unit errors, localization errors, systematic bias, unsupported claims).
- Investigate constrained-output prompting strategies to reduce invalid response rates.
- Compare off-the-shelf model performance with MedVision's SFT and RFT (reinforcement fine-tuning) models to quantify the improvement from measurement-focused training.

---

## References

[1] Yao, Y., Zong, Y., Dutt, R., Yang, Y., Tsaftaris, S. A., & Hospedales, T. (2025). MedVision: Dataset and Benchmark for Quantitative Medical Image Analysis. *arXiv preprint arXiv:2511.18676*.

[2] Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J., Naumann, T., Poon, H., & Gao, J. (2023). LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day. *NeurIPS 2023*.

[3] Tu, T., Azizi, S., Driess, D., Schaekermann, M., Amin, M., Chang, P.-C., Carroll, A., Lau, C., Tanno, R., Ktena, I., et al. (2023). Towards Generalist Biomedical AI. *arXiv preprint arXiv:2307.14334*.

[4] Chen, Z., Cano, A. H., Romanou, A., Bonber, B., Valentino, K., Karim, R., et al. (2024). CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation. *arXiv preprint arXiv:2401.12208*.

---

## Code and Artifacts

- **GitHub Repository:** [https://github.com/YongchengYAO/MedVision](https://github.com/YongchengYAO/MedVision)
- **Dataset (HuggingFace):** [https://huggingface.co/datasets/YongchengYAO/MedVision](https://huggingface.co/datasets/YongchengYAO/MedVision)
- **Google Colab runner:** `script/colab/` (`run_medvision_benchmark.py`, Drive-oriented helpers)
- **UCF Newton SLURM scripts:** `script/ucf_newton/` (benchmark job scripts, analysis scripts, submission scripts)
