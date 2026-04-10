"""
Task suites and model specs for MedVision Colab / Drive workflows.

Paths follow the repo layout: ``<MEDVISION_HOME>/Data``, ``tasks_list/``, ``completed_tasks/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TaskSuite = Literal["AD", "detect", "TL"]


TASK_SUITES: dict[TaskSuite, tuple[str, str, str]] = {
    "AD": (
        "MedVision-AD",
        "tasks_MedVision-AD.json",
        "completed_tasks_MedVision-AD.json",
    ),
    "detect": (
        "MedVision-detect",
        "tasks_MedVision-detect.json",
        "completed_tasks_MedVision-detect.json",
    ),
    "TL": (
        "MedVision-TL",
        "tasks_MedVision-TL.json",
        "completed_tasks_MedVision-TL.json",
    ),
}


def suite_paths(medvision_home: Path, suite: str) -> tuple[str, str, str]:
    if suite not in TASK_SUITES:
        raise KeyError(f"Unknown suite {suite!r}; expected one of {tuple(TASK_SUITES)}")
    tag, tasks_file, status_file = TASK_SUITES[suite]  # type: ignore[index]
    benchmark = medvision_home.resolve()
    result_dir = benchmark / "Results" / tag
    tasks_list = benchmark / "tasks_list" / tasks_file
    task_status = benchmark / "completed_tasks" / status_file
    return str(result_dir), str(tasks_list), str(task_status)


@dataclass(frozen=True)
class ModelSpec:
    """Maps CLI model keys to benchmark modules and Hugging Face identifiers."""

    key: str
    eval_module: str
    model_hf_id: str | None
    model_name: str
    batch_size_default: int
    requirements_txt: str | None
    skip_env_after_manual_install: bool
    install_aiohttp_lt_311: bool
    install_vendored_lmms_eval: bool
    vendored_extra: str | None
    third_party_path_env: list[str]
    gpu_memory_utilization: float | None


MODELS: dict[str, ModelSpec] = {
    "medgemma": ModelSpec(
        key="medgemma",
        eval_module="medvision_bm.benchmark.eval__medgemma",
        model_hf_id="google/medgemma-4b-it",
        model_name="medgemma-4b-it",
        batch_size_default=10,
        requirements_txt="requirements/requirements_eval_medgemma.txt",
        skip_env_after_manual_install=True,
        install_aiohttp_lt_311=False,
        install_vendored_lmms_eval=True,
        vendored_extra=None,
        third_party_path_env=[],
        gpu_memory_utilization=None,
    ),
    "llava_med": ModelSpec(
        key="llava_med",
        eval_module="medvision_bm.benchmark.eval__llava_med",
        model_hf_id="microsoft/llava-med-v1.5-mistral-7b",
        model_name="llava-med-v1.5-mistral-7b",
        batch_size_default=50,
        requirements_txt=None,
        skip_env_after_manual_install=False,
        install_aiohttp_lt_311=False,
        install_vendored_lmms_eval=False,
        vendored_extra=None,
        third_party_path_env=[],
        gpu_memory_utilization=None,
    ),
    "meddr": ModelSpec(
        key="meddr",
        eval_module="medvision_bm.benchmark.eval__meddr",
        model_hf_id="Sunanhe/MedDr_0401",
        model_name="MedDr",
        batch_size_default=1,
        requirements_txt="requirements/requirements_eval_meddr.txt",
        skip_env_after_manual_install=True,
        install_aiohttp_lt_311=True,
        install_vendored_lmms_eval=True,
        vendored_extra="meddr",
        third_party_path_env=["third_party/MedDr"],
        gpu_memory_utilization=None,
    ),
    "qwen25vl": ModelSpec(
        key="qwen25vl",
        eval_module="medvision_bm.benchmark.eval__qwen2_5_vl",
        model_hf_id="Qwen/Qwen2.5-VL-7B-Instruct",
        model_name="Qwen2.5-VL-7B-Instruct",
        batch_size_default=20,
        requirements_txt=None,
        skip_env_after_manual_install=False,
        install_aiohttp_lt_311=False,
        install_vendored_lmms_eval=False,
        vendored_extra="qwen2_5_vl",
        third_party_path_env=[],
        gpu_memory_utilization=0.99,
    ),
    "llama3_vision": ModelSpec(
        key="llama3_vision",
        eval_module="medvision_bm.benchmark.eval__llama3_2_vision",
        model_hf_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        model_name="Llama-3.2-11B-Vision-Instruct",
        batch_size_default=10,
        requirements_txt="requirements/requirements_eval_llama3_vision.txt",
        skip_env_after_manual_install=True,
        install_aiohttp_lt_311=False,
        install_vendored_lmms_eval=True,
        vendored_extra=None,
        third_party_path_env=[],
        gpu_memory_utilization=0.99,
    ),
    "healthgpt": ModelSpec(
        key="healthgpt",
        eval_module="medvision_bm.benchmark.eval__healthgpt",
        model_hf_id=None,
        model_name="HealthGPT-L14",
        batch_size_default=20,
        requirements_txt=None,
        skip_env_after_manual_install=False,
        install_aiohttp_lt_311=False,
        install_vendored_lmms_eval=False,
        vendored_extra=None,
        third_party_path_env=["third_party/HealthGPT"],
        gpu_memory_utilization=None,
    ),
}

MODEL_CHOICES = tuple(MODELS.keys())
