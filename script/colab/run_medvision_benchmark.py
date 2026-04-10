#!/usr/bin/env python3
"""
Run MedVision benchmark evaluations from a Google Drive checkout (or any path).

Designed for Google Colab: mount Drive, set ``--medvision-home`` to your repo root
(containing ``Data/``, ``src/``, ``requirements/``), then run one model and task suite.

Benchmark code expects ``conda`` on PATH for a no-op ``conda env list`` call; when
conda is missing (typical Colab), this script prepends a tiny shim to PATH.

Examples::

    python script/colab/run_medvision_benchmark.py \\
        --medvision-home /content/drive/MyDrive/UCF/MedVision \\
        --model medgemma \\
        --suite AD

    python script/colab/run_medvision_benchmark.py \\
        --medvision-home /content/drive/MyDrive/UCF/MedVision \\
        --model all \\
        --suite all \\
        --install-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from colab_model_tasks import MODELS, MODEL_CHOICES, ModelSpec, TASK_SUITES, suite_paths


def _try_mount_google_drive() -> None:
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except ImportError:
        pass


def _ensure_conda_shim() -> None:
    if shutil.which("conda"):
        return
    bindir = Path("/tmp/medvision_colab_conda_shim")
    bindir.mkdir(parents=True, exist_ok=True)
    shim = bindir / "conda"
    shim.write_text("#!/bin/sh\n# Colab shim: benchmark calls `conda env list`\nexit 0\n")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.environ["PATH"] = f"{bindir}:{os.environ.get('PATH', '')}"


def _prepend_pythonpath(medvision_home: Path, *relative_segments: str) -> None:
    extra = ":".join(str(medvision_home / s) for s in relative_segments)
    cur = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{extra}:{cur}" if cur else extra


def _prepend_data_src(medvision_home: Path) -> None:
    ds = medvision_home / "Data" / "src"
    if ds.is_dir():
        _prepend_pythonpath(medvision_home, "Data/src")


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _pip_install_editable(medvision_home: Path) -> None:
    for p in (medvision_home / "build", medvision_home / "src" / "medvision_bm.egg-info"):
        if p.exists():
            shutil.rmtree(p)
    _run([sys.executable, "-m", "pip", "install", str(medvision_home)])


def _install_vendored(medvision_home: Path, extra: str | None) -> None:
    cmd = [sys.executable, "-m", "medvision_bm.benchmark.install_vendored_lmms_eval"]
    if extra:
        cmd.extend(["--lmms_eval_opt_deps", extra])
    _run(cmd, cwd=medvision_home)


def _pip_install_requirements(medvision_home: Path, rel_req: str) -> None:
    req = medvision_home / rel_req
    if not req.is_file():
        raise FileNotFoundError(f"Requirements file not found: {req}")
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(req),
            "--no-deps",
        ]
    )


def prepare_environment(medvision_home: Path, spec: ModelSpec, suite_key: str) -> None:
    _ensure_conda_shim()
    _prepend_data_src(medvision_home)
    for rel in spec.third_party_path_env:
        _prepend_pythonpath(medvision_home, rel)

    os.chdir(medvision_home)
    _pip_install_editable(medvision_home)

    if spec.key == "medgemma" and suite_key == "TL":
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "datasets>=3.6.0,<4.0.0",
            ]
        )

    if spec.install_aiohttp_lt_311:
        _run([sys.executable, "-m", "pip", "install", "aiohttp<3.11"])

    if spec.install_vendored_lmms_eval:
        _install_vendored(medvision_home, spec.vendored_extra)

    if spec.requirements_txt:
        _pip_install_requirements(medvision_home, spec.requirements_txt)

    if spec.install_aiohttp_lt_311:
        _run([sys.executable, "-m", "pip", "install", "aiohttp<3.11"])


def build_eval_command(
    medvision_home: Path,
    spec: ModelSpec,
    result_dir: str,
    tasks_list: str,
    task_status: str,
    data_dir: str,
    batch_size: int,
    sample_limit: int,
    gpu_memory_utilization: float | None,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "-m",
        spec.eval_module,
        "--results_dir",
        result_dir,
        "--data_dir",
        data_dir,
        "--tasks_list_json_path",
        tasks_list,
        "--task_status_json_path",
        task_status,
        "--batch_size_per_gpu",
        str(batch_size),
        "--sample_limit",
        str(sample_limit),
    ]
    if spec.model_hf_id:
        cmd.extend(["--model_hf_id", spec.model_hf_id])
    cmd.extend(["--model_name", spec.model_name])

    if spec.eval_module.endswith("eval__llava_med") or spec.eval_module.endswith("eval__meddr"):
        cmd.extend(["--dir_third_party", str(medvision_home / "third_party")])

    if spec.eval_module.endswith("eval__healthgpt"):
        cmd.extend(["--dir_third_party", str(medvision_home / "third_party")])

    if spec.skip_env_after_manual_install:
        cmd.append("--skip_env_setup")

    if gpu_memory_utilization is not None:
        cmd.extend(["--gpu_memory_utilization", str(gpu_memory_utilization)])

    return cmd


def run_one(
    medvision_home: Path,
    spec: ModelSpec,
    suite_key: str,
    *,
    batch_size: int | None,
    sample_limit: int,
    gpu_memory_utilization: float | None,
    install_only: bool,
) -> None:
    if suite_key not in TASK_SUITES:
        raise ValueError(f"Unknown suite {suite_key!r}; expected one of {tuple(TASK_SUITES)}")

    result_dir, tasks_list, task_status = suite_paths(medvision_home, suite_key)
    data_dir = str(medvision_home / "Data")
    os.environ.setdefault("MEDVISION_HOME", str(medvision_home))
    os.environ.setdefault("MedVision_DATA_DIR", data_dir)
    bs = batch_size if batch_size is not None else spec.batch_size_default
    gmu = gpu_memory_utilization if gpu_memory_utilization is not None else spec.gpu_memory_utilization

    prepare_environment(medvision_home, spec, suite_key)
    if install_only:
        print("install-only: skipping evaluation.", flush=True)
        return

    eval_cmd = build_eval_command(
        medvision_home,
        spec,
        result_dir,
        tasks_list,
        task_status,
        data_dir,
        bs,
        sample_limit,
        gmu,
    )
    _ensure_conda_shim()
    _prepend_data_src(medvision_home)
    for rel in spec.third_party_path_env:
        _prepend_pythonpath(medvision_home, rel)
    os.chdir(medvision_home)
    print("+", " ".join(eval_cmd), flush=True)
    subprocess.run(eval_cmd, check=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedVision benchmark runner for Colab / Drive.")
    p.add_argument(
        "--medvision-home",
        type=Path,
        default=None,
        help="Root of the MedVision repo (with Data/, src/). Default: env MEDVISION_HOME or cwd.",
    )
    p.add_argument(
        "--mount-google-drive",
        action="store_true",
        help="Mount Google Drive to /content/drive (Colab only).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="medgemma",
        help=f"Model key or 'all'. Choices: {MODEL_CHOICES + ('all',)}",
    )
    p.add_argument(
        "--suite",
        type=str,
        default="AD",
        help="Task suite: AD, detect, TL, or all.",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--sample-limit", type=int, default=1000)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument(
        "--install-only",
        action="store_true",
        help="Only install dependencies (pip / vendored lmms-eval); do not run evaluation.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mount_google_drive:
        _try_mount_google_drive()

    home = args.medvision_home or os.environ.get("MEDVISION_HOME")
    if home:
        medvision_home = Path(home).expanduser().resolve()
    else:
        medvision_home = Path.cwd().resolve()

    if not (medvision_home / "src" / "medvision_bm").is_dir():
        print(
            "Could not find medvision_bm under src/. Set --medvision-home to your repo root.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not (medvision_home / "Data").is_dir():
        print(
            "Could not find Data/ directory. Upload MedVision Data next to the repo or merge into the same tree.",
            file=sys.stderr,
        )
        sys.exit(1)

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    suites = list(TASK_SUITES.keys()) if args.suite == "all" else [args.suite]

    for m in models:
        if m not in MODELS:
            print(f"Unknown model {m!r}. Choices: {MODEL_CHOICES} or all", file=sys.stderr)
            sys.exit(1)
    for s in suites:
        if s not in TASK_SUITES:
            print(f"Unknown suite {s!r}. Choices: {tuple(TASK_SUITES)} or all", file=sys.stderr)
            sys.exit(1)

    for m in models:
        for su in suites:
            print(f"\n=== Model={m} suite={su} ===\n", flush=True)
            run_one(
                medvision_home,
                MODELS[m],
                su,
                batch_size=args.batch_size,
                sample_limit=args.sample_limit,
                gpu_memory_utilization=args.gpu_memory_utilization,
                install_only=args.install_only,
            )


if __name__ == "__main__":
    main()
