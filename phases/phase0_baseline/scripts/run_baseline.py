#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import importlib.metadata as metadata
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PHASE_ROOT.parents[1]
RULER_DIR = REPO_ROOT / "ruler"
RULER_SCRIPTS_DIR = RULER_DIR / "scripts"
# Model + artifact configuration so paths stay centralized and easy to audit.
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_LOCAL_DIR = REPO_ROOT / "models" / "Qwen2.5-7B-Instruct"
ARTIFACTS_DIR = PHASE_ROOT / "artifacts"
RESULTS_DIR = PHASE_ROOT / "results"
LOGS_DIR = PHASE_ROOT / "logs"
BENCHMARK_ROOT = ARTIFACTS_DIR / "benchmark_root" / "qwen2.5-7b-instruct" / "synthetic"
RESULTS_PATH = RESULTS_DIR / "baseline_ruler.json"
DEGRADATION_CSV_PATH = RESULTS_DIR / "degradation_curve.csv"
DEGRADATION_SVG_PATH = RESULTS_DIR / "degradation_curve.svg"
VLLM_LOG_PATH = LOGS_DIR / "vllm_qwen25.log"
# Core task groups and plotting metadata used for reporting.
TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "vt_2hop",
    "fwe",
]
S_NIAH_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
]
PLOT_SERIES = [
    ("S-NIAH", "#005f73"),
    ("vt_2hop", "#bb3e03"),
    ("fwe", "#0a9396"),
]
ACCEPTANCE_THRESHOLDS = {
    "s_niah_4k_min": {
        "length": 4096,
        "aggregate": "S-NIAH",
        "minimum_score": 98.0,
    },
    "s_niah_32k_min": {
        "length": 32768,
        "aggregate": "S-NIAH",
        "minimum_score": 85.0,
    },
}
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen2.5-7B-Instruct RULER baseline.")
    # CLI switches to control the benchmark sweep and server settings.
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--inspect-count", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def build_env() -> dict[str, str]:
    # Ensure subprocesses use the same Python environment as this runner.
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    # Log each subprocess call for reproducibility.
    print(f"$ {format_cmd(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # Lightweight JSONL loader for RULER datasets and predictions.
    rows = []
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as infile:
        return sum(1 for line in infile if line.strip())


def parse_summary_csv(path: Path) -> dict[str, dict[str, Any]]:
    # Summary CSV is written by RULER eval; we parse the rows we need.
    with open(path, "r", encoding="utf-8") as infile:
        rows = list(csv.reader(infile))
    tasks_row = next(row for row in rows if row and row[0] == "Tasks")
    scores_row = next(row for row in rows if row and row[0] == "Score")
    nulls_row = next(row for row in rows if row and row[0] == "Nulls")
    tasks = tasks_row[1:]
    scores = scores_row[1:]
    nulls = nulls_row[1:]
    parsed = {}
    for task, score, null in zip(tasks, scores, nulls):
        parsed[task] = {
            "score": float(score),
            "nulls": null,
        }
    return parsed


def mean_score(summary: dict[str, dict[str, Any]], tasks: list[str]) -> float:
    return sum(summary[task]["score"] for task in tasks) / len(tasks)


def dataset_path(data_dir: Path, task: str) -> Path:
    return data_dir / task / "validation.jsonl"


def prediction_path(pred_dir: Path, task: str) -> Path:
    return pred_dir / f"{task}.jsonl"


def ensure_model(model_dir: Path) -> None:
    # Download only the files needed for inference to keep artifacts small.
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(model_dir),
        allow_patterns=ALLOW_PATTERNS,
    )


def verify_tokenizer(model_dir: Path) -> dict[str, Any]:
    # Sanity check the tokenizer template so prompts match expected Qwen2.5 formatting.
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    sample_content = "Return the word OK."
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    expected = (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{sample_content}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    rendered_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample_content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(templated, "keys") and "input_ids" in templated:
        templated_ids = templated["input_ids"]
    else:
        templated_ids = templated

    if rendered != expected:
        raise RuntimeError("Qwen2.5 chat template does not match the expected rendered prompt.")
    if rendered_ids != templated_ids:
        raise RuntimeError("Rendered prompt tokens do not match tokenizer.apply_chat_template output.")

    return {
        "chat_template_matches_expected": True,
        "rendered_prompt_tokens": len(rendered_ids),
        "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
        "config_max_position_embeddings": getattr(config, "max_position_embeddings", None),
    }


def package_versions() -> dict[str, str]:
    # Capture package versions for reproducibility in the output artifact.
    versions = {}
    for package in ("huggingface_hub", "requests", "transformers", "vllm"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def ensure_ruler_datasets(env: dict[str, str]) -> None:
    # Fetch base RULER datasets if they are missing locally.
    json_dir = RULER_SCRIPTS_DIR / "data" / "synthetic" / "json"
    essay_path = json_dir / "PaulGrahamEssays.json"
    squad_path = json_dir / "squad.json"
    hotpot_path = json_dir / "hotpotqa.json"

    if not essay_path.exists():
        run([sys.executable, "download_paulgraham_essay.py"], cwd=json_dir, env=env)
    if not squad_path.exists() or not hotpot_path.exists():
        run(["bash", "download_qa_dataset.sh"], cwd=json_dir, env=env)


def start_vllm_server(args: argparse.Namespace, env: dict[str, str], model_dir: Path) -> tuple[subprocess.Popen[str], Any]:
    # Launch vLLM server and block until its health endpoint becomes available.
    VLLM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(VLLM_LOG_PATH, "a", encoding="utf-8")
    cmd = [
        sys.executable,
        "pred/serve_vllm.py",
        "--model",
        str(model_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--disable-custom-all-reduce",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
    ]
    print(f"$ {format_cmd(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=RULER_SCRIPTS_DIR,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    health_url = f"http://127.0.0.1:{args.port}/health"
    deadline = time.time() + 900
    while time.time() < deadline:
        if process.poll() is not None:
            log_handle.flush()
            raise RuntimeError(f"vLLM server exited early. Inspect {VLLM_LOG_PATH}.")
        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                return process, log_handle
        except requests.RequestException:
            pass
        time.sleep(5)

    process.terminate()
    raise TimeoutError(f"Timed out waiting for vLLM health check at {health_url}.")


def stop_vllm_server(process: subprocess.Popen[str] | None, log_handle: Any | None) -> None:
    # Best-effort shutdown with a hard kill fallback to avoid orphaned workers.
    try:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
    finally:
        if log_handle is not None:
            log_handle.close()


def prepare_task(length: int, task: str, num_samples: int, model_dir: Path, data_dir: Path, env: dict[str, str]) -> None:
    # Materialize datasets for a given task and context length.
    cmd = [
        sys.executable,
        "data/prepare.py",
        "--save_dir",
        str(data_dir),
        "--benchmark",
        "synthetic",
        "--task",
        task,
        "--tokenizer_path",
        str(model_dir),
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(length),
        "--model_template_type",
        "qwen2.5-instruct",
        "--num_samples",
        str(num_samples),
    ]
    run(cmd, cwd=RULER_SCRIPTS_DIR, env=env)


def should_prepare_task(data_dir: Path, task: str, num_samples: int, force: bool) -> bool:
    # Regenerate if explicitly forced or if we have fewer samples than requested.
    if force:
        return True
    return count_nonempty_lines(dataset_path(data_dir, task)) < num_samples


def predict_task(
    args: argparse.Namespace,
    task: str,
    model_dir: Path,
    data_dir: Path,
    pred_dir: Path,
    env: dict[str, str],
) -> None:
    # Run RULER prediction script against the vLLM server for one task.
    cmd = [
        sys.executable,
        "pred/call_api.py",
        "--data_dir",
        str(data_dir),
        "--save_dir",
        str(pred_dir),
        "--benchmark",
        "synthetic",
        "--task",
        task,
        "--server_type",
        "vllm",
        "--server_host",
        "127.0.0.1",
        "--server_port",
        str(args.port),
        "--model_name_or_path",
        str(model_dir),
        "--temperature",
        "0.0",
        "--top_k",
        "32",
        "--top_p",
        "1.0",
        "--threads",
        str(args.threads),
        "--batch_size",
        str(args.batch_size),
    ]
    run(cmd, cwd=RULER_SCRIPTS_DIR, env=env)


def should_predict_task(pred_dir: Path, task: str, num_samples: int, force: bool) -> bool:
    # Generate predictions if explicitly forced or if existing files are incomplete.
    if force:
        return True
    return count_nonempty_lines(prediction_path(pred_dir, task)) < num_samples


def reset_prediction_file(pred_dir: Path, task: str) -> None:
    # Drop stale predictions when data prep changes.
    path = prediction_path(pred_dir, task)
    if path.exists():
        path.unlink()


def evaluate_predictions(pred_dir: Path, env: dict[str, str]) -> dict[str, dict[str, Any]]:
    # Run RULER evaluation and return parsed summary scores.
    cmd = [
        sys.executable,
        "eval/evaluate.py",
        "--data_dir",
        str(pred_dir),
        "--benchmark",
        "synthetic",
    ]
    run(cmd, cwd=RULER_SCRIPTS_DIR, env=env)
    return parse_summary_csv(pred_dir / "summary.csv")


def inspect_niah_examples(data_dir: Path, count: int = 5) -> list[dict[str, Any]]:
    # Lightweight sanity check to confirm outputs are present in inputs for NIAH.
    samples = read_jsonl(data_dir / "niah_single_1" / "validation.jsonl")[:count]
    inspected = []
    for sample in samples:
        outputs = [str(output) for output in sample["outputs"]]
        inspected.append(
            {
                "index": sample["index"],
                "length": sample["length"],
                "contains_all_outputs": all(output in sample["input"] for output in outputs),
                "outputs": outputs,
                "input_head": sample["input"][:240],
                "input_tail": sample["input"][-240:],
            }
        )
    return inspected


def score_for_series(length_result: dict[str, Any], series_name: str) -> float:
    # Extract the right score for plotting; S-NIAH is an aggregate.
    if series_name == "S-NIAH":
        return float(length_result["aggregates"]["S-NIAH"]["score"])
    return float(length_result["scores"][series_name]["score"])


def write_degradation_curve_csv(results: dict[str, Any], output_path: Path) -> None:
    # Produce a CSV for plotting degradation curves across context lengths.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lengths = sorted(int(length) for length in results["lengths"].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["length", *[name for name, _ in PLOT_SERIES]])
        for length in lengths:
            length_result = results["lengths"][str(length)]
            writer.writerow([length, *[score_for_series(length_result, name) for name, _ in PLOT_SERIES]])


def line_points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def build_svg_text(
    labels: list[str],
    series_points: list[tuple[str, str, list[float]]],
    title: str,
) -> str:
    # Construct an SVG chart by hand to avoid extra plotting dependencies.
    width = 900
    height = 540
    left = 90
    right = 40
    top = 70
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_pos(index: int) -> float:
        if len(labels) == 1:
            return left + plot_width / 2
        return left + (plot_width * index / (len(labels) - 1))

    def y_pos(value: float) -> float:
        return top + plot_height - (plot_height * value / 100.0)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcf8" />',
        f'<text x="{left}" y="36" font-size="24" font-family="Arial, sans-serif" fill="#1f2933">{html.escape(title)}</text>',
    ]

    for tick in range(0, 101, 20):
        y = y_pos(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#d9e2ec" stroke-width="1" />')
        lines.append(
            f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="12" '
            'font-family="Arial, sans-serif" fill="#52606d">'
            f"{tick}</text>"
        )

    lines.append(
        f'<line x1="{left}" y1="{top + plot_height:.2f}" x2="{width - right}" y2="{top + plot_height:.2f}" '
        'stroke="#1f2933" stroke-width="2" />'
    )
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height:.2f}" stroke="#1f2933" stroke-width="2" />')

    for index, label in enumerate(labels):
        x = x_pos(index)
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_height:.2f}" x2="{x:.2f}" y2="{top + plot_height + 6:.2f}" stroke="#1f2933" stroke-width="1.5" />')
        lines.append(
            f'<text x="{x:.2f}" y="{height - 26}" text-anchor="middle" font-size="12" '
            'font-family="Arial, sans-serif" fill="#1f2933">'
            f"{html.escape(label)}</text>"
        )

    lines.append(
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 6}" text-anchor="middle" font-size="14" '
        'font-family="Arial, sans-serif" fill="#1f2933">Context Length</text>'
    )
    lines.append(
        f'<text x="24" y="{top + plot_height / 2:.2f}" text-anchor="middle" font-size="14" '
        'font-family="Arial, sans-serif" fill="#1f2933" transform="rotate(-90 24 '
        f'{top + plot_height / 2:.2f})">Score</text>'
    )

    legend_x = width - right - 180
    legend_y = 28
    for index, (name, color, values) in enumerate(series_points):
        y = legend_y + index * 22
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{color}" stroke-width="3" />')
        lines.append(
            f'<text x="{legend_x + 30}" y="{y + 4}" font-size="12" font-family="Arial, sans-serif" fill="#1f2933">'
            f"{html.escape(name)}</text>"
        )
        polyline_points = [(x_pos(point_index), y_pos(value)) for point_index, value in enumerate(values)]
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" '
            f'stroke-linecap="round" points="{line_points(polyline_points)}" />'
        )
        for point_x, point_y in polyline_points:
            lines.append(f'<circle cx="{point_x:.2f}" cy="{point_y:.2f}" r="4" fill="{color}" />')

    lines.append("</svg>")
    return "\n".join(lines)


def write_degradation_curve_svg(results: dict[str, Any], output_path: Path) -> None:
    # Render the degradation curve SVG alongside the CSV.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lengths = sorted(int(length) for length in results["lengths"].keys())
    labels = [f"{length // 1024}K" for length in lengths]
    series_points = []
    for name, color in PLOT_SERIES:
        values = [score_for_series(results["lengths"][str(length)], name) for length in lengths]
        series_points.append((name, color, values))
    svg = build_svg_text(labels, series_points, "Qwen2.5-7B-Instruct RULER Baseline")
    output_path.write_text(svg, encoding="utf-8")


def build_acceptance_report(results: dict[str, Any]) -> dict[str, Any]:
    # Compare aggregate scores against acceptance thresholds for quick gating.
    report = {}
    for name, config in ACCEPTANCE_THRESHOLDS.items():
        length_key = str(config["length"])
        actual = None
        passed = False
        if length_key in results["lengths"]:
            actual = results["lengths"][length_key]["aggregates"][config["aggregate"]]["score"]
            passed = actual >= config["minimum_score"]
        report[name] = {
            **config,
            "actual_score": actual,
            "pass": passed,
        }
    return report


def run_length(
    args: argparse.Namespace,
    length: int,
    env: dict[str, str],
    results: dict[str, Any],
) -> None:
    # Run the full RULER pipeline for one context length.
    length_root = BENCHMARK_ROOT / str(length)
    data_dir = length_root / "data"
    pred_dir = length_root / "pred"
    data_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Ensure datasets exist for each task at the target length.
    for task in TASKS:
        dataset_regenerated = should_prepare_task(data_dir, task, args.num_samples, args.force)
        if dataset_regenerated:
            prepare_task(length, task, args.num_samples, MODEL_LOCAL_DIR, data_dir, env)
            reset_prediction_file(pred_dir, task)
        else:
            print(f"Skipping data prep for {task} @ {length}; existing dataset is sufficient.")

        # Stage 2: Generate predictions only when missing or explicitly forced.
        if should_predict_task(pred_dir, task, args.num_samples, args.force):
            predict_task(args, task, MODEL_LOCAL_DIR, data_dir, pred_dir, env)
        else:
            print(f"Skipping prediction for {task} @ {length}; existing predictions are sufficient.")

    # Stage 3: Evaluate predictions and persist summary metadata into results.
    summary = evaluate_predictions(pred_dir, env)
    aggregates = {
        "S-NIAH": {
            "score": mean_score(summary, S_NIAH_TASKS),
            "component_tasks": S_NIAH_TASKS,
        },
    }
    results["lengths"][str(length)] = {
        "scores": summary,
        "aggregates": aggregates,
        "paths": {
            "data_dir": str(data_dir),
            "pred_dir": str(pred_dir),
            "summary_csv": str(pred_dir / "summary.csv"),
            "submission_csv": str(pred_dir / "submission.csv"),
        },
        "prediction_files": {
            task: str(prediction_path(pred_dir, task))
            for task in TASKS
        },
    }

    # Stage 4: Capture a small sample inspection for the shortest length.
    if length == 4096:
        results["checks"]["s_niah_4k_sample_inspection"] = inspect_niah_examples(data_dir, count=args.inspect_count)


def finalize_results(results: dict[str, Any]) -> None:
    # Build reporting artifacts and write the canonical results JSON.
    results["checks"]["context_32k_accessible"] = "32768" in results["lengths"]
    write_degradation_curve_csv(results, DEGRADATION_CSV_PATH)
    write_degradation_curve_svg(results, DEGRADATION_SVG_PATH)
    results["artifacts"] = {
        "degradation_curve_csv": str(DEGRADATION_CSV_PATH),
        "degradation_curve_svg": str(DEGRADATION_SVG_PATH),
    }
    results["acceptance"] = build_acceptance_report(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=2)


def main() -> None:
    args = parse_args()
    env = build_env()

    # Stage 0: Bootstrap prerequisites (model files and datasets).
    if not args.skip_download:
        ensure_model(MODEL_LOCAL_DIR)

    tokenizer_check = verify_tokenizer(MODEL_LOCAL_DIR)
    ensure_ruler_datasets(env)

    # Stage 1: Start vLLM, execute lengths, and always tear down cleanly.
    vllm_process = None
    log_handle = None
    try:
        vllm_process, log_handle = start_vllm_server(args, env, MODEL_LOCAL_DIR)

        # Stage 2: Collect results per length and keep artifacts up to date.
        results: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": MODEL_ID,
            "model_path": str(MODEL_LOCAL_DIR),
            "package_versions": package_versions(),
            "server": {
                "type": "vllm",
                "port": args.port,
                "dtype": "bfloat16",
                "tensor_parallel_size": 1,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "log_path": str(VLLM_LOG_PATH),
            },
            "checks": {
                "tokenizer": tokenizer_check,
            },
            "lengths": {},
        }

        for length in args.lengths:
            run_length(args, length, env, results)
            finalize_results(results)

    finally:
        stop_vllm_server(vllm_process, log_handle)


if __name__ == "__main__":
    main()
