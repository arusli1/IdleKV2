"""Aggregate recovery metrics and plots for Phase 5."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(fmean(values)) if values else 0.0


def compute_oracle_recovery(
    rows: list[dict[str, Any]],
    *,
    min_gap: float = 0.05,
) -> dict[str, Any]:
    """
    Summarize recovery while keeping the absolute lift over pure eviction explicit.

    `mean_recovery` answers "what fraction of the eviction gap came back?" while
    `mean_oracle_lift` answers the more concrete question the user cares about:
    "how much better is oracle than pure eviction in absolute task score?".
    """

    enriched_rows: list[dict[str, Any]] = []
    informative_rows: list[dict[str, Any]] = []
    negative_gap_rows: list[dict[str, Any]] = []
    small_gap_count = 0

    for row in rows:
        a_score = float(row["condition_a_score"])
        b_score = float(row["condition_b_score"])
        oracle_score = float(row["oracle_score"])
        eviction_gap = a_score - b_score
        oracle_lift = oracle_score - b_score
        recovery = None
        if eviction_gap >= float(min_gap):
            recovery = oracle_lift / eviction_gap

        enriched = dict(row)
        enriched["eviction_gap"] = round(eviction_gap, 6)
        enriched["oracle_lift"] = round(oracle_lift, 6)
        enriched["recovery"] = None if recovery is None else round(float(recovery), 6)
        enriched_rows.append(enriched)
        if eviction_gap >= float(min_gap):
            informative_rows.append(enriched)
        elif eviction_gap <= -float(min_gap):
            negative_gap_rows.append(enriched)
        else:
            small_gap_count += 1

    mean_a = _mean(float(row["condition_a_score"]) for row in enriched_rows)
    mean_b = _mean(float(row["condition_b_score"]) for row in enriched_rows)
    mean_oracle = _mean(float(row["oracle_score"]) for row in enriched_rows)
    mean_oracle_lift = _mean(float(row["oracle_lift"]) for row in enriched_rows)
    mean_eviction_gap = _mean(float(row["eviction_gap"]) for row in enriched_rows)

    recoveries = [float(row["recovery"]) for row in informative_rows]
    mean_recovery = None if not recoveries else round(_mean(recoveries), 6)
    median_recovery = None if not recoveries else round(float(median(recoveries)), 6)
    informative_mean_oracle_lift = _mean(float(row["oracle_lift"]) for row in informative_rows)
    informative_mean_eviction_gap = _mean(float(row["eviction_gap"]) for row in informative_rows)
    recovery_from_means = None
    if informative_rows:
        recovery_from_means = round(informative_mean_oracle_lift / informative_mean_eviction_gap, 6)

    improved_rows = [
        row
        for row in enriched_rows
        if float(row["oracle_score"]) > float(row["condition_b_score"])
    ]

    return {
        "mean_condition_a": round(mean_a, 6),
        "mean_condition_b": round(mean_b, 6),
        "mean_oracle": round(mean_oracle, 6),
        "mean_eviction_gap": round(mean_eviction_gap, 6),
        "mean_oracle_lift": round(mean_oracle_lift, 6),
        "mean_informative_eviction_gap": round(informative_mean_eviction_gap, 6) if informative_rows else None,
        "mean_informative_oracle_lift": round(informative_mean_oracle_lift, 6) if informative_rows else None,
        "recovery_from_means": recovery_from_means,
        "mean_recovery": mean_recovery,
        "median_recovery": median_recovery,
        "pct_examples_improved_over_b": round(len(improved_rows) / len(enriched_rows), 6) if enriched_rows else 0.0,
        "pct_full_recovery": round(sum(value >= 0.95 for value in recoveries) / len(recoveries), 6) if recoveries else 0.0,
        "pct_no_recovery": round(sum(value <= 0.05 for value in recoveries) / len(recoveries), 6) if recoveries else 0.0,
        "n_examples": len(enriched_rows),
        "n_informative": len(informative_rows),
        "n_negative_gap": len(negative_gap_rows),
        "n_skipped_small_gap": small_gap_count,
        "min_gap": float(min_gap),
        "per_example": enriched_rows,
    }


def format_go_nogo(
    *,
    recovery_table: dict[str, Any],
    serialization_diagnostic: dict[str, Any] | None,
    primary_task_key: str,
    primary_method: str = "snapkv",
    primary_budget: int = 512,
    recovery_threshold: float = 0.60,
    serialization_threshold: float = 1e-2,
    serialization_required_examples: int = 10,
    serialization_required_passes: int = 8,
) -> str:
    """Create the explicit go/no-go note required by the instructions."""

    tasks = recovery_table.get("tasks", {})
    primary_task = tasks.get(primary_task_key)
    slice_payload = None
    if primary_task is not None:
        method_payload = primary_task.get(primary_method, {})
        slice_payload = method_payload.get(f"k{int(primary_budget)}")

    recovery_value = None if slice_payload is None else slice_payload["aggregate"].get("mean_recovery")
    criterion_1 = recovery_value is not None and float(recovery_value) >= float(recovery_threshold)

    criterion_2 = False
    max_logit_diff = None
    n_serialization_examples = None
    n_serialization_passed = None
    n_round_trip_passed = None
    n_round_trip_examples = None
    max_loaded_vs_direct = None
    max_last_token_diff = None
    if serialization_diagnostic is not None:
        aggregate = serialization_diagnostic.get("aggregate", {})
        max_logit_diff = aggregate.get("max_logit_diff")
        n_serialization_examples = int(aggregate.get("n_examples", 0))
        n_serialization_passed = int(aggregate.get("n_passed", 0))
        n_round_trip_examples = int(aggregate.get("n_examples", 0))
        n_round_trip_passed = int(aggregate.get("n_round_trip_passed", 0))
        max_loaded_vs_direct = aggregate.get("max_loaded_vs_direct_logit_diff")
        max_last_token_diff = aggregate.get("max_last_token_diff")
        criterion_2 = (
            n_serialization_examples >= int(serialization_required_examples)
            and n_serialization_passed >= int(serialization_required_passes)
        )

    decision = "GO" if criterion_1 and criterion_2 else "NO-GO"

    lines = [
        f"Primary hop task: {primary_task_key}",
        f"Primary slice: {primary_method} @ k={int(primary_budget)}",
    ]
    if slice_payload is None:
        lines.append("Primary slice result: missing")
    else:
        aggregate = slice_payload["aggregate"]
        lines.extend(
            [
                f"Mean A (full-cache baseline): {aggregate['mean_condition_a']:.3f}",
                f"Mean B (pure eviction): {aggregate['mean_condition_b']:.3f}",
                f"Mean Oracle: {aggregate['mean_oracle']:.3f}",
                f"Mean oracle lift over pure eviction (O-B): {aggregate['mean_oracle_lift']:.3f}",
                f"Mean eviction gap (A-B): {aggregate['mean_eviction_gap']:.3f}",
                f"Mean recovery: {aggregate['mean_recovery'] if aggregate['mean_recovery'] is not None else 'n/a'}",
                f"Informative examples: {aggregate['n_informative']}/{aggregate['n_examples']}",
            ]
        )

    lines.append(
        f"Criterion 1: recovery >= {recovery_threshold:.2f} on the primary hop slice -> "
        f"{'MET' if criterion_1 else 'NOT MET'}"
    )
    if serialization_diagnostic is None:
        lines.append("Criterion 2: exact serialization baseline -> NOT RUN")
    else:
        lines.append(
            f"Criterion 2: exact serialization baseline ({serialization_required_passes}/{serialization_required_examples} pass threshold) -> "
            f"{'MET' if criterion_2 else 'NOT MET'}"
        )
        if n_serialization_passed is not None and n_serialization_examples is not None:
            lines.append(f"Observed structurally equivalent examples: {n_serialization_passed}/{n_serialization_examples}")
        if max_logit_diff is not None:
            lines.append(f"Observed max logit diff: {float(max_logit_diff):.6f}")
            lines.append(f"Reference structural-equivalence threshold: {serialization_threshold:.1e}")
        if max_last_token_diff is not None:
            lines.append(f"Observed last-query-token logit diff: {float(max_last_token_diff):.6f}")
        if max_loaded_vs_direct is not None and n_round_trip_passed is not None and n_round_trip_examples is not None:
            lines.append(
                f"Round-trip cached path exactness: {n_round_trip_passed}/{n_round_trip_examples} "
                f"(max loaded-vs-direct diff {float(max_loaded_vs_direct):.6f})"
            )
    lines.append(f"Decision: {decision}")

    if slice_payload is not None and float(slice_payload["aggregate"]["mean_oracle_lift"]) <= 0.0:
        lines.append("Note: oracle did not improve over pure eviction on average, so recovery claims are not actionable.")
    if slice_payload is not None and int(slice_payload["aggregate"]["n_informative"]) == 0:
        lines.append("Note: the primary slice had no informative A-B gap, so any recovery ratio would be meaningless.")

    return "\n".join(lines) + "\n"


def plot_recovery_distribution(
    recovery_data: dict[str, Any],
    *,
    task_label: str,
    budget: int,
    method: str,
    save_path: str | Path,
) -> bool:
    """Plot the distribution of informative recovery values for one slice."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    save_path = Path(save_path)
    recoveries = [
        float(row["recovery"])
        for row in recovery_data.get("per_example", [])
        if row.get("recovery") is not None
    ]
    display_values = [max(-0.1, min(1.1, value)) for value in recoveries]

    fig, ax = plt.subplots(figsize=(8, 4))
    if display_values:
        ax.hist(display_values, bins=20, range=(-0.1, 1.1), color="#2B8CBE", edgecolor="white")
        ax.axvline(
            _mean(display_values),
            color="#0B3C5D",
            linestyle="--",
            label=f"Mean: {_mean(display_values):.2f}",
        )
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No informative examples",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
    ax.set_xlabel("Oracle recovery rate")
    ax.set_ylabel("Count")
    ax.set_title(f"{task_label} | {method} | k={int(budget)}")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True


def plot_oracle_vs_budget(
    recovery_table: dict[str, Any],
    *,
    save_path: str | Path,
) -> bool:
    """Plot mean recovery versus budget for each task/method pair."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    save_path = Path(save_path)
    series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for task_key, task_payload in recovery_table.get("tasks", {}).items():
        task_label = task_payload.get("display_name", task_key)
        for method, budget_payload in task_payload.items():
            if method == "display_name":
                continue
            for budget_label, slice_payload in budget_payload.items():
                if not budget_label.startswith("k"):
                    continue
                aggregate = slice_payload.get("aggregate", {})
                recovery = aggregate.get("mean_recovery")
                if recovery is None:
                    continue
                series[(task_label, method)].append((int(budget_label[1:]), float(recovery)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (task_label, method), points in sorted(series.items()):
        points.sort(key=lambda item: item[0])
        ax.plot(
            [budget for budget, _ in points],
            [value for _, value in points],
            marker="o",
            label=f"{task_label} | {method}",
        )
    ax.set_xlabel("k_budget")
    ax.set_ylabel("Mean recovery")
    ax.set_title("Oracle Recovery vs. Budget")
    if series:
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True
