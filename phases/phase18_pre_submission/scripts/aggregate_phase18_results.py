"""Aggregate Phase 18 results into a single RESULTS_FINAL.md.

Reads:
- K-sweep redo artifact (Qwen, K∈{32,64,80,96,128})
- Tight-budget sweep artifacts (mult ∈ {0.05, 0.10, 0.30, 1.05})
- Chunk-size sensitivity artifacts (chunk_size ∈ {32, 64, 128, 256})
- Llama low-K artifact (K∈{32, 48})
- GPU verify n=12 K=96 artifact
- Recency-favorable partition artifact (12→34)

Writes a single Markdown summary suitable for paper paragraph
injection or appendix table generation.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


COND = {
    "A": "condition_a_score",
    "B": "condition_b_score",
    "B_match": "b_match_score",
    "RepairKV": "idlekv_score",
    "Refresh-K": "refresh_k_score",
    "Refresh-K-budgeted": "refresh_k_budgeted_score",
    "PageSummary-Quest-inspired": "page_summary_score",
    "RepairKV-no-burst": "repairkv_no_burst_score",
    "RepairKV-chunked": "repairkv_chunked_score",
    "Oracle-K": "oracle_k_score",
    "Random-K": "random_k_score",
    "Oldest-K": "oldest_k_score",
}


def m(xs):
    return np.mean(xs) if xs else float("nan")


def scores_for(rows: list[dict]) -> dict[str, float]:
    out = {}
    for label, field in COND.items():
        vals = [r.get(field) for r in rows if r.get(field) is not None]
        if vals:
            out[label] = float(m(vals))
    return out


def k_sweep_table(art: Path) -> str:
    d = json.load(open(art))
    rows = d["rows"]
    by_k = {}
    for r in rows:
        k = int(r.get("k", 0))
        by_k.setdefault(k, []).append(r)
    ks = sorted(by_k.keys())
    if not ks:
        return ""
    cols = ["A", "B_match", "RepairKV", "Refresh-K", "Refresh-K-budgeted", "PageSummary-Quest-inspired", "RepairKV-no-burst", "Oracle-K"]
    header = "| K | " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    body = []
    for k in ks:
        s = scores_for(by_k[k])
        body.append("| " + str(k) + " | " + " | ".join(f"{s.get(c, float('nan')):.3f}" for c in cols) + " |")
    return f"### K-sweep (Qwen, n=12 × 3 partitions = 36 obs/K)\n\n{header}\n{sep}\n" + "\n".join(body) + "\n"


def tight_sweep_table(in_dir: Path) -> str:
    arts = sorted(glob.glob(str(in_dir / "tight_mult*_clean_suite_*.json")))
    if not arts:
        return ""
    rows = []
    for art in arts:
        m_ = re.match(r"tight_mult([0-9.]+)_", Path(art).name)
        if not m_:
            continue
        mult = float(m_.group(1))
        d = json.load(open(art))
        ex = d["rows"]
        s = scores_for(ex)
        budget_ms = float(np.mean([r.get("refresh_k_budgeted_t_repair_s", 0) * 1000 for r in ex]))
        rkb_caps = sum(1 for r in ex if r.get("refresh_k_budgeted_cap_fired"))
        rkb_pos = float(np.mean([r.get("refresh_k_budgeted_positions_scored", 0) for r in ex]))
        rows.append({"mult": mult, "budget_ms": budget_ms, "rkb_caps": rkb_caps, "rkb_pos": rkb_pos, **s})
    rows.sort(key=lambda r: r["mult"])
    lines = ["### Tight-budget multiplier sweep (Qwen, K=96, n=36)\n"]
    lines.append("| mult | budget (ms) | RepairKV | Refresh-K-budgeted | Δ vs RKB | PageSummary | Δ vs PSum | RKB cap fires | RKB positions scored |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        d_rkb = r.get("RepairKV", float("nan")) - r.get("Refresh-K-budgeted", float("nan"))
        d_ps = r.get("RepairKV", float("nan")) - r.get("PageSummary-Quest-inspired", float("nan"))
        lines.append(
            f"| {r['mult']:.2f} | {r['budget_ms']:.0f} | {r['RepairKV']:.3f} | {r['Refresh-K-budgeted']:.3f} | "
            f"{d_rkb:+.3f} | {r['PageSummary-Quest-inspired']:.3f} | {d_ps:+.3f} | "
            f"{r['rkb_caps']}/36 | {r['rkb_pos']:.0f}/32768 |"
        )
    return "\n".join(lines) + "\n"


def chunk_size_table(in_dir: Path) -> str:
    arts = sorted(glob.glob(str(in_dir / "cs*_clean_suite_*.json")))
    if not arts:
        return ""
    rows = []
    for art in arts:
        m_ = re.match(r"cs([0-9]+)_", Path(art).name)
        if not m_:
            continue
        cs = int(m_.group(1))
        d = json.load(open(art))
        s = scores_for(d["rows"])
        rows.append({"cs": cs, **s})
    rows.sort(key=lambda r: r["cs"])
    lines = ["### Chunk-size sensitivity (Qwen, K=96, n=36)\n"]
    lines.append("| chunk_size | RepairKV | RepairKV-chunked | PageSummary | B_match |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['cs']} | {r.get('RepairKV', float('nan')):.3f} | "
            f"{r.get('RepairKV-chunked', float('nan')):.3f} | "
            f"{r.get('PageSummary-Quest-inspired', float('nan')):.3f} | "
            f"{r.get('B_match', float('nan')):.3f} |"
        )
    return "\n".join(lines) + "\n"


def llama_table(art: Path | None) -> str:
    if art is None or not art.exists():
        return ""
    d = json.load(open(art))
    rows = d["rows"]
    by_k = {}
    for r in rows:
        k = int(r.get("k", 0))
        by_k.setdefault(k, []).append(r)
    ks = sorted(by_k.keys())
    cols = ["A", "B_match", "RepairKV", "Refresh-K-budgeted", "PageSummary-Quest-inspired", "RepairKV-no-burst"]
    lines = ["### Llama-3.1-8B low-K (n=36 per K)\n"]
    lines.append("| K | " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for k in ks:
        s = scores_for(by_k[k])
        lines.append("| " + str(k) + " | " + " | ".join(f"{s.get(c, float('nan')):.3f}" for c in cols) + " |")
    return "\n".join(lines) + "\n"


def gpu_verify_table(art: Path | None) -> str:
    if art is None or not art.exists():
        return ""
    d = json.load(open(art))
    s = scores_for(d["rows"])
    lines = ["### GPU vs CPU scoring agreement (Qwen, K=96, n=12)\n"]
    lines.append("| Path | A | B_match | RepairKV |")
    lines.append("|---|---|---|---|")
    lines.append(f"| GPU (PHASE18_SCORE_ON_GPU=1) | {s.get('A', float('nan')):.3f} | {s.get('B_match', float('nan')):.3f} | {s.get('RepairKV', float('nan')):.3f} |")
    lines.append(f"| CPU (K-sweep redo) | (read from K-sweep) |  |  |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("phases/phase18_pre_submission/RESULTS_FINAL.md"))
    args = parser.parse_args()

    sections = ["# Phase 18 RESULTS FINAL\n"]
    sections.append("Aggregated from queued reruns. NO PAPER EDITS APPLIED.\n")

    # K-sweep redo (sort by mtime so newest artifact wins regardless of filename order)
    import os as _os
    ksw = sorted(
        glob.glob("phases/phase6_repair/results/full/clean_suite_b16384_*n12_k32-64-80-96-128*.json"),
        key=lambda p: _os.path.getmtime(p),
    )
    if ksw:
        sections.append("\n---\n")
        sections.append(k_sweep_table(Path(ksw[-1])))

    # Tight sweep
    tight = Path("phases/phase18_pre_submission/results/w1_tight")
    if tight.exists():
        sections.append("\n---\n")
        sections.append(tight_sweep_table(tight))

    # Chunk-size
    cs = Path("phases/phase18_pre_submission/results/chunk_size_sens")
    if cs.exists():
        sections.append("\n---\n")
        sections.append(chunk_size_table(cs))

    # Llama low-K
    llama_arts = sorted(glob.glob("phases/phase6_repair/results/full/clean_suite_b16384_*mllama318binstruct_n12_k32-48*.json"))
    if llama_arts:
        sections.append("\n---\n")
        sections.append(llama_table(Path(llama_arts[-1])))

    # GPU verify
    gpu_arts = sorted(glob.glob("phases/phase18_pre_submission/results/gpu_verify/gpu_verify_*.json"))
    if gpu_arts:
        sections.append("\n---\n")
        sections.append(gpu_verify_table(Path(gpu_arts[-1])))

    # Tight K-sweep at 150ms abs budget
    tksw_arts = sorted(glob.glob("phases/phase18_pre_submission/results/w1_tight_ksweep/clean_suite_*n12_k32-64-96-128*.json"))
    if tksw_arts:
        sections.append("\n---\n")
        sections.append("### Tight K-sweep at 150 ms absolute budget (Qwen, 4Q, n=36/K)\n")
        d = json.load(open(tksw_arts[-1]))
        rows = d["rows"]
        by_k = {}
        for r in rows:
            k = int(r.get("k", 0))
            by_k.setdefault(k, []).append(r)
        ks = sorted(by_k.keys())
        cols = ["A", "B_match", "RepairKV", "Refresh-K", "Refresh-K-budgeted", "PageSummary-Quest-inspired", "RepairKV-no-burst"]
        sections.append("| K | " + " | ".join(cols) + " |")
        sections.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
        for k in ks:
            s = scores_for(by_k[k])
            sections.append("| " + str(k) + " | " + " | ".join(f"{s.get(c, float('nan')):.3f}" for c in cols) + " |")

    args.out.write_text("\n".join(sections))
    print(f"[aggregate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
