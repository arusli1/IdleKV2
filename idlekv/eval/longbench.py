"""
LongBench benchmark evaluation.

16 subtasks across 6 categories: single-doc QA, multi-doc QA,
summarization, few-shot, synthetic, code.
"""

from dataclasses import dataclass


@dataclass
class LongBenchResult:
    subtask: str
    score: float  # F1 or ROUGE depending on subtask
    num_samples: int


SUBTASKS = [
    # Single-doc QA
    "narrativeqa", "qasper", "multifieldqa_en",
    # Multi-doc QA
    "hotpotqa", "2wikimqa", "musique",
    # Summarization
    "gov_report", "qmsum", "multi_news",
    # Few-shot
    "trec", "triviaqa", "samsum",
    # Synthetic
    "passage_count", "passage_retrieval_en",
    # Code
    "lcc", "repobench-p",
]


def run_longbench(
    model,
    tokenizer,
    subtasks: list[str] | None = None,
    press=None,
    manager=None,
    idle_budget_ms: float = 0,
    phases: str = "1+2",
    max_samples: int = None,
) -> list[LongBenchResult]:
    """
    Run LongBench evaluation.

    Returns:
        List of LongBenchResult, one per subtask.
    """
    tasks = subtasks or SUBTASKS
    results = []

    for task in tasks:
        # TODO: Implement actual LongBench evaluation
        # 1. Load dataset from HuggingFace: datasets.load_dataset("THUDM/LongBench", task)
        # 2. For each sample:
        #    a. Build prompt from context + question
        #    b. Prefill + compress (via press or manager)
        #    c. Generate answer
        #    d. Score (F1 for QA, ROUGE-L for summarization)
        # 3. Aggregate

        results.append(LongBenchResult(
            subtask=task,
            score=0.0,
            num_samples=0,
        ))

    return results
