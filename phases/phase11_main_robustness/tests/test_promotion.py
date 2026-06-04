from phases.phase11_main_robustness.src import evaluate_main_candidate


def test_main_candidate_requires_non_saturated_curve() -> None:
    rows = [
        {"task": "clean_suite", "base_context_budget": 16384, "k": k, "condition_a": 1.0,
         "b_match": 0.2, "idlekv": 1.0, "random_k": 0.2, "oldest_k": 0.2, "gold_k": 1.0}
        for k in [8, 16, 32, 64, 128]
    ]

    decision = evaluate_main_candidate(rows)[0]

    assert decision["main_candidate"] is False
    assert decision["action"] == "appendix_only_saturated_curve"


def test_main_candidate_accepts_clean_non_saturated_frontier() -> None:
    rows = [
        {"task": "clean_suite", "base_context_budget": 16384, "k": 8, "condition_a": 1.0,
         "b_match": 0.25, "idlekv": 0.25, "random_k": 0.25, "oldest_k": 0.25, "gold_k": 0.5},
        {"task": "clean_suite", "base_context_budget": 16384, "k": 16, "condition_a": 1.0,
         "b_match": 0.25, "idlekv": 0.35, "random_k": 0.25, "oldest_k": 0.25, "gold_k": 0.7},
        {"task": "clean_suite", "base_context_budget": 16384, "k": 32, "condition_a": 1.0,
         "b_match": 0.25, "idlekv": 0.55, "random_k": 0.27, "oldest_k": 0.25, "gold_k": 0.9},
        {"task": "clean_suite", "base_context_budget": 16384, "k": 64, "condition_a": 1.0,
         "b_match": 0.25, "idlekv": 0.80, "random_k": 0.27, "oldest_k": 0.25, "gold_k": 1.0},
        {"task": "clean_suite", "base_context_budget": 16384, "k": 128, "condition_a": 1.0,
         "b_match": 0.25, "idlekv": 0.96, "random_k": 0.27, "oldest_k": 0.25, "gold_k": 1.0},
    ]

    decision = evaluate_main_candidate(rows)[0]

    assert decision["main_candidate"] is True
    assert decision["best_eligible_k"] == 128
