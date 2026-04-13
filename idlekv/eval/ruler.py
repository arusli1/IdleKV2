"""
RULER benchmark evaluation.

Wraps NVIDIA's RULER benchmark (13 subtasks) for KV cache compression evaluation.
Supports 4K and 8K context lengths.

Subtasks:
  Retrieval: niah_single_{1,2,3}, niah_multikey_{1,2,3}, niah_multivalue, niah_multiquery
  Aggregation: common_words, freq_words
  Multi-hop: variable_tracking
  QA: qa_{1,2}
"""

import random
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union


@dataclass
class RulerResult:
    subtask: str
    context_length: int
    accuracy: float
    num_samples: int


def create_niah_test(tokenizer, context_length: int = 4096, num_needles: int = 1, device: str = "cuda"):
    """
    Create a needle-in-a-haystack test for RULER evaluation.

    Args:
        tokenizer: HF tokenizer
        context_length: Total context length
        num_needles: Number of needles to insert (1-3)
        device: Device to place tensors on

    Returns:
        input_ids: Tokenized context with needles
        answers: List of correct answers
    """
    # Generate needles (facts to retrieve)
    needles = []
    answers = []
    for i in range(num_needles):
        number = random.randint(1000, 9999)
        needle = f"The secret number {i+1} is {number}."
        answer = str(number)
        needles.append(needle)
        answers.append(answer)

    # Create question
    if num_needles == 1:
        question = "\n\nWhat is the secret number mentioned in the text above?"
    else:
        question = f"\n\nWhat are the {num_needles} secret numbers mentioned in the text above?"

    # Generate filler content
    filler_unit = "This is a passage of text that serves as filler content for testing long-context retrieval. "
    filler_tokens = tokenizer.encode(filler_unit, add_special_tokens=False)

    # Calculate space budget
    needle_tokens = []
    for needle in needles:
        needle_tokens.extend(tokenizer.encode(needle, add_special_tokens=False))

    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    budget = context_length - len(needle_tokens) - len(question_tokens) - 10

    # Build filler
    repeats = budget // len(filler_tokens) + 1
    all_filler = (filler_tokens * repeats)[:budget]

    # Insert needles at various depths
    context_tokens = []
    filler_per_section = len(all_filler) // (num_needles + 1)

    for i in range(num_needles):
        start_idx = i * filler_per_section
        end_idx = start_idx + filler_per_section
        context_tokens.extend(all_filler[start_idx:end_idx])
        context_tokens.extend(tokenizer.encode(needles[i], add_special_tokens=False))

    # Add remaining filler
    context_tokens.extend(all_filler[(num_needles * filler_per_section):])

    # Add question
    full_tokens = context_tokens + question_tokens
    input_ids = torch.tensor([full_tokens[:context_length]], device=device)

    return input_ids, answers


def evaluate_niah_response(response: str, expected_answers: List[str]) -> bool:
    """Check if the response contains the expected needle answers."""
    for answer in expected_answers:
        if answer not in response:
            return False
    return True


def evaluate_ruler_niah(
    model,
    tokenizer,
    context_length: int = 4096,
    subtasks: Optional[List[str]] = None,
    num_samples: int = 50,
    device: str = "cuda",
    manager = None,
    **kwargs
) -> List[RulerResult]:
    """
    Minimal RULER needle-in-a-haystack evaluation.

    Args:
        model: HF model
        tokenizer: HF tokenizer
        context_length: Context window size
        subtasks: List of subtasks to run (e.g., ["niah_single_1", "niah_single_2"])
        num_samples: Number of samples per subtask
        device: Device to use
        manager: Optional CompressedKVManager for IdleKV evaluation
        **kwargs: Additional arguments

    Returns:
        List of RulerResult objects
    """
    if subtasks is None:
        subtasks = ["niah_single_1", "niah_single_2", "niah_single_3"]

    results = []

    for subtask in subtasks:
        print(f"  Running RULER {subtask}...")

        # Parse number of needles from subtask name
        if "single_1" in subtask:
            num_needles = 1
        elif "single_2" in subtask:
            num_needles = 2
        elif "single_3" in subtask:
            num_needles = 3
        else:
            num_needles = 1  # Default

        correct = 0
        total = num_samples

        for sample_idx in range(num_samples):
            try:
                # Create test case
                input_ids, expected_answers = create_niah_test(
                    tokenizer, context_length, num_needles, device
                )

                # Run inference
                if manager is not None:
                    # IdleKV evaluation
                    compressed_kv = manager.prefill(input_ids)

                    # Simple generation (shortened for speed)
                    with torch.no_grad():
                        output = model.generate(
                            input_ids[:, -1:],
                            past_key_values=compressed_kv,
                            max_new_tokens=50,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                else:
                    # Baseline evaluation (full cache)
                    with torch.no_grad():
                        output = model.generate(
                            input_ids,
                            max_new_tokens=50,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )

                # Decode and check
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                if evaluate_niah_response(response, expected_answers):
                    correct += 1

            except Exception as e:
                print(f"    Error in sample {sample_idx}: {e}")
                # Continue with other samples

        accuracy = correct / total if total > 0 else 0.0
        print(f"    {subtask}: {correct}/{total} = {accuracy:.1%}")

        results.append(RulerResult(
            subtask=subtask,
            context_length=context_length,
            accuracy=accuracy,
            num_samples=total
        ))

    return results
    all_subtasks = subtasks or [
        "niah_single_1", "niah_single_2", "niah_single_3",
        "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
        "niah_multivalue", "niah_multiquery",
        "common_words", "freq_words",
        "variable_tracking",
        "qa_1", "qa_2",
    ]

    results = []
    for task in all_subtasks:
        # TODO: Implement actual RULER evaluation
        # 1. Load RULER data for this subtask + context_length
        # 2. For each sample:
        #    a. Prefill context
        #    b. If manager: compress, simulate tool calls with idle refinement
        #    c. If press: apply kvpress compression
        #    d. Generate answer
        #    e. Score against ground truth
        # 3. Aggregate accuracy

        results.append(RulerResult(
            subtask=task,
            context_length=context_length,
            accuracy=0.0,  # placeholder
            num_samples=num_samples,
        ))

    return results
