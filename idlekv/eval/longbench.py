"""
LongBench benchmark evaluation.

This module uses the official LongBench v1 prompt templates and task-specific
metrics for the English subtasks used in this repo. Dataset loading is
implemented directly against the published `data.zip` artifact so evaluation
does not depend on deprecated dataset-script execution.
"""

import gc
import json
import re
import string
import zipfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Callable, List, Optional

import torch
from huggingface_hub import hf_hub_download

from idlekv.eval.inference import generate_text


LONG_BENCH_REPO = "THUDM/LongBench"
LONG_BENCH_ARCHIVE = "data.zip"


@dataclass
class LongBenchResult:
    subtask: str
    score: float  # Official-style 0-100 task score
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


PROMPT_TEMPLATES = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question as concisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, "
        "using a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the "
        "question as concisely as you can, using a single phrase or sentence "
        "if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question "
        "is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
        "Do not provide any explanation.\n\nArticle: {context}\n\n"
        "Answer the question based on the above article as concisely as you "
        "can, using a single phrase or sentence if possible. If the question "
        "cannot be answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give "
        "me the answer and do not output any other words.\n\nQuestion: "
        "{input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page "
        "summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question "
        "or instruction. Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\nNow, answer the query based on the above "
        "meeting transcript in one or more sentences.\n\nQuery: {input}\n"
        "Answer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all "
        "news.\n\nNews:\n{context}\n\nNow, write a one-page summary of all the "
        "news.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some "
        "examples of questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the "
        "answer and do not output any other words. The following are some "
        "examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are "
        "some examples.\n\n{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them "
        "may be duplicates. Please carefully read these paragraphs and "
        "determine how many unique paragraphs there are after removing "
        "duplicates. In other words, how many non-repeating paragraphs are "
        "there in total?\n\n{context}\n\nPlease enter the final count of "
        "unique paragraphs after removing duplicates. The output format should "
        "only contain the number, such as 1, 2, 3, and so on.\n\nThe final "
        "answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. "
        "Please determine which paragraph the abstract is from.\n\n{context}"
        "\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the "
        "number of the paragraph that the abstract is from. The answer format "
        "must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer "
        "is: "
    ),
    "lcc": (
        "Please complete the code given below.\n{context}Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below.\n{context}{input}"
        "Next line of code:\n"
    ),
}


MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}


SINGLE_LINE_DATASETS = {"trec", "triviaqa", "samsum", "lsht"}


def _normalize_qa_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _f1_from_tokens(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    pred_counts = {}
    gold_counts = {}
    for token in prediction_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ground_truth_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, gold_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **_) -> float:
    pred_tokens = _normalize_qa_text(prediction).split()
    gold_tokens = _normalize_qa_text(ground_truth).split()
    return _f1_from_tokens(pred_tokens, gold_tokens)


def classification_score(prediction: str, ground_truth: str, *, all_classes=None, **_) -> float:
    all_classes = all_classes or []
    matches = [class_name for class_name in all_classes if class_name in prediction]
    matches = [
        match
        for match in matches
        if not (match in ground_truth and match != ground_truth)
    ]
    if ground_truth in matches and matches:
        return 1.0 / len(matches)
    return 0.0


def retrieval_score(prediction: str, ground_truth: str, **_) -> float:
    match = re.findall(r"Paragraph (\d+)", ground_truth)
    if not match:
        return 0.0
    target = match[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    hits = sum(1 for number in numbers if number == target)
    return hits / len(numbers)


def count_score(prediction: str, ground_truth: str, **_) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    target = str(ground_truth)
    hits = sum(1 for number in numbers if number == target)
    return hits / len(numbers)


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for left_token in left:
        curr = [0]
        for j, right_token in enumerate(right, start=1):
            if left_token == right_token:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_score(prediction: str, ground_truth: str, **_) -> float:
    pred_tokens = prediction.split()
    gold_tokens = ground_truth.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def code_sim_score(prediction: str, ground_truth: str, **_) -> float:
    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            candidate = line
            break
    return SequenceMatcher(None, candidate, ground_truth).ratio()


DATASET_TO_METRIC: dict[str, Callable[..., float]] = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_l_score,
    "qmsum": rouge_l_score,
    "multi_news": rouge_l_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_l_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


@lru_cache(maxsize=1)
def longbench_archive_path() -> str:
    return hf_hub_download(
        repo_id=LONG_BENCH_REPO,
        filename=LONG_BENCH_ARCHIVE,
        repo_type="dataset",
    )


@lru_cache(maxsize=None)
def load_longbench_records(subtask: str) -> tuple[dict, ...]:
    member_name = f"data/{subtask}.jsonl"
    archive_path = longbench_archive_path()
    records = []

    with zipfile.ZipFile(archive_path) as archive:
        try:
            with archive.open(member_name) as handle:
                for raw_line in handle:
                    if raw_line.strip():
                        records.append(json.loads(raw_line))
        except KeyError as exc:
            raise ValueError(f"LongBench subtask {subtask!r} not found in {archive_path}") from exc

    return tuple(records)


def select_longbench_records(subtask: str, num_samples: int) -> list[dict]:
    records = load_longbench_records(subtask)
    limit = min(num_samples, len(records))
    return list(records[:limit])


def truncate_middle(tokenizer, prompt: str, max_input_length: int) -> str:
    token_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if token_ids.shape[0] <= max_input_length:
        return prompt

    left = max_input_length // 2
    right = max_input_length - left
    return (
        tokenizer.decode(token_ids[:left], skip_special_tokens=True)
        + tokenizer.decode(token_ids[-right:], skip_special_tokens=True)
    )


def normalize_prediction_for_scoring(subtask: str, prediction: str) -> str:
    if subtask in SINGLE_LINE_DATASETS:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


def score_prediction(
    subtask: str,
    prediction: str,
    expected_answers: list[str],
    all_classes=None,
) -> float:
    metric = DATASET_TO_METRIC[subtask]
    normalized_prediction = normalize_prediction_for_scoring(subtask, prediction)
    best_score = 0.0
    for expected in expected_answers:
        best_score = max(
            best_score,
            metric(
                normalized_prediction,
                expected,
                all_classes=all_classes,
            ),
        )
    return best_score


def evaluate_longbench(
    model,
    tokenizer,
    subtasks: Optional[List[str]] = None,
    num_samples: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    manager=None,
    press=None,
    idle_budget_ms: float = 0.0,
    phases="1+2",
    sync_refresh_stride: Optional[int] = None,
    max_input_length: int = 4096,
    **kwargs
) -> List[LongBenchResult]:
    """
    Evaluate a lightweight LongBench v1 slice using official prompts/metrics.

    The repo’s nightly A10G setting intentionally truncates to a 4K prompt
    budget unless the caller overrides `max_input_length`.
    """
    if subtasks is None:
        subtasks = SUBTASKS

    results = []

    for subtask in subtasks:
        print(f"  Running LongBench {subtask}...")

        try:
            dataset = select_longbench_records(subtask, num_samples=num_samples)
            scores = []
            total_samples = len(dataset)
            last_error = None

            for idx, sample in enumerate(dataset):
                try:
                    prompt = PROMPT_TEMPLATES[subtask].format(
                        context=sample.get("context", ""),
                        input=sample.get("input", ""),
                    )
                    prompt = truncate_middle(tokenizer, prompt, max_input_length=max_input_length)
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)

                    generated = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=inputs["input_ids"],
                        max_new_tokens=MAX_NEW_TOKENS[subtask],
                        manager=manager,
                        press=press,
                        idle_budget_ms=idle_budget_ms,
                        phases=phases,
                        sync_refresh_stride=sync_refresh_stride,
                    )
                    prediction = generated["text"].strip()

                    scores.append(
                        score_prediction(
                            subtask,
                            prediction,
                            sample.get("answers", []),
                            all_classes=sample.get("all_classes"),
                        )
                    )

                except Exception as e:
                    print(f"      Error in sample {idx}: {e}")
                    last_error = e
                    continue
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if not scores and total_samples > 0 and last_error is not None:
                raise RuntimeError(
                    f"LongBench {subtask} failed for all {total_samples} samples"
                ) from last_error

            avg_score = 100.0 * sum(scores) / len(scores) if scores else 0.0
            print(f"    {subtask}: {len(scores)} samples, avg score = {avg_score:.2f}")
            results.append(LongBenchResult(
                subtask=subtask,
                score=avg_score,
                num_samples=len(scores),
            ))

        except Exception as e:
            raise RuntimeError(f"Failed to evaluate LongBench {subtask}: {e}") from e

    return results
