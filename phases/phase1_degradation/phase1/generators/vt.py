"""Variable-tracking task generation for multi-hop dependency chains."""

from __future__ import annotations

from ..helpers import make_rng, random_number, random_var
from ..models import PrefillSegment, RelevantSpan, TaskExample
from .common import search_context


# This generator builds variable-tracking tasks by hiding a chain of assignments
# inside long filler text, then asks the model to recover the final numeric value.
# The chain placement (depths and tail anchoring) lets evaluators probe memory
# eviction and multi-hop reasoning stability across context lengths.


def _unique_random_vars(rng, count: int, *, used: set[str] | None = None) -> list[str]:
    """Draw unique variable names so distractors never alias the true chain."""
    seen = set() if used is None else set(used)
    variables: list[str] = []
    while len(variables) < count:
        candidate = random_var(rng)
        if candidate in seen:
            continue
        seen.add(candidate)
        variables.append(candidate)
    return variables


def _spaced_depths(count: int) -> list[float]:
    """Spread inserts across the context body without tail-anchoring any of them."""
    if count <= 0:
        return []
    return [round((index + 1) / (count + 1), 4) for index in range(count)]


def _build_random_permute_vt(
    rng,
    *,
    num_hops: int,
    num_divergences: int,
    final_value: str,
) -> dict[str, object]:
    """Build a fully permuted VT chain with off-chain distractor branches."""
    if num_hops < 4:
        raise ValueError(f"Random-permuted VT probe needs at least 4 hops; got {num_hops}.")
    eligible_divergence_sources = list(range(1, num_hops - 1))
    if num_divergences < 1:
        raise ValueError(f"Random-permuted VT probe needs at least 1 divergence; got {num_divergences}.")
    if num_divergences > len(eligible_divergence_sources):
        raise ValueError(
            "Random-permuted VT probe cannot place "
            f"{num_divergences} divergences across only {len(eligible_divergence_sources)} interior nodes."
        )

    # The true chain is: V0=value, V0=V1, V1=V2, ..., V(n-2)=V(n-1), then ask for V(n-1).
    chain_variables = _unique_random_vars(rng, num_hops)
    distractor_variables = _unique_random_vars(rng, num_divergences, used=set(chain_variables))

    chain_specs: list[tuple[str, str, str, dict]] = [
        (
            "hop_1",
            "hop",
            f"{chain_variables[0]} = {final_value}.",
            {"statement": f"{chain_variables[0]} = {final_value}.", "hop_index": 1, "value": final_value},
        )
    ]
    for hop_index in range(2, num_hops + 1):
        source_var = chain_variables[hop_index - 2]
        target_var = chain_variables[hop_index - 1]
        statement = f"{source_var} = {target_var}."
        chain_specs.append(
            (
                f"hop_{hop_index}",
                "hop",
                statement,
                {
                    "statement": statement,
                    "hop_index": hop_index,
                },
            )
        )

    divergence_sources = sorted(rng.sample(eligible_divergence_sources, k=num_divergences))
    distractor_specs: list[tuple[str, str, str, dict]] = []
    for distractor_index, (source_index, distractor_var) in enumerate(
        zip(divergence_sources, distractor_variables, strict=True),
        start=1,
    ):
        statement = f"{chain_variables[source_index]} = {distractor_var}."
        distractor_specs.append(
            (
                f"distractor_{distractor_index}",
                "distractor",
                statement,
                {
                    "statement": statement,
                    "distractor": True,
                    "source_var": chain_variables[source_index],
                    "source_var_index": source_index,
                    "source_hop_index": source_index + 2,
                    "distractor_var": distractor_var,
                },
            )
        )

    ordered_specs = chain_specs + distractor_specs
    rng.shuffle(ordered_specs)
    body_specs: list[tuple[str, str, str, float, dict]] = []
    hop_depth_by_index: dict[int, float] = {}
    permute_order: list[str] = []
    for depth_fraction, (name, kind, statement, metadata) in zip(_spaced_depths(len(ordered_specs)), ordered_specs, strict=True):
        body_specs.append((name, kind, statement, depth_fraction, metadata))
        permute_order.append(name)
        if kind == "hop":
            hop_depth_by_index[int(metadata["hop_index"])] = depth_fraction

    return {
        "variables": chain_variables,
        "distractor_variables": distractor_variables,
        "body_specs": body_specs,
        "tail_specs": [],
        "query_var": chain_variables[-1],
        "hop_depths": [hop_depth_by_index[hop_index] for hop_index in range(1, num_hops + 1)],
        "tail_anchored_hops": 0,
        "terminal_tail_anchored": False,
        "terminal_value_var": chain_variables[0],
        "num_divergences": num_divergences,
        "metadata_extra": {
            "permute_order": permute_order,
            "divergence_sources": divergence_sources,
            "distractor_variables": distractor_variables,
        },
    }


def build_vt_example(
    task_name: str,
    index: int,
    target_context_length: int,
    tokenizer,
    *,
    num_hops: int,
    depths: list[float],
    terminal_depth: float | None,
    permute: bool = False,
    random_permute: bool = False,
    num_divergences: int = 0,
    filler_kind: str = "essay",
    dataset_seed_offset: int = 0,
) -> TaskExample:
    """Build a variable-assignment chain hidden inside long filler text."""
    if len(depths) > num_hops:
        raise ValueError(f"Expected at most {num_hops} hop depths for {task_name}; got {len(depths)}.")

    # --- Chain setup: deterministic variable and value selection per sample. ---
    # Every sample gets a deterministic chain like VAR_A = VAR_B = ... = 12345
    # so we can later pinpoint exactly which hop was lost.
    rng = make_rng((target_context_length * 2000) + index + num_hops + int(dataset_seed_offset))
    final_value = random_number(rng)
    # --- Depth placement: split essential hops into body vs tail inserts. ---
    if permute and random_permute:
        randomized = _build_random_permute_vt(
            rng,
            num_hops=num_hops,
            num_divergences=num_divergences,
            final_value=final_value,
        )
        variables = randomized["variables"]
        body_specs = randomized["body_specs"]
        tail_specs = randomized["tail_specs"]
        query_var = randomized["query_var"]
        hop_depths = randomized["hop_depths"]
        terminal_tail_anchored = randomized["terminal_tail_anchored"]
        tail_anchored_hops = randomized["tail_anchored_hops"]
        terminal_value_var = randomized["terminal_value_var"]
        effective_num_divergences = randomized["num_divergences"]
        metadata_extra = randomized["metadata_extra"]
    elif permute:
        variables = [random_var(rng) for _ in range(num_hops + 1)]
        if num_hops != 4:
            raise ValueError(f"Permuted VT probe only supports num_hops=4; got {num_hops}.")
        if len(variables) != 5:
            raise ValueError(f"Permuted VT probe expected 5 variables; got {len(variables)}.")

        # Logical chain:
        #   A = value, A = B, B = C, C = D, with C = E as a distractor.
        # Text order:
        #   A = B, C = D, A = value, C = E, B = C
        hop_1_statement = f"{variables[0]} = {variables[1]}."
        hop_3_statement = f"{variables[2]} = {variables[3]}."
        hop_4_statement = f"{variables[0]} = {final_value}."
        distractor_statement = f"{variables[2]} = {variables[4]}."
        hop_2_statement = f"{variables[1]} = {variables[2]}."

        body_specs: list[tuple[str, str, str, float, dict]] = [
            (
                "hop_1",
                "hop",
                hop_1_statement,
                depths[0] if depths else 0.12,
                {"statement": hop_1_statement, "hop_index": 1},
            ),
            (
                "hop_3",
                "hop",
                hop_3_statement,
                depths[1] if len(depths) > 1 else 0.37,
                {"statement": hop_3_statement, "hop_index": 3},
            ),
            (
                "hop_4",
                "hop",
                hop_4_statement,
                depths[2] if len(depths) > 2 else 0.62,
                {"statement": hop_4_statement, "hop_index": 4, "value": final_value},
            ),
        ]
        tail_specs: list[tuple[str, str, str, float, dict]] = [
            (
                "distractor_1",
                "distractor",
                distractor_statement,
                1.0,
                {"statement": distractor_statement, "distractor": True},
            ),
            (
                "hop_2",
                "hop",
                hop_2_statement,
                1.0,
                {"statement": hop_2_statement, "hop_index": 2},
            ),
        ]
        query_var = variables[3]
        hop_depths = [body_specs[0][3], tail_specs[1][3], body_specs[1][3], body_specs[2][3]]
        terminal_tail_anchored = False
        tail_anchored_hops = 1
        terminal_value_var = variables[0]
        effective_num_divergences = 1
        metadata_extra = {}
    else:
        variables = [random_var(rng) for _ in range(num_hops + 1)]
        chain_statements = [f"{variables[hop_index]} = {variables[hop_index + 1]}." for hop_index in range(num_hops)]
        terminal_statement = f"{variables[-1]} = {final_value}."

        body_specs = []
        tail_specs = []
        for hop_index, statement in enumerate(chain_statements, start=1):
            # Earlier hops can be pinned to specific depths; any leftover hops get
            # shoved to the tail to create especially eviction-sensitive links.
            target_specs = body_specs if hop_index <= len(depths) else tail_specs
            target_specs.append(
                (
                    f"hop_{hop_index}",
                    "hop",
                    statement,
                    depths[hop_index - 1] if hop_index <= len(depths) else 1.0,
                    {"statement": statement, "hop_index": hop_index},
                )
            )

        terminal_spec = (
            "terminal_value",
            "terminal",
            terminal_statement,
            1.0 if terminal_depth is None else terminal_depth,
            {"statement": terminal_statement, "value": final_value},
        )
        if terminal_depth is None:
            tail_specs.append(terminal_spec)
        else:
            body_specs.append(terminal_spec)
        query_var = variables[0]
        hop_depths = depths
        terminal_tail_anchored = terminal_depth is None
        tail_anchored_hops = max(num_hops - len(depths), 0)
        terminal_value_var = variables[-1]
        effective_num_divergences = 0
        metadata_extra = {}

    # --- Context construction: embed statements at target depths in filler text. ---
    # Build a long filler context that hides each assignment statement at the
    # requested depth fractions.
    prefix = "Memorize and follow the variable assignment chain hidden in the following text.\n\n"
    built = search_context(
        tokenizer=tokenizer,
        target_context_length=target_context_length,
        prefix=prefix,
        base_filler_kind=filler_kind,
        inserts=[spec[2] for spec in body_specs],
        depths=[spec[3] for spec in body_specs],
        tail_inserts=[spec[2] for spec in tail_specs],
    )

    relevant_spans = []
    for (name, kind, _, depth_fraction, metadata), (char_start, char_end) in zip(body_specs + tail_specs, built.insert_spans):
        # Each hop becomes a traceable span so the evaluator can identify the
        # first place where the reasoning chain was broken by eviction.
        relevant_spans.append(
            RelevantSpan(
                name=name,
                kind=kind,
                char_start=char_start,
                char_end=char_end,
                depth_fraction=depth_fraction,
                metadata=metadata,
            )
        )

    question = f"What is the final numeric value of {query_var}?"
    answer_prefix = f" Answer: The final numeric value of {query_var} is "

    # --- Final task construction: package context, question, and labels. ---
    return TaskExample(
        index=index,
        task_name=task_name,
        task_family="vt",
        context=built.context,
        question=question,
        answer_prefix=answer_prefix,
        outputs=[final_value],
        max_new_tokens=24,
        target_context_length=target_context_length,
        relevant_spans=relevant_spans,
        prefill_segments=[PrefillSegment(name="context", char_start=0, char_end=len(built.context))],
        metadata={
            "query_var": query_var,
            "variables": variables,
            # These fields make the later error taxonomy explainable instead of
            # just reporting "wrong answer" for a broken chain.
            "num_hops": num_hops,
            "hop_depths": hop_depths,
            "terminal_depth": terminal_depth,
            "tail_anchored_hops": tail_anchored_hops,
            "terminal_tail_anchored": terminal_tail_anchored,
            "terminal_value": final_value,
            "terminal_value_var": terminal_value_var,
            "permute": permute,
            "random_permute": random_permute,
            "num_divergences": effective_num_divergences,
            "filler_kind": filler_kind,
            **metadata_extra,
        },
    )
