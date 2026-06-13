from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from eval_kv_frontier import evaluate_with_model, load_model, seed_everything


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_csv_positions(raw: str) -> list[str]:
    allowed = {"prefix", "middle", "recent", "random"}
    values = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown target positions: {unknown}")
    if not values:
        raise argparse.ArgumentTypeError("expected at least one target position")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep target position and retained-token budgets for KV frontier mapping."
    )
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-gpt2")
    parser.add_argument("--items", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--task",
        choices=[
            "kv",
            "alias_bridge",
            "paraphrase_bridge",
            "varied_bridge",
            "format_bridge",
            "id_bridge",
            "descriptor_bridge",
            "shuffled_descriptor_bridge",
            "heldout_descriptor_bridge",
            "heldout_descriptor_bridge_v2",
            "heldout_descriptor_bridge_hard_negative",
        ],
        default="kv",
        help="Synthetic task family to evaluate.",
    )
    parser.add_argument(
        "--target-positions",
        type=parse_csv_positions,
        default=parse_csv_positions("prefix,middle,recent"),
        help="Comma-separated target positions to sweep.",
    )
    parser.add_argument(
        "--budgets",
        type=parse_csv_ints,
        default=parse_csv_ints("128,192,384"),
        help="Comma-separated total retained-token budgets.",
    )
    parser.add_argument(
        "--sink-tokens",
        type=int,
        default=64,
        help="Attention-sink tokens for streaming policy; recent tokens are budget - sink.",
    )
    parser.add_argument(
        "--lexical-records",
        type=int,
        default=1,
        help="Number of record lines to retain for lexical retrieval.",
    )
    parser.add_argument(
        "--prefilter-candidates",
        type=int,
        default=16,
        help="Number of descriptor-prefiltered candidate memos to rerank in prefilter_lm_bridge.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--scoring", choices=["generate", "rank"], default="rank")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--diagnose-final-scores",
        action="store_true",
        help="Emit descriptor-prefilter final-selection score traces and summary metrics.",
    )
    parser.add_argument(
        "--diagnose-contrastive-scores",
        action="store_true",
        help="Emit contrastive descriptor verifier score traces and summary metrics.",
    )
    parser.add_argument(
        "--diagnose-fewshot-scores",
        action="store_true",
        help="Emit few-shot descriptor translation score traces and summary metrics.",
    )
    parser.add_argument(
        "--diagnose-forced-choice-scores",
        action="store_true",
        help="Emit forced-choice descriptor label score traces and summary metrics.",
    )
    parser.add_argument(
        "--diagnose-induced-map-scores",
        action="store_true",
        help="Emit in-context induced descriptor-map score traces and summary metrics.",
    )
    parser.add_argument(
        "--diagnose-external-support-induced-map-scores",
        action="store_true",
        help="Emit task-external support induced descriptor-map score traces and summary metrics.",
    )
    parser.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=5,
        help="Number of top candidates to print per diagnostic scorer.",
    )
    parser.add_argument("--metrics-path", default=os.environ.get("AAD_METRICS_PATH"))
    return parser.parse_args()


def cell_name(position: str, budget: int, metric: str) -> str:
    return f"{position}_b{budget}_{metric}"


def run_sweep(args: argparse.Namespace) -> dict[str, float | int | str]:
    seed_everything(args.seed)
    tokenizer, model, device = load_model(args)
    metrics: dict[str, float | int | str] = {
        "eval_completed": 1,
        "seed": args.seed,
        "items": args.items,
        "distractors": args.distractors,
        "model": args.model,
        "scoring": args.scoring,
        "task": args.task,
        "sink_tokens": args.sink_tokens,
        "lexical_records": args.lexical_records,
        "prefilter_candidates": args.prefilter_candidates,
        "num_positions": len(args.target_positions),
        "num_budgets": len(args.budgets),
        "num_cells": len(args.target_positions) * len(args.budgets),
    }
    streaming_accuracies: list[float] = []
    truncate_accuracies: list[float] = []
    oracle_accuracies: list[float] = []
    lexical_accuracies: list[float] = []
    bridge_accuracies: list[float] = []
    prose_bridge_accuracies: list[float] = []
    entity_bridge_accuracies: list[float] = []
    overlap_bridge_accuracies: list[float] = []
    id_bridge_accuracies: list[float] = []
    char_bridge_accuracies: list[float] = []
    neighbor_bridge_accuracies: list[float] = []
    semantic_bridge_accuracies: list[float] = []
    embedding_bridge_accuracies: list[float] = []
    phrase_lm_bridge_accuracies: list[float] = []
    prefilter_lm_bridge_accuracies: list[float] = []
    prefilter_compressed_lm_bridge_accuracies: list[float] = []
    margin_guarded_prefilter_compressed_lm_bridge_accuracies: list[float] = []
    reverse_descriptor_prefilter_compressed_lm_bridge_accuracies: list[float] = []
    bidirectional_descriptor_prefilter_compressed_lm_bridge_accuracies: list[float] = []
    bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_accuracies: list[float] = []
    lexicon_descriptor_prefilter_compressed_lm_bridge_accuracies: list[float] = []
    fewshot_descriptor_prefilter_compressed_lm_bridge_accuracies: list[float] = []
    forced_choice_descriptor_prefilter_compressed_lm_bridge_accuracies: list[float] = []
    contrastive_descriptor_prefilter_compressed_lm_bridge_accuracies: list[float] = []
    compressed_lm_bridge_accuracies: list[float] = []
    lm_bridge_accuracies: list[float] = []
    full_accuracies: list[float] = []
    streaming_gaps: list[float] = []
    truncate_gaps: list[float] = []
    oracle_gaps: list[float] = []
    lexical_gaps: list[float] = []
    bridge_gaps: list[float] = []
    prose_bridge_gaps: list[float] = []
    entity_bridge_gaps: list[float] = []
    overlap_bridge_gaps: list[float] = []
    id_bridge_gaps: list[float] = []
    char_bridge_gaps: list[float] = []
    neighbor_bridge_gaps: list[float] = []
    semantic_bridge_gaps: list[float] = []
    embedding_bridge_gaps: list[float] = []
    phrase_lm_bridge_gaps: list[float] = []
    prefilter_lm_bridge_gaps: list[float] = []
    prefilter_compressed_lm_bridge_gaps: list[float] = []
    margin_guarded_prefilter_compressed_lm_bridge_gaps: list[float] = []
    reverse_descriptor_prefilter_compressed_lm_bridge_gaps: list[float] = []
    bidirectional_descriptor_prefilter_compressed_lm_bridge_gaps: list[float] = []
    bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_gaps: list[float] = []
    lexicon_descriptor_prefilter_compressed_lm_bridge_gaps: list[float] = []
    fewshot_descriptor_prefilter_compressed_lm_bridge_gaps: list[float] = []
    forced_choice_descriptor_prefilter_compressed_lm_bridge_gaps: list[float] = []
    contrastive_descriptor_prefilter_compressed_lm_bridge_gaps: list[float] = []
    compressed_lm_bridge_gaps: list[float] = []
    lm_bridge_gaps: list[float] = []
    prefilter_lm_bridge_rerank_fractions: list[float] = []
    prefilter_lm_bridge_target_ranks: list[float] = []
    prefilter_lm_bridge_target_survival_rates: list[float] = []
    prefilter_compressed_lm_bridge_rerank_fractions: list[float] = []
    prefilter_compressed_lm_bridge_target_ranks: list[float] = []
    prefilter_compressed_lm_bridge_target_survival_rates: list[float] = []
    margin_guarded_prefilter_compressed_lm_bridge_rerank_fractions: list[float] = []
    margin_guarded_prefilter_compressed_lm_bridge_target_ranks: list[float] = []
    margin_guarded_prefilter_compressed_lm_bridge_target_survival_rates: list[float] = []
    margin_guarded_prefilter_compressed_lm_bridge_override_rates: list[float] = []
    reverse_descriptor_prefilter_compressed_lm_bridge_rerank_fractions: list[float] = []
    reverse_descriptor_prefilter_compressed_lm_bridge_target_ranks: list[float] = []
    reverse_descriptor_prefilter_compressed_lm_bridge_target_survival_rates: list[float] = []
    bidirectional_descriptor_prefilter_compressed_lm_bridge_rerank_fractions: list[float] = []
    bidirectional_descriptor_prefilter_compressed_lm_bridge_target_ranks: list[float] = []
    bidirectional_descriptor_prefilter_compressed_lm_bridge_target_survival_rates: list[float] = []
    bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_rerank_fractions: list[float] = []
    bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_target_ranks: list[float] = []
    bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_target_survival_rates: list[float] = []
    lexicon_descriptor_prefilter_compressed_lm_bridge_rerank_fractions: list[float] = []
    lexicon_descriptor_prefilter_compressed_lm_bridge_target_ranks: list[float] = []
    lexicon_descriptor_prefilter_compressed_lm_bridge_target_survival_rates: list[float] = []
    fewshot_descriptor_prefilter_compressed_lm_bridge_rerank_fractions: list[float] = []
    fewshot_descriptor_prefilter_compressed_lm_bridge_target_ranks: list[float] = []
    fewshot_descriptor_prefilter_compressed_lm_bridge_target_survival_rates: list[float] = []
    forced_choice_descriptor_prefilter_compressed_lm_bridge_rerank_fractions: list[float] = []
    forced_choice_descriptor_prefilter_compressed_lm_bridge_target_ranks: list[float] = []
    forced_choice_descriptor_prefilter_compressed_lm_bridge_target_survival_rates: list[float] = []
    contrastive_descriptor_prefilter_compressed_lm_bridge_rerank_fractions: list[float] = []
    contrastive_descriptor_prefilter_compressed_lm_bridge_target_ranks: list[float] = []
    contrastive_descriptor_prefilter_compressed_lm_bridge_target_survival_rates: list[float] = []
    diagnostic_metric_values: dict[str, list[float]] = {}

    for position in args.target_positions:
        for budget in args.budgets:
            if budget <= args.sink_tokens:
                raise ValueError(
                    f"budget {budget} must be greater than sink tokens {args.sink_tokens}"
                )
            print(
                f"sweep cell position={position} budget={budget}",
                file=sys.stderr,
                flush=True,
            )
            cell_args = argparse.Namespace(**vars(args))
            cell_args.target_position = position
            cell_args.truncate_tokens = budget
            cell_args.recent_tokens = budget - args.sink_tokens
            cell_metrics = evaluate_with_model(cell_args, tokenizer, model, device)

            numeric_names = [
                "full_accuracy",
                "truncate_accuracy",
                "streaming_accuracy",
                "oracle_span_accuracy",
                "lexical_accuracy",
                "bridge_accuracy",
                "prose_bridge_accuracy",
                "entity_bridge_accuracy",
                "overlap_bridge_accuracy",
                "id_bridge_accuracy",
                "char_bridge_accuracy",
                "neighbor_bridge_accuracy",
                "semantic_bridge_accuracy",
                "embedding_bridge_accuracy",
                "phrase_lm_bridge_accuracy",
                "prefilter_lm_bridge_accuracy",
                "prefilter_compressed_lm_bridge_accuracy",
                "margin_guarded_prefilter_compressed_lm_bridge_accuracy",
                "reverse_descriptor_prefilter_compressed_lm_bridge_accuracy",
                "bidirectional_descriptor_prefilter_compressed_lm_bridge_accuracy",
                "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_accuracy",
                "lexicon_descriptor_prefilter_compressed_lm_bridge_accuracy",
                "fewshot_descriptor_prefilter_compressed_lm_bridge_accuracy",
                "forced_choice_descriptor_prefilter_compressed_lm_bridge_accuracy",
                "contrastive_descriptor_prefilter_compressed_lm_bridge_accuracy",
                "compressed_lm_bridge_accuracy",
                "lm_bridge_accuracy",
                "truncate_gap_vs_full",
                "streaming_gap_vs_full",
                "oracle_span_gap_vs_full",
                "lexical_gap_vs_full",
                "bridge_gap_vs_full",
                "prose_bridge_gap_vs_full",
                "entity_bridge_gap_vs_full",
                "overlap_bridge_gap_vs_full",
                "id_bridge_gap_vs_full",
                "char_bridge_gap_vs_full",
                "neighbor_bridge_gap_vs_full",
                "semantic_bridge_gap_vs_full",
                "embedding_bridge_gap_vs_full",
                "phrase_lm_bridge_gap_vs_full",
                "prefilter_lm_bridge_gap_vs_full",
                "prefilter_compressed_lm_bridge_gap_vs_full",
                "margin_guarded_prefilter_compressed_lm_bridge_gap_vs_full",
                "reverse_descriptor_prefilter_compressed_lm_bridge_gap_vs_full",
                "bidirectional_descriptor_prefilter_compressed_lm_bridge_gap_vs_full",
                "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_gap_vs_full",
                "lexicon_descriptor_prefilter_compressed_lm_bridge_gap_vs_full",
                "fewshot_descriptor_prefilter_compressed_lm_bridge_gap_vs_full",
                "forced_choice_descriptor_prefilter_compressed_lm_bridge_gap_vs_full",
                "contrastive_descriptor_prefilter_compressed_lm_bridge_gap_vs_full",
                "compressed_lm_bridge_gap_vs_full",
                "lm_bridge_gap_vs_full",
                "truncate_kept_fraction",
                "streaming_kept_fraction",
                "oracle_span_kept_fraction",
                "lexical_kept_fraction",
                "bridge_kept_fraction",
                "prose_bridge_kept_fraction",
                "entity_bridge_kept_fraction",
                "overlap_bridge_kept_fraction",
                "id_bridge_kept_fraction",
                "char_bridge_kept_fraction",
                "neighbor_bridge_kept_fraction",
                "semantic_bridge_kept_fraction",
                "embedding_bridge_kept_fraction",
                "phrase_lm_bridge_kept_fraction",
                "prefilter_lm_bridge_kept_fraction",
                "prefilter_compressed_lm_bridge_kept_fraction",
                "margin_guarded_prefilter_compressed_lm_bridge_kept_fraction",
                "reverse_descriptor_prefilter_compressed_lm_bridge_kept_fraction",
                "bidirectional_descriptor_prefilter_compressed_lm_bridge_kept_fraction",
                "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_kept_fraction",
                "lexicon_descriptor_prefilter_compressed_lm_bridge_kept_fraction",
                "fewshot_descriptor_prefilter_compressed_lm_bridge_kept_fraction",
                "forced_choice_descriptor_prefilter_compressed_lm_bridge_kept_fraction",
                "contrastive_descriptor_prefilter_compressed_lm_bridge_kept_fraction",
                "compressed_lm_bridge_kept_fraction",
                "lm_bridge_kept_fraction",
                "prefilter_lm_bridge_candidate_count",
                "prefilter_lm_bridge_rerank_count",
                "prefilter_lm_bridge_rerank_fraction",
                "prefilter_lm_bridge_mean_target_rank",
                "prefilter_lm_bridge_min_target_rank",
                "prefilter_lm_bridge_max_target_rank",
                "prefilter_lm_bridge_target_survival_rate",
                "prefilter_compressed_lm_bridge_candidate_count",
                "prefilter_compressed_lm_bridge_rerank_count",
                "prefilter_compressed_lm_bridge_rerank_fraction",
                "prefilter_compressed_lm_bridge_mean_target_rank",
                "prefilter_compressed_lm_bridge_min_target_rank",
                "prefilter_compressed_lm_bridge_max_target_rank",
                "prefilter_compressed_lm_bridge_target_survival_rate",
                "margin_guarded_prefilter_compressed_lm_bridge_candidate_count",
                "margin_guarded_prefilter_compressed_lm_bridge_rerank_count",
                "margin_guarded_prefilter_compressed_lm_bridge_rerank_fraction",
                "margin_guarded_prefilter_compressed_lm_bridge_mean_target_rank",
                "margin_guarded_prefilter_compressed_lm_bridge_min_target_rank",
                "margin_guarded_prefilter_compressed_lm_bridge_max_target_rank",
                "margin_guarded_prefilter_compressed_lm_bridge_target_survival_rate",
                "margin_guarded_prefilter_compressed_lm_bridge_override_rate",
                "reverse_descriptor_prefilter_compressed_lm_bridge_candidate_count",
                "reverse_descriptor_prefilter_compressed_lm_bridge_rerank_count",
                "reverse_descriptor_prefilter_compressed_lm_bridge_rerank_fraction",
                "reverse_descriptor_prefilter_compressed_lm_bridge_mean_target_rank",
                "reverse_descriptor_prefilter_compressed_lm_bridge_min_target_rank",
                "reverse_descriptor_prefilter_compressed_lm_bridge_max_target_rank",
                "reverse_descriptor_prefilter_compressed_lm_bridge_target_survival_rate",
                "bidirectional_descriptor_prefilter_compressed_lm_bridge_candidate_count",
                "bidirectional_descriptor_prefilter_compressed_lm_bridge_rerank_count",
                "bidirectional_descriptor_prefilter_compressed_lm_bridge_rerank_fraction",
                "bidirectional_descriptor_prefilter_compressed_lm_bridge_mean_target_rank",
                "bidirectional_descriptor_prefilter_compressed_lm_bridge_min_target_rank",
                "bidirectional_descriptor_prefilter_compressed_lm_bridge_max_target_rank",
                "bidirectional_descriptor_prefilter_compressed_lm_bridge_target_survival_rate",
                "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_candidate_count",
                "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_rerank_count",
                "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_rerank_fraction",
                "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_mean_target_rank",
                "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_min_target_rank",
                "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_max_target_rank",
                "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_target_survival_rate",
                "lexicon_descriptor_prefilter_compressed_lm_bridge_candidate_count",
                "lexicon_descriptor_prefilter_compressed_lm_bridge_rerank_count",
                "lexicon_descriptor_prefilter_compressed_lm_bridge_rerank_fraction",
                "lexicon_descriptor_prefilter_compressed_lm_bridge_mean_target_rank",
                "lexicon_descriptor_prefilter_compressed_lm_bridge_min_target_rank",
                "lexicon_descriptor_prefilter_compressed_lm_bridge_max_target_rank",
                "lexicon_descriptor_prefilter_compressed_lm_bridge_target_survival_rate",
                "fewshot_descriptor_prefilter_compressed_lm_bridge_candidate_count",
                "fewshot_descriptor_prefilter_compressed_lm_bridge_rerank_count",
                "fewshot_descriptor_prefilter_compressed_lm_bridge_rerank_fraction",
                "fewshot_descriptor_prefilter_compressed_lm_bridge_mean_target_rank",
                "fewshot_descriptor_prefilter_compressed_lm_bridge_min_target_rank",
                "fewshot_descriptor_prefilter_compressed_lm_bridge_max_target_rank",
                "fewshot_descriptor_prefilter_compressed_lm_bridge_target_survival_rate",
                "forced_choice_descriptor_prefilter_compressed_lm_bridge_candidate_count",
                "forced_choice_descriptor_prefilter_compressed_lm_bridge_rerank_count",
                "forced_choice_descriptor_prefilter_compressed_lm_bridge_rerank_fraction",
                "forced_choice_descriptor_prefilter_compressed_lm_bridge_mean_target_rank",
                "forced_choice_descriptor_prefilter_compressed_lm_bridge_min_target_rank",
                "forced_choice_descriptor_prefilter_compressed_lm_bridge_max_target_rank",
                "forced_choice_descriptor_prefilter_compressed_lm_bridge_target_survival_rate",
                "contrastive_descriptor_prefilter_compressed_lm_bridge_candidate_count",
                "contrastive_descriptor_prefilter_compressed_lm_bridge_rerank_count",
                "contrastive_descriptor_prefilter_compressed_lm_bridge_rerank_fraction",
                "contrastive_descriptor_prefilter_compressed_lm_bridge_mean_target_rank",
                "contrastive_descriptor_prefilter_compressed_lm_bridge_min_target_rank",
                "contrastive_descriptor_prefilter_compressed_lm_bridge_max_target_rank",
                "contrastive_descriptor_prefilter_compressed_lm_bridge_target_survival_rate",
                "mean_full_tokens",
                "mean_target_position_fraction",
                "mean_query_position_fraction",
                "mean_target_query_distance_fraction",
            ]
            for name in numeric_names:
                value = cell_metrics[name]
                metrics[cell_name(position, budget, name)] = value
            for name, value in cell_metrics.items():
                if (
                    name.startswith("mean_diagnostic_")
                    or name.startswith("mean_contrastive_diagnostic_")
                    or name.startswith("mean_fewshot_diagnostic_")
                    or name.startswith("mean_forced_choice_diagnostic_")
                    or name.startswith("mean_induced_map_diagnostic_")
                    or name.startswith("mean_external_support_induced_map_diagnostic_")
                ):
                    numeric_value = float(value)
                    metrics[cell_name(position, budget, name)] = numeric_value
                    diagnostic_metric_values.setdefault(name, []).append(numeric_value)

            full_accuracies.append(float(cell_metrics["full_accuracy"]))
            truncate_accuracies.append(float(cell_metrics["truncate_accuracy"]))
            streaming_accuracies.append(float(cell_metrics["streaming_accuracy"]))
            oracle_accuracies.append(float(cell_metrics["oracle_span_accuracy"]))
            lexical_accuracies.append(float(cell_metrics["lexical_accuracy"]))
            bridge_accuracies.append(float(cell_metrics["bridge_accuracy"]))
            prose_bridge_accuracies.append(
                float(cell_metrics["prose_bridge_accuracy"])
            )
            entity_bridge_accuracies.append(
                float(cell_metrics["entity_bridge_accuracy"])
            )
            overlap_bridge_accuracies.append(
                float(cell_metrics["overlap_bridge_accuracy"])
            )
            id_bridge_accuracies.append(float(cell_metrics["id_bridge_accuracy"]))
            char_bridge_accuracies.append(
                float(cell_metrics["char_bridge_accuracy"])
            )
            neighbor_bridge_accuracies.append(
                float(cell_metrics["neighbor_bridge_accuracy"])
            )
            semantic_bridge_accuracies.append(
                float(cell_metrics["semantic_bridge_accuracy"])
            )
            embedding_bridge_accuracies.append(
                float(cell_metrics["embedding_bridge_accuracy"])
            )
            phrase_lm_bridge_accuracies.append(
                float(cell_metrics["phrase_lm_bridge_accuracy"])
            )
            prefilter_lm_bridge_accuracies.append(
                float(cell_metrics["prefilter_lm_bridge_accuracy"])
            )
            prefilter_compressed_lm_bridge_accuracies.append(
                float(cell_metrics["prefilter_compressed_lm_bridge_accuracy"])
            )
            margin_guarded_prefilter_compressed_lm_bridge_accuracies.append(
                float(
                    cell_metrics[
                        "margin_guarded_prefilter_compressed_lm_bridge_accuracy"
                    ]
                )
            )
            reverse_descriptor_prefilter_compressed_lm_bridge_accuracies.append(
                float(
                    cell_metrics[
                        "reverse_descriptor_prefilter_compressed_lm_bridge_accuracy"
                    ]
                )
            )
            bidirectional_descriptor_prefilter_compressed_lm_bridge_accuracies.append(
                float(
                    cell_metrics[
                        "bidirectional_descriptor_prefilter_compressed_lm_bridge_accuracy"
                    ]
                )
            )
            bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_accuracies.append(
                float(
                    cell_metrics[
                        "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_accuracy"
                    ]
                )
            )
            lexicon_descriptor_prefilter_compressed_lm_bridge_accuracies.append(
                float(
                    cell_metrics[
                        "lexicon_descriptor_prefilter_compressed_lm_bridge_accuracy"
                    ]
                )
            )
            fewshot_descriptor_prefilter_compressed_lm_bridge_accuracies.append(
                float(
                    cell_metrics[
                        "fewshot_descriptor_prefilter_compressed_lm_bridge_accuracy"
                    ]
                )
            )
            forced_choice_descriptor_prefilter_compressed_lm_bridge_accuracies.append(
                float(
                    cell_metrics[
                        "forced_choice_descriptor_prefilter_compressed_lm_bridge_accuracy"
                    ]
                )
            )
            contrastive_descriptor_prefilter_compressed_lm_bridge_accuracies.append(
                float(
                    cell_metrics[
                        "contrastive_descriptor_prefilter_compressed_lm_bridge_accuracy"
                    ]
                )
            )
            compressed_lm_bridge_accuracies.append(
                float(cell_metrics["compressed_lm_bridge_accuracy"])
            )
            lm_bridge_accuracies.append(float(cell_metrics["lm_bridge_accuracy"]))
            truncate_gaps.append(float(cell_metrics["truncate_gap_vs_full"]))
            streaming_gaps.append(float(cell_metrics["streaming_gap_vs_full"]))
            oracle_gaps.append(float(cell_metrics["oracle_span_gap_vs_full"]))
            lexical_gaps.append(float(cell_metrics["lexical_gap_vs_full"]))
            bridge_gaps.append(float(cell_metrics["bridge_gap_vs_full"]))
            prose_bridge_gaps.append(float(cell_metrics["prose_bridge_gap_vs_full"]))
            entity_bridge_gaps.append(float(cell_metrics["entity_bridge_gap_vs_full"]))
            overlap_bridge_gaps.append(
                float(cell_metrics["overlap_bridge_gap_vs_full"])
            )
            id_bridge_gaps.append(float(cell_metrics["id_bridge_gap_vs_full"]))
            char_bridge_gaps.append(float(cell_metrics["char_bridge_gap_vs_full"]))
            neighbor_bridge_gaps.append(
                float(cell_metrics["neighbor_bridge_gap_vs_full"])
            )
            semantic_bridge_gaps.append(
                float(cell_metrics["semantic_bridge_gap_vs_full"])
            )
            embedding_bridge_gaps.append(
                float(cell_metrics["embedding_bridge_gap_vs_full"])
            )
            phrase_lm_bridge_gaps.append(
                float(cell_metrics["phrase_lm_bridge_gap_vs_full"])
            )
            prefilter_lm_bridge_gaps.append(
                float(cell_metrics["prefilter_lm_bridge_gap_vs_full"])
            )
            prefilter_compressed_lm_bridge_gaps.append(
                float(cell_metrics["prefilter_compressed_lm_bridge_gap_vs_full"])
            )
            margin_guarded_prefilter_compressed_lm_bridge_gaps.append(
                float(
                    cell_metrics[
                        "margin_guarded_prefilter_compressed_lm_bridge_gap_vs_full"
                    ]
                )
            )
            reverse_descriptor_prefilter_compressed_lm_bridge_gaps.append(
                float(
                    cell_metrics[
                        "reverse_descriptor_prefilter_compressed_lm_bridge_gap_vs_full"
                    ]
                )
            )
            bidirectional_descriptor_prefilter_compressed_lm_bridge_gaps.append(
                float(
                    cell_metrics[
                        "bidirectional_descriptor_prefilter_compressed_lm_bridge_gap_vs_full"
                    ]
                )
            )
            bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_gaps.append(
                float(
                    cell_metrics[
                        "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_gap_vs_full"
                    ]
                )
            )
            lexicon_descriptor_prefilter_compressed_lm_bridge_gaps.append(
                float(
                    cell_metrics[
                        "lexicon_descriptor_prefilter_compressed_lm_bridge_gap_vs_full"
                    ]
                )
            )
            fewshot_descriptor_prefilter_compressed_lm_bridge_gaps.append(
                float(
                    cell_metrics[
                        "fewshot_descriptor_prefilter_compressed_lm_bridge_gap_vs_full"
                    ]
                )
            )
            forced_choice_descriptor_prefilter_compressed_lm_bridge_gaps.append(
                float(
                    cell_metrics[
                        "forced_choice_descriptor_prefilter_compressed_lm_bridge_gap_vs_full"
                    ]
                )
            )
            contrastive_descriptor_prefilter_compressed_lm_bridge_gaps.append(
                float(
                    cell_metrics[
                        "contrastive_descriptor_prefilter_compressed_lm_bridge_gap_vs_full"
                    ]
                )
            )
            compressed_lm_bridge_gaps.append(
                float(cell_metrics["compressed_lm_bridge_gap_vs_full"])
            )
            lm_bridge_gaps.append(float(cell_metrics["lm_bridge_gap_vs_full"]))
            prefilter_lm_bridge_rerank_fractions.append(
                float(cell_metrics["prefilter_lm_bridge_rerank_fraction"])
            )
            prefilter_lm_bridge_target_ranks.append(
                float(cell_metrics["prefilter_lm_bridge_mean_target_rank"])
            )
            prefilter_lm_bridge_target_survival_rates.append(
                float(cell_metrics["prefilter_lm_bridge_target_survival_rate"])
            )
            prefilter_compressed_lm_bridge_rerank_fractions.append(
                float(cell_metrics["prefilter_compressed_lm_bridge_rerank_fraction"])
            )
            prefilter_compressed_lm_bridge_target_ranks.append(
                float(cell_metrics["prefilter_compressed_lm_bridge_mean_target_rank"])
            )
            prefilter_compressed_lm_bridge_target_survival_rates.append(
                float(
                    cell_metrics[
                        "prefilter_compressed_lm_bridge_target_survival_rate"
                    ]
                )
            )
            margin_guarded_prefilter_compressed_lm_bridge_rerank_fractions.append(
                float(
                    cell_metrics[
                        "margin_guarded_prefilter_compressed_lm_bridge_rerank_fraction"
                    ]
                )
            )
            margin_guarded_prefilter_compressed_lm_bridge_target_ranks.append(
                float(
                    cell_metrics[
                        "margin_guarded_prefilter_compressed_lm_bridge_mean_target_rank"
                    ]
                )
            )
            margin_guarded_prefilter_compressed_lm_bridge_target_survival_rates.append(
                float(
                    cell_metrics[
                        "margin_guarded_prefilter_compressed_lm_bridge_target_survival_rate"
                    ]
                )
            )
            margin_guarded_prefilter_compressed_lm_bridge_override_rates.append(
                float(
                    cell_metrics[
                        "margin_guarded_prefilter_compressed_lm_bridge_override_rate"
                    ]
                )
            )
            reverse_descriptor_prefilter_compressed_lm_bridge_rerank_fractions.append(
                float(
                    cell_metrics[
                        "reverse_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"
                    ]
                )
            )
            reverse_descriptor_prefilter_compressed_lm_bridge_target_ranks.append(
                float(
                    cell_metrics[
                        "reverse_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"
                    ]
                )
            )
            reverse_descriptor_prefilter_compressed_lm_bridge_target_survival_rates.append(
                float(
                    cell_metrics[
                        "reverse_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"
                    ]
                )
            )
            bidirectional_descriptor_prefilter_compressed_lm_bridge_rerank_fractions.append(
                float(
                    cell_metrics[
                        "bidirectional_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"
                    ]
                )
            )
            bidirectional_descriptor_prefilter_compressed_lm_bridge_target_ranks.append(
                float(
                    cell_metrics[
                        "bidirectional_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"
                    ]
                )
            )
            bidirectional_descriptor_prefilter_compressed_lm_bridge_target_survival_rates.append(
                float(
                    cell_metrics[
                        "bidirectional_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"
                    ]
                )
            )
            bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_rerank_fractions.append(
                float(
                    cell_metrics[
                        "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_rerank_fraction"
                    ]
                )
            )
            bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_target_ranks.append(
                float(
                    cell_metrics[
                        "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_mean_target_rank"
                    ]
                )
            )
            bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_target_survival_rates.append(
                float(
                    cell_metrics[
                        "bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_target_survival_rate"
                    ]
                )
            )
            lexicon_descriptor_prefilter_compressed_lm_bridge_rerank_fractions.append(
                float(
                    cell_metrics[
                        "lexicon_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"
                    ]
                )
            )
            lexicon_descriptor_prefilter_compressed_lm_bridge_target_ranks.append(
                float(
                    cell_metrics[
                        "lexicon_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"
                    ]
                )
            )
            lexicon_descriptor_prefilter_compressed_lm_bridge_target_survival_rates.append(
                float(
                    cell_metrics[
                        "lexicon_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"
                    ]
                )
            )
            fewshot_descriptor_prefilter_compressed_lm_bridge_rerank_fractions.append(
                float(
                    cell_metrics[
                        "fewshot_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"
                    ]
                )
            )
            fewshot_descriptor_prefilter_compressed_lm_bridge_target_ranks.append(
                float(
                    cell_metrics[
                        "fewshot_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"
                    ]
                )
            )
            fewshot_descriptor_prefilter_compressed_lm_bridge_target_survival_rates.append(
                float(
                    cell_metrics[
                        "fewshot_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"
                    ]
                )
            )
            forced_choice_descriptor_prefilter_compressed_lm_bridge_rerank_fractions.append(
                float(
                    cell_metrics[
                        "forced_choice_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"
                    ]
                )
            )
            forced_choice_descriptor_prefilter_compressed_lm_bridge_target_ranks.append(
                float(
                    cell_metrics[
                        "forced_choice_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"
                    ]
                )
            )
            forced_choice_descriptor_prefilter_compressed_lm_bridge_target_survival_rates.append(
                float(
                    cell_metrics[
                        "forced_choice_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"
                    ]
                )
            )
            contrastive_descriptor_prefilter_compressed_lm_bridge_rerank_fractions.append(
                float(
                    cell_metrics[
                        "contrastive_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"
                    ]
                )
            )
            contrastive_descriptor_prefilter_compressed_lm_bridge_target_ranks.append(
                float(
                    cell_metrics[
                        "contrastive_descriptor_prefilter_compressed_lm_bridge_mean_target_rank"
                    ]
                )
            )
            contrastive_descriptor_prefilter_compressed_lm_bridge_target_survival_rates.append(
                float(
                    cell_metrics[
                        "contrastive_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"
                    ]
                )
            )

    metrics["mean_full_accuracy"] = float(np.mean(full_accuracies))
    metrics["mean_truncate_accuracy"] = float(np.mean(truncate_accuracies))
    metrics["mean_streaming_accuracy"] = float(np.mean(streaming_accuracies))
    metrics["mean_oracle_span_accuracy"] = float(np.mean(oracle_accuracies))
    metrics["mean_lexical_accuracy"] = float(np.mean(lexical_accuracies))
    metrics["mean_bridge_accuracy"] = float(np.mean(bridge_accuracies))
    metrics["mean_prose_bridge_accuracy"] = float(np.mean(prose_bridge_accuracies))
    metrics["mean_entity_bridge_accuracy"] = float(np.mean(entity_bridge_accuracies))
    metrics["mean_overlap_bridge_accuracy"] = float(
        np.mean(overlap_bridge_accuracies)
    )
    metrics["mean_id_bridge_accuracy"] = float(np.mean(id_bridge_accuracies))
    metrics["mean_char_bridge_accuracy"] = float(np.mean(char_bridge_accuracies))
    metrics["mean_neighbor_bridge_accuracy"] = float(
        np.mean(neighbor_bridge_accuracies)
    )
    metrics["mean_semantic_bridge_accuracy"] = float(
        np.mean(semantic_bridge_accuracies)
    )
    metrics["mean_embedding_bridge_accuracy"] = float(
        np.mean(embedding_bridge_accuracies)
    )
    metrics["mean_phrase_lm_bridge_accuracy"] = float(
        np.mean(phrase_lm_bridge_accuracies)
    )
    metrics["mean_prefilter_lm_bridge_accuracy"] = float(
        np.mean(prefilter_lm_bridge_accuracies)
    )
    metrics["mean_prefilter_compressed_lm_bridge_accuracy"] = float(
        np.mean(prefilter_compressed_lm_bridge_accuracies)
    )
    metrics["mean_margin_guarded_prefilter_compressed_lm_bridge_accuracy"] = float(
        np.mean(margin_guarded_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics["mean_reverse_descriptor_prefilter_compressed_lm_bridge_accuracy"] = float(
        np.mean(reverse_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics["mean_bidirectional_descriptor_prefilter_compressed_lm_bridge_accuracy"] = float(
        np.mean(bidirectional_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "mean_bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.mean(bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "mean_lexicon_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.mean(lexicon_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "mean_fewshot_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.mean(fewshot_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "mean_forced_choice_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.mean(forced_choice_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "mean_contrastive_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.mean(contrastive_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics["mean_compressed_lm_bridge_accuracy"] = float(
        np.mean(compressed_lm_bridge_accuracies)
    )
    metrics["mean_lm_bridge_accuracy"] = float(np.mean(lm_bridge_accuracies))
    metrics["mean_truncate_gap_vs_full"] = float(np.mean(truncate_gaps))
    metrics["mean_streaming_gap_vs_full"] = float(np.mean(streaming_gaps))
    metrics["mean_oracle_span_gap_vs_full"] = float(np.mean(oracle_gaps))
    metrics["mean_lexical_gap_vs_full"] = float(np.mean(lexical_gaps))
    metrics["mean_bridge_gap_vs_full"] = float(np.mean(bridge_gaps))
    metrics["mean_prose_bridge_gap_vs_full"] = float(np.mean(prose_bridge_gaps))
    metrics["mean_entity_bridge_gap_vs_full"] = float(np.mean(entity_bridge_gaps))
    metrics["mean_overlap_bridge_gap_vs_full"] = float(
        np.mean(overlap_bridge_gaps)
    )
    metrics["mean_id_bridge_gap_vs_full"] = float(np.mean(id_bridge_gaps))
    metrics["mean_char_bridge_gap_vs_full"] = float(np.mean(char_bridge_gaps))
    metrics["mean_neighbor_bridge_gap_vs_full"] = float(
        np.mean(neighbor_bridge_gaps)
    )
    metrics["mean_semantic_bridge_gap_vs_full"] = float(
        np.mean(semantic_bridge_gaps)
    )
    metrics["mean_embedding_bridge_gap_vs_full"] = float(
        np.mean(embedding_bridge_gaps)
    )
    metrics["mean_phrase_lm_bridge_gap_vs_full"] = float(
        np.mean(phrase_lm_bridge_gaps)
    )
    metrics["mean_prefilter_lm_bridge_gap_vs_full"] = float(
        np.mean(prefilter_lm_bridge_gaps)
    )
    metrics["mean_prefilter_compressed_lm_bridge_gap_vs_full"] = float(
        np.mean(prefilter_compressed_lm_bridge_gaps)
    )
    metrics["mean_margin_guarded_prefilter_compressed_lm_bridge_gap_vs_full"] = float(
        np.mean(margin_guarded_prefilter_compressed_lm_bridge_gaps)
    )
    metrics["mean_reverse_descriptor_prefilter_compressed_lm_bridge_gap_vs_full"] = float(
        np.mean(reverse_descriptor_prefilter_compressed_lm_bridge_gaps)
    )
    metrics[
        "mean_bidirectional_descriptor_prefilter_compressed_lm_bridge_gap_vs_full"
    ] = float(np.mean(bidirectional_descriptor_prefilter_compressed_lm_bridge_gaps))
    metrics[
        "mean_bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_gap_vs_full"
    ] = float(np.mean(bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_gaps))
    metrics[
        "mean_lexicon_descriptor_prefilter_compressed_lm_bridge_gap_vs_full"
    ] = float(np.mean(lexicon_descriptor_prefilter_compressed_lm_bridge_gaps))
    metrics[
        "mean_fewshot_descriptor_prefilter_compressed_lm_bridge_gap_vs_full"
    ] = float(np.mean(fewshot_descriptor_prefilter_compressed_lm_bridge_gaps))
    metrics[
        "mean_forced_choice_descriptor_prefilter_compressed_lm_bridge_gap_vs_full"
    ] = float(np.mean(forced_choice_descriptor_prefilter_compressed_lm_bridge_gaps))
    metrics[
        "mean_contrastive_descriptor_prefilter_compressed_lm_bridge_gap_vs_full"
    ] = float(np.mean(contrastive_descriptor_prefilter_compressed_lm_bridge_gaps))
    metrics["mean_compressed_lm_bridge_gap_vs_full"] = float(
        np.mean(compressed_lm_bridge_gaps)
    )
    metrics["mean_lm_bridge_gap_vs_full"] = float(np.mean(lm_bridge_gaps))
    metrics["mean_prefilter_lm_bridge_rerank_fraction"] = float(
        np.mean(prefilter_lm_bridge_rerank_fractions)
    )
    metrics["mean_prefilter_lm_bridge_target_rank"] = float(
        np.mean(prefilter_lm_bridge_target_ranks)
    )
    metrics["mean_prefilter_lm_bridge_target_survival_rate"] = float(
        np.mean(prefilter_lm_bridge_target_survival_rates)
    )
    metrics["mean_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
        np.mean(prefilter_compressed_lm_bridge_rerank_fractions)
    )
    metrics["mean_prefilter_compressed_lm_bridge_target_rank"] = float(
        np.mean(prefilter_compressed_lm_bridge_target_ranks)
    )
    metrics["mean_prefilter_compressed_lm_bridge_target_survival_rate"] = float(
        np.mean(prefilter_compressed_lm_bridge_target_survival_rates)
    )
    metrics["mean_margin_guarded_prefilter_compressed_lm_bridge_rerank_fraction"] = float(
        np.mean(margin_guarded_prefilter_compressed_lm_bridge_rerank_fractions)
    )
    metrics["mean_margin_guarded_prefilter_compressed_lm_bridge_target_rank"] = float(
        np.mean(margin_guarded_prefilter_compressed_lm_bridge_target_ranks)
    )
    metrics[
        "mean_margin_guarded_prefilter_compressed_lm_bridge_target_survival_rate"
    ] = float(
        np.mean(margin_guarded_prefilter_compressed_lm_bridge_target_survival_rates)
    )
    metrics["mean_margin_guarded_prefilter_compressed_lm_bridge_override_rate"] = float(
        np.mean(margin_guarded_prefilter_compressed_lm_bridge_override_rates)
    )
    metrics[
        "mean_reverse_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"
    ] = float(np.mean(reverse_descriptor_prefilter_compressed_lm_bridge_rerank_fractions))
    metrics["mean_reverse_descriptor_prefilter_compressed_lm_bridge_target_rank"] = float(
        np.mean(reverse_descriptor_prefilter_compressed_lm_bridge_target_ranks)
    )
    metrics[
        "mean_reverse_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"
    ] = float(
        np.mean(reverse_descriptor_prefilter_compressed_lm_bridge_target_survival_rates)
    )
    metrics[
        "mean_bidirectional_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"
    ] = float(
        np.mean(
            bidirectional_descriptor_prefilter_compressed_lm_bridge_rerank_fractions
        )
    )
    metrics[
        "mean_bidirectional_descriptor_prefilter_compressed_lm_bridge_target_rank"
    ] = float(np.mean(bidirectional_descriptor_prefilter_compressed_lm_bridge_target_ranks))
    metrics[
        "mean_bidirectional_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"
    ] = float(
        np.mean(
            bidirectional_descriptor_prefilter_compressed_lm_bridge_target_survival_rates
        )
    )
    metrics[
        "mean_bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_rerank_fraction"
    ] = float(
        np.mean(
            bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_rerank_fractions
        )
    )
    metrics[
        "mean_bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_target_rank"
    ] = float(
        np.mean(bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_target_ranks)
    )
    metrics[
        "mean_bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_target_survival_rate"
    ] = float(
        np.mean(
            bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_target_survival_rates
        )
    )
    metrics[
        "mean_lexicon_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"
    ] = float(
        np.mean(
            lexicon_descriptor_prefilter_compressed_lm_bridge_rerank_fractions
        )
    )
    metrics[
        "mean_lexicon_descriptor_prefilter_compressed_lm_bridge_target_rank"
    ] = float(
        np.mean(lexicon_descriptor_prefilter_compressed_lm_bridge_target_ranks)
    )
    metrics[
        "mean_lexicon_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"
    ] = float(
        np.mean(
            lexicon_descriptor_prefilter_compressed_lm_bridge_target_survival_rates
        )
    )
    metrics[
        "mean_fewshot_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"
    ] = float(
        np.mean(
            fewshot_descriptor_prefilter_compressed_lm_bridge_rerank_fractions
        )
    )
    metrics[
        "mean_fewshot_descriptor_prefilter_compressed_lm_bridge_target_rank"
    ] = float(
        np.mean(fewshot_descriptor_prefilter_compressed_lm_bridge_target_ranks)
    )
    metrics[
        "mean_fewshot_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"
    ] = float(
        np.mean(
            fewshot_descriptor_prefilter_compressed_lm_bridge_target_survival_rates
        )
    )
    metrics[
        "mean_forced_choice_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"
    ] = float(
        np.mean(
            forced_choice_descriptor_prefilter_compressed_lm_bridge_rerank_fractions
        )
    )
    metrics[
        "mean_forced_choice_descriptor_prefilter_compressed_lm_bridge_target_rank"
    ] = float(
        np.mean(forced_choice_descriptor_prefilter_compressed_lm_bridge_target_ranks)
    )
    metrics[
        "mean_forced_choice_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"
    ] = float(
        np.mean(
            forced_choice_descriptor_prefilter_compressed_lm_bridge_target_survival_rates
        )
    )
    metrics[
        "mean_contrastive_descriptor_prefilter_compressed_lm_bridge_rerank_fraction"
    ] = float(
        np.mean(
            contrastive_descriptor_prefilter_compressed_lm_bridge_rerank_fractions
        )
    )
    metrics[
        "mean_contrastive_descriptor_prefilter_compressed_lm_bridge_target_rank"
    ] = float(
        np.mean(contrastive_descriptor_prefilter_compressed_lm_bridge_target_ranks)
    )
    metrics[
        "mean_contrastive_descriptor_prefilter_compressed_lm_bridge_target_survival_rate"
    ] = float(
        np.mean(
            contrastive_descriptor_prefilter_compressed_lm_bridge_target_survival_rates
        )
    )
    for name, values in sorted(diagnostic_metric_values.items()):
        metrics[name] = float(np.mean(values))
    metrics["min_streaming_accuracy"] = float(np.min(streaming_accuracies))
    metrics["max_streaming_accuracy"] = float(np.max(streaming_accuracies))
    metrics["min_oracle_span_accuracy"] = float(np.min(oracle_accuracies))
    metrics["max_oracle_span_accuracy"] = float(np.max(oracle_accuracies))
    metrics["min_lexical_accuracy"] = float(np.min(lexical_accuracies))
    metrics["max_lexical_accuracy"] = float(np.max(lexical_accuracies))
    metrics["min_bridge_accuracy"] = float(np.min(bridge_accuracies))
    metrics["max_bridge_accuracy"] = float(np.max(bridge_accuracies))
    metrics["min_prose_bridge_accuracy"] = float(np.min(prose_bridge_accuracies))
    metrics["max_prose_bridge_accuracy"] = float(np.max(prose_bridge_accuracies))
    metrics["min_entity_bridge_accuracy"] = float(np.min(entity_bridge_accuracies))
    metrics["max_entity_bridge_accuracy"] = float(np.max(entity_bridge_accuracies))
    metrics["min_overlap_bridge_accuracy"] = float(np.min(overlap_bridge_accuracies))
    metrics["max_overlap_bridge_accuracy"] = float(np.max(overlap_bridge_accuracies))
    metrics["min_id_bridge_accuracy"] = float(np.min(id_bridge_accuracies))
    metrics["max_id_bridge_accuracy"] = float(np.max(id_bridge_accuracies))
    metrics["min_char_bridge_accuracy"] = float(np.min(char_bridge_accuracies))
    metrics["max_char_bridge_accuracy"] = float(np.max(char_bridge_accuracies))
    metrics["min_neighbor_bridge_accuracy"] = float(
        np.min(neighbor_bridge_accuracies)
    )
    metrics["max_neighbor_bridge_accuracy"] = float(
        np.max(neighbor_bridge_accuracies)
    )
    metrics["min_semantic_bridge_accuracy"] = float(
        np.min(semantic_bridge_accuracies)
    )
    metrics["max_semantic_bridge_accuracy"] = float(
        np.max(semantic_bridge_accuracies)
    )
    metrics["min_embedding_bridge_accuracy"] = float(
        np.min(embedding_bridge_accuracies)
    )
    metrics["max_embedding_bridge_accuracy"] = float(
        np.max(embedding_bridge_accuracies)
    )
    metrics["min_phrase_lm_bridge_accuracy"] = float(
        np.min(phrase_lm_bridge_accuracies)
    )
    metrics["max_phrase_lm_bridge_accuracy"] = float(
        np.max(phrase_lm_bridge_accuracies)
    )
    metrics["min_prefilter_lm_bridge_accuracy"] = float(
        np.min(prefilter_lm_bridge_accuracies)
    )
    metrics["max_prefilter_lm_bridge_accuracy"] = float(
        np.max(prefilter_lm_bridge_accuracies)
    )
    metrics["min_prefilter_compressed_lm_bridge_accuracy"] = float(
        np.min(prefilter_compressed_lm_bridge_accuracies)
    )
    metrics["max_prefilter_compressed_lm_bridge_accuracy"] = float(
        np.max(prefilter_compressed_lm_bridge_accuracies)
    )
    metrics["min_margin_guarded_prefilter_compressed_lm_bridge_accuracy"] = float(
        np.min(margin_guarded_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics["max_margin_guarded_prefilter_compressed_lm_bridge_accuracy"] = float(
        np.max(margin_guarded_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics["min_reverse_descriptor_prefilter_compressed_lm_bridge_accuracy"] = float(
        np.min(reverse_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics["max_reverse_descriptor_prefilter_compressed_lm_bridge_accuracy"] = float(
        np.max(reverse_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "min_bidirectional_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(np.min(bidirectional_descriptor_prefilter_compressed_lm_bridge_accuracies))
    metrics[
        "max_bidirectional_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(np.max(bidirectional_descriptor_prefilter_compressed_lm_bridge_accuracies))
    metrics[
        "min_bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.min(bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "max_bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.max(bidirectional_descriptor_qa_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "min_lexicon_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.min(lexicon_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "max_lexicon_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.max(lexicon_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "min_fewshot_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.min(fewshot_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "max_fewshot_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.max(fewshot_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "min_forced_choice_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.min(forced_choice_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "max_forced_choice_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.max(forced_choice_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "min_contrastive_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.min(contrastive_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics[
        "max_contrastive_descriptor_prefilter_compressed_lm_bridge_accuracy"
    ] = float(
        np.max(contrastive_descriptor_prefilter_compressed_lm_bridge_accuracies)
    )
    metrics["min_compressed_lm_bridge_accuracy"] = float(
        np.min(compressed_lm_bridge_accuracies)
    )
    metrics["max_compressed_lm_bridge_accuracy"] = float(
        np.max(compressed_lm_bridge_accuracies)
    )
    metrics["min_lm_bridge_accuracy"] = float(np.min(lm_bridge_accuracies))
    metrics["max_lm_bridge_accuracy"] = float(np.max(lm_bridge_accuracies))
    return metrics


def write_metrics(metrics: dict[str, float | int | str], metrics_path: str | None) -> None:
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if not metrics_path:
        return
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(rendered + "\n")


def main() -> None:
    args = parse_args()
    metrics = run_sweep(args)
    write_metrics(metrics, args.metrics_path)


if __name__ == "__main__":
    main()
