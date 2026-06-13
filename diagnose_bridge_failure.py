from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from eval_kv_frontier import (
    assigned_descriptor,
    average_continuation_logprob,
    build_task_items,
    compressed_bridge_prompt,
    load_model,
    query_and_credential_lines,
    rank_answer,
    seed_everything,
    vault_code,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose bridge memo selection and answer ranking per item."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--items", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="heldout_descriptor_bridge_v2")
    parser.add_argument("--target-position", default="middle")
    parser.add_argument("--prefilter-candidates", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--metrics-path", default=os.environ.get("AAD_METRICS_PATH"))
    return parser.parse_args()


@torch.inference_mode()
def ranked_answer_scores(tokenizer, model, device, token_ids, candidates):
    scored = []
    for candidate in candidates:
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if not candidate_ids:
            continue
        input_ids = torch.tensor(
            [token_ids + candidate_ids],
            dtype=torch.long,
            device=device,
        )
        outputs = model(input_ids=input_ids, use_cache=False)
        log_probs = torch.log_softmax(outputs.logits[0], dim=-1)
        total = 0.0
        for offset, token_id in enumerate(candidate_ids):
            total += float(log_probs[len(token_ids) - 1 + offset, token_id].item())
        scored.append(
            {
                "candidate": candidate,
                "avg_logprob": total / len(candidate_ids),
                "total_logprob": total,
            }
        )
    scored.sort(key=lambda row: row["avg_logprob"], reverse=True)
    return scored


def select_lm_bridge(tokenizer, model, device, item):
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    prefix = (
        "Match the personnel memo to the credential memo.\n"
        f"Personnel memo: {query_line}\n"
        "Credential memo:"
    )
    scored = []
    for line_idx, line in enumerate(candidate_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            prefix,
            " " + line,
        )
        scored.append((score, -line_idx, line))
    scored.sort(reverse=True)
    selected = scored[0][2] if scored else ""
    target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(scored, start=1):
        if line == item.target_record:
            target_rank = rank
            break
    prompt = (
        "You are given records. Answer with only the three digit value.\n"
        + "\n".join([query_line, selected])
        + "\nQuestion:"
        + question
    )
    return {
        "query_line": query_line,
        "selected_credential": selected,
        "selected_code": vault_code(selected),
        "target_rank": target_rank,
        "target_selected": selected == item.target_record,
        "top3_credentials": [
            {"rank": rank, "score": score, "code": vault_code(line), "line": line}
            for rank, (score, _line_idx, line) in enumerate(scored[:3], start=1)
        ],
        "prompt": prompt,
        "token_ids": tokenizer.encode(prompt, add_special_tokens=False),
    }


def select_prefilter_compressed(tokenizer, model, device, item, prefilter_candidates):
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    phrase_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    phrase_scored = []
    from eval_kv_frontier import credential_descriptor

    for line_idx, line in enumerate(candidate_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            phrase_prefix,
            " " + credential_descriptor(line),
        )
        phrase_scored.append((score, -line_idx, line))
    phrase_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(phrase_scored)))
    rerank_lines = [line for _score, _line_idx, line in phrase_scored[:rerank_count]]
    prefilter_target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(phrase_scored, start=1):
        if line == item.target_record:
            prefilter_target_rank = rank
            break

    full_prefix = (
        "Match the personnel memo to the credential memo.\n"
        f"Personnel memo: {query_line}\n"
        "Credential memo:"
    )
    full_scored = []
    for line_idx, line in enumerate(rerank_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            full_prefix,
            " " + line,
        )
        full_scored.append((score, -line_idx, line))
    full_scored.sort(reverse=True)
    selected = full_scored[0][2] if full_scored else ""
    rerank_target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(full_scored, start=1):
        if line == item.target_record:
            rerank_target_rank = rank
            break
    prompt = compressed_bridge_prompt(item, question, query_line, selected)
    return {
        "query_line": query_line,
        "selected_credential": selected,
        "selected_code": vault_code(selected),
        "prefilter_target_rank": prefilter_target_rank,
        "prefilter_target_survived": 0 < prefilter_target_rank <= rerank_count,
        "rerank_target_rank": rerank_target_rank,
        "target_selected": selected == item.target_record,
        "top3_prefilter": [
            {"rank": rank, "score": score, "code": vault_code(line), "line": line}
            for rank, (score, _line_idx, line) in enumerate(phrase_scored[:3], start=1)
        ],
        "top3_rerank": [
            {"rank": rank, "score": score, "code": vault_code(line), "line": line}
            for rank, (score, _line_idx, line) in enumerate(full_scored[:3], start=1)
        ],
        "prompt": prompt,
        "token_ids": tokenizer.encode(prompt, add_special_tokens=False),
    }


def select_descriptor_conditioned_prefilter(
    tokenizer,
    model,
    device,
    item,
    prefilter_candidates,
):
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    from eval_kv_frontier import credential_descriptor

    phrase_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    phrase_scored = []
    for line_idx, line in enumerate(candidate_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            phrase_prefix,
            " " + credential_descriptor(line),
        )
        phrase_scored.append((score, -line_idx, line))
    phrase_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(phrase_scored)))
    rerank_lines = [line for _score, _line_idx, line in phrase_scored[:rerank_count]]

    full_prefix = (
        "Select the credential memo for this personnel descriptor.\n"
        f"Personnel descriptor: {assigned_descriptor(query_line)}\n"
        "Credential memo:"
    )
    full_scored = []
    for line_idx, line in enumerate(rerank_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            full_prefix,
            " " + line,
        )
        full_scored.append((score, -line_idx, line))
    full_scored.sort(reverse=True)
    selected = full_scored[0][2] if full_scored else ""
    rerank_target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(full_scored, start=1):
        if line == item.target_record:
            rerank_target_rank = rank
            break
    prompt = compressed_bridge_prompt(item, question, query_line, selected)
    return {
        "selected_credential": selected,
        "selected_code": vault_code(selected),
        "rerank_target_rank": rerank_target_rank,
        "target_selected": selected == item.target_record,
        "top3_rerank": [
            {"rank": rank, "score": score, "code": vault_code(line), "line": line}
            for rank, (score, _line_idx, line) in enumerate(full_scored[:3], start=1)
        ],
        "token_ids": tokenizer.encode(prompt, add_special_tokens=False),
    }


def select_canonical_prefilter(tokenizer, model, device, item, prefilter_candidates):
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    from eval_kv_frontier import credential_descriptor

    phrase_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    phrase_scored = []
    for line_idx, line in enumerate(candidate_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            phrase_prefix,
            " " + credential_descriptor(line),
        )
        phrase_scored.append((score, -line_idx, line))
    phrase_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(phrase_scored)))
    rerank_lines = [line for _score, _line_idx, line in phrase_scored[:rerank_count]]

    full_prefix = (
        "Match the personnel memo to the normalized credential fact.\n"
        f"Personnel memo: {query_line}\n"
        "Credential fact:"
    )
    full_scored = []
    for line_idx, line in enumerate(rerank_lines):
        descriptor = credential_descriptor(line)
        code = vault_code(line)
        canonical = f" The {descriptor} has vault code {code}."
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            full_prefix,
            canonical,
        )
        full_scored.append((score, -line_idx, line, canonical.strip()))
    full_scored.sort(reverse=True)
    selected = full_scored[0][2] if full_scored else ""
    rerank_target_rank = 0
    for rank, (_score, _line_idx, line, _canonical) in enumerate(
        full_scored,
        start=1,
    ):
        if line == item.target_record:
            rerank_target_rank = rank
            break
    prompt = compressed_bridge_prompt(item, question, query_line, selected)
    return {
        "selected_credential": selected,
        "selected_code": vault_code(selected),
        "rerank_target_rank": rerank_target_rank,
        "target_selected": selected == item.target_record,
        "top3_rerank": [
            {
                "rank": rank,
                "score": score,
                "code": vault_code(line),
                "line": line,
                "canonical": canonical,
            }
            for rank, (score, _line_idx, line, canonical) in enumerate(
                full_scored[:3],
                start=1,
            )
        ],
        "token_ids": tokenizer.encode(prompt, add_special_tokens=False),
    }


def select_reverse_descriptor_prefilter(
    tokenizer,
    model,
    device,
    item,
    prefilter_candidates,
):
    question, _record_lines, query_line, candidate_lines = query_and_credential_lines(item)
    from eval_kv_frontier import credential_descriptor

    phrase_prefix = (
        "Match the personnel memo to the credential descriptor.\n"
        f"Personnel memo: {query_line}\n"
        "Credential descriptor:"
    )
    phrase_scored = []
    for line_idx, line in enumerate(candidate_lines):
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            phrase_prefix,
            " " + credential_descriptor(line),
        )
        phrase_scored.append((score, -line_idx, line))
    phrase_scored.sort(reverse=True)
    rerank_count = max(1, min(prefilter_candidates, len(phrase_scored)))
    rerank_lines = [line for _score, _line_idx, line in phrase_scored[:rerank_count]]

    source_descriptor = assigned_descriptor(query_line)
    full_scored = []
    for line_idx, line in enumerate(rerank_lines):
        full_prefix = (
            "Infer the personnel descriptor explained by this credential memo.\n"
            f"Credential memo: {line}\n"
            "Personnel descriptor:"
        )
        score = average_continuation_logprob(
            tokenizer,
            model,
            device,
            full_prefix,
            " " + source_descriptor,
        )
        full_scored.append((score, -line_idx, line))
    full_scored.sort(reverse=True)
    selected = full_scored[0][2] if full_scored else ""
    rerank_target_rank = 0
    for rank, (_score, _line_idx, line) in enumerate(full_scored, start=1):
        if line == item.target_record:
            rerank_target_rank = rank
            break
    prompt = compressed_bridge_prompt(item, question, query_line, selected)
    return {
        "selected_credential": selected,
        "selected_code": vault_code(selected),
        "rerank_target_rank": rerank_target_rank,
        "target_selected": selected == item.target_record,
        "top3_rerank": [
            {"rank": rank, "score": score, "code": vault_code(line), "line": line}
            for rank, (score, _line_idx, line) in enumerate(full_scored[:3], start=1)
        ],
        "token_ids": tokenizer.encode(prompt, add_special_tokens=False),
    }


def select_margin_guarded_canonical_prefilter(
    tokenizer,
    item,
    prefilter,
    canonical_prefilter,
    margin_threshold: float = 0.05,
):
    question, _record_lines, query_line, _candidate_lines = query_and_credential_lines(item)
    rerank_top = prefilter["top3_rerank"]
    full_margin = None
    selected = prefilter["selected_credential"]
    override = False
    if len(rerank_top) >= 2:
        full_margin = rerank_top[0]["score"] - rerank_top[1]["score"]
        canonical_selected = canonical_prefilter["selected_credential"]
        full_runner_up = rerank_top[1]["line"]
        override = (
            full_margin <= margin_threshold
            and canonical_selected == full_runner_up
        )
        if override:
            selected = canonical_selected
    prompt = compressed_bridge_prompt(item, question, query_line, selected)
    return {
        "selected_credential": selected,
        "selected_code": vault_code(selected),
        "target_selected": selected == item.target_record,
        "override": override,
        "full_margin": full_margin,
        "margin_threshold": margin_threshold,
        "canonical_selected_code": canonical_prefilter["selected_code"],
        "full_selected_code": prefilter["selected_code"],
        "full_runner_up_code": rerank_top[1]["code"] if len(rerank_top) >= 2 else "",
        "token_ids": tokenizer.encode(prompt, add_special_tokens=False),
    }


def oracle_prompt(item):
    question = item.prompt.rsplit("\nQuestion:", maxsplit=1)[1]
    return (
        "You are given records. Answer with only the three digit value.\n"
        + "\n".join(item.oracle_records)
        + "\nQuestion:"
        + question
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    tokenizer, model, device = load_model(args)
    items = build_task_items(
        args.task,
        args.items,
        args.distractors,
        args.seed,
        args.target_position,
    )

    rows = []
    for item_idx, item in enumerate(items):
        lm = select_lm_bridge(tokenizer, model, device, item)
        prefilter = select_prefilter_compressed(
            tokenizer,
            model,
            device,
            item,
            args.prefilter_candidates,
        )
        descriptor_prefilter = select_descriptor_conditioned_prefilter(
            tokenizer,
            model,
            device,
            item,
            args.prefilter_candidates,
        )
        canonical_prefilter = select_canonical_prefilter(
            tokenizer,
            model,
            device,
            item,
            args.prefilter_candidates,
        )
        reverse_descriptor_prefilter = select_reverse_descriptor_prefilter(
            tokenizer,
            model,
            device,
            item,
            args.prefilter_candidates,
        )
        guarded_prefilter = select_margin_guarded_canonical_prefilter(
            tokenizer,
            item,
            prefilter,
            canonical_prefilter,
        )
        oracle_ids = tokenizer.encode(oracle_prompt(item), add_special_tokens=False)
        oracle_prediction = rank_answer(
            tokenizer,
            model,
            device,
            oracle_ids,
            item.candidates,
        )
        lm_prediction = rank_answer(
            tokenizer,
            model,
            device,
            lm["token_ids"],
            item.candidates,
        )
        prefilter_prediction = rank_answer(
            tokenizer,
            model,
            device,
            prefilter["token_ids"],
            item.candidates,
        )
        descriptor_prefilter_prediction = rank_answer(
            tokenizer,
            model,
            device,
            descriptor_prefilter["token_ids"],
            item.candidates,
        )
        canonical_prefilter_prediction = rank_answer(
            tokenizer,
            model,
            device,
            canonical_prefilter["token_ids"],
            item.candidates,
        )
        reverse_descriptor_prefilter_prediction = rank_answer(
            tokenizer,
            model,
            device,
            reverse_descriptor_prefilter["token_ids"],
            item.candidates,
        )
        guarded_prefilter_prediction = rank_answer(
            tokenizer,
            model,
            device,
            guarded_prefilter["token_ids"],
            item.candidates,
        )
        row = {
            "item_idx": item_idx,
            "answer": item.answer,
            "target_record": item.target_record,
            "lm_bridge": {
                key: value for key, value in lm.items() if key not in {"token_ids", "prompt"}
            },
            "prefilter_compressed_lm_bridge": {
                key: value
                for key, value in prefilter.items()
                if key not in {"token_ids", "prompt"}
            },
            "rank_predictions": {
                "oracle_span": oracle_prediction,
                "lm_bridge": lm_prediction,
                "prefilter_compressed_lm_bridge": prefilter_prediction,
                "descriptor_conditioned_prefilter": descriptor_prefilter_prediction,
                "canonical_prefilter": canonical_prefilter_prediction,
                "reverse_descriptor_prefilter": reverse_descriptor_prefilter_prediction,
                "margin_guarded_canonical_prefilter": guarded_prefilter_prediction,
            },
            "oracle_correct": oracle_prediction == item.answer,
            "lm_bridge_correct": lm_prediction == item.answer,
            "prefilter_compressed_lm_bridge_correct": prefilter_prediction == item.answer,
            "descriptor_conditioned_prefilter": {
                key: value
                for key, value in descriptor_prefilter.items()
                if key != "token_ids"
            },
            "descriptor_conditioned_prefilter_correct": (
                descriptor_prefilter_prediction == item.answer
            ),
            "canonical_prefilter": {
                key: value for key, value in canonical_prefilter.items() if key != "token_ids"
            },
            "canonical_prefilter_correct": canonical_prefilter_prediction == item.answer,
            "reverse_descriptor_prefilter": {
                key: value
                for key, value in reverse_descriptor_prefilter.items()
                if key != "token_ids"
            },
            "reverse_descriptor_prefilter_correct": (
                reverse_descriptor_prefilter_prediction == item.answer
            ),
            "margin_guarded_canonical_prefilter": {
                key: value for key, value in guarded_prefilter.items() if key != "token_ids"
            },
            "margin_guarded_canonical_prefilter_correct": (
                guarded_prefilter_prediction == item.answer
            ),
            "prefilter_answer_top5": ranked_answer_scores(
                tokenizer,
                model,
                device,
                prefilter["token_ids"],
                item.candidates,
            )[:5],
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)

    metrics = {
        "eval_completed": 1,
        "items": args.items,
        "diagnostic_failures": sum(
            not row["prefilter_compressed_lm_bridge_correct"] for row in rows
        ),
        "target_selection_failures": sum(
            not row["prefilter_compressed_lm_bridge"]["target_selected"]
            for row in rows
        ),
        "answer_ranking_failures_with_target_selected": sum(
            row["prefilter_compressed_lm_bridge"]["target_selected"]
            and not row["prefilter_compressed_lm_bridge_correct"]
            for row in rows
        ),
        "oracle_failures": sum(not row["oracle_correct"] for row in rows),
        "descriptor_conditioned_prefilter_failures": sum(
            not row["descriptor_conditioned_prefilter_correct"] for row in rows
        ),
        "descriptor_conditioned_target_selection_failures": sum(
            not row["descriptor_conditioned_prefilter"]["target_selected"]
            for row in rows
        ),
        "canonical_prefilter_failures": sum(
            not row["canonical_prefilter_correct"] for row in rows
        ),
        "canonical_target_selection_failures": sum(
            not row["canonical_prefilter"]["target_selected"] for row in rows
        ),
        "reverse_descriptor_prefilter_failures": sum(
            not row["reverse_descriptor_prefilter_correct"] for row in rows
        ),
        "reverse_descriptor_target_selection_failures": sum(
            not row["reverse_descriptor_prefilter"]["target_selected"] for row in rows
        ),
        "margin_guarded_canonical_prefilter_failures": sum(
            not row["margin_guarded_canonical_prefilter_correct"] for row in rows
        ),
        "margin_guarded_canonical_target_selection_failures": sum(
            not row["margin_guarded_canonical_prefilter"]["target_selected"]
            for row in rows
        ),
        "margin_guarded_canonical_overrides": sum(
            row["margin_guarded_canonical_prefilter"]["override"] for row in rows
        ),
    }
    if args.metrics_path:
        Path(args.metrics_path).write_text(json.dumps(metrics, indent=2) + "\n")
    else:
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
