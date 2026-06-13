# KV Frontier Ship Summary

This repository is ready to publish as a reproducible small-LM KV-retention
curves artifact. It should not be described as a production KV-cache
implementation or as a speed result. The contribution is a set of registered
and independently re-executed quality-vs-retained-token measurements for
synthetic long-context retrieval tasks under fixed model/backend conditions.

## Scope

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Backend: CPU, `float32`
- Scoring: candidate-rank scoring, no wall-clock claims
- Harness: all claim-bearing numbers come from `aad exp run`; cited claims
  were pre-registered and verified by `aad verify`
- Main implementation files: `eval_kv_frontier.py`, `eval_kv_sweep.py`
- Detailed evidence index: `docs/bidirectional_descriptor_prefilter.md`

## What The Artifact Shows

1. Position matters for naive KV retention.

   On the `kv` task with 64 distractor records, full context is perfect across
   prefix, middle, and recent answer positions, while sink+recent succeeds only
   when the answer record is actually retained. The verified sweep reports
   `mean_streaming_gap_vs_full = 0.3333333333`.

   Evidence: run `r-35a9922e7`, protocol `p-b25c0ce1`, workspace commit
   `e86bf80`.

2. A compact bidirectional descriptor prefilter improves the retained-token
   frontier on held-out descriptor bridge tasks.

   On `heldout_descriptor_bridge_v2` with 33 code-bearing candidates, the
   bidirectional descriptor top-12 policy reaches verified `1.0` accuracy and
   `1.0` target survival on seeds 0, 1, and 2 while retaining about `3.2%` of
   prompt tokens. It repairs the reverse-only top-12 seed-2 recall failure
   without widening final reranking from 12 candidates.

   Evidence: seed 0 `r-404f1edbb6` with protocols `p-c4d6296b`,
   `p-25f841f7`, `p-89d7542e`, `p-7d28b086`; seed 1 `r-b5c34ce7f4` with
   `p-5c751546`, `p-80bfe4b7`, `p-3f7c0cd2`, `p-c22d916c`; seed 2
   `r-15baa35dd6` with `p-6f06634c`, `p-8f545228`, `p-8bceaff5`,
   `p-25a4b9b0`.

3. The same compact policy survives candidate-count scaling in this synthetic
   family.

   With 65 code-bearing candidates and the same top-12 final rerank set, seeds
   0, 1, and 2 each verify `1.0` accuracy, `1.0` target survival, and
   rerank fraction `12/65 = 0.18461538461538463`.

   Evidence: seed 0 `r-474584d695`, seed 1 `r-5c00c85e7b`, seed 2
   `r-18347fbce5`; detailed protocols are listed in
   `docs/bidirectional_descriptor_prefilter.md`.

4. Hard negatives expose the boundary: first-stage recall is not enough.

   On `heldout_descriptor_bridge_hard_negative`, the generic bidirectional
   top-12 selector can keep the true target while still selecting the wrong
   credential. Seed 0 verifies accuracy `0.0` with target survival `1.0`;
   seed 1 independently verifies the same `0.0` accuracy pattern.

   Evidence: seed 0 `r-1442712669`, protocols `p-1e30ba10` and
   `p-c20301bc`; seed 1 `r-c65d8ad073`, protocol `p-5e03b4db`.

5. Oracle and lexicon controls show the compressed candidate set contains the
   answer, but the final selector needs a real target-side translation signal.

   In-context induced mapping reaches verified `1.0` accuracy on hard-negative
   seeds 0, 1, and 2 at about `3.2%` retained tokens, but later diagnostics
   show that less aligned cells can inherit first-stage ordering rather than
   producing an independent margin. A task-external support map reaches a
   verified `0.75` accuracy on seeds 0, 1, and 2, and a verified diagnostic
   localizes the remaining miss to sparse source-token coverage and final-score
   ties.

   Evidence: in-context induced-map runs `r-72068ca7cc`, `r-fdabeeaa73`,
   `r-22327fbebf`; external-support runs `r-0a687a50dd`, `r-8a7991ace4`,
   `r-c888f9e291`; external-support diagnostic run `r-2705edb2c6` with
   protocols `p-762b0212`, `p-fa1083cd`, and `p-e7e5263d`.

## Non-Claims

- No speedup claim: CPU wall-clock measurements are intentionally not used.
- No production KV-cache implementation claim: the policies operate over
  retained prompt spans and candidate reranking, not integrated model-server
  KV memory.
- No broad model generality claim: the verified evidence is scoped to the
  listed Qwen0.5B CPU/float32 cells.
- No unrestricted hard-negative win: the robust public result includes the
  failure boundary and the diagnostic evidence that synthetic support maps are
  sparse tie-breaking aids unless they produce target-side coverage.

## Suggested Public Positioning

Publish as a reproducible curves repository and short writeup:

> We map quality-vs-retained-token frontiers for small-LM long-context
> retrieval probes. A bidirectional descriptor prefilter preserves accuracy at
> roughly 1.6-3.2% retained tokens on held-out descriptor bridge tasks, but
> hard-negative recombinations reveal a final-selection failure even when the
> target survives top-k retention. The repo includes verified AAD run/protocol
> evidence and scripts to reproduce the cells.

The most useful audience is maintainers or researchers comparing KV-retention,
prompt-compression, and retrieval-assisted long-context inference methods for
small local models.
