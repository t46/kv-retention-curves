# Bidirectional descriptor prefilter comparison

This note packages the verified `heldout_descriptor_bridge_v2` comparison and
candidate-count stress for the current best compact prompt policy. It is an
evidence index and method summary, not a new claim beyond the registered AAD
protocols.

## Fixed evaluation cell

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Backend: CPU, `float32`
- Scoring: candidate rank scoring
- Task: `heldout_descriptor_bridge_v2`
- Seeds: `0`, `1`, `2`
- Prompt shape: `items=4`, `distractors=32`, `target-positions=middle`,
  `budgets=192`, `sink-tokens=64`, `lexical-records=1`
- Workspace commits: reverse-only runs at `8536129`, bidirectional runs at
  `df47d06`

For shuffled descriptor tasks, `middle` is the query-alias placement label.
The answer-bearing profile is deliberately separated from that query line; use
the emitted query-position, target-position, and query-target-distance metrics
when describing placement.

## Verified comparison

| policy | final rerank candidates | v2 seeds with verified accuracy 1.0 | v2 seed-2 target survival | max target prefilter rank | retained fraction range | evidence |
|---|---:|---:|---:|---:|---:|---|
| reverse descriptor top-12 | 12/33 | 2/3 | 0.75 | 13 | 0.0321-0.0330 | seed-2 failure: `r-00d6c56dcc`, `p-2bf67a7b`, `p-8e047c97`, `p-e5d67504` |
| reverse descriptor top-16 | 16/33 | 3/3 | 1.0 | 13 | 0.0321-0.0321 | seed-2 repair: `r-2ecc7128df`, `p-e6dd5b49`, `p-6a01b685`, `p-24b4dfdb`, `p-05be4d5e`; seed-0/1 accuracy: `p-1cb7ecdb`, `p-871fd3b7` |
| bidirectional descriptor top-12 | 12/33 | 3/3 | 1.0 | 4 | 0.0321-0.0321 | seed 0: `r-404f1edbb6`, `p-c4d6296b`, `p-25f841f7`, `p-89d7542e`, `p-7d28b086`; seed 1: `r-b5c34ce7f4`, `p-5c751546`, `p-80bfe4b7`, `p-3f7c0cd2`, `p-c22d916c`; seed 2: `r-15baa35dd6`, `p-6f06634c`, `p-8f545228`, `p-8bceaff5`, `p-25a4b9b0` |

The same bidirectional top-12 policy also transfers to the original
`heldout_descriptor_bridge` task on seeds `0` and `1` with verified accuracy,
retained fraction, rerank fraction, and target survival:

| task | seeds | final rerank candidates | accuracy | target survival | retained fraction | evidence |
|---|---:|---:|---:|---:|---:|---|
| `heldout_descriptor_bridge` | 0, 1 | 12/33 | 1.0 | 1.0 | 0.031269027386879915 | seed 0: `r-170475a7e7`, `p-5d9dd7dc`, `p-536c5154`, `p-e837d2d0`, `p-417f310f`; seed 1: `r-564a1d09a9`, `p-23e1ad3e`, `p-3ab02597`, `p-e0b801e4`, `p-aa776e21` |

## Candidate-count stress

The same top-12 bidirectional policy was stress-tested on
`heldout_descriptor_bridge_v2` with `distractors=48`, increasing the
code-bearing candidate set from 33 to 49 while keeping the same model,
backend, scoring mode, budget, and `items=4` cell. The final rerank set stays
fixed at 12 candidates, so the verified rerank fraction drops from `12/33` to
`12/49 = 0.24489795918367346`.

| seed | run | accuracy | target survival | rerank fraction | retained fraction | max target prefilter rank | evidence |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | `r-0dcab51d75` | 1.0 | 1.0 | 0.24489795918367346 | 0.021443213302754026 | 3.0 | `p-4d8157fa`, `p-44b8e4a4`, `p-f8b015c4`, `p-5b113801`, `p-1377345f` |
| 1 | `r-156744c703` | 1.0 | 1.0 | 0.24489795918367346 | 0.021408942493070012 | observed 3.0 | `p-c7ff58a0`, `p-807a6572`, `p-bd124d4a`, `p-c8658292`; max-rank protocol `p-309f23d6` registered the wrong metric name |
| 2 | `r-ba4099d412` | 1.0 | 1.0 | 0.24489795918367346 | 0.021464931206865313 | 3.0 | `p-03e2f347`, `p-11215c3a`, `p-b2310b76`, `p-897fc71f`, `p-936e27a9` |

Same-run contrasts at `distractors=48`: reverse-only descriptor top-12 reached
only `0.5` accuracy and target survival `0.5` on seeds `1` and `2`; the older
unidirectional prefilter/compressed policies reached `0.25` on seed `1` and
`0.5` on seed `2`; oracle span remained `1.0`. Seed `0` also verified
placement diagnostics, with mean query position fraction
`0.5051546391752577` and mean target-query distance fraction
`0.32989690721649484`, but the seed `1` and `2` placement diagnostics have
not been independently verified.

## Larger candidate-count probe

The `distractors=64` probe increased the code-bearing candidate set to 65
while keeping the same top-12 final rerank set. Seeds `0`, `1`, and `2` have
independently verified accuracy and support metrics.

| seed | run | accuracy | target survival | rerank fraction | retained fraction | max target prefilter rank | evidence |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | `r-474584d695` | 1.0 | 1.0 | 0.18461538461538463 | 0.01607996509536399 | 5.0 | `p-0fb78fe3`, `p-fd827f77`, `p-46dddad6`, `p-5c3d8dff`, `p-6cca159b` |
| 1 | `r-5c00c85e7b` | 1.0 | 1.0 | 0.18461538461538463 | 0.016048178993118396 | 4.0 | `p-8d0cee59`, `p-1a3ad6f8`, `p-86432637`, `p-f1988d4a`, `p-3d0896f1` |
| 2 | `r-18347fbce5` | 1.0 | 1.0 | 0.18461538461538463 | 0.016050615888301326 | 5.0 | `p-729ff01c`, `p-2d800e42`, `p-f9292f84`, `p-9a80b941`, `p-62ef63e5` |

Same-run contrasts at `distractors=64`: reverse-only descriptor top-12,
prefilter LM bridge, prefilter compressed LM bridge, and margin-guarded
prefilter compressed LM bridge each reached `0.5` accuracy and `0.5` target
survival on seeds `0`, `1`, and `2`; oracle span remained `1.0`. The
bidirectional policy's final rerank fraction was `12/65 =
0.18461538461538463`.

## Hard-Negative Stress

The next implemented task variant,
`heldout_descriptor_bridge_hard_negative`, keeps the held-out v2 shuffled
layout and varied memo templates, but injects near-miss distractor descriptors
that recombine words from the true source and target descriptors. It is a
different stress axis from candidate-count scaling: it probes whether the
bidirectional descriptor prefilter can separate semantically close credentials
instead of merely finding the right descriptor among unrelated distractors.

Seed 0 has produced a verified failure at commit `bdf8752`. T1 run
`r-1442712669` verified
`mean_bidirectional_descriptor_prefilter_compressed_lm_bridge_accuracy = 0.0`
under protocol `p-1e30ba10`, while target survival independently verified at
`1.0` under protocol `p-c20301bc`. Same-run observed support metrics:
rerank fraction `12/33 = 0.36363636363636365`, retained fraction
`0.031885800746407504`, and max target prefilter rank `6.0`. Those support
metrics are registered (`p-59e7f2f6`, `p-a88bef0b`, `p-9e838250`) but not yet
independently verified.

Interpretation: the hard-negative failure is not caused by top-12 prefilter
recall. The true credential survives, but the compressed reverse final
selection chooses near-miss credentials. The next method work should focus on
the final reranker or compressed prompt rather than widening the first-stage
candidate set.

The same hard-negative baseline has now been extended to seed 1 at workspace
commit `d062956`. Protocols were registered before clean T1 run
`r-c65d8ad073`: bidirectional selector accuracy `p-5e03b4db`, bidirectional
target survival `p-fce1b47f`, lexicon upper-bound selector accuracy
`p-ae179fc9`, and minimized lexicon kept fraction `p-2b756065`. The
bidirectional accuracy protocol independently reproduced `0.0` with no issues
or red flags. Same-run observed metrics:

- bidirectional selector accuracy `0.0`;
- target survival `1.0`;
- rerank fraction `12/33 = 0.36363636363636365`;
- bidirectional kept fraction `0.03181624546052624`;
- mean first-stage target rank `3.75`;
- lexicon upper-bound accuracy `1.0`;
- lexicon kept fraction `0.032127967249218514`.

The support/upper-bound protocols for this seed are registered but not yet
independently verified. As an observed same-run contrast, QA, compressed-LM,
and LM reranking reached only `0.25`, while forced-choice, few-shot,
contrastive, pairwise tournament, source-copy guarded, reverse descriptor, and
the generic bidirectional selector were `0.0`. This makes the generic
bidirectional hard-negative collapse no longer a seed-0-only observation.

A first attempted repair at commit `f4eb1c9` adds
`bidirectional_descriptor_qa_prefilter_compressed_lm_bridge`: the same
bidirectional top-12 first stage, followed by a task-conditioned final rerank
that scores each retained credential's own code as the answer under the
compressed query+credential prompt. T1 run `r-ecd473b719` observed only a
partial repair: accuracy rose from `0.0` to `0.25`, matching the older
`lm_bridge`/`prefilter_compressed_lm_bridge` result but still far below
oracle span `1.0`. It retained target survival `1.0`, reranked `12/33`
candidates, and kept fraction `0.032822914272581015`. Protocols were
registered before the run (`p-7da3dd6f`, `p-2905f545`, `p-b9f90448`,
`p-52fcd639`), but this partial result is not yet independently verified.
Per-item stderr shows the QA reranker fixed item 4, while the older
full-memo prefilter/compressed path already fixed item 2, so the next
diagnostic should inspect final-selection scores rather than treating this
as a solved reranker.

Commit `c6b5d9c` adds opt-in final-score instrumentation
(`--diagnose-final-scores`). Clean T1 diagnostic run `r-1ca21e1cc9` on the
same hard-negative seed-0 cell reproduced the policy metrics above and emitted
candidate-level traces for the bidirectional top-12 set. Observed diagnostic
summary: first-stage target survival `1.0`, mean first-stage target rank
`4.0`, reverse-final selection accuracy `0.0`, full-memo-final selection
accuracy `0.25`, and QA-answer-final selection accuracy `0.25`. Per-item
traces show a consistent near-miss pattern: reverse final scoring prefers
source-word hybrids such as `school nutrition scheduler`, full-memo scoring
prefers target-domain/source-role hybrids such as `cafeteria meal scheduler`,
and the QA-answer scorer only selects the exact target descriptor on item 4.
Registered diagnostic protocols: reverse-final accuracy `p-02838559`,
full-memo-final accuracy `p-0d883d1f`, QA-answer-final accuracy
`p-3111748f`; none are independently verified yet.

Commit `b31cdde` tested the obvious follow-up,
`contrastive_descriptor_prefilter_compressed_lm_bridge`: the same
bidirectional descriptor top-12 first stage, followed by a yes/no
descriptor-pair verifier over the query source descriptor and each retained
credential descriptor. The claim-bearing protocols were registered before T1
run `r-8eda7d862b`: accuracy `p-fa2142e9`, target survival `p-ec1b5cf8`,
rerank fraction `p-624e1609`, and kept fraction `p-0a743068`. Observed
accuracy was `0.0`, while target survival remained `1.0`, rerank fraction
remained `12/33 = 0.36363636363636365`, and kept fraction was
`0.032979555876591043`. This is a clean negative result but is not
independently verified. The yes/no verifier missed every item, including the
two cases previously solved by full-memo or QA reranking, so the next useful
diagnostic is to emit verifier margins for the top-12 set rather than treating
the verifier prompt as a repair.

Commit `551165a` adds that verifier-margin diagnostic as
`--diagnose-contrastive-scores`. Tiny-model smoke run `r-6edeffa9ce` validated
metric emission, and clean Qwen T1 diagnostic run `r-81e82182fe` reused the
same hard-negative seed-0 cell. Protocols were registered before the run:
verifier selection accuracy `p-55bee7ca`, verifier target rank `p-559cc0fa`
(`--minimize`), target margin gap versus best non-target `p-a1398b70`, and
target yes-minus-no margin `p-0049e306`. Observed diagnostic metrics:

- verifier selection accuracy `0.0`;
- mean verifier target rank `8.0`;
- mean first-stage target rank `4.0` and target survival `1.0`;
- mean target yes-minus-no margin `-2.796006917953491`;
- mean best non-target margin `-1.6650190353393555`;
- mean target gap versus best non-target margin `-1.1309878826141357`.

Per-item margins show every true target is scored below at least one
non-target. Item 1 ranks the target `weather loss inspector` seventh behind
`storm loss inspector` and other distractors; item 2 ranks
`clinic inventory organizer` ninth behind unrelated and near-miss descriptors;
item 3 ranks `workshop hazard reviewer` ninth behind
`workshop hazard examiner`; item 4 ranks `cafeteria nutrition scheduler`
seventh behind `dispensary tag checker` and cafeteria near-misses. The yes/no
continuation prompt is therefore a prompt-level negative: it rewards less
negative "no" margins rather than recognizing the intended held-out
descriptor bridge.

Commit `4dce5b6` tests that source-leakage/target-completeness hypothesis as
a diagnostic upper bound. The new
`lexicon_descriptor_prefilter_compressed_lm_bridge` keeps the same
bidirectional top-12 first stage, then uses the committed descriptor lexicon
to score retained candidate descriptors: expected target-token matches are
rewarded, source-token leakage and missing expected target tokens are
penalized. This is not a deployable learned reranker because it uses the
generator's held-out descriptor table, but it isolates whether the retained
candidate set is sufficient.

T0 smoke run `r-710edfed75` validated metric emission. Claim-bearing Qwen T1
run `r-98e23407d2` used the same hard-negative seed-0 cell after registering
accuracy `p-100e1e56`, target survival `p-bd084d3a`, rerank fraction
`p-339ae917`, and kept fraction `p-6c055616`. Observed metrics:

- lexicon selector accuracy `1.0`;
- target survival `1.0`;
- rerank fraction `12/33 = 0.36363636363636365`;
- kept fraction `0.03219771590548423`;
- mean first-stage target rank `4.0`.

Accuracy protocol `p-100e1e56` independently reproduced `1.0` with no issues
or red flags. The support metrics above are same-run observations unless their
registered protocols are verified later. The result says the hard-negative
collapse is not caused by missing information in the compressed prompt: it is
a final-selection signal problem. A useful next step is a non-oracle
forced-choice descriptor translator/ranker that tries to elicit the target
descriptor from the source descriptor, then applies the same completeness and
leakage scoring without reading the ground-truth lexicon.

## Follow-up: few-shot descriptor translation does not repair selection

Commit `a01afe7` added
`fewshot_descriptor_prefilter_compressed_lm_bridge`. It keeps the same
bidirectional descriptor top-12 first stage, then ranks retained credential
descriptors by average continuation log-probability under a few-shot
source-to-target descriptor translation prompt. The prompt examples come from
the non-held-out descriptor families, and the scorer subtracts a small penalty
for candidate descriptor tokens that leak from the source descriptor. It does
not call `descriptor_phrase_map()` or use `HELDOUT_DESCRIPTOR_PAIRS` as the
answer key.

T0 smoke run **r-595f7cefa4** validated metric emission. Protocols were
registered before clean Qwen T1 run **r-3a41cfc710**: accuracy
**p-9b526ebb**, target survival **p-c118ff26**, rerank fraction
**p-5e454c00**, and retained fraction **p-8fe01567**. Observed metrics:

- few-shot selector accuracy `0.0`;
- target survival `1.0`;
- rerank fraction `12/33 = 0.36363636363636365`;
- retained fraction `0.03266558864409933`;
- mean first-stage target rank `4.0`.

This rules out the simple few-shot continuation scorer as a repair path. It
also strengthens the earlier interpretation: the compressed evidence is
present, but final selection needs a more robust target-side descriptor signal
than Qwen's raw continuation probability provides. The next useful step is a
score diagnostic for this scorer, especially whether the source-leakage
penalty is too weak or the continuation prompt prefers generic high-frequency
credential descriptors over the intended target descriptor.

Commit `e52e3aa` adds that score diagnostic as `--diagnose-fewshot-scores`.
It emits the retained top-12 candidates with first-stage score, few-shot
translation score, leakage penalty, and target-rank summary metrics. Tiny
smoke runs `r-4dbd63690c` and `r-78fe5315cd` validated trace and metric
emission; the second smoke confirmed the sweep wrapper propagates the
`mean_fewshot_diagnostic_*` metrics. The exact Qwen hard-negative seed-0
diagnostic command was registered before clean T1 run `r-9396776753` under
selection accuracy `p-8a3ab56f`, final target rank `p-8c667c3a`
(`--minimize`), target gap `p-f721d957`, and target survival `p-e3c3c4e8`.
Observed metrics:

- few-shot diagnostic selection accuracy `0.0`;
- target survival `1.0`;
- mean final target rank `5.0` versus first-stage target rank `4.0`;
- mean target score `-6.131380259990692`;
- mean best non-target score `-4.7531525174776705`;
- mean target gap versus best non-target `-1.378227742513021`;
- mean target leakage penalty `0.0`.

Per-item traces show that the raw continuation scorer prefers source-word
near-misses on items 1-3: `storm loss inspector` over
`weather loss inspector`, `hospital inventory organizer` over
`clinic inventory organizer`, and `workshop safety reviewer` over
`workshop hazard reviewer`. Item 4 is worse: it picks unrelated
`music hall attendance tallyer` over `cafeteria nutrition scheduler`. Because
the true targets have zero source-token leakage penalty, strengthening the
existing leakage penalty alone cannot repair this scorer; the prompt's target
translation likelihood is the weak signal.

## Follow-up: forced-choice descriptor comparison is also negative

Commit `e76cb35` added
`forced_choice_descriptor_prefilter_compressed_lm_bridge`. It keeps the same
bidirectional descriptor top-12 first stage, then presents the retained
credential descriptors as labeled choices in a single prompt and scores each
label as the best match for the source descriptor. This tests whether direct
candidate comparison helps where independent continuation scoring failed.

Failed T0 smoke run `r-142fe81228` used the wrong command shape (`cd
workspace`) after the harness had already entered the workspace checkout. T0
smoke run `r-4b43da967a` then validated metric emission and sweep
propagation. The exact Qwen hard-negative seed-0 command was registered before
clean T1 run `r-39d8d10cf9` under accuracy `p-f736fa8d`, target survival
`p-77309bb2`, rerank fraction `p-331c5046`, and retained fraction
`p-30cdecf4`. Observed metrics:

- forced-choice selector accuracy `0.0`;
- target survival `1.0`;
- rerank fraction `12/33 = 0.36363636363636365`;
- retained fraction `0.031885800746407504`;
- mean first-stage target rank `4.0`.

Same-run controls preserved the existing pattern: the lexicon upper bound
stayed at `1.0`, QA selection stayed at `0.25`, few-shot and contrastive
selection stayed at `0.0`, and oracle span stayed at `1.0`. Direct
candidate-label comparison is therefore not sufficient. The remaining useful
diagnostic is to inspect the forced-choice label scores and selected
descriptors, especially for positional label bias and source-word near-miss
preference, before spending more T1 cycles on prompt variants.

Commit `42dba68` adds that diagnostic as `--diagnose-forced-choice-scores`.
It prints the retained first-stage top candidates, forced-choice label scores,
label ordinals, selected descriptor, and target rank. T0 smoke run
`r-f35928b686` validated metric emission. The exact Qwen hard-negative seed-0
diagnostic command was registered before clean T1 run `r-c43db6e6c2` under
selection accuracy `p-d0839586`, final target rank `p-fe9ee525`
(`--minimize`), target gap versus best non-target `p-7171f358`, and target
survival `p-a823922d`. Observed metrics:

- forced-choice diagnostic selection accuracy `0.0`;
- target survival `1.0`;
- mean final target rank `5.25` versus first-stage target rank `4.0`;
- mean target score `-4.069691598415375`;
- mean best non-target score `-1.1432542353868484`;
- mean target gap versus best non-target `-2.9264373630285263`;
- mean target label ordinal `3.0` and selected label ordinal `0.5`.

Per-item traces show the label prompt chooses source-token hybrids in every
case: `storm loss inspector` over `weather loss inspector`,
`hospital inventory organizer` over `clinic inventory organizer`,
`factory hazard reviewer` over `workshop hazard reviewer`, and
`school nutrition scheduler` over `cafeteria nutrition scheduler`. The early
selected-label ordinal suggests some label-position bias, but the chosen
descriptors are not arbitrary early labels; they are plausible near-misses
that copy source-side words. This makes the forced-choice branch another
prompt-level negative and points away from simple label-list prompting.

## Follow-up: pairwise tournament comparison is also negative

Commit `91ce92c` added
`pairwise_tournament_descriptor_prefilter_compressed_lm_bridge`, which keeps
the bidirectional descriptor top-12 first stage but replaces list-label
scoring with pairwise A/B comparisons. Each challenger is compared against the
current incumbent in both label orders to reduce label-position artifacts, and
the tournament winner is used in the compressed bridge prompt.

T0 smoke **r-c2c82c7cdd** validated metric and diagnostic emission. The exact
Qwen hard-negative seed-0 command was registered before clean T1 run
**r-ed71b7001d** under protocols:

- answer accuracy **p-8924a064**;
- target survival **p-a9c09bdc**;
- rerank fraction **p-d682b89d**;
- diagnostic selector accuracy **p-49e84e73**.

Observed metrics from **r-ed71b7001d**:

- pairwise tournament answer accuracy `0.0`;
- diagnostic selector accuracy `0.0`;
- target survival `1.0`;
- rerank fraction `12/33 = 0.36363636363636365`;
- retained fraction `0.031885800746407504`;
- mean first-stage target rank `4.0`;
- mean target entered step `4.0`, target lost step `4.0`, target won step
  `0.0`.

Per-item tournament winners were the same source-word near-miss family:

- item 1: selected `storm loss inspector`; target `weather loss inspector`
  entered at step 3 and lost by margin `-0.416541`;
- item 2: selected `hospital inventory organizer`; target
  `clinic inventory organizer` entered at step 6 and lost by margin
  `-0.551313`;
- item 3: selected `factory hazard reviewer`; target
  `workshop hazard reviewer` entered at step 3 and lost by margin
  `-0.447124`;
- item 4: selected `school nutrition scheduler`; target
  `cafeteria nutrition scheduler` entered at step 4 and lost by margin
  `-0.449243`.

Interpretation: this is not just list-label bias. Even direct, label-balanced
pairwise comparisons prefer source-copy near-misses over the true target-side
descriptor. Do not spend more T1 budget on generic comparison prompts for this
cell; the next repair should introduce a non-oracle target-side mapping source
or an explicit anti-copy objective.

## Follow-up: source-copy guard is only a partial repair

Commit `342ee70` added
`source_copy_guarded_descriptor_prefilter_compressed_lm_bridge`, which keeps
the same bidirectional descriptor top-12 first stage and reranks retained
credentials by the first-stage descriptor score minus a fixed penalty for
source-side content-word overlap. This is non-oracle: it does not call
`descriptor_phrase_map()` and does not use `HELDOUT_DESCRIPTOR_PAIRS` as a
target descriptor key.

T0 smoke **r-0ea0a1514a** validated metric emission with zero distractors.
T0 smoke **r-dd09210387** exercised nontrivial candidate metrics with
`distractors=4`; failed T0 run **r-638d94fb63** exceeded the tiny GPT-2
position table and should only be treated as a smoke sizing error.

The exact Qwen hard-negative seed-0 command was registered before clean T1
run **r-ea0106fabd** under protocols:

- answer accuracy **p-77935fd7**;
- target survival **p-866345f3**;
- rerank fraction **p-8d39f86e**;
- kept fraction, minimized, **p-b3f5e446**.

The earlier kept-fraction registration **p-6874cabb** used the wrong maximize
direction and should not be cited.

Observed metrics from **r-ea0106fabd**:

- source-copy guarded answer accuracy `0.25`;
- target survival `1.0`;
- rerank fraction `12/33 = 0.36363636363636365`;
- retained fraction `0.03204175832594587`;
- mean first-stage target rank `4.0`;
- lexicon upper-bound accuracy `1.0`;
- QA bidirectional accuracy `0.25`;
- generic bidirectional, forced-choice, pairwise, few-shot, and contrastive
  descriptor selectors all `0.0`.

Per-item stderr shows the guard fixed only item 1:

- item 1: selected the true `weather loss inspector`;
- item 2: selected code `674` instead of target `797`;
- item 3: selected code `367` instead of target `907`;
- item 4: selected code `201` instead of target `216`.

Interpretation: source-copy leakage is a real part of the collapse, but a
simple overlap penalty is not enough. This branch reaches the same `0.25`
partial-recovery level as existing QA/compressed-LM paths and remains well
below the verified lexicon upper bound. The next useful repair should add a
non-oracle target-side mapping source, or a diagnostic that exposes which
candidate words need target-side substitution rather than merely penalizing
copied source words.

## Follow-up: in-context induced descriptor map reaches the upper bound

Commit `252f1dd` added
`induced_map_descriptor_prefilter_compressed_lm_bridge`. It keeps the same
bidirectional descriptor top-12 first stage, then induces source-token to
target-token substitutions from other same-record-id descriptor pairs visible
in the prompt. The queried memo id is excluded, and the selector does not call
`descriptor_phrase_map()` or use `HELDOUT_DESCRIPTOR_PAIRS` as a key for the
queried descriptor. Retained candidates are scored by expected target-token
hits minus missing expected tokens and source-token leakage.

T0 smoke **r-141148a9e0** validated metric emission and diagnostic JSON; it
should be cited only as plumbing evidence because the harness auto-committed
the implementation before the run, so `workspace_dirty=true`.

The exact Qwen hard-negative seed-0 command was registered before clean T1 run
**r-72068ca7cc** under protocols:

- answer accuracy **p-5383e94d**;
- target survival **p-e71713ba**;
- rerank fraction **p-4ae94650**;
- kept fraction, minimized, **p-c269bebb**;
- diagnostic target gap versus best non-target **p-d096d88f**;
- diagnostic source-token coverage **p-fcabb827**.

Primary accuracy protocol **p-5383e94d** independently reproduced
**r-72068ca7cc** with value `1.0`; verifier verdict: `verified=true`, no
issues, no red flags.

Observed metrics from **r-72068ca7cc**:

- induced-map answer accuracy `1.0`;
- target survival `1.0`;
- rerank fraction `12/33 = 0.36363636363636365`;
- retained fraction `0.03219771590548423`;
- mean first-stage target rank `4.0`;
- diagnostic final target rank `1.0`;
- diagnostic target gap versus best non-target `3.25`;
- diagnostic source-token coverage `1.0`;
- lexicon upper-bound accuracy `1.0`;
- QA bidirectional, source-copy guarded, and compressed-LM accuracy `0.25`;
- generic bidirectional, forced-choice, pairwise, few-shot, and contrastive
  descriptor selectors all `0.0`.

Interpretation: the hard-negative failure is not a retained-token recall
problem once the bidirectional top-12 survives. It is a target-side
translation extraction problem. A non-oracle in-context mapping source recovers
the lexicon upper bound on the seed-0 hard-negative cell at the same retained
fraction as the lexicon selector.

The seed-1 robustness command was registered before clean T1 run
**r-fdabeeaa73** under protocols:

- answer accuracy **p-3b324ba4**;
- target survival **p-d3dd7274**;
- kept fraction, minimized, **p-eccd4e8b**;
- diagnostic target gap versus best non-target **p-0da5c340**;
- diagnostic source-token coverage **p-ab1ca51d**.

Primary accuracy protocol **p-3b324ba4** independently reproduced
**r-fdabeeaa73** with value `1.0`; verifier verdict: `verified=true`, no
issues, no red flags.

Observed metrics from **r-fdabeeaa73**:

- induced-map answer accuracy `1.0`;
- target survival `1.0`;
- rerank fraction `12/33 = 0.36363636363636365`;
- retained fraction `0.032127967249218514`;
- mean first-stage target rank `3.75`;
- diagnostic final target rank `1.0`;
- diagnostic target gap versus best non-target `3.0`;
- diagnostic source-token coverage `1.0`;
- lexicon upper-bound accuracy `1.0`;
- QA bidirectional, source-copy guarded, compressed-LM, prefilter-LM, and
  prefilter-compressed-LM accuracy `0.25`;
- generic bidirectional, forced-choice, pairwise, few-shot, and contrastive
  descriptor selectors all `0.0`.

The next useful test is seed 2 or a less directly aligned prompt where
source/target substitutions must be inferred without hard-negative support
pairs.

The seed-2 robustness command was registered before clean T1 run
**r-22327fbebf** under protocols:

- answer accuracy **p-367beffd**;
- target survival **p-4b7b0a4a**;
- kept fraction, minimized, **p-9ac7aaea**;
- diagnostic final target rank, minimized, **p-18c85e89**;
- diagnostic target gap versus best non-target **p-a0574bdc**.

Primary accuracy protocol **p-367beffd** independently reproduced
**r-22327fbebf** with value `1.0`; verifier verdict: `verified=true`, no
issues, no red flags.

Observed metrics from **r-22327fbebf**:

- induced-map answer accuracy `1.0`;
- target survival `1.0`;
- rerank fraction `12/33 = 0.36363636363636365`;
- retained fraction `0.03218553183695408`;
- mean first-stage target rank `3.75`;
- diagnostic final target rank `1.0`;
- diagnostic target gap versus best non-target `3.25`;
- diagnostic source-token coverage `1.0`;
- lexicon upper-bound accuracy `1.0`;
- compressed-LM and LM bridge accuracy `0.5`;
- semantic bridge, prefilter-LM, prefilter-compressed-LM, and
  margin-guarded prefilter accuracy `0.25`;
- generic bidirectional, QA bidirectional, forced-choice, pairwise,
  few-shot, source-copy guarded, and contrastive descriptor selectors all
  `0.0`;
- full-context accuracy `0.25`, truncate and streaming `0.0`, oracle span
  `1.0`.

Across seeds `0`, `1`, and `2`, the induced-map selector matches the lexicon
upper bound and oracle span at about `3.2%` retained tokens, while the generic
descriptor comparison variants remain collapsed.

The less directly aligned `heldout_descriptor_bridge_v2` seed-0 stress command
was then registered before clean T1 run **r-238728cd2b** under protocols:

- answer accuracy **p-f4ada7a9**;
- target survival **p-902d9bd2**;
- diagnostic source-token coverage **p-fabd0670**;
- diagnostic target gap versus best non-target **p-5bd44e50**.

Primary accuracy protocol **p-f4ada7a9** independently reproduced
**r-238728cd2b** with value `0.75`; verifier verdict: `verified=true`, no
issues, no red flags.

Observed metrics from **r-238728cd2b**:

- induced-map answer accuracy `0.75`;
- target survival `1.0`;
- retained fraction `0.03259130616517959`;
- mean first-stage target rank `1.75`;
- diagnostic final target rank `1.75`;
- diagnostic target gap versus best non-target `0.0`;
- diagnostic source-token coverage `0.16666666666666666`;
- lexicon, bidirectional descriptor, margin-guarded, pairwise, reverse
  descriptor, and oracle span accuracy `1.0`;
- prefilter/compressed LM, compressed LM, LM bridge, forced-choice, and
  source-copy guarded accuracy `0.75`;
- full-context, truncate, streaming, lexical, semantic, QA bidirectional, and
  contrastive accuracy `0.0`.

Per-item diagnostics show the caveat: on items 1, 2, and 3, source-token
coverage was `0.0`, all induced scores tied at zero, and the induced-map
selector inherited the first-stage order. Item 2 failed because the true
target was first-stage rank `4`; items 1 and 3 succeeded because the target
was already first-stage rank `1`. Item 4 had partial coverage (`2/3` source
tokens mapped) and still tied the best non-target at score `1.0`, with the
target selected by order. This is therefore a useful robustness datum for the
combined policy, but not evidence that the induced substitution map itself
solves less supported descriptor prompts.

Follow-up diagnostic run **r-0290d4f1ae** at commit **12b30e4** added explicit
first-stage counterfactual metrics to `--diagnose-induced-map-scores` and
reran the same Qwen v2 seed-0 cell. Registered protocols were first-stage
top-1 selection accuracy **p-60658841**, final induced-map diagnostic
selection accuracy **p-a4d8cdcd**, source-token coverage **p-06e116bd**, and
first-stage target gap **p-71004a8d**. The run observed:

- first-stage top-1 selection accuracy `0.75`;
- final induced-map diagnostic selection accuracy `0.75`;
- source-token coverage `0.16666666666666666`;
- mean first-stage target rank `1.75`;
- first-stage target gap versus best non-target `0.5310943007469189`.

Protocol **p-60658841** independently reproduced first-stage top-1 selection
accuracy `0.75` with no issues or red flags. This verifies the caveat above:
on v2 seed 0, the induced-map selector's `0.75` mostly reflects the
bidirectional first-stage ranking already placing the target first on three of
four items.

Commit **a3cea41** adds mechanism counters to the induced-map diagnostic for
zero source-token coverage, zero-coverage success, final-selection changes
relative to the first-stage top choice, and positive induced target gaps.
Clean T1 run **r-b7ec1559c0** reused the same Qwen v2 seed-0 cell after
registering **p-9a922ca0**, **p-1ef1ef34**, **p-257f591f**, and
**p-354025a2**. It observed:

- zero source-token coverage on `0.75` of items;
- zero-coverage selection success on `0.5` of items;
- final induced-map changes to the first-stage top choice `0.0`;
- positive induced target gap `0.0`.

Protocol **p-257f591f** independently reproduced the final-changed-first-stage
value `0.0` with no issues or red flags. Protocol **p-9a922ca0**
independently reproduced zero source-token coverage `0.75`, protocol
**p-1ef1ef34** independently reproduced zero-coverage selection success
`0.5`, and protocol **p-354025a2** independently reproduced the
positive-induced-target-gap value `0.0`, all with no issues or red flags.
Together these verified counters sharpen the same interpretation: on this v2
cell, the induced-map final scorer does not add an independent target-side
substitution signal beyond first-stage order.

Commit **1ac2fd8** adds
`strict_induced_map_descriptor_prefilter_compressed_lm_bridge`, a policy-level
version of the same caveat. It keeps the bidirectional top-12 first stage and
the induced descriptor-token score, but breaks final induced-score ties by
original candidate order instead of by first-stage score. T0 smoke
**r-350b331ec1** validated metric emission. Protocols registered before clean
Qwen v2 seed-0 T1 run **r-4ce6c63efb** were strict accuracy **p-64e38cb9** and
strict target survival **p-cc8f10d6**. Observed metrics:

- strict induced-map accuracy `0.0`;
- strict target survival `1.0`;
- strict retained fraction `0.03399561994071286`;
- strict mean first-stage target rank `1.75`;
- inherited-order induced-map accuracy `0.75` on the same run.

Protocol **p-64e38cb9** independently reproduced strict accuracy `0.0` with no
issues or red flags. Protocol **p-cc8f10d6** independently reproduced strict
target survival `1.0` with no issues or red flags. Together, these verify that
the strict selector failed despite retaining the target, so the earlier v2
induced-map `0.75` combined-policy result was first-stage tie inheritance, not
a robust induced-substitution signal.

## Task-External Support Map

Commit **06ca4fd** adds
`external_support_induced_map_descriptor_prefilter_compressed_lm_bridge`. It
keeps the same bidirectional descriptor top-12 first stage, then builds the
final source-token to target-token descriptor map from committed support
descriptor pairs outside the evaluated prompt, excluding the exact queried
source descriptor. This is meant to test a different mapping source from the
in-context induced-map policy: no query-specific same-record support pairs are
used.

T0 smoke **r-fd1392d9f0** validated metric emission. A prior malformed smoke
**r-9035898579** failed before evaluation because the command tried to
`cd workspace` even though `aad exp run` already executes from the workspace
checkout.

Protocols registered before clean Qwen hard-negative seed-0 T1 run
**r-0a687a50dd**:

- accuracy **p-4976be1c**;
- target survival **p-af878015**.

Observed metrics for **r-0a687a50dd**:

- external-support induced-map accuracy `0.75`;
- target survival `1.0`;
- kept fraction `0.03266764071751431`;
- rerank fraction `12/33 = 0.36363636363636365`;
- mean first-stage target rank `4.0`.

Protocol **p-4976be1c** independently reproduced accuracy `0.75` with
`verified=true`, no issues, and no red flags. Protocol **p-af878015**
independently reproduced target survival `1.0` with `verified=true`, no
issues, and no red flags. This was a prediction miss: the run was launched
with predicted range `[0, 0.5]`. Same-run contrasts on this cell were
bidirectional/reverse/forced-choice/few-shot/contrastive/pairwise accuracy
`0.0`, QA/compressed-LM/LM/source-copy guarded partial baselines `0.25`, and
oracle/lexicon/in-context induced/strict induced accuracy `1.0`.

Seed-1 robustness was tested at commit **ca0b530**. Protocols registered
before clean Qwen hard-negative seed-1 T1 run **r-8a7991ace4**:

- accuracy **p-b1bd9d1c**;
- target survival **p-5ccf55a9**.

Observed metrics for **r-8a7991ace4**:

- external-support induced-map accuracy `0.75`;
- target survival `1.0`;
- kept fraction `0.032753358118511836`;
- rerank fraction `12/33 = 0.36363636363636365`;
- mean first-stage target rank `3.75`.

Protocol **p-b1bd9d1c** independently reproduced accuracy `0.75` with
`verified=true`, no issues, and no red flags. Protocol **p-5ccf55a9**
independently reproduced target survival `1.0` with `verified=true`, no
issues, and no red flags. Same-run contrasts were generic
bidirectional/reverse/forced-choice/few-shot/contrastive/pairwise descriptor
accuracy `0.0`; QA/compressed-LM/LM/prefilter-LM/source-copy guarded/semantic
partial baselines `0.25`; and oracle/lexicon/in-context induced/strict
induced accuracy `1.0`.

Seed-2 robustness was tested at commit **91b30b6**. Protocols registered
before clean Qwen hard-negative seed-2 T1 run **r-c888f9e291**:

- accuracy **p-ba0566e8**;
- target survival **p-77d2d647**.

Observed metrics for **r-c888f9e291**:

- external-support induced-map accuracy `0.75`;
- target survival `1.0`;
- kept fraction `0.032810922706247386`;
- rerank fraction `12/33 = 0.36363636363636365`;
- mean first-stage target rank `3.75`.

Protocol **p-ba0566e8** independently reproduced accuracy `0.75` with
`verified=true`, no issues, and no red flags. Protocol **p-77d2d647**
independently reproduced target survival `1.0` with `verified=true`, no
issues, and no red flags. Same-run contrasts were generic
bidirectional/QA/reverse/forced-choice/few-shot/contrastive/pairwise
descriptor accuracy `0.0`; prefilter-LM/prefilter-compressed/margin-guarded/
semantic partial baselines `0.25`; compressed-LM and LM baselines `0.5`; and
oracle/lexicon/in-context induced/strict induced accuracy `1.0`.

Commit **9a4c80a** adds
`--diagnose-external-support-induced-map-scores`, which emits candidate-level
traces and `mean_external_support_induced_map_diagnostic_*` metrics for the
task-external support map. T0 smoke **r-9e09ce8adc** validated metric emission.
Protocols registered before clean Qwen hard-negative seed-0 diagnostic run
**r-2705edb2c6**:

- diagnostic selection accuracy **p-762b0212**;
- minimized final target rank **p-06f46e88**;
- target gap versus best non-target **p-fa1083cd**;
- source-token coverage **p-e7e5263d**.

Observed diagnostic metrics for **r-2705edb2c6**:

- external-support diagnostic selection accuracy `0.75`;
- final target rank `1.75`;
- target gap versus best non-target `0.0`;
- source-token coverage `0.25`;
- target survival `1.0`;
- positive target gap rate `0.0`.

Protocol **p-762b0212** independently reproduced diagnostic selection accuracy
`0.75` with `verified=true`, no issues, and no red flags. Protocol
**p-fa1083cd** independently reproduced target gap versus best non-target
`0.0`, and protocol **p-e7e5263d** independently reproduced source-token
coverage `0.25`; both verdicts were `verified=true` with no issues and no red
flags. Per-item traces show that the recurring miss is not a prefilter-recall
failure: on item 2, the true target survived the bidirectional top-12 but only
one source token was mapped, leaving the target tied with many non-targets at
final score `-1.0`; the selected descriptor was a tie/order artifact. The
successful items also had zero positive target gap, so this support map is
best interpreted as a sparse tie-breaking aid rather than a robust target-side
translation mechanism.

## Interpretation

The old top-12 reverse-only first stage failed on v2 seed `2` because one true
target ranked `13` and was excluded from the final rerank set. Widening to
top-16 repaired that recall failure, but increased the final rerank fraction
from `12/33 = 0.36363636363636365` to `16/33 =
0.48484848484848486`.

`bidirectional_descriptor_prefilter_compressed_lm_bridge` repairs the same
seed-2 failure without widening the final rerank set. It scores all
code-bearing candidates with both descriptor directions in the first stage,
keeps the top 12, then applies the reverse-descriptor compressed reranker.

This is a recall and retained-prompt result, not a free end-to-end compute
reduction. The bidirectional policy spends an extra descriptor scoring pass
over all code-bearing candidates before the final top-12 rerank, and it is not
a production KV-cache implementation.

## Reproduction shape

All claim-bearing numbers must be produced through AAD. For a fresh run,
register the target metric before running, then use the exact command shape
below with the desired seed, distractor count, and `--prefilter-candidates`
value.

```bash
aad exp run --bet bet-0006 --tier t1 --seed 2 --predict 1,1 -- \
  env UV_CACHE_DIR=.uv-cache uv run --frozen python eval_kv_sweep.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --items 4 \
    --distractors 32 \
    --seed 2 \
    --task heldout_descriptor_bridge_v2 \
    --target-positions middle \
    --budgets 192 \
    --sink-tokens 64 \
    --lexical-records 1 \
    --prefilter-candidates 12 \
    --scoring rank \
    --device cpu \
    --dtype float32
```

The main metrics for the bidirectional policy are:

- `mean_bidirectional_descriptor_prefilter_compressed_lm_bridge_accuracy`
- `middle_b192_bidirectional_descriptor_prefilter_compressed_lm_bridge_kept_fraction`
- `mean_bidirectional_descriptor_prefilter_compressed_lm_bridge_rerank_fraction`
- `mean_bidirectional_descriptor_prefilter_compressed_lm_bridge_target_survival_rate`
- `middle_b192_bidirectional_descriptor_prefilter_compressed_lm_bridge_max_target_rank`
