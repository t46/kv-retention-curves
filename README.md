# KV Frontier

This workspace measures quality vs retained prompt/KV tokens for small causal
LMs. The initial evaluation is deliberately narrow: synthetic retrieval tasks
with multiple prompt-retention and retrieval policies:

Current verified comparison artifact:
`docs/bidirectional_descriptor_prefilter.md` summarizes the reverse top-12,
reverse top-16, and bidirectional top-12 descriptor-prefilter frontier on
`heldout_descriptor_bridge_v2`, including the `distractors=48` candidate-count
stress and the fully support-verified `distractors=64` seed-0/seed-1/seed-2
extension. It also records the first hard-negative descriptor stress result:
seed 0 verified accuracy collapse to `0.0` despite verified target survival
`1.0`, indicating a final-reranker failure rather than prefilter recall loss.

For a concise public handoff, see `docs/ship_summary.md`. It states the
publishable result, evidence IDs, scope, and non-claims without the full
session-by-session evidence log.

- `full`: keep the whole prompt.
- `truncate`: keep only the most recent tokens.
- `streaming`: keep a small attention-sink prefix plus recent tokens.
- `oracle_span`: keep only the answer-bearing record and question tail.
- `lexical`: extract the query key from the question, retain matching record
  line(s), and keep the question tail.
- `bridge`: retrieve the query-matching alias line, parse the linked person
  name from that line, retain the corresponding profile line, and keep the
  question tail.
- `prose_bridge`: retrieve the query-matching line, extract a linked
  capitalized person name from either fixed alias records or prose memo text,
  retain other lines mentioning that name, and keep the question tail.
- `entity_bridge`: retrieve the query-matching line, extract capitalized
  two-word entity spans other than the queried alias, retain other lines that
  mention those spans, and keep the question tail.
- `overlap_bridge`: retrieve the query-matching line, then retain the other
  record line with the largest overlap on non-stopword content tokens from
  that line after removing the queried alias tokens. This is a looser one-hop
  expansion baseline that does not assume capitalized two-word names.
- `id_bridge`: retrieve the query-matching line, normalize employee
  identifiers such as `AB-123` to `AB123`, retain other lines with the same
  normalized identifier, and keep the question tail.
- `char_bridge`: retrieve the query-matching line, then retain the other
  record line with the largest normalized character 3-gram overlap. This is a
  schema-agnostic one-hop expansion baseline for links such as `AB-123` /
  `AB123`.
- `neighbor_bridge`: retrieve the query-matching line, then retain the
  following record line. This is a layout/co-placement baseline
  for tasks where the second hop has no shared name, id, or character key.
- `semantic_bridge`: retrieve the query-matching line, map its descriptor
  through a committed source/paraphrase descriptor table, retain the
  highest-overlap code-bearing record under that mapped phrase, and keep the
  question tail. This is a lightweight non-oracle semantic expansion baseline
  for descriptor tasks.
- `embedding_bridge`: retrieve the query-matching line, then cosine-score
  code-bearing memos against it using mean last-hidden-state embeddings from
  the same causal LM, and retain the highest-scoring memo. This probes whether
  a cheaper model-internal semantic signal can replace continuation reranking.
- `phrase_lm_bridge`: retrieve the query-matching line, then use the same
  causal LM to score only the descriptor phrase extracted from each
  code-bearing memo, retaining the full memo attached to the best descriptor.
  This probes whether the `lm_bridge` result depends on scoring whole
  credential memos including boilerplate and code tokens.
- `prefilter_lm_bridge`: retrieve the query-matching line, use descriptor-only
  LM scoring to keep a top-k subset of code-bearing memos, then rerank only
  that subset with the full-memo `lm_bridge` continuation score. This probes
  whether the expensive full-memo scoring step can be candidate-reduced without
  losing the held-out descriptor recovery.
- `bidirectional_descriptor_prefilter_compressed_lm_bridge`: retrieve the
  query-matching line, score each code-bearing memo with both descriptor
  directions, keep a top-k subset, and then use the reverse-descriptor reranker
  with the compressed bridge prompt. This probes whether top-k recall can be
  improved without widening the final rerank set.
- `contrastive_descriptor_prefilter_compressed_lm_bridge`: use the same
  bidirectional descriptor top-k first stage, then ask the causal LM to score
  whether the query source descriptor and each retained credential descriptor
  are the same role by a `yes` versus `no` continuation margin. This probes
  whether an explicit pair verifier can repair hard-negative final selection.
- `fewshot_descriptor_prefilter_compressed_lm_bridge`: use the same
  bidirectional descriptor top-k first stage, then rank retained credential
  descriptors as continuations of a few-shot source-to-target descriptor
  translation prompt built from non-held-out descriptor families, with a small
  source-token leakage penalty. This probes whether Qwen can elicit the
  target-side descriptor signal without the held-out lexicon.
- `source_copy_guarded_descriptor_prefilter_compressed_lm_bridge`: use the
  same bidirectional descriptor top-k first stage, then rerank retained
  credentials by the first-stage descriptor score minus an explicit penalty
  for copying source-side content words. This probes whether the observed
  source-copy near-miss failure can be reduced without using the held-out
  descriptor lexicon.
- `external_support_induced_map_descriptor_prefilter_compressed_lm_bridge`:
  use the same bidirectional descriptor top-k first stage, then build a
  source-token to target-token descriptor map from committed support descriptor
  pairs outside the evaluated prompt, excluding the exact queried source
  descriptor. This probes whether task-external examples can provide a
  target-side anti-copy signal without relying on query-specific same-record
  support pairs.
- `lm_bridge`: retrieve the query-matching line, then use the same causal LM
  to score each code-bearing memo as a continuation after that line and retain
  the highest-scoring memo. This is a task-agnostic expansion baseline: it has
  no descriptor table, but it does spend model forward passes during
  selection.

Tasks:

- `kv`: single-record key-value lookup. Exact lexical retrieval is expected to
  be near-oracle because the query key appears in the answer-bearing record.
- `alias_bridge`: two-hop alias/profile lookup. The question names an alias,
  one record maps alias to person, and a separate profile record maps person to
  code. The naive lexical policy retrieves the alias line but not the profile
  line, so exact matching is no longer sufficient by construction. The bridge
  policy is a non-oracle structured-text baseline for this task.
- `paraphrase_bridge`: two-hop alias/profile lookup written as prose memos
  rather than fixed `refers to` / `uses access code` records. The current
  bridge parser is expected to collapse to lexical retrieval here because it
  cannot parse the paraphrased link; oracle span remains a recoverability
  control. The `prose_bridge` policy is a non-oracle heuristic baseline for
  this task, but it still assumes entity-like two-word person names.
- `varied_bridge`: a harder prose version that samples multiple alias-link
  and profile-code phrasings. It is designed to break phrase-specific
  extraction such as the `prose_bridge` policy while preserving an oracle
  two-memo recoverability control. The `entity_bridge` policy is a looser
  non-oracle baseline for this task; it still depends on capitalized two-word
  entity spans.
- `format_bridge`: the same varied prose structure, but linked person names
  are lower-case. This preserves the two-memo oracle control while attacking
  the `entity_bridge` assumption that linked names are capitalized entity
  spans. `overlap_bridge` is the intended non-oracle baseline for this task.
- `id_bridge`: two-hop lookup where the alias memo contains an employee id
  with punctuation and the profile memo uses the normalized staff number. This
  attacks literal name-token overlap while preserving a structured non-oracle
  expansion path through the `id_bridge` policy. The `char_bridge` policy tests
  whether that recovery requires an explicit employee-id parser.
- `descriptor_bridge`: two-hop lookup where the alias memo uses one natural
  language descriptor and the adjacent profile memo uses a paraphrased
  descriptor with no intended shared identifier. This attacks the normalized
  character-overlap assumption in `char_bridge`; `neighbor_bridge` tests
  whether retaining a small local window around the query hit can recover the
  answer-bearing span.
- `shuffled_descriptor_bridge`: the same descriptor/paraphrase structure, but
  alias memos and profile memos are shuffled into separate regions and the
  target profile is inserted away from the query-matching alias memo. In this
  task family, `--target-position middle` places the query-matching alias near
  the middle first; the answer-bearing profile may be moved away to enforce
  non-adjacency, which is reported via `mean_target_position_fraction`,
  `mean_query_position_fraction`, and `mean_target_query_distance_fraction`.
  This attacks the co-placement assumption in `neighbor_bridge`;
  `semantic_bridge` is the intended non-oracle expansion baseline.
- `heldout_descriptor_bridge`: the shuffled descriptor task with descriptor
  paraphrase pairs drawn from a separate held-out list that is not included in
  the committed `semantic_bridge` table. This attacks the curated descriptor
  assumption; `lm_bridge` is the intended task-agnostic retrieval baseline.
- `heldout_descriptor_bridge_hard_negative`: the held-out v2 descriptor task
  with several distractor credential descriptors deliberately recombined from
  the target source/target words. This keeps the same shuffled, varied-template
  shape but attacks easy candidate separation: the compact descriptor prefilter
  must rank the true credential above near-miss descriptors that share domain
  and role words.

The current hard-negative diagnostic shows that first-stage survival is not
enough. At commit `bdf8752`, run `r-1442712669` verified
`mean_bidirectional_descriptor_prefilter_compressed_lm_bridge_accuracy = 0.0`
while independently verifying target survival at `1.0`. A later
QA-conditioned final reranker at commit `f4eb1c9` was smoke-tested and run as
T1 `r-ecd473b719`; it observed only `0.25` accuracy, matching older
full-memo LM rerankers rather than repairing the hard-negative collapse.
Commit `c6b5d9c` adds `--diagnose-final-scores` for candidate-level traces;
T1 run `r-1ca21e1cc9` observed that reverse, full-memo, and QA final scorers
all prefer descriptor hybrids over the exact target on most seed-0 items.
Commit `b31cdde` adds the contrastive descriptor-pair verifier above. Clean
T1 run `r-8eda7d862b` observed `0.0` accuracy for that policy, with target
survival still `1.0`, rerank fraction `12/33`, and kept fraction
`0.032979555876591043`. Protocols registered before the run: accuracy
`p-fa2142e9`, target survival `p-ec1b5cf8`, rerank fraction `p-624e1609`,
and kept fraction `p-0a743068`. This is an observed negative result, not
independently verified.

Commit `551165a` adds `--diagnose-contrastive-scores`, which emits the
yes-score, no-score, and yes-minus-no margin for the retained top-12
contrastive verifier candidates. Clean T1 diagnostic run `r-81e82182fe`
reused the same Qwen hard-negative seed-0 cell after registering protocols
`p-55bee7ca`, `p-559cc0fa`, `p-a1398b70`, and `p-0049e306`. The diagnostic
observed verifier selection accuracy `0.0`, mean verifier target rank `8.0`,
mean target margin `-2.796006917953491`, and mean target gap versus the best
non-target margin `-1.1309878826141357`. This indicates that the yes/no
prompt is biased against the true paraphrase pairs, not merely that a final
tie-breaker is wrong.

Commit `4dce5b6` adds `lexicon_descriptor_prefilter_compressed_lm_bridge`, a
diagnostic upper-bound selector that keeps the same bidirectional descriptor
top-12 first stage but chooses among retained credentials by using the
committed descriptor lexicon: reward expected target-side descriptor tokens and
penalize source-side token leakage. T0 smoke run `r-710edfed75` validated
metric emission. Qwen hard-negative seed-0 T1 run `r-98e23407d2` observed
accuracy `1.0`, target survival `1.0`, rerank fraction `12/33`, and kept
fraction `0.03219771590548423`. Accuracy protocol `p-100e1e56` independently
reproduced `1.0` with no issues or red flags. Support protocols
`p-bd084d3a`, `p-339ae917`, and `p-6c055616` are registered but not yet
independently verified. Interpretation: the compressed top-12 set contains
enough information to solve the hard-negative cell when final selection has a
clean target-side translation signal; the remaining research problem is to
learn or elicit that signal without using the generator's descriptor lexicon.

Commit `06ca4fd` adds
`external_support_induced_map_descriptor_prefilter_compressed_lm_bridge`, a
task-external induced-map variant. It keeps the bidirectional top-12 first
stage, but induces descriptor-token substitutions from committed support
descriptor pairs outside the evaluated prompt while excluding the exact
queried source descriptor. T0 smoke `r-fd1392d9f0` validated metric emission.
Protocols registered before Qwen hard-negative seed-0 T1 run `r-0a687a50dd`:
accuracy `p-4976be1c` and target survival `p-af878015`. The run observed
accuracy `0.75`, target survival `1.0`, kept fraction
`0.03266764071751431`, rerank fraction `12/33`, and mean first-stage target
rank `4.0`. Protocol `p-4976be1c` independently reproduced accuracy `0.75`
with no issues or red flags. Protocol `p-af878015` independently reproduced
target survival `1.0` with no issues or red flags. This was a verified
partial repair and a prediction miss: the run was predicted in `[0, 0.5]`.

The seed-1 robustness extension reused the exact hard-negative cell at commit
`ca0b530`. Protocols `p-b1bd9d1c` and `p-5ccf55a9` were registered before
clean T1 run `r-8a7991ace4`. The run observed external-support induced-map
accuracy `0.75`, target survival `1.0`, kept fraction
`0.032753358118511836`, rerank fraction `12/33`, and mean first-stage target
rank `3.75`. Both protocols independently reproduced their metrics with no
issues or red flags. Same-run contrasts again put generic bidirectional,
reverse, forced-choice, few-shot, contrastive, and pairwise descriptor
selectors at `0.0`; QA, compressed-LM, LM, prefilter-LM, source-copy guarded,
and semantic partial baselines at `0.25`; and oracle, lexicon, in-context
induced-map, and strict induced-map selectors at `1.0`.

The seed-2 robustness extension used the same cell at commit `91b30b6`.
Protocols `p-ba0566e8` and `p-77d2d647` were registered before clean T1 run
`r-c888f9e291`. The run observed external-support induced-map accuracy
`0.75`, target survival `1.0`, kept fraction `0.032810922706247386`, rerank
fraction `12/33`, and mean first-stage target rank `3.75`. Both protocols
independently reproduced their metrics with no issues or red flags. Across
seeds 0/1/2, this makes the external-support map a verified three-seed
partial repair, but still not a full frontier claim because it remains below
the lexicon/in-context induced-map upper bound and uses a committed synthetic
support table.

Commit `a01afe7` adds `fewshot_descriptor_prefilter_compressed_lm_bridge`, a
non-oracle attempt at that signal. It keeps the same bidirectional top-12
first stage, then scores each retained credential descriptor as a continuation
of a few-shot source-to-target descriptor translation prompt built only from
the non-held-out descriptor families, with a small penalty for source-token
leakage. T0 smoke run `r-595f7cefa4` validated metric emission. Registered
Qwen seed-0 protocols are accuracy `p-9b526ebb`, target survival
`p-c118ff26`, rerank fraction `p-5e454c00`, and kept fraction `p-8fe01567`.
Clean T1 run `r-3a41cfc710` observed accuracy `0.0`, target survival `1.0`,
rerank fraction `12/33`, and kept fraction `0.03266558864409933`. This is a
negative result: the few-shot continuation scorer does not recover the
lexicon upper bound even though the target remains in the retained set.

Commit `e52e3aa` adds `--diagnose-fewshot-scores`, which emits retained
candidate translation scores, raw continuation scores, leakage penalties, and
target ranks for the few-shot selector. After registering diagnostic protocols
`p-8a3ab56f`, `p-8c667c3a`, `p-f721d957`, and `p-e3c3c4e8`, clean Qwen T1
run `r-9396776753` observed selection accuracy `0.0`, target survival `1.0`,
mean final target rank `5.0`, and mean target score gap versus the best
non-target `-1.378227742513021`. The true target descriptors had zero
source-token leakage penalty, so the miss is not caused by over-penalizing
targets; raw continuation likelihood prefers source-word near-misses or
generic distractors.

Commit `e76cb35` adds
`forced_choice_descriptor_prefilter_compressed_lm_bridge`, a direct
candidate-comparison variant. It keeps the same bidirectional descriptor
top-12 first stage, lists the retained credential descriptors in one prompt,
and scores the label of each retained candidate as the best match for the
source descriptor. T0 smoke run `r-4b43da967a` validated metric emission
through the sweep wrapper; failed smoke `r-142fe81228` only used the wrong
`cd workspace` command shape. Registered Qwen seed-0 protocols are accuracy
`p-f736fa8d`, target survival `p-77309bb2`, rerank fraction `p-331c5046`,
and kept fraction `p-30cdecf4`. Clean T1 run `r-39d8d10cf9` observed
accuracy `0.0`, target survival `1.0`, rerank fraction `12/33`, kept fraction
`0.031885800746407504`, and mean first-stage target rank `4.0`. This is also
a negative result: direct label scoring over the retained list does not
recover the lexicon upper bound.

Commit `42dba68` adds `--diagnose-forced-choice-scores`, which emits the
forced-choice label likelihoods for the retained top-12 descriptors. T0 smoke
run `r-f35928b686` validated metric and stderr-trace emission. Registered
Qwen seed-0 diagnostic protocols are selection accuracy `p-d0839586`, final
target rank `p-fe9ee525`, target gap versus best non-target `p-7171f358`, and
target survival `p-a823922d`. Clean T1 run `r-c43db6e6c2` observed selection
accuracy `0.0`, target survival `1.0`, mean final target rank `5.25`, mean
target label ordinal `3.0`, mean selected label ordinal `0.5`, and mean
target gap versus best non-target `-2.9264373630285263`.

The per-item forced-choice traces show a consistent source-word near-miss
preference rather than a pure label-position artifact: Qwen selects
`storm loss inspector`, `hospital inventory organizer`,
`factory hazard reviewer`, and `school nutrition scheduler` over the true
targets `weather loss inspector`, `clinic inventory organizer`,
`workshop hazard reviewer`, and `cafeteria nutrition scheduler`. The selected
labels skew early, but the chosen descriptors are semantically coherent
source-token hybrids. Further prompt-only label-choice variants are unlikely
to recover the lexicon upper bound without adding a stronger target-side
translation or anti-copy signal.

Commit `91ce92c` adds
`pairwise_tournament_descriptor_prefilter_compressed_lm_bridge`, a
label-balanced pairwise comparison variant. It keeps the same bidirectional
descriptor top-12 first stage, then runs a tournament over retained
credential descriptors, comparing each challenger against the current
incumbent in both A/B orders before retaining the winner. T0 smoke run
`r-c2c82c7cdd` validated metric and diagnostic emission. Registered Qwen
seed-0 protocols are accuracy `p-8924a064`, target survival `p-a9c09bdc`,
rerank fraction `p-d682b89d`, and diagnostic selection accuracy
`p-49e84e73`. Clean T1 run `r-ed71b7001d` observed answer accuracy `0.0`,
diagnostic selection accuracy `0.0`, target survival `1.0`, rerank fraction
`12/33`, kept fraction `0.031885800746407504`, and mean first-stage target
rank `4.0`.

The pairwise traces remove list-label scoring as the main explanation: each
true target entered the tournament and lost its direct comparison to a
source-word near miss. The selected descriptors were `storm loss inspector`,
`hospital inventory organizer`, `factory hazard reviewer`, and
`school nutrition scheduler`. This branch is another prompt-level negative;
the next useful repair needs an explicit target-side translation or anti-copy
signal rather than another generic descriptor comparison prompt.

Commit `342ee70` adds
`source_copy_guarded_descriptor_prefilter_compressed_lm_bridge`, a non-oracle
anti-copy reranker over the same bidirectional top-12 set. It preserves the
first-stage descriptor score but subtracts a fixed penalty for retained
credential descriptors that copy source-side content words. T0 smoke
`r-0ea0a1514a` validated metric emission with zero distractors; harder tiny
GPT-2 smoke `r-dd09210387` exercised nontrivial candidate metrics, while
`r-638d94fb63` only failed because the tiny GPT-2 context window was exceeded.
Registered Qwen seed-0 protocols are accuracy `p-77935fd7`, target survival
`p-866345f3`, rerank fraction `p-8d39f86e`, and minimized kept fraction
`p-b3f5e446` (`p-6874cabb` is the same kept-fraction registration with the
wrong maximize direction and should not be cited). Clean T1 run
`r-ea0106fabd` observed accuracy `0.25`, target survival `1.0`, rerank
fraction `12/33`, kept fraction `0.03204175832594587`, and mean first-stage
target rank `4.0`.

The anti-copy guard fixed only item 1 (`weather loss inspector`) and failed
items 2-4. It therefore improves over the generic list/pairwise/few-shot
descriptor prompt variants at `0.0`, but does not beat existing partial
baselines such as QA bidirectional and compressed LM reranking at `0.25`, and
remains far below the verified lexicon upper bound of `1.0`. Treat this as a
partial negative: source-word leakage is one failure mode, but source-copy
penalty alone is not a sufficient target-side mapping signal.

Commit `252f1dd` adds
`induced_map_descriptor_prefilter_compressed_lm_bridge`, a diagnostic
non-oracle target-side mapping source. It pairs same-record-id source and
credential descriptors visible in the prompt, excluding the queried memo id,
induces position-wise source-token to target-token substitutions from those
in-context distractor pairs, and reranks the retained bidirectional top-12 by
target-token completeness minus source-token leakage. This does not call the
generator lexicon or `HELDOUT_DESCRIPTOR_PAIRS` for the queried descriptor.
T0 smoke `r-141148a9e0` validated metric emission; its workspace was dirty
because the harness auto-committed the implementation before the run, so cite
it only as plumbing evidence.

The exact Qwen hard-negative seed-0 command was registered before clean T1 run
`r-72068ca7cc`: primary accuracy `p-5383e94d`, target survival `p-e71713ba`,
rerank fraction `p-4ae94650`, minimized kept fraction `p-c269bebb`,
diagnostic target gap `p-d096d88f`, and source-token coverage `p-fcabb827`.
The primary accuracy protocol independently verified `1.0` with no issues or
red flags. Observed metrics: induced-map accuracy `1.0`, target survival
`1.0`, rerank fraction `12/33`, kept fraction `0.03219771590548423`, mean
first-stage target rank `4.0`, diagnostic final target rank `1.0`, diagnostic
target gap versus best non-target `3.25`, and source-token coverage `1.0`.
Same-run baselines: lexicon upper bound `1.0`, QA bidirectional `0.25`,
source-copy guarded `0.25`, compressed LM `0.25`, and generic
bidirectional/forced-choice/pairwise/few-shot/contrastive descriptor selectors
`0.0`.

The seed-1 robustness command was registered before clean T1 run
`r-fdabeeaa73`: primary accuracy `p-3b324ba4`, target survival `p-d3dd7274`,
minimized kept fraction `p-eccd4e8b`, diagnostic target gap `p-0da5c340`,
and source-token coverage `p-ab1ca51d`. Primary accuracy independently
verified `1.0` with no issues or red flags. Observed metrics: induced-map
accuracy `1.0`, target survival `1.0`, rerank fraction `12/33`, kept
fraction `0.032127967249218514`, mean first-stage target rank `3.75`,
diagnostic final target rank `1.0`, diagnostic target gap versus best
non-target `3.0`, and source-token coverage `1.0`. Same-run contrasts again
show the lexicon upper bound at `1.0`, QA/source-copy/compressed-LM partial
baselines at `0.25`, and generic descriptor comparison variants at `0.0`.

The seed-2 robustness command was registered before clean T1 run
`r-22327fbebf`: primary accuracy `p-367beffd`, target survival `p-4b7b0a4a`,
minimized kept fraction `p-9ac7aaea`, diagnostic final target rank
`p-18c85e89`, and diagnostic target gap `p-a0574bdc`. Primary accuracy
independently verified `1.0` with no issues or red flags. Observed metrics:
induced-map accuracy `1.0`, target survival `1.0`, rerank fraction `12/33`,
kept fraction `0.03218553183695408`, mean first-stage target rank `3.75`,
diagnostic final target rank `1.0`, diagnostic target gap versus best
non-target `3.25`, and source-token coverage `1.0`. Same-run contrasts show
the lexicon upper bound at `1.0`, compressed-LM and LM bridge baselines at
`0.5`, several partial baselines at `0.25`, and generic descriptor comparison
variants at `0.0`.

The less directly aligned `heldout_descriptor_bridge_v2` seed-0 stress command
was registered before clean T1 run `r-238728cd2b`: primary accuracy
`p-f4ada7a9`, target survival `p-902d9bd2`, source-token coverage
`p-fabd0670`, and diagnostic target gap `p-5bd44e50`. Primary accuracy
independently verified `0.75` with no issues or red flags. Observed metrics:
induced-map accuracy `0.75`, target survival `1.0`, kept fraction
`0.03259130616517959`, mean first-stage target rank `1.75`, diagnostic final
target rank `1.75`, source-token coverage `0.16666666666666666`, and target
gap versus best non-target `0.0`. Same-run contrasts: lexicon, bidirectional,
margin-guarded, pairwise, reverse descriptor, and oracle span reached `1.0`;
prefilter/compressed LM, compressed LM, LM bridge, forced-choice, and
source-copy guarded reached `0.75`; full, truncate, streaming, lexical,
semantic, QA bidirectional, and contrastive were `0.0`.

Interpretation: the retained top-12 evidence is sufficient when a non-oracle
in-context descriptor translation signal is extracted in the hard-negative
construction. The v2 stress result is not a clean induced-mapping win: on
three of four items the induced score tied at zero and inherited the first
stage, while only one item had partial source-token coverage. The next useful
step was to separate first-stage descriptor recall from the final induced-map
reranker on less supported descriptor prompts.

Follow-up diagnostic run `r-0290d4f1ae` at commit `12b30e4` added first-stage
counterfactual metrics to `--diagnose-induced-map-scores` and reran the same
Qwen v2 seed-0 cell. It observed first-stage top-1 selection accuracy `0.75`,
final induced-map diagnostic selection accuracy `0.75`, source-token coverage
`0.16666666666666666`, and first-stage target gap versus best non-target
`0.5310943007469189`. Protocol `p-60658841` independently verified the
first-stage top-1 accuracy at `0.75` with no issues or red flags.
Interpretation: the v2 combined-policy `0.75` is mostly a first-stage ranking
result, not evidence that induced substitutions solve unsupported descriptor
translation. The next useful method work is a mapping source that does not
depend on query-specific support pairs.

Commit `a3cea41` extends the same diagnostic with mechanism counters for
zero-coverage and first-stage inheritance. Clean Qwen T1 run `r-b7ec1559c0`
on the same v2 seed-0 cell observed zero source-token coverage on `0.75` of
items, zero-coverage selection success on `0.5` of items, no final induced-map
changes to the first-stage top choice (`0.0`), and no positive induced target
gap (`0.0`). Registered protocols: `p-9a922ca0`, `p-1ef1ef34`,
`p-257f591f`, and `p-354025a2`. Protocol `p-257f591f` independently
reproduced the final-changed-first-stage value `0.0` with no issues or red
flags; protocol `p-9a922ca0` independently reproduced zero source-token
coverage `0.75` with no issues or red flags; protocol `p-1ef1ef34`
independently reproduced zero-coverage selection success `0.5` with no issues
or red flags; and protocol `p-354025a2` independently reproduced the
no-positive-induced-target-gap value `0.0` with no issues or red flags.

Commit `1ac2fd8` adds
`strict_induced_map_descriptor_prefilter_compressed_lm_bridge`, which keeps
the same top-12 bidirectional first stage and induced descriptor-token score
but removes the first-stage score tie-breaker from final induced-map ranking.
T0 smoke `r-350b331ec1` validated metric emission. Protocols registered
before clean Qwen v2 seed-0 T1 run `r-4ce6c63efb` were strict accuracy
`p-64e38cb9` and strict target survival `p-cc8f10d6`. The run observed
strict accuracy `0.0`, target survival `1.0`, retained fraction
`0.03399561994071286`, and mean first-stage target rank `1.75`; accuracy
protocol `p-64e38cb9` independently reproduced `0.0` with no issues or red
flags. Same-run inherited-order induced-map accuracy remained `0.75`. This
turns the v2 caveat into verified policy evidence: without first-stage
tie inheritance, the induced-map final score contributes no answer accuracy
on this low-coverage cell.

Hard-negative seed-1 has now been rerun at workspace commit `d062956` to
check whether the generic bidirectional selector collapse was seed-0-specific.
Registered protocols before clean T1 run `r-c65d8ad073` were bidirectional
accuracy `p-5e03b4db`, bidirectional target survival `p-fce1b47f`, lexicon
upper-bound accuracy `p-ae179fc9`, and minimized lexicon kept fraction
`p-2b756065`. The bidirectional accuracy protocol independently reproduced
`0.0` with no issues or red flags. Same-run observed metrics preserved the
seed-0 pattern: bidirectional accuracy `0.0`, target survival `1.0`, lexicon
upper-bound accuracy `1.0`, lexicon kept fraction `0.032127967249218514`,
QA/compressed-LM/LM partial baselines `0.25`, and forced-choice, few-shot,
contrastive, and pairwise descriptor selectors `0.0`. The support protocols
are registered but not yet independently verified.

The T0 protocol uses `hf-internal-testing/tiny-random-gpt2` only to validate
that the harness, dependency lockfile, metric emission, and clean-checkout
command work. It is not evidence for or against the charter thesis. T1 triage
should switch `--model` to a small instruction model and increase item/context
counts once the smoke path is stable.

Commit `9a4c80a` adds `--diagnose-external-support-induced-map-scores` for
the task-external support map. T0 smoke `r-9e09ce8adc` validated the new
metric emission. Registered seed-0 Qwen diagnostic protocols were selection
accuracy `p-762b0212`, final target rank `p-06f46e88`, target gap
`p-fa1083cd`, and source-token coverage `p-e7e5263d`; clean T1 run
`r-2705edb2c6` observed diagnostic selection accuracy `0.75`, final target
rank `1.75`, target gap `0.0`, source-token coverage `0.25`, and target
survival `1.0`. Protocol `p-762b0212` independently reproduced diagnostic
selection accuracy `0.75` with no issues or red flags. Follow-up protocols
`p-fa1083cd` and `p-e7e5263d` independently reproduced target gap `0.0` and
source-token coverage `0.25`, respectively, also with no issues or red flags.
The recurring seed-0 miss is now localized to sparse support-map coverage plus
final-score ties: the target survives top-12, but the support map does not
create a positive target gap.

For real-model triage, prefer `--scoring rank`. It scores the known three-digit
record values by continuation log-probability instead of relying on exact
generation formatting, which makes base-model runs less brittle.

Use `--target-position {prefix,middle,recent,random}` to control where the
answer-bearing record is placed for single-record and adjacent two-hop tasks.
For shuffled descriptor tasks it controls the query alias placement, and the
answer-bearing profile is deliberately separated from that query line. The
placement metrics distinguish these cases. The initial T1 runs used the
default `prefix`, which is a favorable case for attention-sink retention;
position sweeps should include `middle` and `recent` before interpreting any
sink+recent result as a general frontier point.

`eval_kv_sweep.py` runs a compact position/budget grid while loading the model
once. Budgets are total retained tokens for both `truncate` and `streaming`;
`streaming` splits each budget into `--sink-tokens` plus `budget - sink` recent
tokens. The script emits stable per-cell metrics such as
`middle_b384_streaming_accuracy` and aggregate metrics such as
`mean_streaming_gap_vs_full`. It also emits oracle-control metrics such as
`middle_b384_oracle_span_accuracy` and `mean_oracle_span_gap_vs_full`. The
oracle span policy is not deployable because it uses the known target record;
it is a control for whether retaining the right middle span is sufficient to
recover quality at small retained fractions.

Lexical retrieval is a non-oracle synthetic baseline: it only uses the query
key visible in the question and record text. On this synthetic task it is
expected to be near-oracle; if it fails, the prompt reconstruction or scorer is
broken. If it succeeds, the next task should make record selection less
trivially exact-match.

Run through AAD, not directly, for citeable numbers:

```bash
aad exp run --bet bet-0006 --tier t0 --seed 0 --predict 1,1 -- \
  env UV_CACHE_DIR=.uv-cache uv run --frozen python eval_kv_frontier.py \
    --seed 0 --items 4 --distractors 8 --target-position middle \
    --max-new-tokens 2
```

Example sweep:

```bash
aad exp run --bet bet-0006 --tier t1 --seed 0 --predict 0.3,0.8 -- \
  env UV_CACHE_DIR=.uv-cache uv run --frozen python eval_kv_sweep.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --task kv --seed 0 --items 6 \
    --distractors 64 --target-positions prefix,middle,recent \
    --budgets 128,192,384 --sink-tokens 64 --scoring rank \
    --device cpu --dtype float32
```
