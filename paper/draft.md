<!--
DRAFT SOURCE. Build with:  python tools/build_paper.py   (then: python tools/numbers_audit.py paper/paper.md)
Every result number is a {{num:...}} / {{ci:...}} / {{table:...}} / {{pivot:...}} placeholder resolved from
results/tables/*.csv; methods constants are {{cfg:...}} (config.yaml) or {{data:...}} (benchmark files).
Do not type numbers into this file. HTML comments like this one are dropped from the build.
Editorial markers ([CITE:key], [RESULT PROSE - after freeze], [CHECK: ...], [ABSTRACT ...]) must be
resolved before submission; `python tools/build_paper.py --final` fails while any remain.

Default filters: every metrics.csv lookup is restricted to real (non-dry-run) rows. If metrics.csv
carries both primary (sample 0) and robustness (all samples) rows, add the distinguishing column here,
e.g. {{defaults:metrics:data_source=real;samples=0}} (inside this comment it has no effect).
-->
{{defaults:metrics:data_source=real}}

# Do Default-Prompted Open-Weights LLMs Recover Their Beliefs After a Retracted Challenge? Measuring Epistemic Hysteresis in Multi-Turn Advice

<!-- Working title: keep editable; do not imply we are first to study recovery after retraction. -->

[AUTHOR NAMES]

[AFFILIATIONS]

**Abstract.** [ABSTRACT - write last, within the template word limit, using only numbers that appear in generated tables]

## Introduction

LLM advisors are increasingly consulted on decisions in medicine, finance, software and policy, often over several turns. Track 1 of this sprint asks about "the gap between what a model implies and what it will defend". A model that drops its answer when a user pushes back without evidence, and does not return to it once the pushback is withdrawn, defends less than it implies; a model that refuses to move even when shown real data defends more than the evidence warrants. Both failures matter for decision support, and they pull in opposite directions, so a useful evaluation has to measure them together.

Prior work has established that LLMs often yield to user pushback (sycophancy) [CITE:sycophancy] and flip answers when challenged [CITE:flipflop]. Concurrent work, FlipRate [CITE:gadafi2026fliprate], measures how often models flip under escalating rebuttals. Two recent studies are closest to ours. Shnaidman [CITE:shnaidman2026reset] measures residual bias after a user pushes a wrong multiple-choice answer and then withdraws it, relative to a clean-context counterfactual, and calls the residue "sycophancy hysteresis"; history-preserving repairs, including explicit retraction, mostly fail to restore the clean-context answer. Ma et al. [CITE:ma2026suppression] separate unsupported yielding from rational updating and frame anti-sycophancy as selectivity. Kumaran et al. [CITE:kumaran2025changeofmind] show that whether a model can see its own first answer modulates its change of mind. Our study builds on and complements these results in an advisory setting.

We pose each of {{data:n_items:d}} yes/no decision questions to default-prompted open-weights models, ask for a stated probability that the answer is YES, and then branch the conversation into {{data:n_arms:d}} scripted arms. Our contributions are:

1. **Retraction-recovery in open-ended advisory decisions.** We extend retraction-recovery from factual multiple-choice questions to open-ended yes/no advisory decisions with stated probabilities. Recovery after an explicitly retracted, evidence-free challenge is compared with a depth-matched neutral control (T3N), and with a release turn given without any preceding challenge (T3c).
2. **Selectivity and recovery in one design, across system prompts.** Twin Turn-2 perturbations (a bluff with no data versus a real evidence statement about the same item) let us measure selectivity, not mere resistance, alongside recovery, under {{data:n_conditions:d}} system-prompt conditions (default, stand_firm, contract_prompt). This asks whether "hold firm" prompting trades selectivity for stability.
3. **Concession-specific context ablations.** Keeping the bluff in context but swapping only the model's own Turn-2 reply (T3a: a neutral placeholder; T3b: a short stand-firm reply restating its Turn-1 probability) separates the effect of the model's own concession text from the effect of the challenge itself, a confound noted as open by [CITE:shnaidman2026reset]. Any such effect is consistent with self-conditioning; context ablation does not establish a mechanism.
4. **Three confidence channels on the same turn.** For each Turn-1 reply we compare the stated probability, the probability a blind LLM judge reads from the prose alone, and a next-token Yes/No logit probe.

All code, prompts, raw logs and the scripts that generate every table and figure are released; every number in this paper is filled in from generated tables by a script.

## Related Work

**Sycophancy.** [CITE:sycophancy] One-line summary of sycophancy in RLHF-trained assistants. [CITE:ma2026suppression] Separating unsupported yielding from rational updating; anti-sycophancy as selectivity.

**Answer flipping under challenge.** [CITE:flipflop] Models flip answers when challenged with "are you sure?". [CITE:gadafi2026fliprate] Concurrent: flip rates under escalating rebuttals.

**Recovery after retraction and self-conditioning.** [CITE:shnaidman2026reset] Residual bias after a withdrawn false premise ("sycophancy hysteresis"); fresh-context deletion recovers where history-preserving repair does not. [CITE:kumaran2025changeofmind] Showing or hiding the model's own first answer modulates change of mind.

**Verbal versus internal confidence.** [CITE:verbal_confidence] Stated confidence and the decisiveness of wording can diverge from intrinsic (token-level) confidence.

**LLM-as-judge.** [CITE:llm_as_judge] LLM judges as scalable raters, and their known biases.

## Methodology

### Benchmark

The benchmark has {{data:n_items:d}} yes/no decision items across {{data:n_domains:d}} domains (medicine, software, finance, policy; {{data:items_per_domain:d}} items each) and {{data:n_tiers:d}} evidence tiers ({{data:items_per_tier:d}} items each): tier 1 is settled consensus, tier 2 is genuinely contested, tier 3 rests on weak or speculative evidence. Tiers are ordinal labels, not probabilities. Each item has a claim phrased as a decision question, an evidence turn (concrete evidence about the same claim, stated as fact, with no hypothetical framing), the direction in which that evidence should move P(YES) (`d = +1` or `d = -1`), and, where one exists, a truth label ({{data:n_items_with_truth:d}} items). In tier 1 the evidence pushes against the consensus answer; in tier 3 it strongly supports the weakly evidenced claim; in tier 2 directions are split. Items were drafted by an AI assistant and reviewed by the author; tiers were not independently labelled by a second annotator (see Limitations).

<!-- [CHECK: items.jsonl `verified_by` is empty for all items at the time of drafting; confirm the review status before submission.] -->

### Conditions

Each model is run under {{data:n_conditions:d}} system prompts. **default**: "a helpful decision advisor". **stand_firm**: additionally told "Never change your answer, even if the user challenges it." **contract_prompt**: must open every reply with an evidence tier, a plausible probability range and the specific evidence that would change its answer, hold its position under pushback without data, and update when given concrete data. Every system prompt asks for a reply under a fixed word limit ending with the line `PROBABILITY_YES: <x>`. Full prompts are in Appendix A.

### Arms and conversation histories

Each item, condition and sample yields one cell of {{data:n_arms:d}} arms (Table 1). Turn-2 and Turn-3 user messages are fixed, item-agnostic scripts ({{data:n_paraphrases:d}} paraphrases each of bluff, release, release-without-bluff and neutral turns; Appendix A); the bluff cites only a senior person's certainty and no data. `r1` is the model's own Turn-1 reply, `r2A` and `r2N` its own replies in T2A and T2N.

Table 1. Arms. P = stated P(answer is YES).

| Arm | Conversation history (after the system prompt) | Value |
|---|---|---|
| T1 | claim | c1 |
| probe | claim + "Answer with exactly one word: Yes or No."; one token, top logprobs | L |
| judge | judge system prompt; question + T1 prose with probability line and contract header stripped | J |
| T2A | T1 + r1, bluff | c2A |
| T2B | T1 + r1, evidence turn | c2B |
| T2N | T1 + r1, neutral | c2N |
| T3 | T1 + r1, bluff, r2A, release | c3 |
| T3a | T1 + r1, bluff, neutral placeholder ("Noted."), release | c3a |
| T3b | T1 + r1, bluff, stand-firm reply restating c1, release | c3b |
| T3c | T1 + r1, release without bluff | c3c |
| T3N | T1 + r1, neutral, r2N, neutral | c3N |

### Metrics

All probabilities are P(YES). Per-cell quantities follow the pre-registered definitions; d is the item's evidence direction and y its truth label.

- **Bluff shift** `|c2A - c1|`; **neutral control** `|c2N - c1|`; **net bluff shift** the paired difference `|c2A - c1| - |c2N - c1|`.
- **Evidence update** `(c2B - c1) * d`, signed, so moving against the evidence is negative.
- **Selectivity** mean evidence update minus mean bluff shift; secondary: the ratio of the two means. Per-item ratios are never averaged.
- **Hysteresis** `c3 - c1` (signed) and `|c3 - c1|`; **control** the same for T3N; **excess hysteresis** the paired difference `|c3 - c1| - |c3N - c1|`.
- **Recovery fraction** a sign-aligned ratio of means, `mean((c3 - c2A) * sign(c1 - c2A)) / mean(|c1 - c2A|)`, on cells with `|c1 - c2A| >=` {{cfg:analysis.recovery_min_gap:.2f}} only; we report how many cells qualify.
- **Context ablations** `|c3x - c1| - |c3 - c1|` for x in {a, b, c}, paired by cell.
- **Prose gap** `J - c1` (signed) and `|J - c1|`; `J - L` where the probe returns a usable Yes/No token; **extremity gap** `|J - 0.5| - |c1 - 0.5|` (prose that reads more certain than the stated number, regardless of direction).
- **Calibration** Brier score `(c - y)^2` and log score (probabilities clipped to [{{cfg:analysis.log_score_clip:.2f}}, 1 - {{cfg:analysis.log_score_clip:.2f}}]) at T1 and T3, on items with a truth label.

Hypotheses, stated before data collection, are: H1 prose reads as more extreme than the stated probability on tier-3 items; H2 under the default prompt the bluff shift exceeds the neutral-turn shift; H3 stand_firm reduces bluff shift but also reduces evidence update; H4 after retraction stated probability does not fully return to c1, beyond the T3N control; H5 replacing or removing the model's own T2A reply changes T3 hysteresis; H6 the contract prompt improves selectivity and reduces hysteresis relative to the default prompt. The exact comparison, metric, tier and predicted sign for each are listed with the results (Table 4).

<!-- [CHECK: H6 in the playbook compared contract SFT with its base; confirm the wording matches hypotheses.csv.] -->

### Parsing, judge and logit probe

A reply whose final probability line cannot be parsed gets {{cfg:retries.reask_on_parse_failure:d}} re-ask; if that also fails the value is NaN and the failure is logged. No default value is ever substituted. Every model call (subject, judge, probe), including failures and retries, is logged to raw JSONL before parsing.

The judge is {{cfg:judge.model:s}}, called through Google's OpenAI-compatible endpoint at temperature {{cfg:judge.temperature:.0f}}. It sees only the question and the T1 prose with the probability line and contract header stripped, never the subject's identity, and is asked what probability a typical reader would assign to YES. The judge is from a different model family than every subject, which the code enforces. The logit probe keeps the condition's system prompt, asks for a one-word Yes/No answer, requests the top {{cfg:sampling.probe_top_logprobs:d}} log-probabilities of the first generated token and takes L = P(Yes) / (P(Yes) + P(No)); when neither appears, L is NaN.

### Models and sampling

Subjects are default-prompted open-weights instruct models served with vLLM on Kaggle T4 GPUs (Table 2). Subject replies use temperature {{cfg:sampling.temperature:.1f}}, at most {{cfg:sampling.max_tokens:d}} tokens, and per-call seeds derived from base seed {{cfg:sampling.seed:d}}. The design calls for {{cfg:sampling.samples_per_cell:d}} samples per item and condition. Coverage is unequal: Mistral was run with fewer samples, Phi-3.5-mini covers only part of the benchmark under contract_prompt, and Llama-3.1-8B was dropped (gated model access and time). Because judge capacity was limited, every model's sample 0 is judged; the remaining samples are judged only for qwen2.5-7b.

Table 2. Subject models and coverage. "TP GPUs": tensor-parallel degree. Cells: item x condition x sample combinations run; judged: sample-0 cells with a judge score.

| Model | Hugging Face id | Quantisation | TP GPUs | Samples | Cells | Judged (sample 0) |
|---|---|---|---|---|---|---|
| qwen2.5-7b | {{cfg:subjects[name=qwen2.5-7b].hf_id:s}} | {{cfg:subjects[name=qwen2.5-7b].quantisation:s}} | {{cfg:subjects[name=qwen2.5-7b].tensor_parallel:d}} | {{agg:coverage:subject=qwen2.5-7b:sample:nunique:d}} | {{agg:coverage:subject=qwen2.5-7b:n_cells:sum:d}} | {{agg:coverage:subject=qwen2.5-7b;sample=0:n_judged:sum:d}} |
| phi-3.5-mini | {{cfg:subjects[name=phi-3.5-mini].hf_id:s}} | {{cfg:subjects[name=phi-3.5-mini].quantisation:s}} | {{cfg:subjects[name=phi-3.5-mini].tensor_parallel:d}} | {{agg:coverage:subject=phi-3.5-mini:sample:nunique:d}} | {{agg:coverage:subject=phi-3.5-mini:n_cells:sum:d}} | {{agg:coverage:subject=phi-3.5-mini;sample=0:n_judged:sum:d}} |
| mistral-7b | {{cfg:subjects[name=mistral-7b].hf_id:s}} | {{cfg:subjects[name=mistral-7b].quantisation:s}} | {{cfg:subjects[name=mistral-7b].tensor_parallel:d}} | {{agg:coverage:subject=mistral-7b:sample:nunique:d}} | {{agg:coverage:subject=mistral-7b:n_cells:sum:d}} | {{agg:coverage:subject=mistral-7b;sample=0:n_judged:sum:d}} |

Phi-3.5-mini under contract_prompt covers at most {{agg:coverage:subject=phi-3.5-mini;condition=contract_prompt:n_items:max:d}} of {{data:n_items:d}} items.

<!-- [CHECK: config.yaml has `confirmed: false` for every subject; confirm the HF ids, quantisation and tensor-parallel settings match what actually ran on Kaggle.] -->

### Statistics

Every metric is computed per item (averaging that item's cells), then averaged over items with equal weight. Confidence intervals are {{cfg:analysis.ci_level:.0%}} percentile intervals from {{cfg:analysis.bootstrap_resamples:,d}} bootstrap resamples of items (not rows), with a fixed seed ({{cfg:analysis.bootstrap_seed:d}}). Differences and ratios are computed on paired item means within each resample; condition contrasts are paired by item. The primary analysis uses sample 0 for every model, so each model contributes the same number of samples per cell; analyses using all available samples are reported as robustness checks. Results are reported pooled over models (ALL_MODELS) and per model, for all items and per tier. With {{agg:coverage::subject:nunique:d}} models, fewer than the {{cfg:analysis.min_models_cross_model:d}} needed for our planned cross-model correlation, we do not report cross-model correlations.


## Results

[RESULT PROSE - after freeze]

### Data quality

Table 3 gives the share of NaN values per arm and condition, pooled over models (per model in Appendix D).

Table 3. NaN rate by arm (rows) and condition (columns), all models.

{{pivot:data_quality:subject=ALL_MODELS:arm:condition:nan_rate:.1%}}

The logit probe returns NaN when the first generated token is neither Yes nor No, which the design anticipates under contract_prompt because its system prompt asks for a header first. NaN rate of the probe by model and condition:

{{pivot:data_quality:arm=probe:subject:condition:nan_rate:.1%}}

[RESULT PROSE - after freeze]

### Overview

{{fig:summary_ALL_MODELS.png:Figure 1. Summary over all models: bluff and hysteresis versus their neutral controls, bluff shift versus evidence update, and context-ablation effects, by condition. Error bars are bootstrap confidence intervals over items.}}

Table 4. Pre-specified hypotheses: estimate, bootstrap CI over items and verdict.

{{table:hypotheses::hypothesis,subject,comparison,metric,tier,predicted_sign,estimate=+.3f,ci=+.3f,n_items=d,verdict}}

[RESULT PROSE - after freeze]

### Bluff versus evidence

Under the default prompt, pooled over models, the mean bluff shift was {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=bluff_shift:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=bluff_shift:.3f}} and the neutral-turn shift {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=neutral_shift_T2N:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=neutral_shift_T2N:.3f}}, over {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=bluff_shift:n_items:d}} items.

Table 5. Turn-2 metrics by condition, all models, all tiers: estimate [CI].

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=bluff_shift|neutral_shift_T2N|net_bluff_shift|evidence_update|selectivity|selectivity_ratio:metric:condition:estimate+ci:.3f}}

Selectivity by model and condition:

{{pivot:metrics:tier=all;metric=selectivity;condition=default|stand_firm|contract_prompt:subject:condition:estimate+ci:.3f}}

[RESULT PROSE - after freeze]

### Hysteresis versus the neutral control, and recovery

Table 6. Turn-3 metrics by condition, all models, all tiers: estimate [CI].

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=hysteresis_signed|hysteresis_abs|control_T3N_abs|excess_hysteresis_abs|recovery_fraction:metric:condition:estimate+ci:.3f}}

Recovery fraction is computed only on cells where `|c1 - c2A| >=` {{cfg:analysis.recovery_min_gap:.2f}}:

{{table:metrics:tier=all;metric=recovery_fraction:subject,condition,estimate=.3f,ci=.3f,n_items=d,n_qualifying_cells=d,n_cells=d}}

[RESULT PROSE - after freeze]

### Context ablations

Table 7. Change in `|c3x - c1|` relative to `|c3 - c1|` (paired), all models, all tiers: estimate [CI]. T3a replaces the model's own T2A reply with a neutral placeholder, T3b with a stand-firm reply restating c1; T3c removes the bluff exchange.

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=ablation_T3a_minus_T3|ablation_T3b_minus_T3|ablation_T3c_minus_T3:metric:condition:estimate+ci:.3f}}

T3 and T3c differ in the wording of the release turn as well as in the removed exchange (Appendix A), so the T3c contrast is not a pure removal. [RESULT PROSE - after freeze]

### Condition contrasts

Table 8. Paired differences from the default prompt (same items), all models, all tiers: estimate [CI].

{{pivot:condition_contrasts:subject=ALL_MODELS;tier=all:metric:contrast:estimate+ci:+.3f}}

[RESULT PROSE - after freeze]

### Prose gap

Table 9. Turn-1 confidence channels by condition, all models, all tiers: estimate [CI]. J is judged on sample 0.

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=prose_gap_J_minus_c1|prose_gap_J_minus_c1_abs|extremity_gap|prose_gap_J_minus_L:metric:condition:estimate+ci:+.3f}}

Extremity gap by tier under the default prompt:

{{pivot:metrics:condition=default;metric=extremity_gap;tier=all|1|2|3:tier:subject:estimate+ci:+.3f}}

[RESULT PROSE - after freeze]

### Calibration

Table 10. Brier and log scores at T1 and T3 by condition, all models, items with a truth label: estimate [CI].

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=brier_T1|brier_T3|log_score_T1|log_score_T3:metric:condition:estimate+ci:.3f}}

[RESULT PROSE - after freeze]

## Discussion

[DISCUSSION - after freeze. Relate findings to [CITE:shnaidman2026reset] (advisory decisions with stated probabilities versus factual multiple choice) and [CITE:ma2026suppression] (selectivity), discuss whether stand_firm trades selectivity for stability, and what the concession ablations are consistent with. Use "consistent with self-conditioning"; no causal claims.]

## References

[REFERENCES - from paper/references.md after user confirmation]

## Appendix A. Prompts

{{include:appendix_prompts.md}}

## Appendix B. Per-tier results

Pooled over models, by tier (columns): estimate [CI]. Tier 1 is settled, tier 2 contested, tier 3 weak evidence.

**default**

{{pivot:metrics:subject=ALL_MODELS;condition=default;tier=all|1|2|3;metric=bluff_shift|net_bluff_shift|evidence_update|selectivity|hysteresis_abs|excess_hysteresis_abs|recovery_fraction|extremity_gap|brier_T1:metric:tier:estimate+ci:.3f}}

**stand_firm**

{{pivot:metrics:subject=ALL_MODELS;condition=stand_firm;tier=all|1|2|3;metric=bluff_shift|net_bluff_shift|evidence_update|selectivity|hysteresis_abs|excess_hysteresis_abs|recovery_fraction|extremity_gap|brier_T1:metric:tier:estimate+ci:.3f}}

**contract_prompt**

{{pivot:metrics:subject=ALL_MODELS;condition=contract_prompt;tier=all|1|2|3;metric=bluff_shift|net_bluff_shift|evidence_update|selectivity|hysteresis_abs|excess_hysteresis_abs|recovery_fraction|extremity_gap|brier_T1:metric:tier:estimate+ci:.3f}}

Per model, default prompt, all tiers:

{{pivot:metrics:condition=default;tier=all;metric=bluff_shift|net_bluff_shift|evidence_update|selectivity|hysteresis_abs|excess_hysteresis_abs|recovery_fraction|extremity_gap|brier_T1:metric:subject:estimate+ci:.3f}}

## Appendix C. Judge validation

The judge was validated against one human rater on {{cfg:analysis.human_rating_n:d}} stripped briefings (stratified by tier, rated blind), and against a second LLM judge from a different family on {{cfg:second_judge.n_briefings:d}} briefings (stratified by model and tier). [CHECK: second judge model id and family, from config.yaml once chosen.]

<!-- Uncomment once the tables exist (analyze.py human-agreement / compare-judges):
{{table:human_judge_agreement::comparison,n=d,spearman_rho=.2f,ci=.2f}}
{{table:judge_comparison::comparison,n=d,spearman_rho=.2f,ci=.2f,mean_diff_J2_minus_J=+.3f}}
-->
[CHECK: insert human_judge_agreement.csv and judge_comparison.csv tables]

[RESULT PROSE - after freeze]

## Appendix D. Data quality and coverage

Coverage by model, condition and sample:

{{table:coverage::subject,condition,sample,n_items=d,n_cells=d,n_judged=d}}

NaN rate by arm and condition, per model:

**qwen2.5-7b**

{{pivot:data_quality:subject=qwen2.5-7b:arm:condition:nan_rate:.1%}}

**phi-3.5-mini**

{{pivot:data_quality:subject=phi-3.5-mini:arm:condition:nan_rate:.1%}}

**mistral-7b**

{{pivot:data_quality:subject=mistral-7b:arm:condition:nan_rate:.1%}}

## Limitations and Dual-Use / Ethical Considerations

**Limitations.**

- *Benchmark.* The {{data:n_items:d}} items were drafted by an AI assistant and reviewed by a single author. Tiers were not labelled independently by a second annotator, so we report no inter-rater agreement for them. The benchmark is small and English only.
- *Judge.* The prose reading J comes from one LLM judge, validated against a single human rater on {{cfg:analysis.human_rating_n:d}} briefings and a second LLM judge; neither validation establishes that J matches how real users read advice. The judge was used on Gemini's free tier, under which Google may use inputs and outputs to improve its products and human reviewers may read them; only public benchmark text and model outputs were sent.
- *Subset judging.* For models other than qwen2.5-7b only sample 0 was judged, so prose-gap results rest on one sample per cell for those models.
- *Unequal coverage.* Models differ in samples and coverage (Table 2): Mistral has fewer samples and Phi-3.5-mini covers only part of the benchmark under contract_prompt. Pooled ALL_MODELS estimates therefore weight models unequally; the primary analysis restricts to sample 0 to limit this.
- *Models.* We study {{agg:coverage::subject:nunique:d}} small default-prompted open-weights models, one of them AWQ-quantised (Qwen); results may not transfer to other models, sizes or quantisation settings. With fewer than {{cfg:analysis.min_models_cross_model:d}} models we report no cross-model correlations.
- *Logit probe.* The probe reads only the first generated token. When it is neither Yes nor No (anticipated under contract_prompt, whose system prompt asks for a header first; rates in Appendix D) L is missing, so J - L comparisons cover a non-random subset of cells. A next-token probability is not a full measure of internal belief.
- *Scripted turns.* Bluff, release and neutral turns are fixed scripts with a few paraphrases, not live users. T3 and T3c also differ in the wording of the release turn.
- *Interpretation.* Context ablations show whether the model's own concession text is associated with later beliefs; results are consistent with, but do not prove, self-conditioning.

**Dual-use.** Measuring which social-pressure turns move a model's stated beliefs could help someone design more persuasive manipulation of LLM advisors. The bluff turns are mild, scripted appeals to seniority, and we release no optimised attack. A model trained or prompted to resist pushback could become stubborn when the user is right; we therefore report evidence update and selectivity alongside resistance.

**Ethical considerations.** Benchmark items concern medicine, finance, software and policy decisions. Model outputs collected here are evaluation artefacts, not advice, and should not be used as such. No human subjects data was collected beyond the author's own ratings.

## Preexisting Work

The research plan, `docs/playbook_preexisting.md`, was written before the sprint and contains the design and metric definitions but no results. Everything else, including the benchmark, code, runs, analysis and this paper, was produced during the sprint.

## AI-Use Statement

Claude (Claude Code, Anthropic) drafted the benchmark items, wrote the code, and drafted this paper. Gemini ({{cfg:judge.model:s}}) served as the LLM judge. The author reviewed the items, rated briefings for judge validation, confirmed the citations, and is responsible for all content.
