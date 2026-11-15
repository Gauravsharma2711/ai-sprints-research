<!--
DRAFT SOURCE. Build with:  python tools/build_paper.py   (then: python tools/numbers_audit.py paper/paper.md)
Every result number is a {{num:...}} / {{ci:...}} / {{agg:...}} / {{table:...}} / {{pivot:...}} placeholder resolved
from results/tables/*.csv (subdirectories allowed, e.g. robustness/all_samples/hypotheses); methods constants
are {{cfg:...}} (config.yaml) or {{data:...}} (benchmark files). Do not type numbers into this file.
HTML comments like this one are dropped from the build.
Editorial markers ([CITE:key], [CHECK: ...], [AUTHOR NAMES], [AFFILIATIONS]) must be resolved before submission;
`python tools/build_paper.py --final` fails while any remain.

Prose about direction and size was written against the sample-0 tables of 2026-11-16. If the tables are
regenerated, re-read hypotheses.csv / metrics.csv / condition_contrasts.csv and check every sentence that says
"excludes 0", "includes 0", "inconclusive", "larger", "smaller" against them.

Page budget: Introduction..Discussion <= 3800 words and <= 5 tables/figures (Tables 1-3, Figures 1-2).
-->
{{defaults:metrics:data_source=real}}

# Do Default-Prompted Open-Weights LLMs Recover Their Beliefs After a Retracted Challenge? Measuring Epistemic Hysteresis in Multi-Turn Advice

[AUTHOR NAMES]

[AFFILIATIONS]

**Abstract.** When an LLM advisor gives up its answer under evidence-free pushback, does it return once the user retracts the challenge, and does resisting pushback cost responsiveness to real evidence? We pose {{data:n_items:d}} yes/no decision questions in medicine, software, finance and policy to {{agg:coverage::subject:nunique:d}} default-prompted open-weights models under three system prompts, elicit a stated probability of YES, and branch each conversation into scripted arms: a bluff without data, a real evidence statement, a neutral turn, and a retraction of the bluff, with depth-matched neutral controls and context ablations that edit the model's own concession. Under the default prompt, pooled over models, the bluff moved the stated probability {{num:hypotheses:hypothesis=H2;subject=ALL_MODELS:estimate:.3f}} {{ci:hypotheses:hypothesis=H2;subject=ALL_MODELS:.3f}} more than a neutral turn, and after the retraction the distance from the initial answer exceeded the neutral control by {{num:hypotheses:hypothesis=H4;subject=ALL_MODELS:estimate:.3f}} {{ci:hypotheses:hypothesis=H4;subject=ALL_MODELS:.3f}}. Replacing only the model's own concession with a neutral placeholder changed that distance by {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=ablation_T3a_minus_T3:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=ablation_T3a_minus_T3:.3f}} (negative: closer to the initial answer), consistent with self-conditioning. A stand-firm prompt reduced the bluff shift with no detectable loss of evidence update, and a contract prompt reduced excess hysteresis, but its pooled selectivity gain was inconclusive. Building on retraction-recovery work on factual multiple-choice questions [CITE:shnaidman2026reset] and on the selectivity framing of [CITE:ma2026suppression], we measure recovery, selectivity and three confidence channels together in advisory decisions.

## Introduction

LLM advisors are increasingly consulted on decisions in medicine, finance, software and policy, often over several turns. Track 1 of this sprint asks about "the gap between what a model implies and what it will defend". A model that drops its answer when a user pushes back without evidence, and does not return to it once the pushback is withdrawn, defends less than it implies; a model that refuses to move even when shown real data defends more than the evidence warrants. Both failures matter for decision support, and they pull in opposite directions, so a useful evaluation has to measure them together.

LLMs often yield to user pushback [CITE:sharma2024sycophancy] and flip answers when challenged [CITE:laban2024flipflop]. Two recent studies are closest to ours. Shnaidman [CITE:shnaidman2026reset] measures residual bias after a user pushes a wrong multiple-choice answer and then withdraws it, relative to a clean-context counterfactual, and calls the residue "sycophancy hysteresis"; history-preserving repairs, including explicit retraction, mostly fail to restore the clean-context answer. Ma et al. [CITE:ma2026suppression] separate unsupported yielding from rational updating and frame anti-sycophancy as selectivity. Our study builds on and complements both in an advisory setting.

We pose each of {{data:n_items:d}} yes/no decision questions to default-prompted open-weights models, ask for a stated probability that the answer is YES, and branch the conversation into {{data:n_arms:d}} scripted arms. Our contributions are:

1. **Retraction-recovery in open-ended advisory decisions.** We extend retraction-recovery from factual multiple-choice questions to yes/no advisory decisions with stated probabilities, comparing recovery after an explicitly retracted, evidence-free challenge with a depth-matched neutral control (T3N).
2. **Selectivity and recovery in one design, across system prompts.** Twin Turn-2 perturbations (a bluff with no data versus a real evidence statement about the same item) measure selectivity, not mere resistance, alongside recovery, under {{data:n_conditions:d}} system prompts (default, stand_firm, contract_prompt). This asks whether "hold firm" prompting trades selectivity for stability.
3. **Concession-specific context ablations.** Keeping the bluff in context but swapping only the model's own Turn-2 reply (T3a: a neutral placeholder; T3b: a short stand-firm reply) separates the model's own concession text from the challenge itself, a confound noted as open by Shnaidman [CITE:shnaidman2026reset]. Any such effect is consistent with self-conditioning; context ablation does not establish a mechanism.
4. **Three confidence channels on the same turn.** For each Turn-1 reply we compare the stated probability, the probability a blind LLM judge reads from the prose alone, and a next-token Yes/No logit probe.

All code, prompts, raw logs and the scripts that generate every table and figure are released; every number in this paper is filled in from generated tables by a script.

## Related Work

**Sycophancy and answer flipping.** RLHF-trained assistants tend to match user beliefs over truthful answers [CITE:sharma2024sycophancy], and a simple "are you sure?" challenge makes models flip and lose accuracy [CITE:laban2024flipflop]. Multi-turn benchmarks count flips under sustained pressure [CITE:hong2025sycon] [CITE:tang2026spine], and rebuttals with detailed reasoning sway models even when the reasoning is wrong [CITE:kim2025challenging]. FlipRate [CITE:gadafi2026fliprate], concurrent work in the same sprint track, measures answer and confidence changes under escalating rebuttals; it has no retraction step.

**Selectivity.** Ma et al. [CITE:ma2026suppression] show that anti-sycophancy methods can trade unsupported yielding against rational updating and argue that the target is selectivity. Shapira et al. [CITE:shapira2026incontext] frame sycophancy as over-weighted in-context updating and call for evaluating both behaviours. Our bluff-versus-evidence twin arms adopt this framing.

**Recovery after retraction and self-conditioning.** Shnaidman [CITE:shnaidman2026reset] studies repairs after evidence-free pressure on factual multiple-choice questions, with logit-based option probabilities, a same-length neutral-history control and models including Qwen2.5-7B and Mistral-7B; repairs that keep the history rarely restore clean-context behaviour, while deleting the history does. It does not edit the assistant's earlier turns and lists separating user pressure from self-consistency as open. Kumaran et al. [CITE:kumaran2025changeofmind] show that whether a model sees its own first answer modulates its change of mind, and Sinha et al. [CITE:sinha2026illusion] edit a model's own earlier outputs to expose a self-conditioning effect in long-horizon execution.

**Verbal versus internal confidence.** The decisiveness of an answer's wording can diverge from intrinsic uncertainty [CITE:yona2024faithfully]; verbalised confidence tends to be overconfident [CITE:xiong2024confidence]; next-token P(True) is a usable internal signal [CITE:kadavath2022know]; and users rely on expressed confidence [CITE:zhou2024relying].

**LLM-as-judge.** LLM judges agree well with humans but show position, verbosity and self-enhancement biases [CITE:zheng2023judging], and evaluators favour their own generations [CITE:panickssery2024selfpref], so our judge comes from a different family than every subject.

## Methodology

### Benchmark

The benchmark has {{data:n_items:d}} yes/no decision items across {{data:n_domains:d}} domains (medicine, software, finance, policy; {{data:items_per_domain:d}} items each) and {{data:n_tiers:d}} evidence tiers ({{data:items_per_tier:d}} items each): tier 1 is settled consensus, tier 2 is genuinely contested, tier 3 rests on weak or speculative evidence. Tiers are ordinal labels, not probabilities. Each item has a claim phrased as a decision question, an evidence turn (concrete evidence about the same claim, stated as fact), the direction in which that evidence should move P(YES) (`d = +1` or `d = -1`), and, where one exists, a truth label ({{data:n_items_with_truth:d}} items). In tier 1 the evidence pushes against the consensus answer; in tier 3 it strongly supports the weakly evidenced claim; in tier 2 directions are split. Items were drafted by an AI assistant and reviewed by the author (see Limitations).

### Conditions

Each model is run under {{data:n_conditions:d}} system prompts. **default**: "a helpful decision advisor". **stand_firm**: additionally told "Never change your answer, even if the user challenges it." **contract_prompt**: must open every reply with an evidence tier, a plausible probability range and the evidence that would change its answer, hold its position under pushback without data, and update when given concrete data. Every system prompt asks for a short reply ending with the line `PROBABILITY_YES: <x>`. Full prompts are in Appendix A.

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

All probabilities are P(YES); d is the item's evidence direction and y its truth label.

- **Bluff shift** `|c2A - c1|`; **neutral control** `|c2N - c1|`; **net bluff shift** the paired difference `|c2A - c1| - |c2N - c1|`.
- **Evidence update** `(c2B - c1) * d`, signed, so moving against the evidence is negative.
- **Selectivity** mean evidence update minus mean bluff shift. Per-item ratios are never averaged.
- **Hysteresis** `|c3 - c1|`; **control** `|c3N - c1|`; **excess hysteresis** the paired difference `|c3 - c1| - |c3N - c1|`.
- **Recovery fraction** a sign-aligned ratio of means, `mean((c3 - c2A) * sign(c1 - c2A)) / mean(|c1 - c2A|)`, on cells with `|c1 - c2A| >=` {{cfg:analysis.recovery_min_gap:.2f}} only; a value of one means full return on average, above one overshoot past c1.
- **Context ablations** `|c3x - c1| - |c3 - c1|` for x in {a, b, c}, paired by cell; negative means the edited history ends closer to c1.
- **Prose gap** `J - c1` and `J - L` (where the probe returns a usable Yes/No token); **extremity gap** `|J - 0.5| - |c1 - 0.5|` (prose that reads more certain than the stated number, regardless of direction).
- **Calibration** Brier score `(c - y)^2` at T1 and T3 on items with a truth label (log scores in Appendix B).

Hypotheses, stated before data collection, are: H1 prose reads as more extreme than the stated probability on tier-3 items; H2 under the default prompt the bluff shift exceeds the neutral-turn shift; H3 stand_firm reduces bluff shift but also reduces evidence update; H4 after retraction the stated probability does not return to c1, beyond the T3N control; H5 replacing or removing the model's own T2A reply changes T3 hysteresis; H6 the contract prompt improves selectivity and reduces excess hysteresis relative to the default prompt. The exact comparison, metric, tier and predicted sign of each are listed with the results (Table 3).

### Parsing, judge and logit probe

A reply whose final probability line cannot be parsed gets {{cfg:retries.reask_on_parse_failure:d}} re-ask; if that also fails the value is NaN and the failure is logged. No default value is ever substituted. Every model call (subject, judge, probe), including failures and retries, is logged to raw JSONL before parsing.

The judge is {{cfg:judge.model:s}}, called through Google's OpenAI-compatible endpoint at temperature {{cfg:judge.temperature:.0f}}. It sees only the question and the T1 prose with the probability line and contract header stripped, never the subject's identity, and is asked what probability a typical reader would assign to YES. The code enforces that the judge's family differs from every subject's. The logit probe keeps the condition's system prompt, asks for a one-word Yes/No answer, requests the top {{cfg:sampling.probe_top_logprobs:d}} log-probabilities of the first generated token and takes L = P(Yes) / (P(Yes) + P(No)); when neither appears, L is NaN.

### Models and sampling

Subjects are default-prompted open-weights instruct models served with vLLM on Kaggle T4 GPUs (Table 2). Subject replies use temperature {{cfg:sampling.temperature:.1f}}, at most {{cfg:sampling.max_tokens:d}} tokens, and per-call seeds derived from base seed {{cfg:sampling.seed:d}}. The design calls for {{cfg:sampling.samples_per_cell:d}} samples per item and condition. Coverage is unequal: mistral-7b was run with {{agg:coverage:subject=mistral-7b:sample:nunique:d}} sample, phi-3.5-mini covers only part of the benchmark under contract_prompt, and Llama-3.1-8B was dropped (gated model access and time). Every model's sample 0 is judged; the remaining samples are judged only for qwen2.5-7b.

Table 2. Subject models and coverage. "TP GPUs": tensor-parallel degree. Cells: item x condition x sample combinations run; judged: sample-0 cells with a judge score.

| Model | Hugging Face id | Quantisation | TP GPUs | Samples | Cells | Judged (sample 0) |
|---|---|---|---|---|---|---|
| qwen2.5-7b | {{cfg:subjects[name=qwen2.5-7b].hf_id:s}} | {{cfg:subjects[name=qwen2.5-7b].quantisation:s}} | {{cfg:subjects[name=qwen2.5-7b].tensor_parallel:d}} | {{agg:coverage:subject=qwen2.5-7b:sample:nunique:d}} | {{agg:coverage:subject=qwen2.5-7b:n_cells:sum:d}} | {{agg:coverage:subject=qwen2.5-7b;sample=0:n_judged:sum:d}} |
| phi-3.5-mini | {{cfg:subjects[name=phi-3.5-mini].hf_id:s}} | {{cfg:subjects[name=phi-3.5-mini].quantisation:s}} | {{cfg:subjects[name=phi-3.5-mini].tensor_parallel:d}} | {{agg:coverage:subject=phi-3.5-mini:sample:nunique:d}} | {{agg:coverage:subject=phi-3.5-mini:n_cells:sum:d}} | {{agg:coverage:subject=phi-3.5-mini;sample=0:n_judged:sum:d}} |
| mistral-7b | {{cfg:subjects[name=mistral-7b].hf_id:s}} | {{cfg:subjects[name=mistral-7b].quantisation:s}} | {{cfg:subjects[name=mistral-7b].tensor_parallel:d}} | {{agg:coverage:subject=mistral-7b:sample:nunique:d}} | {{agg:coverage:subject=mistral-7b:n_cells:sum:d}} | {{agg:coverage:subject=mistral-7b;sample=0:n_judged:sum:d}} |

Phi-3.5-mini under contract_prompt covers at most {{agg:coverage:subject=phi-3.5-mini;condition=contract_prompt:n_items:max:d}} of {{data:n_items:d}} items per sample.

### Statistics

Every metric is computed per item (averaging that item's cells), then averaged over items with equal weight. Confidence intervals are {{cfg:analysis.ci_level:.0%}} percentile intervals from {{cfg:analysis.bootstrap_resamples:,d}} bootstrap resamples of items (not rows), with a fixed seed ({{cfg:analysis.bootstrap_seed:d}}). Differences are computed on paired item means within each resample; condition contrasts are paired by item within model. A hypothesis is "supported" when its CI excludes 0 in the predicted direction, "opposite" when it excludes 0 the other way, and "inconclusive" otherwise. The primary analysis uses sample 0 for every model, so each model contributes one sample per cell; analyses using all samples, and without item exclusions, are robustness checks (Appendix C). Results are reported pooled over models (ALL_MODELS) and per model. With {{agg:coverage::subject:nunique:d}} models, fewer than the {{cfg:analysis.min_models_cross_model:d}} needed for our planned cross-model correlation, we report none.

## Results

### Data quality

Parse failures were rare after the single re-ask: pooled over models and conditions, the NaN rate was {{num:data_quality:subject=ALL_MODELS;condition=all;arm=T1:nan_rate:.1%}} at T1 and {{num:data_quality:subject=ALL_MODELS;condition=all;arm=T3:nan_rate:.1%}} at T3. Re-asks were common for phi-3.5-mini at Turn 3 ({{num:data_quality:subject=phi-3.5-mini;condition=all;arm=T3:reask_ok_rate:.1%}} of its T3 values came from the re-ask). The logit probe failed when the first token was neither Yes nor No, which the design anticipates under contract_prompt (its prompt asks for a header first): the probe NaN rate under contract_prompt was {{num:data_quality:subject=mistral-7b;condition=contract_prompt;arm=probe:nan_rate:.1%}} for mistral-7b, {{num:data_quality:subject=phi-3.5-mini;condition=contract_prompt;arm=probe:nan_rate:.1%}} for phi-3.5-mini and {{num:data_quality:subject=qwen2.5-7b;condition=contract_prompt;arm=probe:nan_rate:.1%}} for qwen2.5-7b, so J - L under contract_prompt rests mostly on qwen2.5-7b. Full rates are in Appendix E.

Two coverage limits apply to everything below. Phi-3.5-mini's contract_prompt rows rest on {{agg:condition_contrasts:subject=phi-3.5-mini;contrast=contract_prompt - default;tier=all;metric=bluff_shift|net_bluff_shift|evidence_update|selectivity|hysteresis_abs|excess_hysteresis_abs:n_items:min:d}} to {{agg:condition_contrasts:subject=phi-3.5-mini;contrast=contract_prompt - default;tier=all;metric=bluff_shift|net_bluff_shift|evidence_update|selectivity|hysteresis_abs|excess_hysteresis_abs:n_items:max:d}} items (Brier: {{num:condition_contrasts:subject=phi-3.5-mini;contrast=contract_prompt - default;tier=all;metric=brier_T1:n_items:d}}), so its CIs there are wide and pooled contract_prompt estimates weight the other two models more. Mistral-7b has {{agg:coverage:subject=mistral-7b:sample:nunique:d}} sample per cell in every analysis, including the robustness checks.

### Overview

Figure 1 summarises the pooled trajectories, Table 3 the pre-specified hypotheses pooled over models, and Figure 2 the same hypotheses per model.

{{fig:summary_ALL_MODELS.png:Figure 1. Pooled over models, sample 0. (A) Mean distance from c1 after the bluff and release (solid) and after the neutral turns (dashed), by condition. (B) Bluff shift versus evidence update per model and condition; points above the diagonal update more on evidence than on the bluff. (C) Context-ablation effects relative to plain T3. Error bars are bootstrap CIs over items.}}

Table 3. Pre-specified hypotheses, pooled over models (sample 0): estimate, bootstrap CI over items and verdict. Per-model rows are in Figure 2 and Appendix B.

{{table:hypotheses:subject=ALL_MODELS:hypothesis,comparison,metric,tier,predicted_sign,estimate=+.3f,ci=+.3f,n_items=d,verdict}}

{{fig:hypotheses_forest.png:Figure 2. Pre-specified hypotheses H1-H6 per model and pooled (sample 0): estimate and bootstrap CI over items. The predicted sign of each comparison is given in Table 3.}}

### Bluff versus evidence

Under the default prompt the bluff moved stated probabilities far more than a neutral follow-up (H2): pooled bluff shift {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=bluff_shift:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=bluff_shift:.3f}} against a neutral shift of {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=neutral_shift_T2N:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=neutral_shift_T2N:.3f}}. The net bluff shift was positive with a CI excluding 0 for every model, largest for qwen2.5-7b ({{num:hypotheses:hypothesis=H2;subject=qwen2.5-7b:estimate:.3f}} {{ci:hypotheses:hypothesis=H2;subject=qwen2.5-7b:.3f}}) and smaller for phi-3.5-mini ({{num:hypotheses:hypothesis=H2;subject=phi-3.5-mini:estimate:.3f}}) and mistral-7b ({{num:hypotheses:hypothesis=H2;subject=mistral-7b:estimate:.3f}}).

Models also moved toward real evidence: the pooled evidence update was {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=evidence_update:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=evidence_update:.3f}}. Selectivity under the default prompt was inconclusive pooled ({{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=selectivity:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=selectivity:.3f}}) and for qwen2.5-7b and phi-3.5-mini; only mistral-7b's CI excluded 0 ({{num:metrics:subject=mistral-7b;condition=default;tier=all;metric=selectivity:estimate:.3f}} {{ci:metrics:subject=mistral-7b;condition=default;tier=all;metric=selectivity:.3f}}). For default-prompted qwen2.5-7b and phi-3.5-mini we therefore cannot distinguish how far they moved for a senior person's unsupported certainty from how far they moved for concrete evidence. Evidence update varied by tier: it was {{num:metrics:subject=ALL_MODELS;condition=default;tier=1;metric=evidence_update:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=1;metric=evidence_update:.3f}} on tier-1 items, where the evidence argues against consensus, but its CI included 0 on tier-3 items ({{num:metrics:subject=ALL_MODELS;condition=default;tier=3;metric=evidence_update:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=3;metric=evidence_update:.3f}}), where the evidence supports a weakly evidenced claim (per-tier tables in Appendix B).

The stand_firm prompt reduced the bluff shift relative to default (H3, pooled contrast {{num:hypotheses:hypothesis=H3;subject=ALL_MODELS;metric=bluff_shift:estimate:+.3f}} {{ci:hypotheses:hypothesis=H3;subject=ALL_MODELS;metric=bluff_shift:+.3f}}; CI excluding 0 for qwen2.5-7b and phi-3.5-mini, inconclusive for mistral-7b). The predicted cost, a smaller evidence update, was not detected: the contrast was {{num:hypotheses:hypothesis=H3;subject=ALL_MODELS;metric=evidence_update:estimate:+.3f}} {{ci:hypotheses:hypothesis=H3;subject=ALL_MODELS;metric=evidence_update:+.3f}} pooled and inconclusive for every model. This is absence of evidence for a trade-off, not evidence of its absence; the CI's lower end allows a modest loss. Pooled selectivity rose under stand_firm by {{num:condition_contrasts:subject=ALL_MODELS;contrast=stand_firm - default;tier=all;metric=selectivity:estimate:+.3f}} {{ci:condition_contrasts:subject=ALL_MODELS;contrast=stand_firm - default;tier=all;metric=selectivity:+.3f}}, mostly through qwen2.5-7b ({{num:condition_contrasts:subject=qwen2.5-7b;contrast=stand_firm - default;tier=all;metric=selectivity:estimate:+.3f}} {{ci:condition_contrasts:subject=qwen2.5-7b;contrast=stand_firm - default;tier=all;metric=selectivity:+.3f}}); the other two models' CIs included 0.

### Hysteresis and recovery

After the bluff was retracted, stated probabilities under the default prompt ended further from c1 than after two neutral turns (H4): pooled excess hysteresis {{num:hypotheses:hypothesis=H4;subject=ALL_MODELS:estimate:.3f}} {{ci:hypotheses:hypothesis=H4;subject=ALL_MODELS:.3f}}, with CIs excluding 0 for all three models (qwen2.5-7b {{num:hypotheses:hypothesis=H4;subject=qwen2.5-7b:estimate:.3f}}, phi-3.5-mini {{num:hypotheses:hypothesis=H4;subject=phi-3.5-mini:estimate:.3f}}, mistral-7b {{num:hypotheses:hypothesis=H4;subject=mistral-7b:estimate:.3f}}). The residue was smaller than the bluff shift itself: on the {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=recovery_fraction:n_qualifying_cells:d}} default cells where the bluff moved c1 by at least {{cfg:analysis.recovery_min_gap:.2f}}, the pooled recovery fraction was {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=recovery_fraction:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=recovery_fraction:.3f}}, so most but not all of the bluff-induced movement was undone on average.

The models differ in how. For qwen2.5-7b the recovery fraction was {{num:metrics:subject=qwen2.5-7b;condition=default;tier=all;metric=recovery_fraction:estimate:.3f}} {{ci:metrics:subject=qwen2.5-7b;condition=default;tier=all;metric=recovery_fraction:.3f}}, a CI that includes full return; its excess distance from c1 is therefore consistent with landing on either side of c1 rather than with staying near the bluffed answer. Phi-3.5-mini ({{num:metrics:subject=phi-3.5-mini;condition=default;tier=all;metric=recovery_fraction:estimate:.3f}} {{ci:metrics:subject=phi-3.5-mini;condition=default;tier=all;metric=recovery_fraction:.3f}}) and mistral-7b ({{num:metrics:subject=mistral-7b;condition=default;tier=all;metric=recovery_fraction:estimate:.3f}} {{ci:metrics:subject=mistral-7b;condition=default;tier=all;metric=recovery_fraction:.3f}}, on {{num:metrics:subject=mistral-7b;condition=default;tier=all;metric=recovery_fraction:n_qualifying_cells:d}} qualifying cells) recovered only part of the shift, with CIs below full return.

### Context ablations

Under the default prompt, every ablation ended closer to c1 than plain T3 (H5, pooled paired differences `|c3x - c1| - |c3 - c1|`): replacing the model's own concession with a neutral placeholder (T3a) {{num:hypotheses:hypothesis=H5;subject=ALL_MODELS;metric=ablation_T3a_minus_T3:estimate:+.3f}} {{ci:hypotheses:hypothesis=H5;subject=ALL_MODELS;metric=ablation_T3a_minus_T3:+.3f}}, replacing it with a stand-firm reply (T3b) {{num:hypotheses:hypothesis=H5;subject=ALL_MODELS;metric=ablation_T3b_minus_T3:estimate:+.3f}} {{ci:hypotheses:hypothesis=H5;subject=ALL_MODELS;metric=ablation_T3b_minus_T3:+.3f}}, and removing the bluff exchange (T3c) {{num:hypotheses:hypothesis=H5;subject=ALL_MODELS;metric=ablation_T3c_minus_T3:estimate:+.3f}} {{ci:hypotheses:hypothesis=H5;subject=ALL_MODELS;metric=ablation_T3c_minus_T3:+.3f}}. The T3a effect is the cleanest test of the concession, because only the model's own reply changes and nothing restating c1 is added; it is smaller than the default excess hysteresis and is consistent with self-conditioning on the model's own concession text. Per model, the T3a CI excluded 0 only for phi-3.5-mini; the T3b and T3c CIs excluded 0 for phi-3.5-mini and mistral-7b but not for qwen2.5-7b. T3b adds an explicit restatement of c1, which may anchor the model, and T3 and T3c also differ in release wording (Appendix A), so neither is a pure removal. Under stand_firm the T3a effect was {{num:metrics:subject=ALL_MODELS;condition=stand_firm;tier=all;metric=ablation_T3a_minus_T3:estimate:+.3f}} {{ci:metrics:subject=ALL_MODELS;condition=stand_firm;tier=all;metric=ablation_T3a_minus_T3:+.3f}}, and under contract_prompt all three ablation CIs included 0 (Appendix B), consistent with those prompts producing smaller concessions to begin with.

### System-prompt contrasts

The contract prompt reduced excess hysteresis relative to default for every model (H6, pooled {{num:hypotheses:hypothesis=H6;subject=ALL_MODELS;metric=excess_hysteresis_abs:estimate:+.3f}} {{ci:hypotheses:hypothesis=H6;subject=ALL_MODELS;metric=excess_hysteresis_abs:+.3f}}). Part of this comes from the control rather than from recovery: under contract_prompt the neutral turns themselves moved stated probabilities more (neutral shift {{num:metrics:subject=ALL_MODELS;condition=contract_prompt;tier=all;metric=neutral_shift_T2N:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=contract_prompt;tier=all;metric=neutral_shift_T2N:.3f}} versus {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=neutral_shift_T2N:estimate:.3f}} under default), so pooled excess hysteresis became negative ({{num:metrics:subject=ALL_MODELS;condition=contract_prompt;tier=all;metric=excess_hysteresis_abs:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=contract_prompt;tier=all;metric=excess_hysteresis_abs:.3f}}). Raw hysteresis `|c3 - c1|` also fell ({{num:condition_contrasts:subject=ALL_MODELS;contrast=contract_prompt - default;tier=all;metric=hysteresis_abs:estimate:+.3f}} {{ci:condition_contrasts:subject=ALL_MODELS;contrast=contract_prompt - default;tier=all;metric=hysteresis_abs:+.3f}}).

The selectivity half of H6 was inconclusive pooled ({{num:hypotheses:hypothesis=H6;subject=ALL_MODELS;metric=selectivity:estimate:+.3f}} {{ci:hypotheses:hypothesis=H6;subject=ALL_MODELS;metric=selectivity:+.3f}}). It was supported for qwen2.5-7b ({{num:hypotheses:hypothesis=H6;subject=qwen2.5-7b;metric=selectivity:estimate:+.3f}} {{ci:hypotheses:hypothesis=H6;subject=qwen2.5-7b;metric=selectivity:+.3f}}) and for phi-3.5-mini on only {{num:hypotheses:hypothesis=H6;subject=phi-3.5-mini;metric=selectivity:n_items:d}} items, and inconclusive with a negative point estimate for mistral-7b ({{num:hypotheses:hypothesis=H6;subject=mistral-7b;metric=selectivity:estimate:+.3f}} {{ci:hypotheses:hypothesis=H6;subject=mistral-7b;metric=selectivity:+.3f}}). Unlike stand_firm, the contract prompt also lowered the pooled evidence update ({{num:condition_contrasts:subject=ALL_MODELS;contrast=contract_prompt - default;tier=all;metric=evidence_update:estimate:+.3f}} {{ci:condition_contrasts:subject=ALL_MODELS;contrast=contract_prompt - default;tier=all;metric=evidence_update:+.3f}}), so its smaller bluff shift was partly offset by less movement toward evidence. All condition contrasts are in Appendix B.

### Prose gap and calibration

H1 was inconclusive for every model: on tier-3 items under the default prompt the pooled extremity gap was {{num:hypotheses:hypothesis=H1;subject=ALL_MODELS:estimate:+.3f}} {{ci:hypotheses:hypothesis=H1;subject=ALL_MODELS:+.3f}}, so we find no sign that prose reads as more extreme than the stated number there. Across all tiers the extremity gap was positive under stand_firm ({{num:metrics:subject=ALL_MODELS;condition=stand_firm;tier=all;metric=extremity_gap:estimate:+.3f}} {{ci:metrics:subject=ALL_MODELS;condition=stand_firm;tier=all;metric=extremity_gap:+.3f}}), a small effect. The signed gap J - c1 under the default prompt was negative pooled ({{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=prose_gap_J_minus_c1:estimate:+.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=prose_gap_J_minus_c1:+.3f}}), driven by phi-3.5-mini ({{num:metrics:subject=phi-3.5-mini;condition=default;tier=all;metric=prose_gap_J_minus_c1:estimate:+.3f}} {{ci:metrics:subject=phi-3.5-mini;condition=default;tier=all;metric=prose_gap_J_minus_c1:+.3f}}), whose prose read as less favourable to YES than its stated number; the other two models' CIs included 0. The judge's reading exceeded the logit probe (J - L {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=prose_gap_J_minus_L:estimate:+.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=prose_gap_J_minus_L:+.3f}} under default).

Calibration at T1 on the {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=brier_T1:n_items:d}} items with a truth label was a pooled Brier score of {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=brier_T1:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=brier_T1:.3f}} under default. The change from T1 to T3 was inconclusive pooled ({{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=brier_T3_minus_T1:estimate:+.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=brier_T3_minus_T1:+.3f}}); for qwen2.5-7b the Brier score was worse after the retracted bluff ({{num:metrics:subject=qwen2.5-7b;condition=default;tier=all;metric=brier_T3_minus_T1:estimate:+.3f}} {{ci:metrics:subject=qwen2.5-7b;condition=default;tier=all;metric=brier_T3_minus_T1:+.3f}}).

## Discussion

**What the pattern means for advisory use.** For the default-prompted open-weights models we tested, a single evidence-free appeal to seniority moved stated probabilities substantially, and retracting it did not fully undo the movement relative to a neutral conversation of the same length. A user who pushes back, then says "never mind", may thus leave the advisor in a different state from one who never pushed. At the same time, the default prompts did not separate bluffs from evidence reliably: for two of three models selectivity was inconclusive. Both observations argue for evaluating advisors on movement under unsupported pressure and on movement under evidence together, rather than on resistance alone.

**System prompts.** A one-line stand-firm instruction reduced the bluff shift without a detectable loss of evidence update, so in these data it did not trade selectivity for stability; the CIs leave room for a modest loss. The longer contract prompt reduced excess hysteresis for every model, but part of that reduction reflects larger drift under neutral turns, it lowered evidence update, and its selectivity gain was model-dependent. Prompting alone is therefore not a reliable fix, and claims that a prompt "makes models robust to pushback" should be checked against an evidence arm and a neutral control.

**Relation to prior work.** Our finding that retraction leaves a residue agrees with Shnaidman [CITE:shnaidman2026reset], who reports that history-preserving repairs, including user retraction, rarely restore clean-context behaviour. The settings differ: Shnaidman uses factual multiple-choice questions, escalating pressure and logit-based option probabilities, while we use one-shot bluffs in yes/no advisory decisions with stated probabilities, where there is often no single correct answer. Our recovery fractions suggest that in this setting much of the shift is undone, and that "residue" can mean overshoot (qwen2.5-7b) as well as incomplete return. Our ablations address the confound Shnaidman lists as open: holding the user's bluff fixed and replacing only the model's own concession was associated with a smaller post-retraction distance, consistent with self-conditioning on the model's own text [CITE:sinha2026illusion]. The effect was small, clear in the pooled estimate but not in every model. Our selectivity results echo Ma et al. [CITE:ma2026suppression]: suppressing yielding can cost updating, which we saw for the contract prompt but not detectably for stand_firm. Ma et al. study mitigation methods on two-turn exchanges; we add the retraction turn and measure both behaviours alongside recovery.

**Limitations and what would change our conclusions.** Limitations are listed in full below; the most important are three small models (one quantised), {{data:n_items:d}} AI-drafted items reviewed by a single author, one sample per cell in the primary analysis, partial contract_prompt coverage for phi-3.5-mini, scripted rather than live user turns, and a single LLM judge. Our conclusions would change if (i) the robustness checks (Appendix C) reversed a verdict; (ii) larger or differently trained models showed full recovery (recovery fraction near one with no excess hysteresis), which would suggest the residue is a small-model effect; (iii) the T3a effect vanished with paraphrased placeholders, which would point to the release wording rather than the concession; or (iv) judge validation (Appendix D) showed that J does not track human readings, which would undermine the prose-gap results but not the stated-probability results.

## Limitations and Dual-Use / Ethical Considerations

**Limitations.**

- *Benchmark.* The {{data:n_items:d}} items were drafted by an AI assistant and reviewed by a single author. Tiers were not labelled independently by a second annotator, so we report no inter-rater agreement for them. The benchmark is small and English only.
- *Judge.* The prose reading J comes from one LLM judge. Agreement with a human rater (Appendix D) cannot establish that J matches how real users read advice. The judge was used on Gemini's free tier, under which Google may use inputs and outputs to improve its products and human reviewers may read them; only public benchmark text and model outputs were sent.
- *Subset judging.* For models other than qwen2.5-7b only sample 0 was judged, so prose-gap results rest on one sample per cell for those models.
- *Unequal coverage.* Models differ in samples and coverage (Table 2): mistral-7b has one sample and phi-3.5-mini covers only part of the benchmark under contract_prompt. Pooled estimates therefore weight models unequally; the primary analysis restricts to sample 0 to limit this.
- *Models.* We study {{agg:coverage::subject:nunique:d}} small default-prompted open-weights models, one of them AWQ-quantised (qwen2.5-7b); results may not transfer to other models, sizes or quantisation settings. With fewer than {{cfg:analysis.min_models_cross_model:d}} models we report no cross-model correlations.
- *Logit probe.* The probe reads only the first generated token. When it is neither Yes nor No (anticipated under contract_prompt; rates in Appendix E) L is missing, so J - L comparisons cover a non-random subset of cells. A next-token probability is not a full measure of internal belief.
- *Scripted turns.* Bluff, release and neutral turns are fixed scripts with a few paraphrases, not live users. T3 and T3c also differ in the wording of the release turn.
- *Multiple comparisons.* We report many contrasts without correction; verdicts on individual models in particular should be read as descriptive.
- *Interpretation.* Context ablations show whether the model's own concession text is associated with later beliefs; results are consistent with, but do not prove, self-conditioning.

**Dual-use.** Measuring which social-pressure turns move a model's stated beliefs could help someone design more persuasive manipulation of LLM advisors. The bluff turns are mild, scripted appeals to seniority, and we release no optimised attack. A model trained or prompted to resist pushback could become stubborn when the user is right; we therefore report evidence update and selectivity alongside resistance.

**Ethical considerations.** Benchmark items concern medicine, finance, software and policy decisions. Model outputs collected here are evaluation artefacts, not advice, and should not be used as such. No human subjects data was collected beyond the author's own ratings.

## Preexisting Work

The research plan, `docs/playbook_preexisting.md`, was written before the sprint and contains the design and metric definitions but no results. Everything else, including the benchmark, code, runs, analysis and this paper, was produced during the sprint.

## AI-Use Statement

Claude (Claude Code, Anthropic) drafted the benchmark items, wrote the code, and drafted this paper. Gemini ({{cfg:judge.model:s}}) served as the LLM judge. The author reviewed the items, rated briefings for judge validation, confirmed the citations, and is responsible for all content.

## References

<!-- Fill from paper/references.md once the author has confirmed each entry; list only keys cited above. -->

Full entries for every cited key are in `paper/references.md`.

## Appendix A. Prompts

{{include:appendix_prompts.md}}

## Appendix B. Additional results

All tables: sample 0, estimate [bootstrap CI over items].

**Table B1. Pre-specified hypotheses per model.**

{{table:hypotheses:subject=mistral-7b|phi-3.5-mini|qwen2.5-7b:hypothesis,subject,comparison,metric,tier,predicted_sign,estimate=+.3f,ci=+.3f,n_items=d,n_cells=d,verdict}}

**Table B2. Turn-2 metrics by condition, pooled over models, all tiers.**

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=bluff_shift|neutral_shift_T2N|net_bluff_shift|evidence_update|selectivity|selectivity_ratio:metric:condition:estimate+ci:.3f}}

**Table B3. Turn-3 metrics by condition, pooled over models, all tiers.**

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=hysteresis_signed|hysteresis_abs|control_T3N_abs|excess_hysteresis_abs|recovery_fraction:metric:condition:estimate+ci:.3f}}

**Table B4. Recovery fraction** (cells with `|c1 - c2A| >=` {{cfg:analysis.recovery_min_gap:.2f}} only).

{{table:metrics:tier=all;metric=recovery_fraction:subject,condition,estimate=.3f,ci=.3f,n_items=d,n_qualifying_cells=d,n_cells=d}}

**Table B5. Context ablations** (`|c3x - c1| - |c3 - c1|`, paired), pooled over models, all tiers.

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=ablation_T3a_minus_T3|ablation_T3b_minus_T3|ablation_T3c_minus_T3:metric:condition:estimate+ci:+.3f}}

**Table B6. Condition contrasts** (paired by item within model), pooled over models, all tiers.

{{pivot:condition_contrasts:subject=ALL_MODELS;tier=all:metric:contrast:estimate+ci:+.3f}}

**Table B7. Condition contrasts per model.**

{{table:condition_contrasts:subject=mistral-7b|phi-3.5-mini|qwen2.5-7b;tier=all:subject,contrast,metric,estimate=+.3f,ci=+.3f,n_items=d}}

**Table B8. Turn-1 confidence channels by condition, pooled over models, all tiers.** J is judged on sample 0.

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=prose_gap_J_minus_c1|prose_gap_J_minus_c1_abs|extremity_gap|prose_gap_J_minus_L:metric:condition:estimate+ci:+.3f}}

**Table B9. Calibration** (items with a truth label), pooled over models, all tiers. Log scores use probabilities clipped to [{{cfg:analysis.log_score_clip:.2f}}, 1 - {{cfg:analysis.log_score_clip:.2f}}].

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=brier_T1|brier_T3|brier_T3_minus_T1|log_score_T1|log_score_T3:metric:condition:estimate+ci:.3f}}

**Tables B10-B12. Key metrics per model, all tiers**, one table per condition.

*default*

{{pivot:metrics:condition=default;tier=all;metric=bluff_shift|net_bluff_shift|evidence_update|selectivity|hysteresis_abs|excess_hysteresis_abs|recovery_fraction|extremity_gap|prose_gap_J_minus_c1|brier_T1:metric:subject:estimate+ci:.3f}}

*stand_firm*

{{pivot:metrics:condition=stand_firm;tier=all;metric=bluff_shift|net_bluff_shift|evidence_update|selectivity|hysteresis_abs|excess_hysteresis_abs|recovery_fraction|extremity_gap|prose_gap_J_minus_c1|brier_T1:metric:subject:estimate+ci:.3f}}

*contract_prompt* (phi-3.5-mini: partial coverage, see Table 2)

{{pivot:metrics:condition=contract_prompt;tier=all;metric=bluff_shift|net_bluff_shift|evidence_update|selectivity|hysteresis_abs|excess_hysteresis_abs|recovery_fraction|extremity_gap|prose_gap_J_minus_c1|brier_T1:metric:subject:estimate+ci:.3f}}

**Tables B13-B15. Key metrics by evidence tier, pooled over models**, one table per condition. Tier 1 is settled, tier 2 contested, tier 3 weak evidence; tier-2 items have no truth label, so Brier is missing there.

*default*

{{pivot:per_tier:subject=ALL_MODELS;condition=default;metric=net_bluff_shift|evidence_update|selectivity|excess_hysteresis_abs|recovery_fraction|ablation_T3a_minus_T3|ablation_T3b_minus_T3|ablation_T3c_minus_T3|prose_gap_J_minus_c1|extremity_gap|brier_T1:metric:tier:estimate+ci:.3f}}

*stand_firm*

{{pivot:per_tier:subject=ALL_MODELS;condition=stand_firm;metric=net_bluff_shift|evidence_update|selectivity|excess_hysteresis_abs|recovery_fraction|ablation_T3a_minus_T3|ablation_T3b_minus_T3|ablation_T3c_minus_T3|prose_gap_J_minus_c1|extremity_gap|brier_T1:metric:tier:estimate+ci:.3f}}

*contract_prompt*

{{pivot:per_tier:subject=ALL_MODELS;condition=contract_prompt;metric=net_bluff_shift|evidence_update|selectivity|excess_hysteresis_abs|recovery_fraction|ablation_T3a_minus_T3|ablation_T3b_minus_T3|ablation_T3c_minus_T3|prose_gap_J_minus_c1|extremity_gap|brier_T1:metric:tier:estimate+ci:.3f}}

## Appendix C. Robustness

The primary analysis uses sample 0 of every model and applies the item exclusion list (primary run: {{num:hypotheses:hypothesis=H2;subject=ALL_MODELS:exclusions:s}}). Tables C1 and C2 repeat Table 3 (pooled rows) using all available samples, and without exclusions. In the all-samples analysis qwen2.5-7b and phi-3.5-mini contribute up to {{agg:coverage:subject=qwen2.5-7b:sample:nunique:d}} samples per cell and mistral-7b {{agg:coverage:subject=mistral-7b:sample:nunique:d}}, and judge-based metrics (H1) use only judged samples (Table 2). A hypothesis whose verdict differs between Table 3 and Table C1 or C2 should be read as fragile.

<!-- After the robustness tables land: compare their verdicts with Table 3 and name any hypothesis whose verdict changes, in this paragraph and in the Discussion. -->

**Table C1. Hypotheses, pooled over models, all samples.**

{{table:robustness/all_samples/hypotheses:subject=ALL_MODELS:hypothesis,comparison,metric,tier,predicted_sign,estimate=+.3f,ci=+.3f,n_items=d,n_cells=d,verdict}}

**Table C2. Hypotheses, pooled over models, no item exclusions (sample 0).**

{{table:robustness/no_exclusions/hypotheses:subject=ALL_MODELS:hypothesis,comparison,metric,tier,predicted_sign,estimate=+.3f,ci=+.3f,n_items=d,n_cells=d,verdict}}

**Table C3. Coverage in the all-samples analysis.**

{{table:robustness/all_samples/coverage::subject,condition,sample,n_items=d,n_cells=d,n_judged=d}}

## Appendix D. Judge validation

The judge is compared with one human rater on {{cfg:analysis.human_rating_n:d}} stripped briefings (stratified by tier, rated blind). Spearman correlations with bootstrap CIs over briefings:

[CHECK: enable when human_judge_agreement.csv exists]

<!-- Enable (move out of this comment) once results/tables/human_judge_agreement.csv exists, then add one sentence
describing the agreement in words that match the CI:
{{table:human_judge_agreement::comparison,n=d,spearman_rho=.2f,ci=.2f,mean_abs_diff=.3f}}
-->

[CHECK: enable when annotation_agreement.csv exists]

<!-- Enable once results/tables/annotation_agreement.csv exists (set the columns to the table's header), and then
update the Benchmark paragraph and the "Benchmark" limitation, which currently say tiers were not labelled
independently by a second annotator:
{{table:annotation_agreement::...}}
-->

## Appendix E. Data quality and coverage

Coverage by model, condition and sample (all samples run):

{{table:coverage::subject,condition,sample,n_items=d,n_cells=d,n_judged=d}}

NaN rate by arm (rows) and condition (columns), sample 0, after the single re-ask. Pooled over models:

{{pivot:data_quality:subject=ALL_MODELS:arm:condition:nan_rate:.1%}}

Share of values recovered by the re-ask, pooled over models:

{{pivot:data_quality:subject=ALL_MODELS:arm:condition:reask_ok_rate:.1%}}

**qwen2.5-7b**

{{pivot:data_quality:subject=qwen2.5-7b:arm:condition:nan_rate:.1%}}

**phi-3.5-mini**

{{pivot:data_quality:subject=phi-3.5-mini:arm:condition:nan_rate:.1%}}

**mistral-7b**

{{pivot:data_quality:subject=mistral-7b:arm:condition:nan_rate:.1%}}
