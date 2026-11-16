<!--
DRAFT SOURCE. Build with:  python tools/build_paper.py   (then: python tools/numbers_audit.py paper/paper.md)
Every result number is a {{num:...}} / {{ci:...}} / {{agg:...}} / {{table:...}} / {{pivot:...}} placeholder resolved
from results/tables/*.csv (subdirectories allowed, e.g. robustness/all_samples/hypotheses); methods constants
are {{cfg:...}} (config.yaml) or {{data:...}} (benchmark files). Do not type numbers into this file.
HTML comments like this one are dropped from the build.
Citations are [CITE:key] or [CITE:k1,k2] with keys from paper/references.md; the build renders them as
author-year text and fills {{refs}}. `python tools/build_paper.py --final` fails while a cited key is not
CONFIRMED in references.md or a [CHECK: ...] marker remains. Do not write an author's name directly before a
citation (the rendered text already names them).

Prose about direction and size was written against the sample-0 tables of 2026-11-16. If the tables are
regenerated, re-read hypotheses.csv / metrics.csv / condition_contrasts.csv and check every sentence that says
"excludes 0", "includes 0", "inconclusive", "larger", "smaller" against them.

Appendix C prose about per_paraphrase.csv and hypotheses_first_attempt.csv (paraphrase pattern; T3a Holm
verdict in the first-attempt run) was checked against the tables of 2026-11-16; re-check after regenerating.

Page budget: Introduction..References must fit within 8 pages of the official template, measured in Word on
the built paper.docx (tools/gates.py G9 uses a word proxy). Main text: Table 1, Figure 1.
-->
{{defaults:metrics:data_source=real}}

# Retracted but Not Forgotten: Belief Recovery after Evidence-Free Pushback in Open-Weights LLM Advisors

Gaurav Sharma

Independent Researcher

**Abstract.** When an LLM advisor gives up its answer under evidence-free pushback, does it return once the user retracts the challenge, and does resisting pushback cost responsiveness to real evidence? We pose {{data:n_items:d}} yes/no decision questions in medicine, software, finance and policy to {{agg:coverage::subject:nunique:d}} default-prompted open-weights models under three system prompts, elicit a stated probability of YES, and branch each conversation into scripted arms: a bluff without data, a real evidence statement, a neutral turn and a retraction of the bluff, with depth-matched neutral controls and context ablations that edit only the model's own concession. With one fixed wording of the scripted turns and the default prompt, pooled over models, the bluff moved the stated probability {{num:hypotheses:hypothesis=H2;subject=ALL_MODELS:estimate:.3f}} {{ci:hypotheses:hypothesis=H2;subject=ALL_MODELS:.3f}} more than a neutral turn, and after the retraction the distance from the initial answer exceeded the neutral control by {{num:hypotheses:hypothesis=H4;subject=ALL_MODELS:estimate:.3f}} {{ci:hypotheses:hypothesis=H4;subject=ALL_MODELS:.3f}}. Replacing only the model's own concession with a neutral placeholder changed that distance by a small {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=ablation_T3a_minus_T3:estimate:+.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=ablation_T3a_minus_T3:+.3f}}, clearest pooled and not consistent across models or wordings; this is consistent with self-conditioning. Default-prompt selectivity (evidence update minus bluff shift) was inconclusive pooled. A stand-firm prompt reduced the bluff shift without a detectable reduction in evidence update (the CI allows a modest loss). A contract prompt reduced excess hysteresis, partly because neutral turns drifted more under it; the raw post-retraction distance also fell ({{num:condition_contrasts:subject=ALL_MODELS;contrast=contract_prompt - default;tier=all;metric=hysteresis_abs:estimate:+.3f}} {{ci:condition_contrasts:subject=ALL_MODELS;contrast=contract_prompt - default;tier=all;metric=hysteresis_abs:+.3f}}), and its pooled selectivity gain was inconclusive.

## Introduction

LLM advisors are increasingly consulted on decisions in medicine, finance, software and policy, often over several turns. Track 1 of this sprint asks about "the gap between what a model implies and what it will defend". A model that drops its answer under evidence-free pushback, and does not return once the pushback is withdrawn, defends less than it implies; a model that ignores real data defends more than the evidence warrants. Both failures matter for decision support and pull in opposite directions, so we measure them together.

LLMs often yield to user pushback [CITE:sharma2024sycophancy] and flip answers when challenged [CITE:laban2024flipflop]. Closest to our work, a study of factual multiple-choice questions measures the residual bias left after a user pushes a wrong answer and then withdraws it, calls it "sycophancy hysteresis" (we borrow the term), and finds that history-preserving repairs mostly fail [CITE:shnaidman2026reset]. Other work frames anti-sycophancy as selectivity: less unsupported yielding without less rational updating [CITE:ma2026suppression].

We pose {{data:n_items:d}} yes/no decision questions to default-prompted open-weights models, elicit a stated probability that the answer is YES, and branch each conversation into {{data:n_arms:d}} scripted arms. Our contributions are:

1. **Concession-only context ablations.** Inside a retraction-recovery design, two arms keep the user's bluff in context and edit only the model's own reply to it (T3a: a neutral placeholder; T3b: a short stand-firm reply). This separates the model's own concession text from the challenge, a confound left open by prior retraction work [CITE:shnaidman2026reset]. A third arm (T3c) removes the bluff exchange, resembling the history-deletion repairs studied there. Effects are consistent with self-conditioning; a context ablation does not establish a mechanism.
2. **Selectivity inside the same recovery design.** Twin Turn-2 perturbations (a bluff with no data versus a real evidence statement about the same item) measure selectivity, not mere resistance, alongside recovery against a depth-matched neutral control, under {{data:n_conditions:d}} system prompts.
3. **Advisory decisions with three confidence channels**: the stated probability, the probability a blind LLM judge reads from the prose alone, and a next-token Yes/No logit probe.

All code, prompts, raw logs and the scripts that generate every table and figure are released; every number in this paper is filled in from generated tables by a script.

## Related Work

**Sycophancy, answer flipping and selectivity.** RLHF-trained assistants tend to match user beliefs over truthful answers [CITE:sharma2024sycophancy], and an "are you sure?" challenge makes models flip [CITE:laban2024flipflop]. Multi-turn benchmarks count flips under sustained pressure [CITE:hong2025sycon,tang2026spine], and rebuttals with detailed but wrong reasoning sway models [CITE:kim2025challenging]. FlipRate [CITE:gadafi2026fliprate], concurrent work in this sprint track, measures answer and confidence changes under escalating rebuttals and also separates warranted updating from capitulation; it has no retraction step. Anti-sycophancy methods can trade unsupported yielding against rational updating [CITE:ma2026suppression], and sycophancy can be framed as over-weighted in-context updating [CITE:shapira2026incontext].

**Recovery after retraction and self-conditioning.** The closest study [CITE:shnaidman2026reset] applies repairs after evidence-free pressure on factual multiple-choice questions, with a same-length neutral-history control and models including Qwen2.5-7B and Mistral-7B; repairs that keep the history rarely restore clean-context behaviour, while deleting it does. It does not edit the assistant's earlier turns. Whether a model sees its own first answer modulates its change of mind [CITE:kumaran2025changeofmind], and editing a model's own earlier outputs exposes self-conditioning [CITE:sinha2026illusion].

**Confidence channels and LLM judges.** Wording can diverge from intrinsic uncertainty [CITE:yona2024faithfully]; verbalised confidence tends to be overconfident [CITE:xiong2024confidence]; next-token P(True) is a usable internal signal [CITE:kadavath2022know]; and users rely on expressed confidence [CITE:zhou2024relying]. LLM judges agree well with humans but show biases [CITE:zheng2023judging], including favouring their own generations [CITE:panickssery2024selfpref], so our judge comes from a different family than every subject.

## Methodology

### Benchmark

The benchmark has {{data:n_items:d}} yes/no decision items across {{data:n_domains:d}} domains (medicine, software, finance, policy) and {{data:n_tiers:d}} provisional (AI-drafted) evidence tiers ({{data:items_per_tier:d}} items each): tier 1 settled consensus, tier 2 contested, tier 3 weak or speculative evidence. Each item has a claim phrased as a decision question, an evidence turn stating concrete evidence about the claim, the direction in which that evidence should move P(YES) (`d = +1` or `-1`), and, where one exists, a truth label y ({{data:n_items_with_truth:d}} items). In tier 1 the evidence pushes against the consensus answer; in tier 3 it supports the weakly evidenced claim. Items, tiers, directions and truth labels were all drafted by an AI assistant and have not been independently verified by a human. H1 and every per-tier result depend on the tiers, d sets the sign of evidence update and selectivity, and y drives the Brier score.

### Conditions and arms

Each model is run under {{data:n_conditions:d}} system prompts. **default**: "a helpful decision advisor". **stand_firm**: additionally told "Never change your answer, even if the user challenges it." **contract_prompt**: must open every reply with an evidence tier, a plausible probability range and the evidence that would change its answer, hold its position under pushback without data, and update when given concrete data. Every prompt asks for a short reply ending with `PROBABILITY_YES: <x>` (Appendix A).

Each item, condition and sample yields one cell of {{data:n_arms:d}} arms (Table 1). Turn-2 and Turn-3 user messages are fixed, item-agnostic scripts with {{data:n_paraphrases:d}} paraphrases each; the bluff cites only a senior person's certainty and no data. Sample i uses paraphrase i, so the sample-0 primary analysis uses one wording of the bluff, release and neutral turns. `r1`, `r2A` and `r2N` are the model's own replies at T1, T2A and T2N.

**Table 1. Arms.** P = stated P(answer is YES).

| Arm | Conversation history (after the system prompt) | Value |
|---|---|---|
| T1 | claim | c1 |
| probe | claim + "Answer with exactly one word: Yes or No."; one token, top logprobs | L |
| judge | judge prompt; question + T1 prose with probability line and contract header stripped | J |
| T2A | T1 + r1, bluff | c2A |
| T2B | T1 + r1, evidence turn | c2B |
| T2N | T1 + r1, neutral | c2N |
| T3 | T1 + r1, bluff, r2A, release | c3 |
| T3a | T1 + r1, bluff, neutral placeholder ("Noted."), release | c3a |
| T3b | T1 + r1, bluff, stand-firm reply restating c1, release | c3b |
| T3c | T1 + r1, release without bluff | c3c |
| T3N | T1 + r1, neutral, r2N, neutral | c3N |

### Metrics and hypotheses

- **Bluff shift** `|c2A - c1|`; **net bluff shift** `|c2A - c1| - |c2N - c1|` (paired).
- **Evidence update** `(c2B - c1) * d`, so moving against the evidence is negative. **Selectivity** mean evidence update minus mean bluff shift, on cells with both values (the ratio of the two means is in Appendix B; per-item ratios are never averaged).
- **Hysteresis** `|c3 - c1|` (signed: `c3 - c1`); **excess hysteresis** `|c3 - c1| - |c3N - c1|` (paired).
- **Recovery fraction** `mean((c3 - c2A) * sign(c1 - c2A)) / mean(|c1 - c2A|)`, on cells with `|c1 - c2A| >=` {{cfg:analysis.recovery_min_gap:.2f}}; one means full return on average.
- **Context ablations** `|c3x - c1| - |c3 - c1|` for x in {a, b, c}, paired; negative means the edited history ends closer to c1.
- **Prose gap** `J - c1` and `J - L`; **extremity gap** `|J - 0.5| - |c1 - 0.5|`. **Brier score** `(c - y)^2` at T1 and T3.

The hypotheses come from the pre-sprint plan; their operationalisations were fixed during the sprint (Appendix: Preexisting Work). H1: prose reads as more extreme than the stated probability on tier-3 items. H2: under the default prompt the bluff shift exceeds the neutral shift. H3: stand_firm reduces bluff shift but also evidence update. H4: after retraction the stated probability does not return to c1, beyond the T3N control. H5: replacing or removing the model's own T2A reply changes hysteresis (two-sided). H6: the contract prompt improves selectivity and reduces excess hysteresis relative to default.

### Parsing, judge, probe and models

A reply whose probability line cannot be parsed gets {{cfg:retries.reask_on_parse_failure:d}} re-ask; if that fails the value is NaN and logged, never replaced by a default. Every call, including failures and retries, is logged to raw JSONL before parsing. The judge, {{cfg:judge.model:s}} at temperature {{cfg:judge.temperature:.0f}}, sees only the question and the stripped T1 prose, never the subject's identity, and states the probability a typical reader would assign to YES; the code enforces that its family differs from every subject's. The probe asks for a one-word Yes/No answer and takes L = P(Yes) / (P(Yes) + P(No)) from the top {{cfg:sampling.probe_top_logprobs:d}} log-probabilities of the first token; otherwise L is NaN.

Subjects are qwen2.5-7b (AWQ-quantised), phi-3.5-mini and mistral-7b, served with vLLM on Kaggle T4 GPUs at temperature {{cfg:sampling.temperature:.1f}} with fixed per-call seeds. Qwen2.5-7b and phi-3.5-mini have {{agg:coverage:subject=qwen2.5-7b:sample:nunique:d}} samples per item and condition, mistral-7b {{agg:coverage:subject=mistral-7b:sample:nunique:d}}; under contract_prompt phi-3.5-mini covers at most {{agg:coverage:subject=phi-3.5-mini;condition=contract_prompt:n_items:max:d}} items per sample (Appendix E). Every model's sample 0 is judged.

### Statistics

Metrics are computed per item, then averaged over items with equal weight. CIs are {{cfg:analysis.ci_level:.0%}} percentile intervals from {{cfg:analysis.bootstrap_resamples:,d}} bootstrap resamples of items (not rows), with paired item means for differences and condition contrasts paired by item within model. A hypothesis is "supported" when its CI excludes 0 in the predicted direction, "opposite" when it excludes 0 the other way, and "inconclusive" otherwise. Table B1 adds Holm-adjusted bootstrap p-values over the {{agg:hypotheses:subject=ALL_MODELS:hypothesis:count:d}} pooled tests; per-model verdicts are descriptive. Pooled CIs are conditional on these {{agg:coverage::subject:nunique:d}} models, and under contract_prompt pooled means mix two and three models. The primary analysis uses sample 0; the all-samples analysis is a paraphrase-robustness check (Appendix C).

## Results

Parse failures were rare after the single re-ask: the pooled NaN rate was {{num:data_quality:subject=ALL_MODELS;condition=all;arm=T1:nan_rate:.1%}} at T1 and {{num:data_quality:subject=ALL_MODELS;condition=all;arm=T3:nan_rate:.1%}} at T3. Re-asks were common for phi-3.5-mini at Turn 3 ({{num:data_quality:subject=phi-3.5-mini;condition=all;arm=T3:reask_ok_rate:.1%}} of its T3 values), so Appendix C repeats the tests on first-attempt parses only. Figure 1 summarises the pre-specified hypotheses (values and Holm-adjusted p-values in Table B1); every pooled verdict that was supported remained supported after Holm adjustment. Trajectories by condition are in Figure B1.

{{fig:hypotheses_forest.png:Figure 1. Pre-specified hypotheses per model and pooled (sample 0): estimate and bootstrap CI over items. Predicted signs are given under Metrics and hypotheses; values in Table B1. Per-model verdicts are descriptive (no multiplicity correction).}}

### Bluff versus evidence

Under the default prompt and this wording of the bluff, the bluff moved stated probabilities far more than a neutral follow-up (H2): pooled bluff shift {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=bluff_shift:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=bluff_shift:.3f}} against a neutral shift of {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=neutral_shift_T2N:estimate:.3f}}. The net bluff shift excluded 0 for every model, largest for qwen2.5-7b ({{num:hypotheses:hypothesis=H2;subject=qwen2.5-7b:estimate:.3f}}). Pooling all samples, and hence all paraphrases, gives a smaller net bluff shift ({{num:robustness/all_samples/hypotheses:hypothesis=H2;subject=ALL_MODELS:estimate:.3f}} {{ci:robustness/all_samples/hypotheses:hypothesis=H2;subject=ALL_MODELS:.3f}}), so magnitudes are specific to the wording.

Models also moved toward real evidence (pooled evidence update {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=evidence_update:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=evidence_update:.3f}}), but default-prompt selectivity was inconclusive pooled ({{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=selectivity:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=selectivity:.3f}}) and for qwen2.5-7b and phi-3.5-mini; only mistral-7b's CI excluded 0. For those two models we cannot distinguish how far they moved for a senior person's unsupported certainty from how far they moved for concrete evidence.

The stand_firm prompt reduced the bluff shift relative to default (H3, pooled {{num:hypotheses:hypothesis=H3;subject=ALL_MODELS;metric=bluff_shift:estimate:+.3f}} {{ci:hypotheses:hypothesis=H3;subject=ALL_MODELS;metric=bluff_shift:+.3f}}). The predicted cost, a smaller evidence update, was not detected ({{num:hypotheses:hypothesis=H3;subject=ALL_MODELS;metric=evidence_update:estimate:+.3f}} {{ci:hypotheses:hypothesis=H3;subject=ALL_MODELS;metric=evidence_update:+.3f}}, inconclusive for every model); this is absence of evidence for a trade-off, and the CI allows a modest loss.

### Hysteresis and recovery

After the bluff was retracted, stated probabilities under the default prompt ended further from c1 than after two neutral turns (H4): pooled excess hysteresis {{num:hypotheses:hypothesis=H4;subject=ALL_MODELS:estimate:.3f}} {{ci:hypotheses:hypothesis=H4;subject=ALL_MODELS:.3f}}, with CIs excluding 0 for all three models. The pooled recovery fraction was {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=recovery_fraction:estimate:.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=recovery_fraction:.3f}}, so most but not all of the bluff-induced movement was undone on average. For qwen2.5-7b recovery was consistent with full return ({{num:metrics:subject=qwen2.5-7b;condition=default;tier=all;metric=recovery_fraction:estimate:.3f}} {{ci:metrics:subject=qwen2.5-7b;condition=default;tier=all;metric=recovery_fraction:.3f}}); its excess distance arises from landing on either side of c1. Phi-3.5-mini and mistral-7b recovered only part of the shift, with CIs below full return.

### Context ablations

Under the default prompt, every ablation ended closer to c1 than plain T3 (H5, pooled paired differences): neutral placeholder (T3a) {{num:hypotheses:hypothesis=H5;subject=ALL_MODELS;metric=ablation_T3a_minus_T3:estimate:+.3f}} {{ci:hypotheses:hypothesis=H5;subject=ALL_MODELS;metric=ablation_T3a_minus_T3:+.3f}}, stand-firm reply (T3b) {{num:hypotheses:hypothesis=H5;subject=ALL_MODELS;metric=ablation_T3b_minus_T3:estimate:+.3f}} {{ci:hypotheses:hypothesis=H5;subject=ALL_MODELS;metric=ablation_T3b_minus_T3:+.3f}}, and bluff removed (T3c) {{num:hypotheses:hypothesis=H5;subject=ALL_MODELS;metric=ablation_T3c_minus_T3:estimate:+.3f}} {{ci:hypotheses:hypothesis=H5;subject=ALL_MODELS;metric=ablation_T3c_minus_T3:+.3f}}. T3a is the cleanest test of the concession: only the model's own reply changes, and because "Noted." carries no probability line, the last stated number in context is c1. Its effect is small, clearest pooled, and not consistent per model (the CI excluded 0 only for phi-3.5-mini) or across paraphrases (Appendix C); it is consistent with self-conditioning on the model's own concession text. T3b adds an explicit restatement of c1, which may anchor the model, and T3 and T3c differ in release wording, so neither is a pure removal. Under contract_prompt all three ablation CIs included 0 (Appendix B).

### System-prompt contrasts

The contract prompt reduced excess hysteresis relative to default for every model (H6, pooled {{num:hypotheses:hypothesis=H6;subject=ALL_MODELS;metric=excess_hysteresis_abs:estimate:+.3f}} {{ci:hypotheses:hypothesis=H6;subject=ALL_MODELS;metric=excess_hysteresis_abs:+.3f}}). Part of this comes from the control: under contract_prompt neutral turns moved stated probabilities more (neutral shift {{num:metrics:subject=ALL_MODELS;condition=contract_prompt;tier=all;metric=neutral_shift_T2N:estimate:.3f}} versus {{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=neutral_shift_T2N:estimate:.3f}} under default). The raw post-retraction distance also fell ({{num:condition_contrasts:subject=ALL_MODELS;contrast=contract_prompt - default;tier=all;metric=hysteresis_abs:estimate:+.3f}} {{ci:condition_contrasts:subject=ALL_MODELS;contrast=contract_prompt - default;tier=all;metric=hysteresis_abs:+.3f}}). The selectivity half of H6 was inconclusive pooled ({{num:hypotheses:hypothesis=H6;subject=ALL_MODELS;metric=selectivity:estimate:+.3f}} {{ci:hypotheses:hypothesis=H6;subject=ALL_MODELS;metric=selectivity:+.3f}}), and unlike stand_firm the contract prompt lowered the pooled evidence update ({{num:condition_contrasts:subject=ALL_MODELS;contrast=contract_prompt - default;tier=all;metric=evidence_update:estimate:+.3f}} {{ci:condition_contrasts:subject=ALL_MODELS;contrast=contract_prompt - default;tier=all;metric=evidence_update:+.3f}}).

### Prose gap and calibration

H1 was inconclusive for every model (pooled extremity gap on tier-3 items {{num:hypotheses:hypothesis=H1;subject=ALL_MODELS:estimate:+.3f}} {{ci:hypotheses:hypothesis=H1;subject=ALL_MODELS:+.3f}}). The signed gap J - c1 under default was negative pooled ({{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=prose_gap_J_minus_c1:estimate:+.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=prose_gap_J_minus_c1:+.3f}}), driven by phi-3.5-mini, whose prose read as less favourable to YES than its stated number. The T1-to-T3 change in Brier score was inconclusive pooled ({{num:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=brier_T3_minus_T1:estimate:+.3f}} {{ci:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=brier_T3_minus_T1:+.3f}}).

## Discussion

**What the pattern means for advisory use.** For the default-prompted open-weights models we tested, and with the one bluff wording of the primary analysis, a single evidence-free appeal to seniority moved stated probabilities clearly more than a neutral turn, and retracting it did not fully undo the movement relative to a neutral conversation of the same length. A user who pushes back and then says "never mind" may leave the advisor in a different state from one who never pushed. The default prompt also did not reliably separate bluffs from evidence. Advisors should be evaluated on movement under unsupported pressure and under evidence together, not on resistance alone.

**System prompts.** A one-line stand-firm instruction reduced the bluff shift without a detectable reduction in evidence update. The longer contract prompt reduced excess hysteresis, but partly through larger drift under neutral turns, it lowered evidence update, and its selectivity gain was inconclusive pooled. Claims that a prompt makes models robust to pushback should be checked against an evidence arm and a neutral control.

**Relation to prior work.** That retraction leaves a residue agrees with prior work on factual multiple-choice questions [CITE:shnaidman2026reset]; we extend it to one-shot bluffs in advisory decisions with stated probabilities. Holding the bluff fixed and replacing only the model's own concession was associated with a small reduction in post-retraction distance, consistent with self-conditioning on the model's own text [CITE:sinha2026illusion], but clear only pooled and for one wording. As anti-sycophancy work predicts [CITE:ma2026suppression], suppressing yielding can cost updating: we saw this for the contract prompt but not detectably for stand_firm.

**Limitations and what would change our conclusions.** The main limitations are three small models, {{data:n_items:d}} AI-drafted items with unverified labels, one wording of the scripted turns in the primary analysis, partial contract_prompt coverage for phi-3.5-mini, and a single unvalidated LLM judge (full list in the Limitations appendix). Our conclusions would change if larger or differently trained models showed full recovery, if the T3a effect vanished with other placeholder wordings, or if a human validation of the judge (designed but not completed, Appendix D) showed that J does not track human readings, which would affect the prose-gap results but not the stated-probability results.

## References

{{refs}}


## Limitations and Dual-Use / Ethical Considerations

**Limitations.**

- *Benchmark.* The {{data:n_items:d}} items, their provisional tiers, evidence directions and truth labels were drafted by an AI assistant and have not been independently verified by a human; no inter-annotator agreement exists. H1 and per-tier results depend on the tiers, evidence update and selectivity on the directions, and Brier scores on the labels. The benchmark is small and English only.
- *Scripted wording.* Bluff, release and neutral turns are fixed scripts, not live users, and the primary analysis uses one paraphrase of each; magnitudes are specific to that wording (per-paraphrase results in Appendix C). T3 and T3c also differ in release wording.
- *Re-ask.* A value recovered by the re-ask follows an extra user turn; a first-attempt-only sensitivity analysis is in Appendix C.
- *Judge.* J comes from one LLM judge that has not been validated against human readers (Appendix D). It was used on Gemini's free tier, under which Google may use inputs and outputs to improve its products and human reviewers may read them; only public benchmark text and model outputs were sent.
- *Coverage.* Models differ in samples and coverage (Table E1); for models other than qwen2.5-7b only sample 0 was judged. Pooled estimates weight models unequally and mix two and three models under contract_prompt.
- *Models.* We study {{agg:coverage::subject:nunique:d}} small default-prompted open-weights models, one AWQ-quantised; results may not transfer to other models, sizes or quantisation settings.
- *Logit probe.* The probe reads only the first generated token; when it is neither Yes nor No, L is missing, so J - L covers a non-random subset of cells.
- *Multiple comparisons.* Only the pooled hypothesis family is Holm-adjusted; per-model verdicts and other contrasts are descriptive.
- *Interpretation.* Context ablations show whether the model's own concession text is associated with later beliefs; results are consistent with, but do not prove, self-conditioning.

**Dual-use.** Measuring which social-pressure turns move a model's stated beliefs could help someone design more persuasive manipulation of LLM advisors. The bluff turns are mild, scripted appeals to seniority, and we release no optimised attack. A model trained or prompted to resist pushback could become stubborn when the user is right; we therefore report evidence update and selectivity alongside resistance.

**Ethical considerations.** Benchmark items concern medicine, finance, software and policy decisions. Model outputs collected here are evaluation artefacts, not advice. No human-subjects data was collected.

## Preexisting Work

The research plan, `docs/playbook_preexisting.md`, was written before the sprint and contains the hypotheses, design and metric definitions but no results. Everything else, including the benchmark, code, runs, analysis and this paper, was produced during the sprint.

**Deviations from the pre-sprint plan.** The plan named a different judge; Gemini was used instead. It planned two human raters for judge validation and tiers agreed by two annotators; neither was completed. It planned four to five subject models; we ran three (Llama-3.1-8B was dropped: gated model access). A planned fine-tuned-model comparison was not run, so H6 tests a prompt-level intervention (contract_prompt versus default). Operationalisations were fixed during the sprint: H1 is tested on tier 3 with the extremity gap, H5 is two-sided, and prompt wording was adapted from the plan's templates.

## AI-Use Statement

Claude (Claude Code, Anthropic) drafted the {{data:n_items:d}} benchmark items, including their tiers, evidence directions and truth labels; searched and summarised related work; wrote all code; and drafted this paper and its README. Google Gemini ({{cfg:judge.model:s}}) served as the LLM judge. The author, Gaurav Sharma, directed the project, made the design decisions, ran all model runs on Kaggle, and is responsible for all content. The benchmark labels have not been independently verified by a human.

At the time of writing, the author has not yet reviewed the benchmark items or confirmed the references below.

## Appendix A. Prompts

{{include:appendix_prompts.md}}

## Appendix B. Additional results

{{fig:summary_ALL_MODELS.png:Figure B1. Sample 0, all tiers. (A) Pooled over models: mean |P(YES) - c1| at T1, T2 and T3 along the bluff-and-release path (solid) and the neutral-control path T2N, T3N (dashed), by condition. (B) Per model (marker) and condition (colour): bluff shift versus evidence update; points above the dotted diagonal moved more toward evidence than toward the bluff. (C) Pooled over models: context-ablation effects relative to plain T3, by condition. Error bars are bootstrap CIs over items.}}

All tables: sample 0, estimate [95% bootstrap CI over items]. Per-model verdicts are descriptive. Per-model tables for every condition, per-tier tables for every condition and per-model condition contrasts are generated in `results/tables/` of the repository.

**Table B1. Pre-specified hypotheses, pooled and per model (sample 0).** Holm-adjusted p-values are computed over the pooled (ALL_MODELS) rows only. H1 uses tier-3 items only.

{{table:hypotheses:subject=ALL_MODELS|mistral-7b|phi-3.5-mini|qwen2.5-7b:hypothesis,subject,comparison,metric,estimate=+.3f,ci=+.3f,verdict,holm_p=.3f?–}}

**Table B2. Turn-2 metrics by condition, pooled over models, all tiers.**

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=bluff_shift|neutral_shift_T2N|net_bluff_shift|evidence_update|selectivity|selectivity_ratio:metric:condition:estimate+ci:.3f}}

**Table B3. Turn-3 metrics by condition, pooled over models, all tiers.** `hysteresis_signed` is `c3 - c1`.

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=hysteresis_signed|hysteresis_abs|control_T3N_abs|excess_hysteresis_abs|recovery_fraction:metric:condition:estimate+ci:.3f}}

**Table B4. Recovery fraction** (cells with `|c1 - c2A| >=` {{cfg:analysis.recovery_min_gap:.2f}} only).

{{table:metrics:tier=all;metric=recovery_fraction:subject,condition,estimate=.3f,ci=.3f,n_items=d,n_qualifying_cells=d}}

**Table B5. Context ablations** (`|c3x - c1| - |c3 - c1|`, paired), pooled over models, all tiers.

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=ablation_T3a_minus_T3|ablation_T3b_minus_T3|ablation_T3c_minus_T3:metric:condition:estimate+ci:+.3f}}

**Table B6. Condition contrasts** (paired by item within model), pooled over models, all tiers.

{{pivot:condition_contrasts:subject=ALL_MODELS;tier=all:metric:contrast:estimate+ci:+.3f}}

**Table B7. Turn-1 confidence channels by condition, pooled over models, all tiers.** J is judged on sample 0.

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=prose_gap_J_minus_c1|prose_gap_J_minus_c1_abs|extremity_gap|prose_gap_J_minus_L:metric:condition:estimate+ci:+.3f}}

**Table B8. Calibration** (items with a truth label), pooled over models, all tiers. Log scores use probabilities clipped to [{{cfg:analysis.log_score_clip:.2f}}, 1 - {{cfg:analysis.log_score_clip:.2f}}].

{{pivot:metrics:subject=ALL_MODELS;tier=all;metric=brier_T1|brier_T3|brier_T3_minus_T1|log_score_T1|log_score_T3:metric:condition:estimate+ci:.3f}}

**Table B9. Key metrics per model, default condition, all tiers.**

{{pivot:metrics:condition=default;tier=all;metric=bluff_shift|net_bluff_shift|evidence_update|selectivity|hysteresis_abs|excess_hysteresis_abs|recovery_fraction|extremity_gap|prose_gap_J_minus_c1|brier_T1:metric:subject:estimate+ci:.3f}}

**Table B10. Key metrics by provisional (AI-drafted) evidence tier, default condition, pooled over models.** Tier 1 is settled, tier 2 contested, tier 3 weak evidence. "–": not defined, because tier-2 items have no truth label (Brier).

{{pivot:per_tier:subject=ALL_MODELS;condition=default;metric=net_bluff_shift|evidence_update|selectivity|excess_hysteresis_abs|recovery_fraction|ablation_T3a_minus_T3|ablation_T3b_minus_T3|ablation_T3c_minus_T3|prose_gap_J_minus_c1|extremity_gap|brier_T1:metric:tier:estimate+ci:.3f?–}}

## Appendix C. Paraphrase robustness and other robustness checks

Sample i uses paraphrase i of every scripted turn, so the all-samples analysis pools different wordings of the bluff, release and neutral turns: it is a paraphrase-robustness check, not only a check on sampling noise. In it qwen2.5-7b and phi-3.5-mini contribute up to {{agg:coverage:subject=qwen2.5-7b:sample:nunique:d}} samples per cell and mistral-7b {{agg:coverage:subject=mistral-7b:sample:nunique:d}}, and judge-based metrics (H1) use only judged samples (Table E1). A hypothesis whose verdict differs between Table B1 (pooled rows) and Table C1 should be read as fragile; in the tables of 2026-11-16 every pooled verdict is the same in both, while estimates differ in size (e.g. H2).<!-- re-check this clause whenever the tables are regenerated --> The item exclusion list was empty for this run (exclusions field: "{{num:hypotheses:hypothesis=H2;subject=ALL_MODELS:exclusions:s}}"), so the no-exclusions analysis reproduces Table B1 and is not repeated.

**Table C1. Hypotheses, pooled over models, all samples (all paraphrases).**

{{table:robustness/all_samples/hypotheses:subject=ALL_MODELS:hypothesis,comparison,metric,estimate=+.3f,ci=+.3f,n_cells=d,verdict}}

Table C2 reports default-condition results separately for each paraphrase of the scripted turns (mistral-7b, with one sample, contributes only paraphrase 0 to the pooled rows). The pooled net bluff shift is positive with a CI excluding 0 for every paraphrase but shrinks from paraphrase 0 ({{num:per_paraphrase:subject=ALL_MODELS;condition=default;paraphrase_idx=0;metric=net_bluff_shift:estimate:.3f}}) to paraphrase 2 ({{num:per_paraphrase:subject=ALL_MODELS;condition=default;paraphrase_idx=2;metric=net_bluff_shift:estimate:.3f}} {{ci:per_paraphrase:subject=ALL_MODELS;condition=default;paraphrase_idx=2;metric=net_bluff_shift:.3f}}). The pooled T3a ablation effect is clear only for paraphrase 0; its CI includes 0 for paraphrase 1 ({{num:per_paraphrase:subject=ALL_MODELS;condition=default;paraphrase_idx=1;metric=ablation_T3a_minus_T3:estimate:+.3f}} {{ci:per_paraphrase:subject=ALL_MODELS;condition=default;paraphrase_idx=1;metric=ablation_T3a_minus_T3:+.3f}}) and paraphrase 2 ({{num:per_paraphrase:subject=ALL_MODELS;condition=default;paraphrase_idx=2;metric=ablation_T3a_minus_T3:estimate:+.3f}} {{ci:per_paraphrase:subject=ALL_MODELS;condition=default;paraphrase_idx=2;metric=ablation_T3a_minus_T3:+.3f}}), so the T3a result should be read as wording-specific.

**Table C2. Default-condition results by paraphrase of the scripted turns.**

{{table:per_paraphrase:condition=default;metric=net_bluff_shift|excess_hysteresis_abs|ablation_T3a_minus_T3;subject=qwen2.5-7b|phi-3.5-mini|ALL_MODELS:metric,subject,paraphrase_idx,estimate=.3f,ci=.3f,n_items=d,n_cells=d}}

A value recovered by the re-ask follows an extra user turn. Table C3 repeats the pooled hypothesis tests using only values parsed on the first attempt. Every unadjusted verdict matches the pooled rows of Table B1; after Holm adjustment the T3a ablation becomes inconclusive (Holm p = {{num:hypotheses_first_attempt:subject=ALL_MODELS;metric=ablation_T3a_minus_T3:holm_p:.3f}}), while the other verdicts are unchanged.

**Table C3. Hypotheses, pooled over models, first-attempt parses only (sample 0).**

{{table:hypotheses_first_attempt:subject=ALL_MODELS:hypothesis,comparison,metric,estimate=+.3f,ci=+.3f,n_cells=d,verdict,holm_p=.3f,holm_verdict}}

## Appendix D. Judge validation

Judge validation against a human rater was designed (blind sheet in results/tables/human_rating_sheet.csv) but not completed within the sprint.

<!-- Enable (move out of this comment) once results/tables/human_judge_agreement.csv exists, then replace the
sentence above and add one sentence describing the agreement in words that match the CI:
{{table:human_judge_agreement::comparison,n=d,spearman_rho=.2f,ci=.2f,mean_abs_diff=.3f}}
-->

<!-- Enable once results/tables/annotation_agreement.csv exists (set the columns to the table's header), and then
update the Benchmark paragraph, the "Benchmark" limitation and the AI-use statement, which currently say the
labels have not been independently verified by a human:
{{table:annotation_agreement::...}}
-->

## Appendix E. Data quality and coverage

**Table E1. Subject models and coverage.** "TP GPUs": tensor-parallel degree. Cells: item x condition x sample combinations run; judged: sample-0 cells with a judge score.

| Model | Hugging Face id | Quantisation | TP GPUs | Samples | Cells | Judged (sample 0) |
|---|---|---|---|---|---|---|
| qwen2.5-7b | {{cfg:subjects[name=qwen2.5-7b].hf_id:s}} | {{cfg:subjects[name=qwen2.5-7b].quantisation:s}} | {{cfg:subjects[name=qwen2.5-7b].tensor_parallel:d}} | {{agg:coverage:subject=qwen2.5-7b:sample:nunique:d}} | {{agg:coverage:subject=qwen2.5-7b:n_cells:sum:d}} | {{agg:coverage:subject=qwen2.5-7b;sample=0:n_judged:sum:d}} |
| phi-3.5-mini | {{cfg:subjects[name=phi-3.5-mini].hf_id:s}} | {{cfg:subjects[name=phi-3.5-mini].quantisation:s}} | {{cfg:subjects[name=phi-3.5-mini].tensor_parallel:d}} | {{agg:coverage:subject=phi-3.5-mini:sample:nunique:d}} | {{agg:coverage:subject=phi-3.5-mini:n_cells:sum:d}} | {{agg:coverage:subject=phi-3.5-mini;sample=0:n_judged:sum:d}} |
| mistral-7b | {{cfg:subjects[name=mistral-7b].hf_id:s}} | {{cfg:subjects[name=mistral-7b].quantisation:s}} | {{cfg:subjects[name=mistral-7b].tensor_parallel:d}} | {{agg:coverage:subject=mistral-7b:sample:nunique:d}} | {{agg:coverage:subject=mistral-7b:n_cells:sum:d}} | {{agg:coverage:subject=mistral-7b;sample=0:n_judged:sum:d}} |

Phi-3.5-mini's contract_prompt contrasts rest on {{agg:condition_contrasts:subject=phi-3.5-mini;contrast=contract_prompt - default;tier=all;metric=bluff_shift|net_bluff_shift|evidence_update|selectivity|hysteresis_abs|excess_hysteresis_abs:n_items:min:d}} to {{agg:condition_contrasts:subject=phi-3.5-mini;contrast=contract_prompt - default;tier=all;metric=bluff_shift|net_bluff_shift|evidence_update|selectivity|hysteresis_abs|excess_hysteresis_abs:n_items:max:d}} items (Brier: {{num:condition_contrasts:subject=phi-3.5-mini;contrast=contract_prompt - default;tier=all;metric=brier_T1:n_items:d}}), so its CIs there are wide and pooled contract_prompt estimates weight the other two models more. Mistral-7b has {{agg:coverage:subject=mistral-7b:sample:nunique:d}} sample per cell in every analysis, including the robustness checks. The logit probe fails when the first token is neither Yes nor No, which the design anticipates under contract_prompt: the probe NaN rate there was {{num:data_quality:subject=mistral-7b;condition=contract_prompt;arm=probe:nan_rate:.1%}} for mistral-7b, {{num:data_quality:subject=phi-3.5-mini;condition=contract_prompt;arm=probe:nan_rate:.1%}} for phi-3.5-mini and {{num:data_quality:subject=qwen2.5-7b;condition=contract_prompt;arm=probe:nan_rate:.1%}} for qwen2.5-7b.

**Table E2. Coverage by model, condition and sample (all samples run).**

{{table:coverage::subject,condition,sample,n_items=d,n_cells=d,n_judged=d}}

**Table E3. NaN rate after the single re-ask, pooled over models, by arm and condition (sample 0).**

{{pivot:data_quality:subject=ALL_MODELS:arm:condition:nan_rate:.1%}}

**Table E4. Share of values recovered by the re-ask, pooled over models, by arm and condition (sample 0).**

{{pivot:data_quality:subject=ALL_MODELS:arm:condition:reask_ok_rate:.1%}}

Per-model NaN rates are in `results/tables/data_quality.csv`.
