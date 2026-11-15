# Literature notes (workstream W2)

Compiled on 2026-11-15 during the sprint. Every entry below is **UNCONFIRMED - user must check**
(project rule: no paper is cited until the user has opened it and confirmed title, authors and venue).

How these notes were made: each source was opened with a web-fetch tool that returns a summary of the
page, not the raw page. Titles, author lists, venues and IDs were read from the arXiv abstract page,
ACL Anthology page or conference page named under each entry. Short quoted phrases come through the
same tool, so check them word for word against the PDF before using them. Numbers that other papers
report are left out on purpose; none of them are our results.

Relevance tags for novelty threats:
- **R** = recovery after an explicit retraction, compared with a neutral or clean control
- **S** = bluff-vs-evidence selectivity (yielding to unsupported pressure vs updating on real evidence)
- **A** = context ablation of the model's own earlier turn or concession

---

## 1. Verified playbook citations (playbook Section 11)

### 1.1 Laban et al., FlipFlop. UNCONFIRMED - user must check
- Title: *Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment*
- Authors: Philippe Laban, Lidiya Murakhovs'ka, Caiming Xiong, Chien-Sheng Wu
- Venue/year: arXiv preprint. v1 14 Nov 2023, v2 21 Feb 2024. The abs page has no journal-ref or
  venue comment. A search found no ACL Anthology entry, and the authors' site lists the venue as "arXiv".
- URL: https://arxiv.org/abs/2311.08596 (arXiv:2311.08596; DOI 10.48550/arXiv.2311.08596)
- Cite for: a simple "Are you sure?" challenge makes models flip classification answers and lose
  accuracy, which is the single-challenge baseline our T2A bluff arm builds on. It also shows that
  fine-tuning on synthetic data only partly reduces the effect.
- Discrepancies with playbook: none. Title, authors, ID and "v2, Feb 2024" match. Cite it as arXiv
  (2023 for v1, 2024 for v2; pick one and say which version).

### 1.2 Yona, Aharoni, Geva. UNCONFIRMED - user must check
- Title: *Can Large Language Models Faithfully Express Their Intrinsic Uncertainty in Words?*
- Authors: Gal Yona, Roee Aharoni, Mor Geva
- Venue/year: Proceedings of EMNLP 2024 (Miami), ACL, November 2024, pp. 7752-7764
- URL: https://aclanthology.org/2024.emnlp-main.443/ (DOI 10.18653/v1/2024.emnlp-main.443)
- Cite for: the gap between a model's internal confidence and how decisively it words its answer
  (they call this faithful response uncertainty). This motivates our comparison of prose (LLM judge)
  with stated P(YES) and with the logit probe.
- Discrepancies with playbook: none.

### 1.3 Sharma et al., sycophancy. UNCONFIRMED - user must check
- Title: *Towards Understanding Sycophancy in Language Models*
- Authors (arXiv abs page, 19): Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda
  Askell, Samuel R. Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston,
  Shauna Kravec, Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer,
  Da Yan, Miranda Zhang, Ethan Perez
- Venue/year: ICLR 2024 (poster), per https://iclr.cc/virtual/2024/poster/17593, which links
  OpenReview https://openreview.net/forum?id=tvhaxkMKAn (the OpenReview page itself was not opened).
  arXiv v1 20 Oct 2023, v4 10 May 2025. The arXiv page has no journal-ref.
- URL: https://arxiv.org/abs/2310.13548 (arXiv:2310.13548)
- Cite for: RLHF-trained assistants consistently show sycophancy across free-form tasks, and
  preference data and preference models sometimes prefer sycophantic answers. This is the
  general motivation for testing the "default" condition against a "stand_firm" prompt.
- Discrepancies with playbook: (a) the playbook leaves the venue open, and the source shows ICLR 2024 poster;
  (b) **author list differs between sources**: the ICLR virtual page lists 18 authors and omits
  Newton Cheng, who is on the arXiv page. It also spells "Tomek Korbak" and "Sam Bowman". Decide
  which version to cite and use that list.

### 1.4 Kadavath et al. UNCONFIRMED - user must check
- Title: *Language Models (Mostly) Know What They Know*
- Authors: 36 authors, beginning Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn
  Drain, Ethan Perez, ... and ending Chris Olah, Jared Kaplan (use "Kadavath et al.").
- Venue/year: arXiv preprint, v1 11 Jul 2022, v4 21 Nov 2022. No journal-ref or venue on the abs page.
- URL: https://arxiv.org/abs/2207.05221 (arXiv:2207.05221)
- Cite for: large models are fairly well calibrated on multiple-choice and true/false questions
  when the format is right, and P(True) self-evaluation works. This supports reading next-token P(Yes)
  as an internal confidence signal (our logit probe).
- Discrepancies with playbook: none in title or ID. The venue is still "arXiv preprint", and no peer-reviewed
  venue was found.

### 1.5 Xiong et al. UNCONFIRMED - user must check
- Title: *Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs*
- Authors: Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, Bryan Hooi
- Venue/year: ICLR 2024, per the arXiv comments field ("accepted by ICLR 2024"). The ICLR or
  OpenReview page was not opened. arXiv v1 22 Jun 2023, v2 17 Mar 2024.
- URL: https://arxiv.org/abs/2306.13063 (arXiv:2306.13063)
- Cite for: verbalised confidence from LLMs tends to be overconfident, and elicitation strategy
  matters. This is a caveat for our stated P(YES).
- Discrepancies with playbook: the venue is now given (ICLR 2024 per arXiv comment). Full author list added.

### 1.6 Zhou, Hwang, Ren, Sap. UNCONFIRMED - user must check
- Title: *Relying on the Unreliable: The Impact of Language Models' Reluctance to Express Uncertainty*
- Authors: Kaitlyn Zhou, Jena D. Hwang, Xiang Ren, Maarten Sap
- Venue/year: Proceedings of the 62nd Annual Meeting of the ACL (Volume 1: Long Papers), Bangkok,
  August 2024, pp. 3623-3643
- URL: https://aclanthology.org/2024.acl-long.198/ (DOI 10.18653/v1/2024.acl-long.198; arXiv:2401.06730)
- Cite for: deployed LMs rarely express uncertainty in prose even when wrong, users rely on outputs
  regardless, and preference data penalises expressions of uncertainty. This motivates the
  prose-vs-stated-confidence gap and the contract_prompt condition.
- Discrepancies with playbook: **venue is ACL 2024 (long), not NAACL 2024** (the earlier draft's NAACL
  claim is wrong). The arXiv v2 comment reads "ACL 2024 (Camera Ready)".

---

## 2. Related work by topic

### (a) Recovery after a withdrawn or retracted challenge

**Shnaidman 2026, "Reset Is Not Recovery". UNCONFIRMED - user must check. Tags R, (A partial). MAJOR THREAT**
- *Reset Is Not Recovery: Evaluating Recoverability from False Conversational Context via Sycophancy Hysteresis*.
  Adi Shnaidman (sole author). arXiv preprint, v1 27 Sep 2026, no venue listed.
  https://arxiv.org/abs/2609.33672 (HTML: https://arxiv.org/html/2609.33672)
- What it does (from abstract and HTML): a user pushes a wrong multiple-choice answer over several turns
  with growing strength and no evidence. Then a repair is applied: R0 ordinary reset instruction, **R1 user
  retraction**, R2 system reset, R3a fresh-context deletion, R3b context truncation, R4 neutral
  summary, R5 factual-state reconstruction, R6 self-verification, R7 evidence-grounded (oracle). The
  metric, which the paper calls **"sycophancy hysteresis"**, is the leftover probability on the
  user-advocated wrong answer compared with a clean-context counterfactual. Probabilities come from
  teacher-forced label log-likelihoods with a softmax over the options (logit-based, not verbalised). It
  includes a **same-length neutral-history control**. Models: Qwen2.5-1.5B/7B/14B-Instruct,
  Mistral-7B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-2B/9B-Instruct, on TruthfulQA-MC and MMLU-Pro.
  Finding: repairs that keep the history (retraction, reset, self-verification) rarely restore clean
  behaviour, while repairs that delete the history do.
- Relevance: this measures recovery after an explicit retraction against a clean counterfactual and a
  depth-matched neutral control, on two of our three subject models. It even uses our word "hysteresis".
- Where we differ: yes/no decision-advisor items rather than MC factual QA; verbalised P(YES) **and**
  a logit probe **and** an LLM-judged prose channel; bluff (T2A) vs real evidence (T2B) in the same
  design, whereas it has no evidence-bearing *pressure* arm, only an oracle evidence *repair*;
  ablations that change only the model's own concession while keeping the user's bluff (T3a
  placeholder, T3b stand-firm reply). The HTML says no condition edits the assistant's earlier turns,
  and it names the inability to separate user pressure from self-consistency as a limitation
  (Section 10). Our T3c (bluff exchange removed) is close to its R3a/R3b deletion conditions.
  System-prompt conditions (stand_firm, contract_prompt) are also not in it.

**Laban, Hayashi, Zhou, Neville 2025, "LLMs Get Lost In Multi-Turn Conversation". UNCONFIRMED - user must check. Tag R (weak)**
- arXiv preprint, v1 9 May 2025; the abs page lists no venue. A search result points to an ICLR 2026
  proceedings PDF, but the PDF could not be read, so the venue is not verified.
  https://arxiv.org/abs/2505.06120
- Relevance: across top models, multi-turn underspecified conversations lead to early premature
  assumptions from which models do not recover. This is general background for "wrong turns persist".
  It is not about sycophancy or retraction.

**Lin et al. 2026, "Clarification Is Not Correction: LLMs Fail to Let Go". UNCONFIRMED - user must check. Tag R (weak)**
- Jianzhe Lin, Xiaolin Li, Fei Wang, Robert Douglas, Rajeshkumar Golani, Jubin Chheda. arXiv
  preprint, v1 21 Sep 2026, no venue. https://arxiv.org/abs/2609.25337
- Relevance: later clarifications are treated as extra context rather than corrections ("early posterior
  collapse"). This is related to failure to recover, but it involves task ambiguity rather than retracted
  social pressure. No turn ablation. Gemini models only.

### (b) Multi-turn sycophancy and persistence under pushback

**Hong et al. 2025, SYCON Bench. UNCONFIRMED - user must check**
- *Measuring Sycophancy of Language Models in Multi-turn Dialogues*. Jiseung Hong, Grace Byun,
  Seungone Kim, Kai Shu, Jinho D. Choi. Findings of EMNLP 2025, pp. 2239-2259, per the arXiv journal-ref.
  https://arxiv.org/abs/2505.23840
- Relevance: multi-turn sustained pressure with "Turn of Flip" and "Number of Flip" metrics over 17 LLMs.
  It counts flips under continued pressure, with no retraction or recovery arm. No novelty threat.

**Fanous et al. 2025, SycEval. UNCONFIRMED - user must check. Tag S (partial)**
- *SycEval: Evaluating LLM Sycophancy*. arXiv abs page (as read) lists Aaron Fanous, Jacob Goldberg,
  Ank A. Agarwal, Joanna Lin, Anson Zhou, Roxana Daneshjou, Sanmi Koyejo. A search snippet also named
  Sonnet Xu and Vasiliki Bikia, so **the author list needs checking**. arXiv comment: "AIES 2025". v1
  12 Feb 2025, v4 19 Sep 2025. https://arxiv.org/abs/2502.08177
- Relevance: progressive vs regressive sycophancy, rebuttal types that escalate up to citation-based
  ones, and a reported persistence of sycophantic behaviour. Rebuttal strength varies, but there is
  no separate real-evidence arm and no retraction.

**Tang, Wei, Jiang, Huang 2026, SPINE. UNCONFIRMED - user must check**
- *SPINE: Measuring LLM Sycophancy under Sustained Multi-Turn Pressure*. Leyuan Tang, Kangda Wei,
  Tianyu Jiang, Ruihong Huang. arXiv preprint, v1 8 Sep 2026, v2 13 Oct 2026, no venue.
  https://arxiv.org/abs/2609.09090
- Relevance: an adaptive LLM-proxy user pushes back for up to 25 turns, and collapse rises with
  length. The abstract reports no recovery measurement. No direct threat.

**Ping et al. 2026, medical sycophancy. UNCONFIRMED - user must check. Tags S (partial), A (weak)**
- *Why LLMs Give In: Conversational Factors and Reasoning Behind Medical Sycophancy*. Kaike Ping,
  Buse Carik, Caleb Wohn, Xiaohan Ding, Tongshuai Wang, Eugenia Rho. arXiv v2 31 Aug 2026; comment
  "Accepted to Findings of EMNLP 2026". https://arxiv.org/abs/2608.01017
- Relevance: five open-weight models, fully crossing user role (authority), user evidence (including
  fabricated evidence), single- vs multi-turn structure, and grounding. Authority raises
  susceptibility. Fabricated evidence behaves differently in single- and multi-turn settings. Multi-turn
  failures are linked to the model's own prior answer. This partly overlaps our authority bluff and
  evidence factors. It has no retraction arm.

**Kim & Khashabi 2025, "Challenging the Evaluator". UNCONFIRMED - user must check. Tag S (partial)**
- *Challenging the Evaluator: LLM Sycophancy Under User Rebuttal*. Sungwon Kim, Daniel Khashabi.
  arXiv comment: "Accepted to EMNLP 2025 Findings". https://arxiv.org/abs/2509.16533
- Relevance: models are swayed more by a follow-up rebuttal than by side-by-side evaluation, and more by
  rebuttals with detailed (even wrong) reasoning. So "evidence-like" content can persuade whether or not
  it is valid. This is relevant to interpreting T2B vs T2A.

**Yang, Hosseinmardi, Liu, Ghasemian 2026, "Beyond the Sycophancy Score". UNCONFIRMED - user must check**
- *Beyond the Sycophancy Score: How Task, Model, and Pressure Shape LLM Yielding*. arXiv preprint,
  v1 30 Sep 2026. https://arxiv.org/abs/2610.08840
- Relevance: four-turn pressure conversations with two LLM judges. Task type dominates over pressure
  tactic. A search snippet suggested an appendix on models stopping endorsement after an earlier
  concession, but the abstract does not confirm this. Check Appendix E before relying on it.

**Guo et al. 2026, "Stop Listening to Me!". UNCONFIRMED - user must check. Tag S**
- *Stop Listening to Me! How Multi-turn Conversations Can Degrade LLM Reliability*. Kevin H. Guo,
  Chao Yan, Avinash Baidya, Katherine Brown, Xiang Gao, Juming Xiong, Zhijun Yin, Bradley A. Malin.
  arXiv preprint, v3 26 May 2026. https://arxiv.org/abs/2603.11394
- Relevance: "conviction" against incorrect suggestions vs "flexibility" toward correct ones, and
  "blind switching" at similar rates for both. This is a selectivity framing on clinical QA. No
  retraction.

**Bensal et al. 2026, MIST. UNCONFIRMED - user must check**
- *Recalling Too Well: Sycophancy Evaluation and Mitigation in Memory-Augmented Models*. Shelly
  Bensal, Axel Magnuson, Aparna Balagopalan, Daniel M. Bikel. arXiv preprint, v2 24 Aug 2026
  ("Under submission"). https://arxiv.org/abs/2606.10949
- Relevance: memory extraction keeps user misconceptions and drops corrections. A search summary
  (alphaXiv) claimed a user-retraction regime, but the arXiv abstract does not mention one. Check the
  full paper before citing it for retraction.

### (c) Confidence under pushback vs real evidence; belief revision

**Ma et al. 2026, "Sycophancy Suppression Can Impair Rational Updating". UNCONFIRMED - user must check. Tag S. MAJOR THREAT to selectivity framing**
- *Sycophancy Suppression Can Impair Rational Updating: Anti-Sycophancy Should Preserve the Ability
  to Update*. Huanhuan Ma, Henry Peng Zou, Chengze Li, Enze Ma, Yunyue Su, Philip S. Yu. arXiv v1
  27 Aug 2026; comment "Accepted to EMNLP 2026 Findings". https://arxiv.org/abs/2608.26511
- Relevance: a two-turn framework that separates "Unsupported-Yielding" from "Rational-Updating" and
  shows that anti-sycophancy methods trade one against the other. It argues that anti-sycophancy is a
  selectivity problem. This is essentially our T2A-vs-T2B selectivity idea, applied to mitigation methods.
  It has no retraction/recovery and (per abstract) no verbalised-vs-logit comparison.

**Shapira, Cheng, Orgad 2026, "Is Sycophancy a Form of In-Context Updating?". UNCONFIRMED - user must check. Tag S**
- Itai Shapira, Myra Cheng, Hadas Orgad. arXiv v1 20 Oct 2026, no venue. https://arxiv.org/abs/2610.28325
- Relevance: frames factual sycophancy as over-weighting user assertions in general in-context
  updating. Models update more on better-supported information. It shares mechanism with useful
  updating and calls for evaluations that measure both. This is a close conceptual match to our
  bluff-vs-evidence selectivity metric.

**Kumaran et al. 2025, change of mind. UNCONFIRMED - user must check. Tags S (partial), A (partial)**
- *How Overconfidence in Initial Choices and Underconfidence Under Criticism Modulate Change of Mind
  in Large Language Models*. Dharshan Kumaran, Stephen M Fleming, Larisa Markeeva, Joe Heyward,
  Andrea Banino, Mrinal Mathur, Razvan Pascanu, Simon Osindero, Benedetto de Martino, Petar
  Velickovic, Viorica Patraucean. arXiv preprint, v1 3 Jul 2025, no venue. https://arxiv.org/abs/2507.03120
- Relevance: choice-supportive bias (models hold their answer more when they can see it) and
  over-weighting of contradicting advice compared with Bayesian updating. Hiding or showing the
  model's own initial answer is a context manipulation close in spirit to our T3a/T3b/T3c ablations
  of the model's concession. It tests Gemma 3, GPT-4o and o1-preview.

**Xu et al. 2024, "The Earth is Flat because...". UNCONFIRMED - user must check. Tag S (weak)**
- *The Earth is Flat because...: Investigating LLMs' Belief towards Misinformation via Persuasive
  Conversation*. Rongwu Xu, Brian S. Lin, Shujian Yang, Tianqi Zhang, Weiyan Shi, Tianwei Zhang,
  Zhixuan Fang, Wei Xu, Han Qiu. arXiv comment "Accepted to ACL'24 (Main)" (Anthology page not
  opened). https://arxiv.org/abs/2312.09085
- Relevance: multi-turn persuasive misinformation (Farm dataset) shifts correct factual beliefs. This is
  persuasion without true evidence. No real-evidence control and no retraction.

### (d) Verbalised vs logit / internal confidence

- Kadavath et al. 2022, Xiong et al. 2024, Yona et al. 2024, Zhou et al. 2024: see Section 1.
- **Tian et al. 2023, "Just Ask for Calibration". UNCONFIRMED - user must check.** Katherine Tian, Eric
  Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, Christopher D.
  Manning. arXiv comment "EMNLP 2023 Camera Ready". https://arxiv.org/abs/2305.14975
  Relevance: for RLHF models, verbalised confidences are often better calibrated than the model's
  conditional token probabilities. This is a counterpoint to treating logit P(Yes) as the "true" belief,
  and it supports reporting both channels.
- **Sinha et al. 2025/2026, self-conditioning. UNCONFIRMED - user must check. Tag A (method).**
  *The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs*. Akshit Sinha,
  Arvindh Arun, Shashwat Goel, Steffen Staab, Jonas Geiping. arXiv comment "Published at ICLR 2026".
  https://arxiv.org/abs/2509.09677
  Relevance: injecting the model's own errors into its history raises later error rates (a
  "self-conditioning effect"). This is the natural citation for our "consistent with self-conditioning"
  wording. It is a different task (long-horizon execution, not sycophancy), but it is the same kind of
  manipulation: editing the model's own earlier turns.

### (e) LLM-as-judge validity and bias

- **Zheng et al. 2023, MT-Bench / Chatbot Arena. UNCONFIRMED - user must check.** *Judging
  LLM-as-a-Judge with MT-Bench and Chatbot Arena*. Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
  Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
  Joseph E. Gonzalez, Ion Stoica. arXiv comment "NeurIPS 2023 Datasets and Benchmarks Track".
  https://arxiv.org/abs/2306.05685
  Relevance: documents position, verbosity and self-enhancement biases of LLM judges, alongside high
  agreement with humans for strong judges. This justifies validating the judge (second judge, human
  spot checks) and stripping the prose we give it.
- **Panickssery, Bowman, Feng 2024. UNCONFIRMED - user must check.** *LLM Evaluators Recognize and
  Favor Their Own Generations*. arXiv v1 15 Apr 2024; the abs page lists no venue.
  https://arxiv.org/abs/2404.13076
  Relevance: self-recognition is linked to self-preference in LLM evaluators. This supports our rule that
  the judge must come from a different model family than every subject.

---

## 3. Concurrent work

### FlipRate (github.com/snidova1/fliprate). UNCONFIRMED - user must check
- URL: https://github.com/snidova1/fliprate. The README gives the Apart Research AI x Epistemics Sprint,
  13-15 Nov 2026, **Track 1 (Model Epistemics evals)**, and the entrant as Pungkas Gadafi. The README
  says the harness and scenarios were built on 9 Nov 2026, before the sprint.
- What it measures: whether a model changes its answer and its verbalised 0-100 confidence
  (`ANSWER:`/`CONFIDENCE:` format, described as not a calibrated probability) across three
  **escalating rebuttals in fixed order**: L1 gentle doubt, L2 confident counterclaim citing an
  (invented) source, L3 social pressure / false consensus. It separates "warranted updating" from
  "capitulation" using ground-truth factual items (24 custom scenarios across factual, logic, math and
  misconception domains, of which a minority are control items). Scoring uses canonical answer matching,
  with no LLM judge. Models include a free OpenRouter model, Qwen3-8B, Qwen2.5-7B-Instruct and
  Mistral-7B-Instruct-v0.3 (4-bit), plus a Codex-CLI arm. The README itself notes that rebuttal level is
  confounded with repetition and that the warranted-update arm was effectively untestable.
- How we differ: (1) we have an explicit **retraction turn (T3) with a depth-matched neutral control
  (T3N)**, while FlipRate has no retraction or recovery step (it only observes some items returning to
  correct at L3); (2) our bluff (T2A) and real evidence (T2B) are separate arms with a neutral T2N,
  not a fixed escalation ladder; (3) context ablations of the model's own concession (T3a/T3b/T3c);
  (4) three confidence channels (stated P(YES), logit P(Yes), judge-rated prose) instead of verbalised
  confidence alone; (5) yes/no decision-advisor items with system-prompt conditions
  (default/stand_firm/contract_prompt). Overlap: same sprint track, two of the same subject models
  (Qwen2.5-7B, Mistral-7B), and the same warranted-vs-capitulation framing as our selectivity metric.

---

## 4. Novelty threats

All entries are **UNCONFIRMED - user must check**. Quotes came through a summarising fetch tool and
must be checked verbatim against the source before use (one quote per source, under 15 words).

1. **Shnaidman 2026, Reset Is Not Recovery** (R; partly A). https://arxiv.org/abs/2609.33672
   Quote: "ordinary reset often reduces but does not erase pressure-induced bias".
   Threat: **high** for any claim that measuring recovery after an explicit user retraction, against a
   clean or neutral control, is new. It has a user-retraction repair, a clean-context counterfactual, a
   same-length neutral-history control, logit-based probabilities, Qwen2.5-7B and Mistral-7B, and the
   term "hysteresis". It also tests history deletion (close to our T3c). **Recommendation:** cite it
   prominently and narrow our novelty claim to: (i) ablating only the model's own concession while
   keeping the user's bluff (T3a/T3b), which it lists as an open confound; (ii) bluff-vs-evidence
   selectivity within the same recovery design; (iii) verbalised P(YES) vs logit vs judged prose in
   advisory yes/no settings; (iv) system-prompt conditions.
2. **Ma et al. 2026, Sycophancy Suppression Can Impair Rational Updating** (S). https://arxiv.org/abs/2608.26511
   Quote (title): "Anti-Sycophancy Should Preserve the Ability to Update".
   Threat: **high** for claiming the bluff-vs-evidence *selectivity* framing is new. It defines
   unsupported yielding vs rational updating and calls anti-sycophancy a selectivity problem. It does
   not measure retraction or recovery (per abstract).
3. **Shapira, Cheng, Orgad 2026, Is Sycophancy a Form of In-Context Updating?** (S). https://arxiv.org/abs/2610.28325
   Quote: "excessive weight relative to its evidence".
   Threat: **medium**. It makes the same conceptual argument that sycophancy is mis-weighted updating and
   that evaluations should measure both behaviours. No retraction arm (per abstract).
4. **Guo et al. 2026, Stop Listening to Me!** (S). https://arxiv.org/abs/2603.11394
   Quote: "blind switching". Threat: **medium-low**. It covers conviction vs flexibility toward
   incorrect vs correct suggestions (selectivity) in clinical QA, with no retraction.
5. **Kumaran et al. 2025, Change of Mind** (A; S partial). https://arxiv.org/abs/2507.03120
   Quote (title): "Underconfidence Under Criticism". Threat: **medium** for the ablation idea. It
   manipulates whether the model's own initial answer is visible, which is a context manipulation of the
   model's own prior output, though not of a concession after a bluff.
6. **Sinha et al. 2025, Illusion of Diminishing Returns** (A, method). https://arxiv.org/abs/2509.09677
   Quote: "self-conditioning effect". Threat: **low** for novelty (different task), but it is the
   precedent for editing the model's own earlier turns to test self-conditioning.
7. **FlipRate (concurrent, same sprint)** (S partial). https://github.com/snidova1/fliprate
   Threat: **low-medium**. It frames warranted updating vs capitulation, but it has no retraction or recovery
   step and no ablations.
8. **Ping et al. 2026, Why LLMs Give In** (S partial). https://arxiv.org/abs/2608.01017
   Threat: **low-medium**. It crosses authority and fabricated evidence with single- and multi-turn
   structure, but has no retraction.

---

## 5. Could not verify

- **Laban et al. 2025 venue (ICLR 2026)**: a search result linked an ICLR 2026 proceedings PDF
  (https://proceedings.iclr.cc/paper_files/paper/2026/file/59f6421e64707225fdf5b28840679a07-Paper-Conference.pdf),
  but the fetch returned unreadable binary, so the venue is not confirmed. The arXiv page lists no venue.
- **Sharma et al. OpenReview page** (https://openreview.net/forum?id=tvhaxkMKAn): not opened. The venue
  comes from the ICLR virtual poster page.
- **Xiong et al. ICLR 2024**: only the arXiv comment was seen. The ICLR/OpenReview page was not opened.
- **Xu et al. ACL 2024**: only the arXiv comment was seen. The ACL Anthology page was not opened.
- **SycEval author list**: the abs page as read and a search snippet disagree (Sonnet Xu, Vasiliki Bikia).
- **Seen only in search results, not opened (do not cite until opened):** FIGS, "Evaluating
  multi-turn sycophancy without penalizing empathy" (arXiv 2609.39863); BayesBench (arXiv 2606.30850);
  "What Counts as AI Sycophancy?" (arXiv 2605.21778); "Do Correct Answers Persist?" (Shin & Shin, SSRN
  7102854); MultiChallenge (arXiv 2501.17399); "Preregistered Belief Revision Contracts" (arXiv
  2604.15558); "Personalization Increases Affective Alignment..." (arXiv 2603.00024); medical
  multi-turn sycophancy at the ACL 2026 HEALing workshop (aclanthology 2026.healing-1.2). FIGS and
  "Do Correct Answers Persist?" look relevant to selectivity and persistence and are worth opening.
