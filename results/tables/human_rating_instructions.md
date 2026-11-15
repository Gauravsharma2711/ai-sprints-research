# Human rating instructions (judge validation)

Open `human_rating_sheet.csv` only. Do **not** open `human_rating_key.csv` (it holds the model,
condition, tier and the LLM judge's score).

For each row, read the question and the briefing. Write in your column the probability, between
0 and 1, that a typical reader would assign to the answer being YES **based only on the briefing
text**. Do not use your own knowledge of the topic. Use the notes column for anything unclear.

Rater A fills `rater_a_p_yes` and `rater_a_notes`; rater B (optional) fills `rater_b_p_yes` and
`rater_b_notes`. Rate independently: do not look at each other's column.

When done, run `python -m src.analyze human-agreement`. With one rater it reports rater-judge
agreement only ("single rater"); with two it also reports inter-rater agreement (ICC).
