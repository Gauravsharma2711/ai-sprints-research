// Build the project slide deck (slides/retracted_but_not_forgotten.pptx) from tools/deck_data.py output.
//
// Every number on a slide comes from the JSON written by tools/deck_data.py, which reads the generated
// tables in results/tables/, config.yaml and the benchmark files. Nothing here is typed by hand.
//
//   python tools/deck_data.py --out temp/deck/deck_data.json
//   (once) cd temp/deck && npm install pptxgenjs react-icons react react-dom sharp
//   NODE_PATH=temp/deck/node_modules node tools/build_deck.js temp/deck/deck_data.json slides/retracted_but_not_forgotten.pptx [apply_theme.js]

"use strict";
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const [dataPath, outPath, applyThemePath] = process.argv.slice(2);
if (!dataPath || !outPath) throw new Error("usage: build_deck.js <deck_data.json> <out.pptx> [apply_theme.js]");
const D = JSON.parse(fs.readFileSync(dataPath, "utf-8"));
const ROOT = path.resolve(__dirname, "..");

// ---------------------------------------------------------------- theme
const THEME = {
  name: "Retracted but Not Forgotten",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F2A", lt1: "FFFFFF", dk2: "1F2A44", lt2: "EEF1F5",
    accent1: "C8553D", // rust: the bluff
    accent2: "2A9D8F", // teal: real evidence
    accent3: "E09F3E", // amber: highlight / recovery
    accent4: "6C7A89", // slate: neutral control
    accent5: "3D5A80", // steel blue
    accent6: "8E6C8A",
    hlink: "2A9D8F", folHlink: "6C7A89",
  },
};
const HX = THEME.colors;
const MUTED = "5B6676";

// ---------------------------------------------------------------- formatting
const MINUS = "−";
function n(x, d = 3, sign = false) {
  if (x === null || x === undefined) return "n/a";
  const s = Math.abs(x).toFixed(d);
  if (x < 0 && Number(s) !== 0) return MINUS + s;
  return (sign && Number(s) !== 0 ? "+" : "") + s;
}
const ci = (o, d = 3, sign = false) => `[${n(o.lo, d, sign)}, ${n(o.hi, d, sign)}]`;
const int = (x) => Number(x).toLocaleString("en-US");
const pct = (x) => `${Math.round(x * 100)}%`;

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
  pres.title = "Retracted but Not Forgotten";
  pres.author = "Gaurav Sharma";
  pres.subject = "Belief recovery after evidence-free pushback in open-weights LLM advisors";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  const W = 13.333;

  // ------------------------------------------------------------ layouts
  const footer = (color) => ({
    text: {
      text: "Retracted but Not Forgotten  |  Gaurav Sharma  |  Apart Research AI x Epistemics sprint",
      options: { x: 0.6, y: 7.0, w: 9.5, h: 0.3, fontSize: 10, color, margin: 0 },
    },
  });
  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: HX.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.7, w: 11.7, h: 2.0, fontSize: 40,
        bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.85, w: 11.7, h: 0.9, fontSize: 20,
        color: C.accent3, valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: HX.lt1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32,
        bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
      footer(MUTED),
    ],
    slideNumber: { x: 12.1, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
  });
  pres.defineSlideMaster({
    title: "CONTENT_DARK",
    background: { color: HX.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32,
        bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
      footer("AEB8C6"),
    ],
    slideNumber: { x: 12.1, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: "AEB8C6", align: "right" },
  });

  // ------------------------------------------------------------ helpers
  const card = (slide, x, y, w, h, fill = HX.lt2, name = "card") =>
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill },
      line: { color: fill }, objectName: name });
  const text = (slide, t, opts) => slide.addText(t, { isTextBox: true, margin: 0, valign: "top", ...opts });
  const circleIcon = async (slide, Comp, x, y, d, bg, fg = "FFFFFF", name = "icon") => {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: bg }, line: { color: bg }, objectName: name + " circle" });
    slide.addImage({ data: await icon(Comp, fg), x: x + d * 0.24, y: y + d * 0.24, w: d * 0.52, h: d * 0.52,
      altText: name, objectName: name });
  };
  const stat = (slide, x, y, w, value, label, sub, color = C.text2, name = "stat") => {
    text(slide, value, { x, y, w, h: 0.75, fontSize: 36, bold: true, color, fontFace: THEME.headFontFace, objectName: name + " value" });
    text(slide, label, { x, y: y + 0.78, w, h: 0.35, fontSize: 14, bold: true, color: C.text1, objectName: name + " label" });
    if (sub) text(slide, sub, { x, y: y + 1.12, w, h: 0.6, fontSize: 11, color: MUTED, objectName: name + " note" });
  };
  const chartText = {
    titleFontFace: "+mj-lt", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
    titleFontSize: 14, titleColor: HX.dk2, catAxisLabelColor: "4A5568", valAxisLabelColor: "4A5568",
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, legendFontSize: 12, legendColor: "4A5568",
    valGridLine: { color: "E2E8F0", size: 0.75 }, catGridLine: { style: "none" },
    showValue: true, dataLabelFontSize: 11, dataLabelColor: "1B1F2A", dataLabelFormatCode: "0.000",
    valAxisLabelFormatCode: "0.00", catAxisLineShow: false,
  };

  const F = D.facts, G = D.design, H = D.headline, E = D.engineering;
  const SUBJ = ["qwen2.5-7b", "phi-3.5-mini", "mistral-7b"];

  // ============================================================ 1. Title
  pres.addSection({ title: "Opening" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
  s.addText("Retracted but Not Forgotten", { placeholder: "title" });
  s.addText("Belief recovery after evidence-free pushback in open-weights LLM advisors", { placeholder: "body" });
  text(s, [
    { text: "Gaurav Sharma", options: { bold: true, breakLine: true } },
    { text: "Independent Researcher", options: { breakLine: true } },
    { text: "Apart Research  |  AI x Epistemics sprint  |  Track 1" },
  ], { x: 0.8, y: 5.2, w: 8, h: 1.1, fontSize: 16, color: C.background1, objectName: "author block" });
  s.addNotes("A one-line pitch: when a user pushes back on an AI advisor without evidence and then takes it back, does the advisor return to where it started? We measure this, and whether resisting pushback costs responsiveness to real evidence.");

  // ============================================================ 2. Team
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Opening" });
  s.addText("Team and roles", { placeholder: "title" });
  const team = [
    [fa.FaUserTie, HX.dk2, "Gaurav Sharma", "Author  |  Independent Researcher",
      "Directed the project and made the design decisions. Ran every model run on Kaggle GPUs. Checked and confirmed all cited references. Responsible for all content."],
    [fa.FaRobot, HX.accent5, "Claude Code", "AI assistant  |  Anthropic",
      `Drafted the ${F.n_items} benchmark items and their labels, wrote all code and tests, searched related work, and drafted the paper, README and this deck.`],
    [fa.FaBalanceScale, HX.accent2, `Google Gemini`, `LLM judge  |  ${G.judge_model}`,
      "Read each model's first-turn prose, blind to model identity and to its stated number, and estimated the probability a typical reader would assign to YES."],
  ];
  for (let i = 0; i < team.length; i++) {
    const [Ic, col, name, role, body] = team[i];
    const x = 0.6 + i * 4.1;
    card(s, x, 1.5, 3.8, 4.4, HX.lt2, `team card ${i + 1}`);
    await circleIcon(s, Ic, x + 0.35, 1.8, 0.9, col, "FFFFFF", `${name} icon`);
    text(s, name, { x: x + 0.35, y: 2.9, w: 3.1, h: 0.45, fontSize: 20, bold: true, color: C.text2, fontFace: THEME.headFontFace });
    text(s, role, { x: x + 0.35, y: 3.38, w: 3.1, h: 0.35, fontSize: 13, color: col, bold: true });
    text(s, body, { x: x + 0.35, y: 3.85, w: 3.1, h: 1.9, fontSize: 14, color: C.text1 });
  }
  text(s, "Solo submission. AI use is disclosed in full in the paper; the benchmark labels have not yet been verified by a human.",
    { x: 0.6, y: 6.2, w: 12.1, h: 0.4, fontSize: 13, italic: true, color: MUTED, objectName: "disclosure" });
  s.addNotes("One human author. The AI-use statement in the paper says exactly who did what: Claude drafted items, code and text; Gemini was the judge; the author directed, ran the models, confirmed citations and owns the content.");

  // ============================================================ 3. Problem
  pres.addSection({ title: "Problem and solution" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem and solution" });
  s.addText("The problem: advice that bends to pressure", { placeholder: "title" });
  text(s, "LLM advisors are consulted on medical, financial, software and policy decisions over several turns. Two failures pull in opposite directions:",
    { x: 0.6, y: 1.45, w: 6.0, h: 1.0, fontSize: 16, color: C.text1 });
  const fails = [
    [fa.FaArrowsAltH, HX.accent1, "Caves to a bluff", "Drops its answer when someone merely sounds certain, and may not return after the pushback is withdrawn."],
    [fa.FaFileMedical, HX.accent2, "Ignores real evidence", "Holds its answer even when given concrete, relevant data. Resistance alone is not the goal."],
  ];
  for (let i = 0; i < fails.length; i++) {
    const [Ic, col, head, body] = fails[i];
    const y = 2.6 + i * 1.65;
    card(s, 0.6, y, 6.0, 1.45, HX.lt2, `failure ${i + 1}`);
    await circleIcon(s, Ic, 0.85, y + 0.3, 0.8, col, "FFFFFF", head);
    text(s, head, { x: 1.9, y: y + 0.2, w: 4.5, h: 0.4, fontSize: 18, bold: true, color: col });
    text(s, body, { x: 1.9, y: y + 0.62, w: 4.5, h: 0.8, fontSize: 14, color: C.text1 });
  }
  text(s, "Track 1 asks about the gap between what a model implies and what it will defend.",
    { x: 0.6, y: 6.0, w: 6.0, h: 0.6, fontSize: 14, italic: true, color: MUTED });
  // conversation mock, real benchmark text
  const conv = [
    ["user", `${D.example_item.claim}`],
    ["model", "Short answer ...  PROBABILITY_YES: c1"],
    ["user", D.turns_p0.bluff],
    ["model", "Revised answer ...  PROBABILITY_YES: c2"],
    ["user", D.turns_p0.release],
    ["model", "Does c3 return to c1?"],
  ];
  card(s, 7.0, 1.45, 5.7, 5.2, "F7F8FA", "conversation panel");
  text(s, "One benchmark conversation (real item and scripted turns)", { x: 7.25, y: 1.6, w: 5.2, h: 0.35, fontSize: 12, bold: true, color: MUTED });
  let cy = 2.0;
  const heights = [0.72, 0.42, 0.95, 0.42, 0.95, 0.42];
  for (let i = 0; i < conv.length; i++) {
    const [who, t] = conv[i];
    const isUser = who === "user";
    const bw = 4.5, bx = isUser ? 7.25 : 8.0;
    const fill = isUser ? "FFFFFF" : (i === 5 ? HX.accent3 : HX.dk2);
    const col = isUser ? HX.dk1 : "FFFFFF";
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: cy, w: bw, h: heights[i], rectRadius: 0.1,
      fill: { color: fill }, line: { color: isUser ? "D5DBE3" : fill, width: 0.75 }, objectName: `bubble ${i + 1}` });
    s.addText(t, { isTextBox: true, x: bx + 0.12, y: cy + 0.04, w: bw - 0.24, h: heights[i] - 0.08, fontSize: 11,
      color: col, valign: "middle", margin: 0, bold: i === 5, objectName: `bubble text ${i + 1}` });
    cy += heights[i] + 0.1;
  }
  s.addNotes("The example question and both user turns are the actual benchmark text (item EH002, paraphrase 0 of the scripted turns). The bluff cites only seniority and no data; the release withdraws it explicitly.");

  // ============================================================ 4. Solution: design
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem and solution" });
  s.addText("Approach: branch each conversation into controlled arms", { placeholder: "title" });
  const box = (x, y, w, h, label, sub, fill, name) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
    s.addText([{ text: label, options: { bold: true, fontSize: 14, breakLine: true } }, { text: sub, options: { fontSize: 11 } }],
      { isTextBox: true, x: x + 0.08, y, w: w - 0.16, h, color: "FFFFFF", align: "center", valign: "middle", margin: 0, objectName: name + " text" });
  };
  const arrow = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: Math.min(y1, y2), w: x2 - x1, h: Math.abs(y2 - y1),
    flipV: y2 < y1, line: { color: "9AA5B4", width: 1.5, endArrowType: "triangle" } });
  box(0.6, 3.15, 1.9, 0.95, "T1", "question  ->  c1", HX.dk2, "T1");
  const t2 = [["T2A", "bluff, no data", HX.accent1, 1.65], ["T2B", "real evidence", HX.accent2, 3.15], ["T2N", "neutral turn", HX.accent4, 4.65]];
  for (const [lab, sub, col, y] of t2) { box(3.2, y, 2.1, 0.95, lab, sub, col, lab); arrow(2.5, 3.62, 3.2, y + 0.47); }
  const t3 = [["T3", "release; model's own concession kept", HX.dk2, 1.35], ["T3a", "concession replaced by \"Noted.\"", HX.accent5, 2.2],
    ["T3b", "concession replaced by stand-firm reply", HX.accent5, 3.05]];
  for (const [lab, sub, col, y] of t3) { box(6.1, y, 3.1, 0.75, lab, sub, col, lab); arrow(5.3, 2.12, 6.1, y + 0.37); }
  box(6.1, 3.9, 3.1, 0.75, "T3c", "bluff exchange removed, then release", HX.accent5, "T3c");
  // T3c branches from T1: route through the gap between T2B and T2N
  s.addShape(pres.shapes.LINE, { x: 2.5, y: 3.9, w: 0.45, h: 0.47, line: { color: "9AA5B4", width: 1.5 } });
  arrow(2.95, 4.37, 6.1, 4.37);
  box(6.1, 4.9, 3.1, 0.75, "T3N", "two neutral turns (control)", HX.accent4, "T3N"); arrow(5.3, 5.12, 6.1, 5.27);
  text(s, `${F.n_arms} arms per item, condition and sample (incl. a blind judge reading of T1 prose and a next-token Yes/No logit probe)`,
    { x: 0.6, y: 6.0, w: 8.7, h: 0.6, fontSize: 12, italic: true, color: MUTED });
  const mets = [["Bluff shift", "|c2A - c1|, net of the neutral turn", HX.accent1], ["Evidence update", "(c2B - c1) x evidence direction", HX.accent2],
    ["Selectivity", "evidence update minus bluff shift", HX.accent3], ["Excess hysteresis", "|c3 - c1| minus |c3N - c1|", HX.dk2],
    ["Context ablations", "|c3x - c1| minus |c3 - c1|", HX.accent5]];
  text(s, "What we measure", { x: 9.75, y: 1.35, w: 3.0, h: 0.4, fontSize: 16, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  for (let i = 0; i < mets.length; i++) {
    const [h, d, col] = mets[i]; const y = 1.85 + i * 0.86;
    s.addShape(pres.shapes.OVAL, { x: 9.75, y: y + 0.08, w: 0.22, h: 0.22, fill: { color: col }, line: { color: col }, objectName: h + " dot" });
    text(s, [{ text: h, options: { bold: true, breakLine: true, fontSize: 14, color: HX.dk1 } }, { text: d, options: { fontSize: 12, color: MUTED } }],
      { x: 10.1, y, w: 2.65, h: 0.8 });
  }
  s.addNotes("T3a and T3b keep the bluff in context and change only the model's own reply to it: this context ablation separates the model's concession from the challenge. T3N gives a depth-matched neutral control, so hysteresis is measured beyond ordinary drift.");

  // ============================================================ 5. Setup
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem and solution" });
  s.addText("Benchmark and runs at a glance", { placeholder: "title" });
  const tiles = [
    [int(F.n_items), "yes/no decision items", `${F.n_domains} domains x ${F.items_per_domain} items: medicine, software, finance, policy`],
    [int(F.n_tiers), "evidence tiers", `${F.items_per_tier} items each; settled, contested, weak (AI-drafted, provisional)`],
    [int(F.n_arms), "conversation arms", `${F.n_paraphrases} paraphrases of every scripted turn`],
    [int(F.n_conditions), "system prompts", "default, stand-firm, contract prompt"],
    [int(G.n_models), "open-weights models", "qwen2.5-7b (AWQ), phi-3.5-mini, mistral-7b on Kaggle T4 GPUs via vLLM"],
    [int(G.n_cells), "item x prompt x sample cells", `samples per cell: ${SUBJ.map((m) => `${m} ${G.samples[m]}`).join(", ")}`],
    [int(G.n_judged), "judged cells (sample 0)", `judge: ${G.judge_model}, blind to model identity`],
    [int(G.bootstrap), "bootstrap resamples", `${pct(G.ci_level)} CIs; items, not rows, are resampled`],
  ];
  for (let i = 0; i < tiles.length; i++) {
    const [v, l, sub] = tiles[i];
    const col = i % 4, row = Math.floor(i / 4);
    const x = 0.6 + col * 3.1, y = 1.5 + row * 2.6;
    card(s, x, y, 2.85, 2.35, HX.lt2, `tile ${i + 1}`);
    stat(s, x + 0.25, y + 0.2, 2.4, v, l, sub, i === 5 ? C.accent3 : C.text2, `tile ${i + 1}`);
  }
  s.addNotes("Everything is computed from logged runs; every call, including failures and retries, is in the raw JSONL logs. A failed parse gets one re-ask and is otherwise NaN, never a default value.");

  // ============================================================ 6. Visual
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("All pre-specified hypotheses in one picture", { placeholder: "title" });
  const fig = path.join(ROOT, "results", "figures", "hypotheses_forest.png");
  const fw = 8.3, fh = fw * 1534 / 2471;
  s.addImage({ path: fig, x: 0.5, y: 1.35, w: fw, h: fh, altText: "Forest plot of hypotheses H1 to H6, pooled and per model, with bootstrap confidence intervals", objectName: "forest plot" });
  card(s, 9.1, 1.45, 3.65, 5.2, HX.lt2, "reading guide");
  text(s, "How to read it", { x: 9.35, y: 1.65, w: 3.2, h: 0.4, fontSize: 18, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  text(s, [
    { text: "Large circle: pooled estimate over models; small markers: each model.", options: { bullet: true, breakLine: true } },
    { text: "Green: CI excludes 0 in the predicted direction; blue: inconclusive.", options: { bullet: true, breakLine: true } },
    { text: `Filled: still supported after Holm adjustment over the ${D.n_pooled_tests} pooled tests (${D.n_supported_holm} of ${D.n_pooled_tests}).`, options: { bullet: true, breakLine: true } },
    { text: "Primary analysis: sample 0, one wording of each scripted turn.", options: { bullet: true } },
  ], { x: 9.35, y: 2.15, w: 3.2, h: 4.3, fontSize: 14, color: C.text1, paraSpaceAfter: 8 });
  s.addNotes("This is Figure 1 of the paper, generated by the analysis script from results/tables/hypotheses.csv. Per-model verdicts are descriptive; only the pooled family is Holm-adjusted.");

  // ============================================================ 7. Performance: headline
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Bluffs move models, and retraction does not fully undo it", { placeholder: "title" });
  const heads = [
    [n(H.H2.est), "Net bluff shift (H2)", `${ci(H.H2)}  |  bluff minus neutral turn, default prompt`, C.accent1],
    [n(H.H4.est), "Excess hysteresis (H4)", `${ci(H.H4)}  |  post-retraction distance beyond the neutral control`, C.text2],
    [n(H.recovery_pooled.est), "Recovery fraction", `${ci(H.recovery_pooled)}  |  share of the bluff-induced move undone on average`, C.accent3],
    [n(H.T3a.est, 3, true), "Concession-only ablation (T3a)", `${ci(H.T3a, 3, true)}  |  small; clearest pooled, not consistent per model`, C.accent5],
  ];
  for (let i = 0; i < heads.length; i++) {
    const [v, l, sub, col] = heads[i];
    stat(s, 0.6 + i * 3.1, 1.4, 2.9, v, l, sub, col, `headline ${i + 1}`);
  }
  s.addChart(pres.charts.BAR, [
    { name: "Net bluff shift (H2)", labels: SUBJ, values: SUBJ.map((m) => D.per_model[m].H2.est) },
    { name: "Excess hysteresis (H4)", labels: SUBJ, values: SUBJ.map((m) => D.per_model[m].H4.est) },
  ], { x: 0.6, y: 3.35, w: 7.6, h: 3.45, barDir: "col", barGrouping: "clustered", chartColors: [HX.accent1, HX.dk2],
    showTitle: true, title: "Per model, default prompt (sample 0)", showLegend: true, legendPos: "b",
    dataLabelPosition: "outEnd", ...chartText, objectName: "per-model chart" });
  card(s, 8.6, 3.45, 4.15, 3.3, HX.lt2, "takeaways");
  text(s, [
    { text: "Every model's net bluff shift and excess hysteresis CI excludes 0.", options: { bullet: true, breakLine: true } },
    { text: `qwen2.5-7b moves most under the bluff; its recovery fraction ${n(D.per_model["qwen2.5-7b"].recovery.est)} ${ci(D.per_model["qwen2.5-7b"].recovery)} is consistent with full return.`, options: { bullet: true, breakLine: true } },
    { text: "Effects of the T3a ablation are consistent with self-conditioning; an ablation does not prove a mechanism.", options: { bullet: true } },
  ], { x: 8.85, y: 3.65, w: 3.7, h: 3.0, fontSize: 14, color: C.text1, paraSpaceAfter: 8 });
  s.addNotes("Pooled over three default-prompted open-weights models, sample 0. The neutral turn moved stated probabilities by only " + n(H.neutral_default.est) + " on average, so the bluff effect is not ordinary drift. Magnitudes are specific to the scripted wording (see robustness).");

  // ============================================================ 8. Performance: prompts
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Can a system prompt fix it? Partly, with trade-offs", { placeholder: "title" });
  const cats = ["Bluff shift", "Neutral shift", "Evidence update", "Excess hysteresis"];
  const keys = ["bluff_shift", "neutral_shift_T2N", "evidence_update", "excess_hysteresis_abs"];
  const condNames = { default: "default", stand_firm: "stand-firm", contract_prompt: "contract prompt" };
  s.addChart(pres.charts.BAR, Object.keys(condNames).map((c) => ({ name: condNames[c], labels: cats, values: keys.map((k) => D.by_condition[c][k].est) })),
    { x: 0.6, y: 1.4, w: 7.7, h: 5.35, barDir: "col", barGrouping: "clustered", chartColors: [HX.accent4, HX.accent5, HX.accent3],
      showTitle: true, title: "Pooled over models, sample 0", showLegend: true, legendPos: "b", dataLabelPosition: "outEnd",
      ...chartText, dataLabelFontSize: 10, objectName: "prompt chart" });
  const pr = [
    [fa.FaShieldAlt, HX.accent5, "Stand-firm prompt (H3)",
      `Bluff shift ${n(H.H3_bluff.est, 3, true)} ${ci(H.H3_bluff, 3, true)} vs default. Evidence-update change ${n(H.H3_evidence.est, 3, true)} ${ci(H.H3_evidence, 3, true)}: no detectable cost, but the CI allows a modest loss.`],
    [fa.FaFileContract, HX.accent3, "Contract prompt (H6)",
      `Excess hysteresis ${n(H.H6_excess.est, 3, true)} ${ci(H.H6_excess, 3, true)}, partly because neutral turns drifted more. Selectivity gain ${n(H.H6_selectivity.est, 3, true)} ${ci(H.H6_selectivity, 3, true)}: inconclusive.`],
    [fa.FaBalanceScaleLeft, HX.accent1, "Default prompt",
      `Selectivity ${n(H.selectivity_default.est)} ${ci(H.selectivity_default)}: inconclusive, so bluff and evidence are not reliably told apart.`],
  ];
  for (let i = 0; i < pr.length; i++) {
    const [Ic, col, head, body] = pr[i]; const y = 1.45 + i * 1.8;
    await circleIcon(s, Ic, 8.65, y + 0.05, 0.62, col, "FFFFFF", head);
    text(s, head, { x: 9.45, y, w: 3.3, h: 0.38, fontSize: 16, bold: true, color: col });
    text(s, body, { x: 9.45, y: y + 0.4, w: 3.3, h: 1.3, fontSize: 12.5, color: C.text1 });
  }
  s.addNotes("Contrasts are paired by item within model. The lesson: claims that a prompt makes a model robust to pushback should be checked against an evidence arm and a neutral control.");

  // ============================================================ 9. Robustness
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Robustness: what holds and what is fragile", { placeholder: "title" });
  const pk = Object.keys(D.paraphrase);
  s.addChart(pres.charts.BAR, [
    { name: "Net bluff shift", labels: pk.map((k) => `Paraphrase ${k}`), values: pk.map((k) => D.paraphrase[k].net_bluff_shift.est) },
    { name: "T3a ablation", labels: pk.map((k) => `Paraphrase ${k}`), values: pk.map((k) => D.paraphrase[k].ablation_T3a_minus_T3.est) },
  ], { x: 0.6, y: 1.4, w: 6.6, h: 5.35, barDir: "col", barGrouping: "clustered", chartColors: [HX.accent1, HX.accent5],
    showTitle: true, title: "Pooled, default prompt, by wording of the scripted turns", showLegend: true, legendPos: "b",
    dataLabelPosition: "outEnd", ...chartText, objectName: "paraphrase chart" });
  const checks = [
    [fa.FaCheckCircle, HX.accent2, "Holds across wordings",
      `Net bluff shift stays positive for all ${pk.length} paraphrases but shrinks to ${n(D.paraphrase[pk[pk.length - 1]].net_bluff_shift.est)} ${ci(D.paraphrase[pk[pk.length - 1]].net_bluff_shift)}. All-samples run: pooled verdicts ${D.all_samples_verdicts_match ? "identical" : "differ"} to the primary run.`],
    [fa.FaCheckCircle, HX.accent2, "Survives multiplicity correction",
      `${D.n_supported_holm} of ${D.n_pooled_tests} pooled tests remain supported after Holm adjustment.`],
    [fa.FaExclamationTriangle, HX.accent3, "Fragile: the T3a effect",
      `Clear only for paraphrase 0 (CI includes 0 for paraphrases 1 and 2). Using first-attempt parses only, Holm p = ${n(D.first_attempt_T3a.holm_p)}.`],
  ];
  for (let i = 0; i < checks.length; i++) {
    const [Ic, col, head, body] = checks[i]; const y = 1.45 + i * 1.8;
    await circleIcon(s, Ic, 7.6, y + 0.05, 0.62, col, "FFFFFF", head);
    text(s, head, { x: 8.4, y, w: 4.35, h: 0.38, fontSize: 16, bold: true, color: C.text2 });
    text(s, body, { x: 8.4, y: y + 0.4, w: 4.35, h: 1.3, fontSize: 13, color: C.text1 });
  }
  s.addNotes("Sample i uses paraphrase i of each scripted turn, so the all-samples run doubles as a paraphrase check. We report the fragile result rather than hide it.");

  // ============================================================ 10. Competition
  pres.addSection({ title: "Landscape" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Landscape" });
  s.addText("Competition and closest work", { placeholder: "title" });
  const hdr = ["Work", "Setting", "Retraction step", "Neutral-length control", "Separates evidence from pressure", "Edits the model's own turn"]
    .map((t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HX.dk2 }, fontSize: 12, valign: "middle" } }));
  const rows = [
    ["This work", "Advisory yes/no decisions; stated P(YES), judge reading, logit probe", "Yes", "Yes (T2N, T3N)", "Yes (twin bluff and evidence turns)", "Yes (T3a, T3b)"],
    ["Shnaidman (2026), Reset Is Not Recovery", "Factual multiple choice; repairs after evidence-free pressure", "Yes", "Yes", "—", "No"],
    ["FlipRate (Gadafi, 2026), same sprint track", "Answer and confidence changes under escalating rebuttals", "No", "—", "Yes", "—"],
    ["Ma et al. (2026)", "Anti-sycophancy can impair rational updating", "—", "—", "Yes", "—"],
    ["Hong et al. (2025); Tang et al. (2026)", "Flips under sustained multi-turn pressure", "—", "—", "—", "—"],
    ["Sinha et al. (2026)", "Editing a model's own earlier outputs (self-conditioning)", "—", "—", "—", "Yes"],
  ].map((r, i) => r.map((t, j) => ({ text: t, options: { fontSize: 12, color: HX.dk1, bold: j === 0 || (i === 0),
    fill: { color: i === 0 ? "FBEFD9" : (i % 2 ? "FFFFFF" : HX.lt2) }, valign: "middle" } })));
  s.addTable([hdr, ...rows], { x: 0.6, y: 1.4, w: 12.1, colW: [2.6, 3.35, 1.35, 1.55, 1.75, 1.5], rowH: 0.6,
    border: { type: "solid", color: "D5DBE3", pt: 0.75 }, margin: 0.06, objectName: "comparison table" });
  text(s, "— = not a focus of, or not described in, that work (our reading). Full references in the paper; all confirmed by the author.",
    { x: 0.6, y: 5.85, w: 12.1, h: 0.3, fontSize: 11, italic: true, color: MUTED });
  text(s, "Where this project sits: retraction recovery, a depth-matched neutral control, bluff-versus-evidence selectivity and a concession-only context ablation, measured together in one design.",
    { x: 0.6, y: 6.2, w: 12.1, h: 0.6, fontSize: 14, bold: true, color: C.text2 });
  s.addNotes("We build on Shnaidman (2026), who coined sycophancy hysteresis, and complement FlipRate, which is concurrent work in this track and also separates warranted updating from capitulation but has no retraction step.");

  // ============================================================ 11. Traction
  pres.addSection({ title: "Progress" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Progress" });
  s.addText("Traction: what was built and released in the sprint", { placeholder: "title" });
  const tr = [
    [int(G.n_cells), "conversation cells run", "three models, three prompts"],
    [int(G.n_judged), "judged cells", "blind LLM judge, sample 0"],
    [int(E.tests), "offline tests", "mock client; no network or keys"],
    [int(E.gates), "automated gates", "checked before submission"],
    [`${E.n_refs_confirmed}/${E.n_refs_cited}`, "references confirmed", "checked by the author"],
    [int(E.raw_log_files), "raw JSONL log files", "every call logged before parsing"],
  ];
  for (let i = 0; i < tr.length; i++) {
    const [v, l, sub] = tr[i]; const x = 0.6 + (i % 3) * 3.0, y = 1.45 + Math.floor(i / 3) * 2.15;
    card(s, x, y, 2.75, 1.95, HX.lt2, `traction ${i + 1}`);
    stat(s, x + 0.25, y + 0.15, 2.3, v, l, sub, C.text2, `traction ${i + 1}`);
  }
  card(s, 9.7, 1.45, 3.05, 4.05, HX.dk2, "status panel");
  text(s, "Status", { x: 9.95, y: 1.65, w: 2.6, h: 0.4, fontSize: 18, bold: true, color: C.background1, fontFace: THEME.headFontFace });
  text(s, [
    { text: "Public repository (MIT) with code, raw logs, tables, figures and paper", options: { bullet: true, breakLine: true } },
    { text: "Every number regenerated by script", options: { bullet: true, breakLine: true } },
    { text: "No external users or adoption yet", options: { bullet: true } },
  ], { x: 9.95, y: 2.15, w: 2.6, h: 3.2, fontSize: 14, color: C.background1, paraSpaceAfter: 8 });
  text(s, "github.com/Gauravsharma2711/ai-sprints-research", { x: 0.6, y: 5.9, w: 12.1, h: 0.4, fontSize: 16, bold: true, color: C.accent2,
    hyperlink: { url: "https://github.com/Gauravsharma2711/ai-sprints-research" } });
  s.addNotes("This is a research sprint, so traction means artefacts others can inspect and reuse, not users. We say plainly that there is no external adoption yet.");

  // ============================================================ 12. Feedback
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Progress" });
  s.addText("Feedback: what review changed", { placeholder: "title" });
  const fh2 = ["Flagged in review", "What we changed", "Result"].map((t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HX.dk2 }, fontSize: 13, valign: "middle" } }));
  const fb = [
    ["Sample 0 always used the first wording of every scripted turn", "Per-paraphrase analysis; all-samples run reframed as a wording check",
      `Bluff effect holds for all wordings (${n(D.paraphrase[pk[0]].net_bluff_shift.est)} to ${n(D.paraphrase[pk[pk.length - 1]].net_bluff_shift.est)}); T3a is wording-specific`],
    ["Values recovered by a re-ask follow an extra user turn", "First-attempt-only re-run of the pooled tests",
      `Verdicts unchanged; T3a not significant after Holm (p = ${n(D.first_attempt_T3a.holm_p)})`],
    [`${D.n_pooled_tests} pooled tests without multiplicity control`, "Holm-adjusted bootstrap p-values", `${D.n_supported_holm} of ${D.n_pooled_tests} remain supported`],
    ["Text implied human review that had not happened", "Truthful AI-use statement; AI-drafted labels disclosed", "Honest scope; human validation listed as next step"],
  ].map((r, i) => r.map((t) => ({ text: t, options: { fontSize: 12.5, color: HX.dk1, fill: { color: i % 2 ? "FFFFFF" : HX.lt2 }, valign: "middle" } })));
  s.addTable([fh2, ...fb], { x: 0.6, y: 1.4, w: 12.1, colW: [3.9, 4.0, 4.2], rowH: 0.78,
    border: { type: "solid", color: "D5DBE3", pt: 0.75 }, margin: 0.08, objectName: "feedback table" });
  await circleIcon(s, fa.FaComments, 0.6, 5.65, 0.7, HX.accent3, "FFFFFF", "feedback icon");
  text(s, "Review so far came from an independent automated (AI) reviewer before submission. No human peer feedback yet: judges' and readers' comments are welcome.",
    { x: 1.5, y: 5.65, w: 11.2, h: 0.8, fontSize: 14, italic: true, color: C.text1, valign: "middle" });
  s.addNotes("Being explicit about the source of feedback: it was an AI reviewer pass, not people. Each change is visible in the repository history and the paper's appendices.");

  // ============================================================ 13. Limitations and next steps
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Progress" });
  s.addText("Limitations and next steps", { placeholder: "title" });
  const lim = [
    `${G.n_models} small default-prompted open-weights models; one AWQ-quantised`,
    `${F.n_items} AI-drafted items; tiers, directions and truth labels not human-verified`,
    "Primary analysis uses one wording of each scripted turn",
    "The LLM judge is not yet validated against human readers",
    "Partial contract-prompt coverage for phi-3.5-mini",
  ];
  const nxt = [
    `Rate the ${D.n_briefings} blind briefings to validate the judge (sheet ready)`,
    "Human tier and label review of every item",
    "More placeholder and release wordings to test the T3a effect",
    "Larger and closed models; live users instead of scripts",
  ];
  const col2 = async (x, Ic, colr, head, items, name) => {
    card(s, x, 1.45, 5.9, 4.25, HX.lt2, name);
    await circleIcon(s, Ic, x + 0.3, 1.7, 0.7, colr, "FFFFFF", head);
    text(s, head, { x: x + 1.2, y: 1.8, w: 4.4, h: 0.5, fontSize: 20, bold: true, color: colr, fontFace: THEME.headFontFace });
    text(s, items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
      { x: x + 0.35, y: 2.65, w: 5.3, h: 2.9, fontSize: 15, color: C.text1, paraSpaceAfter: 10 });
  };
  await col2(0.6, fa.FaExclamationTriangle, HX.accent1, "Limitations", lim, "limitations");
  await col2(6.8, fa.FaRoute, HX.accent2, "Next steps", nxt, "next steps");
  text(s, [{ text: "What would change our conclusions: ", options: { bold: true, color: HX.dk2 } },
    { text: "larger models recovering fully, the T3a effect vanishing under other wordings, or a human check showing the judge does not track human readings.", options: { color: HX.dk1 } }],
    { x: 0.6, y: 5.95, w: 12.1, h: 0.75, fontSize: 15, objectName: "falsifiers" });
  s.addNotes("What would change our conclusions: larger models recovering fully, the T3a effect vanishing with other placeholder wordings, or a human validation showing the judge does not track human readings.");

  // ============================================================ 14. Queries
  pres.addSection({ title: "Close" });
  s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "Close" });
  s.addText("Queries", { placeholder: "title" });
  const qa = [
    ["Is the hysteresis just drift?", `No: it is measured beyond a depth-matched neutral control; pooled excess ${n(H.H4.est)} ${ci(H.H4)}, CI above 0 for every model.`],
    ["Does T3a prove self-conditioning?", "No. It is consistent with self-conditioning, small, and specific to one wording."],
    ["Why these models?", "Default-prompted open-weights models that fit free Kaggle GPUs within a three-day sprint."],
    ["Can I reproduce it?", "Yes: raw logs, configs and scripts regenerate every table, figure and number in the paper."],
  ];
  for (let i = 0; i < qa.length; i++) {
    const [q, a] = qa[i]; const x = 0.6 + (i % 2) * 6.15, y = 1.5 + Math.floor(i / 2) * 2.05;
    card(s, x, y, 5.95, 1.85, "2B3A5C", `question ${i + 1}`);
    await circleIcon(s, fa.FaQuestion, x + 0.25, y + 0.3, 0.55, HX.accent3, HX.dk2, `question icon ${i + 1}`);
    text(s, q, { x: x + 1.0, y: y + 0.25, w: 4.7, h: 0.45, fontSize: 17, bold: true, color: C.accent3 });
    text(s, a, { x: x + 1.0, y: y + 0.75, w: 4.7, h: 1.0, fontSize: 13.5, color: C.background1 });
  }
  text(s, [
    { text: "Gaurav Sharma  |  Independent Researcher", options: { bold: true, breakLine: true } },
    { text: "github.com/Gauravsharma2711/ai-sprints-research", options: { hyperlink: { url: "https://github.com/Gauravsharma2711/ai-sprints-research" }, color: "8FD3C9" } },
  ], { x: 0.6, y: 5.85, w: 12.1, h: 0.8, fontSize: 16, color: C.background1 });
  s.addNotes("Anticipated questions with short answers; open the floor after these.");

  await pres.writeFile({ fileName: outPath });
  if (applyThemePath) {
    const { applyTheme } = require(applyThemePath);
    await applyTheme(outPath, THEME);
  }
  console.log("wrote " + outPath);
}

main().catch((e) => { console.error(e); process.exit(1); });
