// RenameRisk dashboard. All user-controlled text is inserted with textContent; no HTML strings are ever parsed.
"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };
const fmt = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v)) ? "n/a" : Number(v).toFixed(d);
const LABEL = { main: "Main 40 (primary)", comparison: "Comparison 42 (secondary)" };
const RING = 326.7;

let MODELS = null, RESULT = null, FILE = null, toastTimer = null;

function toast(title, text, isError) {
  const t = $("toast"); t.replaceChildren(el("b", title), el("span", text));
  t.className = "toast" + (isError ? " err" : ""); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 4500);
}

async function init() {
  try {
    const h = await (await fetch("/health")).json();
    $("health").textContent = h.models_loaded ? "● Models loaded" : "● Models unavailable";
    $("health").className = "status " + (h.models_loaded ? "ok" : "bad");
    if (!h.models_loaded) { showError("Models unavailable: " + (h.error || "unknown error")); return; }
    MODELS = await (await fetch("/api/models")).json();
    renderModelCards();
  } catch (e) { showError("The backend is not reachable."); }
}

function renderModelCards() {
  const cards = $("model-cards"); cards.replaceChildren();
  for (const fs of ["main", "comparison"]) {
    const m = MODELS.models[fs], t = MODELS.evaluation.test_results[fs], d = MODELS.evaluation.development_cv_ap[fs];
    const c = el("div", undefined, "mcard");
    c.append(el("h4", LABEL[fs] + " · Logistic Regression + Yeo-Johnson"));
    c.append(el("div", "Held-out AP " + fmt(t.average_precision), "big"));
    c.append(el("div", `random baseline ${fmt(t.prevalence)} · ROC-AUC ${fmt(t.roc_auc, 3)}`, "muted small"));
    c.append(el("div", `development CV AP ${fmt(d.mean)} ± ${fmt(d.std)}`, "muted small"));
    c.append(el("div", `threshold ${fmt(m.threshold, 6)} · held-out TP ${t.tp}, FP ${t.fp}, FN ${t.fn}`, "muted small"));
    cards.append(c);
  }
  const ds = MODELS.dataset, info = el("div", undefined, "mcard");
  info.append(el("h4", "Training data"));
  info.append(el("div", `${ds.rows.toLocaleString()} methods`, "big"));
  info.append(el("div", `${ds.positives} automatic rename labels · ${ds.projects} projects`, "muted small"));
  info.append(el("div", `fitted on ${ds.development_projects} development projects; evaluated once on ${ds.test_projects} unseen projects`, "muted small"));
  cards.append(info);
  $("about").textContent = MODELS.review_cutoff_note + " " + MODELS.threshold_note + " Comparison 42 adds file size (file_loc, file_n_methods); 5 of its "
    + "6 held-out true positives came from a single project, so its higher score is fragile.";
}

function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = !msg; }

function chooseFile(f) {
  FILE = f || null; showError("");
  if (!FILE) { $("file-info").textContent = "No file selected"; $("analyze-btn").disabled = true; return; }
  $("file-info").textContent = `${FILE.name} · ${(FILE.size / 1024).toFixed(1)} KB`;
  if (!FILE.name.toLowerCase().endsWith(".py")) { showError("Only .py files can be analysed."); $("analyze-btn").disabled = true; return; }
  if (FILE.size > 2 * 1024 * 1024) { showError("File too large (limit 2 MB)."); $("analyze-btn").disabled = true; return; }
  $("analyze-btn").disabled = !MODELS;
}

async function analyze() {
  if (!FILE) return;
  const body = new FormData(); body.append("file", FILE);
  const rp = $("relpath").value.trim(); if (rp) body.append("relative_path", rp);
  $("analyze-btn").disabled = true; $("spinner").hidden = false; showError(""); $("detail").hidden = true;
  try {
    const r = await fetch("/api/analyze", { method: "POST", body });
    const data = await r.json();
    if (!r.ok) { showError(data.detail || "Analysis failed."); toast("Analysis failed", data.detail || "", true); return; }
    RESULT = data; renderResults();
    toast("Analysis complete", `${data.n_scored} method(s) scored · ${data.above_review_cutoff.main} at or above the ${fmt(data.review_cutoff, 2)} review cutoff (Main 40)`);
  } catch (e) { showError("The request failed."); toast("Request failed", "Is the server still running?", true); }
  finally { $("analyze-btn").disabled = false; $("spinner").hidden = true; }
}

function renderResults() {
  $("results").hidden = false; $("res-file").textContent = "· " + RESULT.file;
  $("s-scored").textContent = RESULT.n_scored;
  const cut = fmt(RESULT.review_cutoff, 2);
  $("s-main").textContent = RESULT.above_review_cutoff.main;
  $("s-comp").textContent = RESULT.above_review_cutoff.comparison;
  $("s-main-lbl").textContent = `At or above ${cut} review cutoff · Main 40`;
  $("s-comp-lbl").textContent = `At or above ${cut} review cutoff · Comparison 42`;
  $("cutoff-note").textContent = `Review cutoff ${cut} is user-chosen and not validated: it suggests where to look, not which methods will be renamed. Validated threshold (method details): above it · Main 40: ${RESULT.above_threshold.main}, Comparison 42: ${RESULT.above_threshold.comparison}.`;
  $("s-skip").textContent = RESULT.skipped.length;
  const top = RESULT.methods[0];                      // the backend sorts by the main 40 score
  const best = top ? top.models.main.score : 0;
  $("ring-val").textContent = top ? fmt(best, 2) : "–";
  $("ring-fg").style.strokeDashoffset = String(RING * (1 - best));
  $("top-method").textContent = top ? `${top.class ? top.class + "." : ""}${top.name}()  ·  line ${top.lineno}  ·  `
    + (top.models.main.above_review_cutoff ? `at or above the ${cut} review cutoff` : `below the ${cut} review cutoff`) : "No method in scope.";
  $("skipped").textContent = RESULT.n_scored === 0
    ? "No method in this file is inside the training scope (module-level functions and class methods, excluding dunder and test* names)."
    : (RESULT.skipped.length ? "Not scored: " + RESULT.skipped.map(x => `${x.name} (line ${x.lineno}: ${x.reason})`).join("; ") : "");
  renderList();
  $("results-anchor").scrollIntoView({ behavior: "smooth" });
}

function badge(above, cutoff) { return el("span", above ? `▲ REVIEW (≥ ${fmt(cutoff, 2)})` : `BELOW ${fmt(cutoff, 2)} CUTOFF`, "badge " + (above ? "above" : "below")); }

function scoreBlock(label, score) {
  const s = el("div", undefined, "score"); s.append(el("span", label, "lbl"), el("b", fmt(score, 3)));
  const bar = el("div", undefined, "bar"), fill = el("i"); fill.style.width = Math.round(score * 100) + "%"; bar.append(fill); s.append(bar);
  return s;
}

function renderList() {
  const q = $("search").value.trim().toLowerCase(), key = $("sort").value;
  let rows = RESULT.methods.filter(m => !q || m.name.toLowerCase().includes(q) || (m.class || "").toLowerCase().includes(q));
  rows = rows.slice().sort((a, b) => key === "name" ? a.name.localeCompare(b.name)
    : key === "lineno" ? a.lineno - b.lineno : b.models[key].score - a.models[key].score);
  const list = $("list"); list.replaceChildren();
  if (!rows.length) { list.append(el("div", RESULT.methods.length ? "No method matches the search." : "No methods to show.", "card empty muted")); return; }
  rows.forEach((m, i) => {
    const item = el("div", undefined, "item"); item.tabIndex = 0;
    const who = el("div"); who.append(el("div", m.name + "()", "mname"),
      el("div", `${m.class || "module level"} · line ${m.lineno}` + (m.is_async ? " · async" : "") + (m.decorators ? ` · ${m.decorators} decorator(s)` : ""), "mloc"));
    const scores = el("div", undefined, "score"); scores.append(scoreBlock("Main 40", m.models.main.score), scoreBlock("Comparison 42", m.models.comparison.score));
    const badges = el("div", undefined, "badges"); badges.append(badge(m.models.main.above_review_cutoff, m.models.main.review_cutoff));
    item.append(el("div", "#" + (i + 1), "rank"), who, scores, badges);
    const open = () => showDetail(m, item);
    item.addEventListener("click", open);
    item.addEventListener("keydown", e => { if (e.key === "Enter") open(); });
    list.append(item);
  });
}

function showDetail(m, item) {
  document.querySelectorAll(".item.sel").forEach(r => r.classList.remove("sel")); item.classList.add("sel");
  $("detail").hidden = false;
  $("d-title").textContent = (m.class ? m.class + "." : "") + m.name + "()";
  $("d-loc").textContent = `${RESULT.file} · lines ${m.lineno}–${m.end_lineno}` + (m.is_async ? " · async" : "") + (m.decorators ? ` · ${m.decorators} decorator(s)` : "");
  const cards = $("d-models"); cards.replaceChildren(); const contrib = $("d-contrib"); contrib.replaceChildren();
  for (const fs of ["main", "comparison"]) {
    const mm = m.models[fs], c = el("div", undefined, "mcard");
    const head = el("h4", LABEL[fs] + " "); head.append(badge(mm.above_review_cutoff, mm.review_cutoff)); c.append(head);
    c.append(el("div", "Model score " + fmt(mm.score, 4), "big"));
    c.append(el("div", `${mm.above_review_cutoff ? "At or above" : "Below"} the ${fmt(mm.review_cutoff, 2)} review cutoff (user-chosen, not validated) · rank ${mm.rank} of ${RESULT.n_scored} in this file`, "muted small"));
    c.append(el("div", `Validated Stage 8 threshold ${fmt(mm.threshold, 4)}: ${mm.above_threshold ? "above" : "below"}`, "muted small"));
    cards.append(c);
    const box = el("div", undefined, "contrib"); box.append(el("h4", LABEL[fs] + ": largest contributions", "muted"));
    for (const t of mm.top_contributions) {
      const row = el("div"); row.append(el("span", `${t.feature} = ${t.raw_value}`), el("span", (t.contribution >= 0 ? "+" : "") + fmt(t.contribution, 3), t.contribution >= 0 ? "pos" : "neg"));
      box.append(row);
    }
    box.append(el("div", `intercept ${fmt(mm.intercept, 3)} · logit ${fmt(mm.logit, 3)}`, "muted"));
    contrib.append(box);
  }
  const mainSet = new Set(MODELS.models.main.features), tb = $("d-features").querySelector("tbody"); tb.replaceChildren();
  for (const [f, v] of Object.entries(m.features)) {
    const tr = el("tr"); tr.append(el("td", f), el("td", String(v), "num"), el("td", mainSet.has(f) ? "both models" : "Comparison 42 only", "muted")); tb.append(tr);
  }
  $("d-source").textContent = m.source;
  $("detail").scrollIntoView({ behavior: "smooth" });
}

const drop = $("drop");
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => chooseFile(e.dataTransfer.files[0]));
drop.addEventListener("keydown", e => { if (e.key === "Enter") $("file").click(); });
$("file").addEventListener("change", e => chooseFile(e.target.files[0]));
$("analyze-btn").addEventListener("click", analyze);
$("search").addEventListener("input", () => RESULT && renderList());
$("sort").addEventListener("change", () => RESULT && renderList());
$("d-close").addEventListener("click", () => { $("detail").hidden = true; document.querySelectorAll(".item.sel").forEach(r => r.classList.remove("sel")); });
document.querySelectorAll(".nav").forEach(a => a.addEventListener("click", () => {
  document.querySelectorAll(".nav").forEach(x => x.classList.remove("active")); a.classList.add("active");
}));
init();
