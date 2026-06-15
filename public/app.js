const COLORS = {
  llmlingua: "#2dd4bf",
  embedding_chunk_drop: "#818cf8",
  kv_cache_eviction: "#f472b6",
  hard_prompt_pruning: "#f59e0b"
};
const FALLBACK = "#93a1b5";

const LABELS = {
  rag_qa: "RAG QA",
  agent_traces: "Agent traces",
  long_context_summarization: "Long-context summ."
};

const state = { data: null, task: "all", model: "all", hover: null, points: [] };

const $ = (s) => document.querySelector(s);
const fmt = (n, d = 1) => Number(n).toFixed(d);
const prettyStrat = (s) => s.replaceAll("_", " ");
const prettyTask = (t) => LABELS[t] || t;
const color = (s) => COLORS[s] || FALLBACK;

async function load() {
  const res = await fetch("/results/latest_results.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`latest_results.json → HTTP ${res.status}`);
  state.data = await res.json();
  let clf = null;
  try {
    const cr = await fetch("/results/classifier_report.json", { cache: "no-store" });
    if (cr.ok) clf = await cr.json();
  } catch (_) {}
  state.clf = clf;

  bindStats();
  bindSegments();
  renderFindings();
  renderStrategies();
  renderClassifier();
  render();

  const c = $("#pareto");
  c.addEventListener("mousemove", onMove);
  c.addEventListener("mouseleave", () => {
    state.hover = null;
    $("#tooltip").style.opacity = 0;
    drawChart(currentPoints());
  });
}

function aggregates() {
  // Prefer full aggregates (with on_frontier); fall back to legacy `pareto`.
  const d = state.data;
  if (d.aggregates && d.aggregates.length) return d.aggregates;
  return (d.pareto || []).map((p) => ({ ...p, on_frontier: true }));
}

function bindStats() {
  const d = state.data;
  const strats = new Set(aggregates().map((p) => p.strategy));
  $("#s-models").textContent = (d.models || []).length;
  $("#s-strats").textContent = strats.size;
  $("#s-tasks").textContent = (d.tasks || []).length;
  $("#s-examples").textContent = d.n_examples ?? "—";
  $("#s-records").textContent = (d.n_records ?? 0).toLocaleString();
  $("#f-run").textContent = d.run_id || "—";
  $("#f-date").textContent = (d.created_at || "").replace("T", " ").replace("Z", " UTC");
}

function bindSegments() {
  const aggs = aggregates();
  const tasks = ["all", ...new Set(aggs.map((p) => p.task))];
  const models = ["all", ...new Set(aggs.map((p) => p.model))];
  fillSeg("#task-seg", tasks, "task", (v) => (v === "all" ? "All tasks" : prettyTask(v)));
  fillSeg("#model-seg", models, "model", (v) => (v === "all" ? "All" : v));
}

function fillSeg(sel, values, key, label) {
  const el = $(sel);
  el.innerHTML = "";
  values.forEach((v) => {
    const b = document.createElement("button");
    b.textContent = label(v);
    if (state[key] === v) b.classList.add("active");
    b.onclick = () => {
      state[key] = v;
      [...el.children].forEach((c) => c.classList.remove("active"));
      b.classList.add("active");
      render();
    };
    el.appendChild(b);
  });
}

function currentPoints() {
  return aggregates().filter(
    (p) => (state.task === "all" || p.task === state.task) && (state.model === "all" || p.model === state.model)
  );
}

function render() {
  const pts = currentPoints();
  renderLegend(pts);
  drawChart(pts);
  renderTable(pts);
}

/* ---------------- Findings ---------------- */
function renderFindings() {
  const aggs = aggregates();
  const tasks = [...new Set(aggs.map((p) => p.task))];
  const wrap = $("#findings-cards");
  wrap.innerHTML = "";
  tasks.forEach((task) => {
    const pool = aggs.filter((p) => p.task === task && p.on_frontier && p.tokens_saved_pct >= 40);
    const candidates = pool.length ? pool : aggs.filter((p) => p.task === task && p.on_frontier);
    if (!candidates.length) return;
    const best = candidates.slice().sort((a, b) => b.quality_retained - a.quality_retained)[0];
    const el = document.createElement("div");
    el.className = "finding";
    el.style.setProperty("--accent", color(best.strategy));
    el.innerHTML = `
      <div class="task-tag">${prettyTask(task)}</div>
      <div class="winner">${prettyStrat(best.strategy)}</div>
      <div class="detail"><b>${fmt(best.tokens_saved_pct)}%</b> tokens saved · <b>${fmt(best.quality_retained * 100)}%</b> quality · <b>${fmt(best.latency_ms)} ms</b> · ${best.model}</div>`;
    wrap.appendChild(el);
  });
}

/* ---------------- Legend ---------------- */
function renderLegend(points) {
  const strats = [...new Set(points.map((p) => p.strategy))];
  $("#legend").innerHTML = strats
    .map((s) => `<span class="item"><span class="dot" style="background:${color(s)}"></span>${prettyStrat(s)}</span>`)
    .join("");
}

/* ---------------- Chart ---------------- */
function chartGeom(canvas) {
  return { pad: { l: 70, r: 28, t: 26, b: 64 }, w: canvas.width, h: canvas.height };
}

function projectPoint(p, g) {
  const plotW = g.w - g.pad.l - g.pad.r;
  const plotH = g.h - g.pad.t - g.pad.b;
  const x = g.pad.l + (p.tokens_saved_pct / 100) * plotW;
  const y = g.pad.t + (1 - clampQ(p.quality_retained)) * plotH;
  const r = Math.max(6, Math.min(22, 6 + Math.sqrt(p.latency_ms) * 1.6));
  return { x, y, r };
}
const clampQ = (q) => Math.max(0, Math.min(1, q));

function drawChart(points) {
  const canvas = $("#pareto");
  const ctx = canvas.getContext("2d");
  const g = chartGeom(canvas);
  const plotW = g.w - g.pad.l - g.pad.r;
  const plotH = g.h - g.pad.t - g.pad.b;
  ctx.clearRect(0, 0, g.w, g.h);

  // grid
  ctx.font = "13px JetBrains Mono, monospace";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 5; i++) {
    const gx = g.pad.l + (plotW * i) / 5;
    const gy = g.pad.t + (plotH * i) / 5;
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(gx, g.pad.t);
    ctx.lineTo(gx, g.pad.t + plotH);
    ctx.moveTo(g.pad.l, gy);
    ctx.lineTo(g.pad.l + plotW, gy);
    ctx.stroke();
    ctx.fillStyle = "#5f6e84";
    ctx.textAlign = "center";
    ctx.fillText(`${(i / 5) * 100}`, gx, g.pad.t + plotH + 18);
    ctx.textAlign = "right";
    ctx.fillText(`${(1 - i / 5).toFixed(1)}`, g.pad.l - 12, gy);
  }

  // axis labels
  ctx.fillStyle = "#93a1b5";
  ctx.font = "13px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Tokens saved (%)", g.pad.l + plotW / 2, g.h - 14);
  ctx.save();
  ctx.translate(20, g.pad.t + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Quality retained", 0, 0);
  ctx.restore();

  // frontier connector — only meaningful within a single (task, tokenizer) slice
  const singleSlice = state.task !== "all" && state.model !== "all";
  const frontier = singleSlice
    ? points.filter((p) => p.on_frontier).sort((a, b) => a.tokens_saved_pct - b.tokens_saved_pct)
    : [];
  if (frontier.length > 1) {
    ctx.strokeStyle = "rgba(45,212,191,0.35)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    frontier.forEach((p, i) => {
      const { x, y } = projectPoint(p, g);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // points (draw hovered last)
  const ordered = points.slice().sort((a, b) => (a === state.hover ? 1 : 0) - (b === state.hover ? 1 : 0));
  state.points = [];
  ordered.forEach((p) => {
    const { x, y, r } = projectPoint(p, g);
    state.points.push({ p, x, y, r });
    const c = color(p.strategy);
    const isHover = p === state.hover;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = c;
    ctx.globalAlpha = isHover ? 0.95 : 0.7;
    ctx.fill();
    ctx.globalAlpha = 1;
    if (p.on_frontier) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = isHover ? "#ffffff" : c;
      ctx.beginPath();
      ctx.arc(x, y, r + 3.5, 0, Math.PI * 2);
      ctx.stroke();
    }
    if (isHover) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.stroke();
    }
  });

  if (!points.length) {
    ctx.fillStyle = "#5f6e84";
    ctx.font = "16px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No points for this filter.", g.w / 2, g.h / 2);
  }
}

function onMove(e) {
  const canvas = $("#pareto");
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;
  let found = null;
  let bd = Infinity;
  state.points.forEach(({ p, x, y, r }) => {
    const d = Math.hypot(mx - x, my - y);
    if (d <= r + 3 && d < bd) {
      bd = d;
      found = { p, x, y };
    }
  });
  const tip = $("#tooltip");
  if (found) {
    state.hover = found.p;
    tip.innerHTML = `
      <div class="t-strat" style="color:${color(found.p.strategy)}">${prettyStrat(found.p.strategy)}${found.p.on_frontier ? " ◯" : ""}</div>
      <div class="t-row">${prettyTask(found.p.task)} · ${found.p.model} · ratio ${fmt(found.p.ratio, 2)}</div>
      <div class="t-row">saved ${fmt(found.p.tokens_saved_pct)}% · quality ${fmt(found.p.quality_retained * 100)}% · ${fmt(found.p.latency_ms)} ms</div>`;
    tip.style.left = `${found.x / scaleX}px`;
    tip.style.top = `${found.y / scaleY}px`;
    tip.style.opacity = 1;
    canvas.style.cursor = "pointer";
  } else {
    state.hover = null;
    tip.style.opacity = 0;
    canvas.style.cursor = "default";
  }
  drawChart(currentPoints());
}

/* ---------------- Strategy scorecard ---------------- */
const STRAT_DESC = {
  hard_prompt_pruning: "Deterministic head/tail token-budget truncation. Near-zero latency baseline.",
  embedding_chunk_drop: "Drops least-relevant chunks by TF-IDF cosine similarity to the query.",
  kv_cache_eviction: "Keeps prefix + recent suffix + salient middle — a text proxy for cache eviction.",
  llmlingua: "Wraps Microsoft LLMLingua-2 learned token compression (optional dependency)."
};

function renderStrategies() {
  const rows = (state.data.by_strategy || []).slice();
  const grid = $("#strat-grid");
  grid.innerHTML = "";
  if (!rows.length) return;
  const maxSaved = Math.max(...rows.map((r) => r.tokens_saved_pct), 1);
  const maxLat = Math.max(...rows.map((r) => r.latency_ms), 1);
  rows.forEach((r) => {
    const c = color(r.strategy);
    const el = document.createElement("div");
    el.className = "panel strat";
    el.innerHTML = `
      <h3><span class="dot" style="background:${c}"></span>${prettyStrat(r.strategy)}</h3>
      <div class="sub">${STRAT_DESC[r.strategy] || ""}</div>
      ${bar("Quality retained", r.quality_retained * 100, 100, "%", c)}
      ${bar("Tokens saved", r.tokens_saved_pct, maxSaved, "%", c)}
      ${bar("Latency (median)", r.latency_ms, maxLat, " ms", "#5f6e84")}`;
    grid.appendChild(el);
  });
}

function bar(label, value, max, unit, c) {
  const pct = Math.max(2, Math.min(100, (value / max) * 100));
  const shown = unit === "%" ? fmt(value) : fmt(value, value < 10 ? 2 : 1);
  return `
    <div class="bar">
      <div class="bl"><span>${label}</span><b>${shown}${unit}</b></div>
      <div class="track"><div class="fill" style="width:${pct}%;background:${c}"></div></div>
    </div>`;
}

/* ---------------- Table ---------------- */
function renderTable(points) {
  const tbody = $("#frontier-table");
  tbody.innerHTML = "";
  points
    .slice()
    .sort((a, b) => Number(b.on_frontier) - Number(a.on_frontier) || b.tokens_saved_pct - a.tokens_saved_pct)
    .forEach((p) => {
      const tr = document.createElement("tr");
      if (p.on_frontier) tr.className = "frontier";
      tr.innerHTML = `
        <td>${prettyTask(p.task)}</td>
        <td>${p.model}</td>
        <td><span class="strat-tag"><span class="dot" style="background:${color(p.strategy)}"></span>${prettyStrat(p.strategy)}</span></td>
        <td class="num">${fmt(p.ratio, 2)}</td>
        <td class="num">${fmt(p.tokens_saved_pct)}%</td>
        <td class="num">${fmt(p.quality_retained, 3)}</td>
        <td class="num">${fmt(p.latency_ms)} ms</td>
        <td class="num">${p.n_examples}</td>
        <td>${p.on_frontier ? '<span class="pill front">frontier</span>' : ""}</td>`;
      tbody.appendChild(tr);
    });
}

/* ---------------- Classifier ---------------- */
function renderClassifier() {
  const grid = $("#clf-grid");
  const c = state.clf;
  if (!c) {
    grid.innerHTML = `<div class="panel clf-card"><h3>Classifier report not found</h3><div class="sub">Run <code>compress-bench train-classifier</code> to generate <code>public/results/classifier_report.json</code>.</div></div>`;
    return;
  }
  const drop = (c.top_droppable_tokens || []).slice(0, 10);
  const keep = (c.top_keep_tokens || []).slice(0, 10);
  grid.innerHTML = `
    <div class="panel clf-card">
      <h3>Held-out performance</h3>
      <div class="sub">${(c.n_tokens || 0).toLocaleString()} labeled tokens · trained on <b>${c.model_id || "gpt-4o"}</b> · ${(c.droppable_rate * 100).toFixed(0)}% droppable baseline</div>
      <div class="metric-row">
        <div class="kmetric"><div class="v">${fmt(c.roc_auc, 3)}</div><div class="k">ROC-AUC</div></div>
        <div class="kmetric"><div class="v">${fmt(c.f1, 3)}</div><div class="k">F1 (droppable)</div></div>
        <div class="kmetric"><div class="v">${fmt(c.accuracy * 100)}%</div><div class="k">Accuracy</div></div>
        <div class="kmetric"><div class="v">${fmt(c.precision, 2)}/${fmt(c.recall, 2)}</div><div class="k">Precision / Recall</div></div>
      </div>
    </div>
    <div class="panel clf-card">
      <h3>What it learned</h3>
      <div class="sub">Most confidently <span style="color:var(--accent-2)">droppable</span> vs. <span style="color:var(--good)">keep</span> tokens on held-out data.</div>
      <div style="font-size:12px;color:var(--faint);margin-bottom:4px">Droppable</div>
      <div class="chips">${drop.map((t) => `<span class="chip drop">${escapeHtml(t)}</span>`).join("") || "—"}</div>
      <div style="font-size:12px;color:var(--faint);margin:14px 0 4px">Keep</div>
      <div class="chips">${keep.map((t) => `<span class="chip keep">${escapeHtml(t)}</span>`).join("") || "—"}</div>
    </div>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

load().catch((err) => {
  document.querySelector("main").insertAdjacentHTML(
    "afterbegin",
    `<div class="error-box">Failed to load benchmark data: ${escapeHtml(err.message)}<br/>Run <code>compress-bench run</code> first, then reload.</div>`
  );
});
