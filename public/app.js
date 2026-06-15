const state = {
  results: null,
  task: "all",
  model: "all"
};

const colors = {
  llmlingua: "#0f766e",
  hard_prompt_pruning: "#b45309",
  embedding_chunk_drop: "#2563eb",
  kv_cache_eviction: "#7c3aed"
};

async function loadResults() {
  const response = await fetch("/results/latest_results.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load latest_results.json: ${response.status}`);
  state.results = await response.json();
  bindSummary();
  bindFilters();
  render();
}

function bindSummary() {
  const { run_id, n_records, n_examples, pareto = [] } = state.results;
  document.querySelector("#run-id").textContent = run_id || "unknown";
  document.querySelector("#records").textContent = n_records ?? 0;
  document.querySelector("#examples").textContent = n_examples ?? 0;
  document.querySelector("#points").textContent = pareto.length;
}

function bindFilters() {
  const pareto = state.results.pareto || [];
  const tasks = ["all", ...new Set(pareto.map((p) => p.task))];
  const models = ["all", ...new Set(pareto.map((p) => p.model))];
  fillSelect("#task-filter", tasks);
  fillSelect("#model-filter", models);
  document.querySelector("#task-filter").addEventListener("change", (event) => {
    state.task = event.target.value;
    render();
  });
  document.querySelector("#model-filter").addEventListener("change", (event) => {
    state.model = event.target.value;
    render();
  });
}

function fillSelect(selector, values) {
  const select = document.querySelector(selector);
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
}

function filteredPoints() {
  return (state.results.pareto || []).filter((point) => {
    const taskOk = state.task === "all" || point.task === state.task;
    const modelOk = state.model === "all" || point.model === state.model;
    return taskOk && modelOk;
  });
}

function render() {
  const points = filteredPoints();
  drawChart(points);
  renderTable(points);
}

function drawChart(points) {
  const canvas = document.querySelector("#pareto");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const pad = { left: 76, right: 32, top: 38, bottom: 76 };
  const plotW = canvas.width - pad.left - pad.right;
  const plotH = canvas.height - pad.top - pad.bottom;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#d8dee8";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.left, pad.top, plotW, plotH);

  for (let i = 0; i <= 5; i += 1) {
    const x = pad.left + (plotW * i) / 5;
    const y = pad.top + (plotH * i) / 5;
    ctx.strokeStyle = "#edf1f7";
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#5d6779";
  ctx.font = "14px system-ui";
  ctx.fillText("Tokens saved (%)", pad.left + plotW / 2 - 54, canvas.height - 24);
  ctx.save();
  ctx.translate(22, pad.top + plotH / 2 + 54);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Quality retained", 0, 0);
  ctx.restore();

  points.forEach((point) => {
    const x = pad.left + (point.tokens_saved_pct / 100) * plotW;
    const y = pad.top + (1 - point.quality_retained) * plotH;
    const radius = Math.max(5, Math.min(18, 5 + point.latency_ms / 80));
    ctx.fillStyle = colors[point.strategy] || "#1d2433";
    ctx.globalAlpha = 0.86;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#1d2433";
    ctx.font = "12px system-ui";
    ctx.fillText(point.strategy.replaceAll("_", " "), x + radius + 5, y + 4);
  });

  if (!points.length) {
    ctx.fillStyle = "#5d6779";
    ctx.font = "18px system-ui";
    ctx.fillText("No points for this filter.", pad.left + 24, pad.top + 44);
  }
}

function renderTable(points) {
  const tbody = document.querySelector("#frontier-table");
  tbody.innerHTML = "";
  points
    .slice()
    .sort((a, b) => b.tokens_saved_pct - a.tokens_saved_pct)
    .forEach((point) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${point.task}</td>
        <td>${point.model}</td>
        <td>${point.strategy}</td>
        <td>${Number(point.ratio).toFixed(2)}</td>
        <td>${Number(point.tokens_saved_pct).toFixed(1)}%</td>
        <td>${Number(point.quality_retained).toFixed(3)}</td>
        <td>${Number(point.latency_ms).toFixed(1)} ms</td>
        <td>${point.n_examples}</td>
      `;
      tbody.appendChild(row);
    });
}

loadResults().catch((error) => {
  document.querySelector("#summary").innerHTML = `<div class="metric"><strong>${error.message}</strong></div>`;
});
