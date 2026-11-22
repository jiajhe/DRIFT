// Frozen DRIFT trajectories for two independent single-edge interventions.
const el = id => document.getElementById(id);
const svgNS = "http://www.w3.org/2000/svg";
const years = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
const schedules = {
  today: [false, false, false, false, true],
  five: [true, true, true, true, true],
  restored: [true, true, true, true, false],
};
const explanation = {
  today: "Only day five is masked, so both curves stay at zero through the first four sessions.",
  five: "Each close passes the altered flow state into the next session, compounding with day five’s mask.",
  restored: "The edge is open again on day five. Any remaining difference comes purely from the flow state carried over from the four masked days.",
};

let archive, loading, market = "SP500", mode = "restored";
const selections = { SP500: 0, CSI300: 0 };
const selected = () => archive.markets[market].cases[selections[market]];
const signed = (value, full = false) => {
  if (value === 0) return "0";
  return (value > 0 ? "+" : "") + (full ? String(value) : value.toPrecision(8));
};
const signedEdgeMilli = w => {
  const v = Number(w) * 1000;
  return (v > 0 ? "+" : "") + v.toFixed(2);
};

function mark(parent, tag, attrs, text) {
  const n = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (text !== undefined) n.textContent = text;
  parent.append(n);
  return n;
}

function draw() {
  if (!archive || el("multiday").hidden) return;
  const c = selected(), svg = el("md-chart");
  const width = Math.max(280, Math.round(svg.getBoundingClientRect().width));
  const narrow = width < 540, height = narrow ? 250 : 272;
  const left = narrow ? 58 : 82, right = narrow ? 30 : 52, top = 28, bottom = height - 54;
  const x = i => left + i * (width - left - right) / 4;
  const stocks = [c.sender_a, c.sender_b];
  const series = stocks.map(s => c.trajectories[s][mode].delta);
  const values = stocks.flatMap(s => Object.values(c.trajectories[s]).flatMap(t => t.delta));
  const min = Math.min(0, ...values), max = Math.max(0, ...values);
  const rawStep = (max - min) / 4 || 1, magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const step = [1, 2, 2.5, 5, 10].find(n => n >= rawStep / magnitude) * magnitude;
  const lo = Math.floor(min / step), hi = Math.ceil(max / step);
  const y = v => bottom - (v / step - lo) / (hi - lo) * (bottom - top);

  svg.setAttribute("viewBox", "0 0 " + width + " " + height);
  svg.replaceChildren();
  const defs = mark(svg, "defs", {});
  const ga = mark(defs, "linearGradient", { id: "mdGradA", x1: 0, y1: 0, x2: 0, y2: 1 });
  mark(ga, "stop", { offset: "0%", "stop-color": "#0b6e64", "stop-opacity": "0.14" });
  mark(ga, "stop", { offset: "100%", "stop-color": "#0b6e64", "stop-opacity": "0.01" });
  const gb = mark(defs, "linearGradient", { id: "mdGradB", x1: 0, y1: 0, x2: 0, y2: 1 });
  mark(gb, "stop", { offset: "0%", "stop-color": "#b45309", "stop-opacity": "0.12" });
  mark(gb, "stop", { offset: "100%", "stop-color": "#b45309", "stop-opacity": "0.01" });

  mark(svg, "title", { id: "md-chart-title" }, c.target + " score differences for " + stocks.join(" and ") + ". " + explanation[mode]);
  mark(svg, "rect", { x: x(4) - (narrow ? 20 : 36), y: 12, width: narrow ? 40 : 72, height: bottom - 4, class: "md-final-band" });

  for (let tick = lo; tick <= hi; tick++) {
    const v = tick * step;
    mark(svg, "line", { x1: left - 3, y1: y(v), x2: width - right + 7, y2: y(v), class: tick === 0 ? "md-zero" : "md-gridline" });
    const label = tick === 0 ? "0" : (v > 0 ? "+" : "") + Number(v.toPrecision(4));
    mark(svg, "text", { x: left - 10, y: y(v) + 4, "text-anchor": "end", class: "md-axis" }, label);
  }
  c.window_dates.forEach((date, i) => {
    const masked = schedules[mode][i];
    mark(svg, "line", { x1: x(i), y1: top, x2: x(i), y2: bottom, stroke: "rgba(18,21,26,0.05)" });
    mark(svg, "text", { x: x(i), y: height - 30, "text-anchor": "middle", class: "md-axis md-date" }, years.format(new Date(date + "T00:00:00Z")));
    mark(svg, "text", { x: x(i), y: height - 11, "text-anchor": "middle", class: "md-axis " + (masked ? "md-closed" : "md-open") }, masked ? "MASKED" : i === 4 && mode === "restored" ? "RESTORED" : "OPEN");
  });

  const yZero = y(0);
  series.forEach((points, j) => {
    const areaPts = `${x(0)},${yZero} ` + points.map((v, i) => `${x(i)},${y(v)}`).join(" ") + ` ${x(4)},${yZero}`;
    mark(svg, "polygon", { points: areaPts, fill: j ? "url(#mdGradB)" : "url(#mdGradA)" });
    const group = mark(svg, "g", { class: j ? "md-line-b" : "md-line-a", "data-channel": stocks[j] });
    mark(group, "polyline", { points: points.map((v, i) => x(i) + "," + y(v)).join(" "), fill: "none" });
    points.forEach((v, i) => {
      const dot = mark(group, "circle", { cx: x(i), cy: y(v), r: i === 4 ? 5 : 3.4, "data-session": i + 1 });
      mark(dot, "title", {}, stocks[j] + " · " + c.window_dates[i] + " · " + signed(v, true));
    });
  });
}

function renderTable() {
  const c = selected(), full = el("md-precision").checked;
  const [a, b] = [c.sender_a, c.sender_b].map(s => c.trajectories[s][mode]);
  el("md-th-a").textContent = c.sender_a + " score";
  el("md-th-b").textContent = c.sender_b + " score";
  const body = el("md-table").tBodies[0];
  body.replaceChildren(...c.window_dates.map((d, i) => {
    const row = document.createElement("tr");
    const date = document.createElement("th");
    date.scope = "row"; date.textContent = d; row.append(date);
    for (const value of [c.base_scores[i], a.scores[i], a.delta[i], b.scores[i], b.delta[i]]) {
      const td = row.insertCell(); td.textContent = signed(value, full); td.title = signed(value, true);
    }
    return row;
  }));
}

function render() {
  const c = selected();
  for (const b of el("md-markets").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.market === market));
  for (const b of el("md-modes").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
  const stocks = [c.sender_a, c.sender_b];
  const delta = stocks.map(s => c.trajectories[s][mode].delta[4]);
  stocks.forEach((s, i) => {
    const key = i ? "b" : "a", node = el("md-delta-" + key), edge = c.edges_on_final_day[s];
    node.textContent = signed(delta[i]); node.title = signed(delta[i], true);
    el("md-key-" + key).textContent = s;
    el("md-name-" + key).textContent = "Sender " + key.toUpperCase() + ": " + s;
    const edgeEl = el("md-edge-" + key);
    edgeEl.innerHTML = `${s} → ${c.target}, <span class="math"><var>K</var><sub><var>i</var>,<var>j</var></sub></span> = ${signedEdgeMilli(edge)} <span class="math">×10<sup>−3</sup></span>`;
    edgeEl.title = `${s} → ${c.target}, exact K = ${signed(edge, true)}`;
  });
  const magnitudes = delta.map(Math.abs), largest = Math.max(...magnitudes);
  const gap = largest ? Math.abs(magnitudes[0] - magnitudes[1]) / largest : 0;
  const tie = gap < 0.05;
  el("md-winner").textContent = tie ? "Near Tie" : stocks[magnitudes[0] > magnitudes[1] ? 0 : 1];
  el("md-comparison").textContent = (100 * gap).toFixed(1) + "% gap relative to larger |Δscore|" + (tie ? " (within 5%)" : "");
  el("md-target").textContent = c.target + " from " + c.window_dates[0] + " to " + c.date;
  el("md-mechanism").textContent = explanation[mode];
  el("md-mechanism").classList.toggle("is-restored", mode === "restored");
  el("md-results").hidden = false;
  renderTable(); draw();
}

function setMarket(value) {
  market = value;
  const list = el("md-case");
  list.replaceChildren(...archive.markets[market].cases.map((c, index) => new Option(c.date + " — " + c.target, String(index))));
  list.value = String(selections[market]); list.disabled = false;
  render();
}

el("md-case").addEventListener("change", event => { selections[market] = Number(event.target.value); render(); });
el("md-markets").addEventListener("click", event => { if (archive && event.target.dataset.market) setMarket(event.target.dataset.market); });
el("md-modes").addEventListener("click", event => { if (archive && event.target.dataset.mode) { mode = event.target.dataset.mode; render(); } });
el("md-precision").addEventListener("change", renderTable);
new ResizeObserver(draw).observe(el("md-chart"));

export async function showMultiday() {
  try {
    if (!archive) {
      el("md-status").textContent = "Loading saved replays…";
      loading ||= fetch("./multiday-replay.json").then(r => {
        if (!r.ok) throw Error("Could not load the replay archive.");
        return r.json();
      });
      archive = await loading;
    }
    el("md-status").textContent = "";
    setMarket(market);
  } catch (error) {
    loading = null;
    el("md-status").textContent = error.message;
  }
}
