import { staticRequest } from "./archive.js";
import { showMultiday } from "./multiday.js";
import { showStarmap } from "./starmap.js";

const $ = id => document.getElementById(id);
const svgNS = "http://www.w3.org/2000/svg";
const SECTORS = {
  AAPL: "Hardware", MSFT: "Software", NVDA: "Semiconductors",
  GOOGL: "Internet", AMZN: "Retail & Cloud", META: "Social & Ads",
  TSLA: "Autos & EV", JPM: "Banking",
};

const state = {
  dates: [], targets: [], target: "", date: "", day: null,
  selected: "", maskedSender: "", maskedDelta: null,
  busy: false, playing: false, version: 0,
  hoveredSender: "", nodes: [], animFrame: 0, shockStart: 0,
};

const base = (window.MARKETFLOW_API || "").replace(/\/$/, "");
const fmt = n => Number(n).toPrecision(5);
const signed = n => (n > 0 ? "+" : "") + fmt(n);
const signedEdgeMilli = w => {
  const v = Number(w) * 1000;
  return (v > 0 ? "+" : "") + v.toFixed(2);
};
const number = n => new Intl.NumberFormat("en-US").format(n);
const title = word => word.toLowerCase().replaceAll("_", " ").replace(/^./, c => c.toUpperCase());
const pane = () => ["starmap", "explorer", "replay", "multiday", "model"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "starmap";

function node(tag, value = "", className = "") {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (value !== undefined && value !== null) el.textContent = value;
  return el;
}
function notice(message) { $("status").textContent = message; $("status").hidden = !message; }

let offline = false;
async function request(path, body) {
  if (!offline) {
    let res;
    try {
      res = await fetch(base + "/api/" + path, {
        ...(body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}),
        signal: AbortSignal.timeout(path === "copilot" ? 60000 : 2500),
      });
    } catch { /* Static host uses packaged real model outputs */ }
    if (res && (res.headers.get("content-type") || "").includes("application/json")) {
      const result = await res.json();
      if (res.ok) return result;
      const staticNotFound = res.status === 404 && (!result.detail || result.detail === "Not Found");
      if (res.status < 500 && !staticNotFound)
        throw Error(typeof result.detail === "string" ? result.detail : "Observation not found.");
    }
    offline = true;
    const catalog = await staticRequest("catalog");
    $("source-mode").textContent = `Archived inference (${catalog.dates.length} dates, ${catalog.targets.length} stocks)`;
    $("ask").textContent = "Read analysis";
    state.dates = catalog.dates;
    options($("date"), catalog.dates, true);
    const requested = new URL(path, "https://marketflow.invalid/").searchParams.get("date") || body?.date || state.date;
    if (requested && !catalog.dates.includes(requested)) {
      if (path.startsWith("catalog")) return catalog;
      if (path !== "evaluation") {
        await load(catalog.date, state.target);
        throw Error("Switched to the first saved date in the archive.");
      }
    }
    if (requested && catalog.dates.includes(requested)) {
      const available = await staticRequest("catalog?date=" + encodeURIComponent(requested));
      $("date").value = requested;
      options($("symbols"), available.targets, false);
      options($("pair-symbols"), available.targets.filter(s => s !== state.target), false);
      if (state.target && !available.targets.includes(state.target) && !path.startsWith("catalog") && path !== "evaluation") {
        await load(requested, available.targets[0]);
        throw Error("That ticker is not in the saved archive for this date. Showing " + available.targets[0] + ".");
      }
    }
  }
  return staticRequest(path, body);
}

const options = (el, values, select) => el.replaceChildren(...values.map(value => {
  const item = document.createElement("option");
  item.value = value;
  if (select) item.textContent = value;
  return item;
}));
const payload = () => ({ market: "SP500", date: state.date, target: state.target });

function stop() { state.playing = false; $("play").textContent = "Play"; }
function busy(value) {
  state.busy = value;
  $("filters").classList.toggle("is-loading", value);
  for (const id of ["date", "ticker", "load", "ask", "mask"]) $(id).disabled = value;
  const i = state.dates.indexOf(state.date);
  $("prev").disabled = value || i <= 0;
  $("next").disabled = value || i >= state.dates.length - 1;
}

function renderQuickTickers(targets) {
  const wrap = $("quick-tickers");
  wrap.replaceChildren(...targets.slice(0, 10).map(sym => {
    const b = node("button", sym, "ticker-pill");
    b.type = "button";
    b.setAttribute("aria-pressed", String(sym === state.target));
    b.onclick = () => { if (sym !== state.target) { stop(); load(state.date, sym); } };
    return b;
  }));
}

function renderPairChips(targets) {
  const rivals = targets.filter(s => s !== state.target);
  const current = $("pair").value.trim().toUpperCase();
  $("pair-chips").replaceChildren(...rivals.slice(0, 8).map(sym => {
    const b = node("button", sym, "pair-chip");
    b.type = "button";
    b.setAttribute("aria-pressed", String(sym === current));
    b.onclick = () => {
      $("pair").value = sym;
      renderPairChips(state.targets);
      evaluate("ask");
    };
    return b;
  }));
}

async function load(date, ticker, chooseAvailable = true) {
  const version = ++state.version;
  stopProgress();
  busy(true);
  notice("");
  $("content").hidden = true;
  try {
    const data = await request("catalog" + (date ? "?date=" + encodeURIComponent(date) : ""));
    if (version !== state.version) return;
    if (state.dates.join() !== data.dates.join()) {
      state.dates = data.dates;
      options($("date"), data.dates, true);
    }
    state.targets = data.targets;
    $("date").value = data.date;
    options($("symbols"), data.targets, false);
    let target = ticker || (data.targets.includes("AAPL") ? "AAPL" : data.targets[0]);
    if (!data.targets.includes(target) && chooseAvailable) {
      target = data.targets[0];
      notice("Previous ticker is unavailable on this date; switched to " + target + ".");
    }
    $("ticker").value = target;
    const day = await request("snapshot?" + new URLSearchParams({ date: data.date, target, market: "SP500" }));
    if (version !== state.version) return;
    state.day = day;
    state.date = day.date;
    state.target = day.target;
    state.maskedSender = "";
    state.maskedDelta = null;
    options($("pair-symbols"), data.targets.filter(s => s !== target), false);
    $("pair").value = data.targets.find(s => s !== target && s === "NVDA") || data.targets.find(s => s !== target && s === "MSFT") || data.targets.find(s => s !== target) || "";
    state.selected = [...day.edges].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))[0].source;
    $("content").hidden = false;
    renderQuickTickers(data.targets);
    renderPairChips(data.targets);
    renderDay();
    if (offline) evaluate("ask");
  } catch (e) {
    if (version === state.version) { state.day = null; stop(); notice(e.message); }
  } finally {
    if (version === state.version) {
      busy(false);
      if (!state.dates.length) for (const id of ["load", "date", "ticker", "play"]) $(id).disabled = true;
    }
  }
}

function renderDay() {
  const d = state.day;
  const score = d.forecast.drift_score;
  $("target-metric").textContent = d.target;
  $("sector-badge").textContent = SECTORS[d.target] || "S&P 500";
  $("window-sub").textContent = `${d.window_start} to ${d.date}`;
  $("score").textContent = signed(score);
  $("score").style.color = score >= 0 ? "var(--teal)" : "var(--crimson)";
  const stance = $("stance-badge");
  stance.textContent = score >= 0.35 ? "LONG BIAS" : score <= -0.35 ? "SHORT BIAS" : "NEUTRAL";
  stance.className = "tag " + (score >= 0.35 ? "tag-teal" : score <= -0.35 ? "tag-crimson" : "tag-rust");

  const tel = d.telemetry;
  if (tel) {
    $("score-sub").textContent = `Seed-0: ${signed(tel.seed0_score)} (4-seed std ${fmt(tel.ens_std)})`;
    $("tel-local").textContent = signed(tel.p_local);
    $("tel-flow").textContent = signed(tel.flow_delta);
    const lAbs = Math.abs(tel.p_local), fAbs = Math.abs(tel.flow_delta), tot = lAbs + fAbs || 1;
    $("tel-local-bar").style.width = (100 * lAbs / tot).toFixed(1) + "%";
    $("tel-flow-bar").style.width = (100 * fAbs / tot).toFixed(1) + "%";
    const sameSign = Math.sign(tel.p_local) === Math.sign(tel.flow_delta);
    $("driver-badge").textContent = fAbs / tot < 0.12 ? "LOCAL" : sameSign ? "REINFORCED" : "OFFSET";
    const netRole = tel.g_s - tel.g_r;
    $("role-badge").textContent = netRole > 0.05 ? "NET SENDER" : netRole < -0.05 ? "NET RECEIVER" : "BALANCED";
    $("role-sub").innerHTML = `<span class="math"><var>g</var><sub><var>s</var></sub></span> = ${tel.g_s.toFixed(2)}, <span class="math"><var>g</var><sub><var>r</var></sub></span> = ${tel.g_r.toFixed(2)}, 20d shift ${tel.role_tv_20d.toFixed(3)}`;
  } else {
    $("score-sub").textContent = "Cross-sectional z-score";
    $("tel-local").textContent = "—";
    $("tel-flow").textContent = "—";
    $("tel-local-bar").style.width = "65%";
    $("tel-flow-bar").style.width = "35%";
    $("role-sub").textContent = "20-day flow state";
  }

  $("universe").textContent = number(d.universe_size);
  $("window").textContent = d.window_start + " to " + d.date;
  $("mask-result").textContent = "Mask this edge to zero it out in DRIFT and recompute the score.";
  $("mask-box").classList.remove("is-active");
  $("canvas-hud").hidden = true;
  $("masked-key").hidden = true;
  $("ask").textContent = offline ? "Read analysis" : "Run model";
  $("answer").replaceChildren();

  renderSenders();
  renderSparkline();

  const entries = Object.entries(d.components);
  const sum = entries.reduce((a, x) => a + x[1], 0) || 1;
  const max = Math.max(...entries.map(x => x[1])) || 1;
  $("components").replaceChildren(...entries.map(([label, n]) => {
    const item = node("div", "", "bar");
    const name = node("div", "", "bar-name");
    const right = node("span", fmt(n));
    right.append(node("span", `(${(100 * n / sum).toFixed(0)}%)`, "bar-share"));
    name.append(node("span", title(label)), right);
    const track = node("div", "", "track");
    const fill = node("i", "");
    fill.style.width = (100 * n / max).toFixed(1) + "%";
    track.append(fill);
    item.append(name, track);
    return item;
  }));
}

function renderSparkline() {
  const svg = $("sparkline");
  const d = state.day;
  const retMap = d?.returns_20d;
  svg.replaceChildren();
  if (!retMap || !retMap[d.target]) {
    $("spark-caption").textContent = "20-day return series unavailable for this snapshot.";
    return;
  }
  const sender = state.selected;
  const toCum = arr => {
    let acc = 1;
    return arr.map(r => { acc *= (1 + r); return (acc - 1) * 100; });
  };
  const sT = toCum(retMap[d.target]);
  const sS = retMap[sender] ? toCum(retMap[sender]) : null;
  $("sp-target-lbl").textContent = `${d.target} ${sT.at(-1) >= 0 ? "+" : ""}${sT.at(-1).toFixed(1)}%`;
  $("sp-sender-lbl").textContent = sS ? `${sender} ${sS.at(-1) >= 0 ? "+" : ""}${sS.at(-1).toFixed(1)}%` : "";

  const w = 360, h = 118, padL = 36, padR = 10, padT = 10, padB = 18;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const all = sS ? [...sT, ...sS, 0] : [...sT, 0];
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi - lo < 1) { hi += 0.5; lo -= 0.5; }
  const x = i => padL + (i / (sT.length - 1)) * (w - padL - padR);
  const y = v => padT + ((hi - v) / (hi - lo)) * (h - padT - padB);

  const mk = (tag, attrs, txt) => {
    const el = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
    if (txt !== undefined) el.textContent = txt;
    svg.append(el);
    return el;
  };
  const y0 = y(0);
  mk("line", { x1: padL, y1: y0, x2: w - padR, y2: y0, stroke: "#9e907e", "stroke-dasharray": "3 3" });
  mk("text", { x: padL - 6, y: y(hi) + 4, "text-anchor": "end", fill: "#7c7063", "font-size": 10, "font-family": "var(--mono)" }, `${hi.toFixed(0)}%`);
  if (Math.abs(y0 - y(hi)) > 14 && Math.abs(y(lo) - y0) > 14) {
    mk("text", { x: padL - 6, y: y0 + 3, "text-anchor": "end", fill: "#584e43", "font-size": 10, "font-family": "var(--mono)" }, "0%");
  }
  mk("text", { x: padL - 6, y: y(lo) + 3, "text-anchor": "end", fill: "#7c7063", "font-size": 10, "font-family": "var(--mono)" }, `${lo.toFixed(0)}%`);

  if (sS) {
    const ptsS = sS.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    mk("polyline", { points: ptsS, fill: "none", stroke: "#0f6b5e", "stroke-width": 1.9, "stroke-dasharray": "4 3" });
  }
  const ptsT = sT.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  mk("polyline", { points: ptsT, fill: "none", stroke: "#b84a1e", "stroke-width": 2.3 });
  mk("rect", { x: x(sT.length - 1) - 3, y: y(sT.at(-1)) - 3, width: 6, height: 6, fill: "#b84a1e" });
  if (sS) mk("rect", { x: x(sS.length - 1) - 2.5, y: y(sS.at(-1)) - 2.5, width: 5, height: 5, fill: "#0f6b5e" });
  $("spark-caption").textContent = `${d.target} vs. ${sender} over the last 20 trading days.`;
}

function renderSenders() {
  const rows = [...state.day.edges].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  const maxW = Math.max(...rows.map(r => Math.abs(r.weight))) || 1;
  $("senders").replaceChildren(...rows.map(e => {
    const b = node("button", "", "row");
    b.type = "button";
    b.setAttribute("aria-pressed", String(e.source === state.selected));
    const bar = node("div", "", "row-bar " + (e.weight >= 0 ? "pos-bar" : "neg-bar"));
    bar.style.width = (100 * Math.abs(e.weight) / maxW).toFixed(1) + "%";
    b.append(bar, node("span", e.source), node("span", signedEdgeMilli(e.weight), e.weight >= 0 ? "pos" : "neg"));
    b.title = `K[${state.target}, ${e.source}] = ${signed(e.weight)} (${signedEdgeMilli(e.weight)} × 10⁻³)`;
    b.onclick = () => selectSender(e.source);
    return b;
  }));
  $("mask").textContent = "Mask " + state.selected + " → " + state.target;
  startNetworkLoop();
}

function selectSender(sym) {
  state.selected = sym;
  state.maskedSender = "";
  state.maskedDelta = null;
  $("canvas-hud").hidden = true;
  $("masked-key").hidden = true;
  $("mask-box").classList.remove("is-active");
  $("mask-result").textContent = "Selected " + sym + " → " + state.target + ". Click above to mask this channel.";
  renderSenders();
  renderSparkline();
  if ($("task").value === "COUNTERFACTUAL_COT") {
    if (offline) evaluate("ask");
    else taskChanged();
  }
}

/* Sharp Architectural Flow Graph */
const COORDS = [[0.18, 0.29], [0.82, 0.23], [0.84, 0.69], [0.50, 0.82], [0.16, 0.72]];
function startNetworkLoop() {
  if (state.animFrame) cancelAnimationFrame(state.animFrame);
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const render = now => {
    if (!state.day || $("workspace").hidden) { state.animFrame = 0; return; }
    drawNetworkFrame(now || 0);
    if (!reduced) state.animFrame = requestAnimationFrame(render);
  };
  render(performance.now());
}

function boxAnchor(cx, cy, halfW, halfH, dx, dy) {
  const tx = Math.abs(dx) > 1e-6 ? halfW / Math.abs(dx) : Infinity;
  const ty = Math.abs(dy) > 1e-6 ? halfH / Math.abs(dy) : Infinity;
  const t = Math.min(tx, ty);
  return [cx + dx * t, cy + dy * t];
}

function drawNetworkFrame(now) {
  const canvas = $("network");
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  const dpr = devicePixelRatio || 1;
  const tw = Math.round(w * dpr), th = Math.round(h * dpr);
  if (canvas.width !== tw || canvas.height !== th) { canvas.width = tw; canvas.height = th; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  // Subtle warm archival grid
  ctx.strokeStyle = "rgba(26, 22, 19, 0.05)";
  ctx.lineWidth = 1;
  const step = 32;
  for (let gx = step; gx < w; gx += step) {
    ctx.beginPath(); ctx.moveTo(gx + 0.5, 0); ctx.lineTo(gx + 0.5, h); ctx.stroke();
  }
  for (let gy = step; gy < h; gy += step) {
    ctx.beginPath(); ctx.moveTo(0, gy + 0.5); ctx.lineTo(w, gy + 0.5); ctx.stroke();
  }

  const cx = w * 0.5, cy = h * 0.50;
  const compact = w < 440;
  const nodeW = compact ? 74 : 92, nodeH = compact ? 38 : 44;
  const hubW = compact ? 88 : 108, hubH = compact ? 48 : 54;
  const edges = state.day.edges;
  const maxW = Math.max(...edges.map(e => Math.abs(e.weight))) || 1;
  state.nodes = [];

  // Pass 1: Draw all edges first
  edges.forEach((e, i) => {
    const x = w * COORDS[i][0], y = h * COORDS[i][1];
    state.nodes.push({ source: e.source, x, y, w: nodeW, h: nodeH });
    const dx = cx - x, dy = cy - y, len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;
    const active = e.source === state.selected;
    const hovered = e.source === state.hoveredSender;
    const severed = e.source === state.maskedSender;
    const color = e.weight >= 0 ? "#0f6b5e" : "#a61c3c";

    const [startX, startY] = boxAnchor(x, y, nodeW / 2 + 2, nodeH / 2 + 2, ux, uy);
    const [endX, endY] = boxAnchor(cx, cy, hubW / 2 + 5, hubH / 2 + 5, -ux, -uy);

    ctx.save();
    if (severed) {
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = "#a61c3c";
      ctx.lineWidth = 1.8;
    } else {
      ctx.strokeStyle = active || hovered ? color : color + "85";
      ctx.lineWidth = active ? 2.4 : hovered ? 2.0 : 1.3;
    }
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();
    ctx.restore();

    if (!severed) {
      const speed = 0.00028 + 0.00024 * (Math.abs(e.weight) / maxW);
      const count = active ? 3 : 2;
      for (let k = 0; k < count; k++) {
        const t = ((now * speed) + k / count + i * 0.19) % 1;
        const px = startX + (endX - startX) * t;
        const py = startY + (endY - startY) * t;
        const sz = active ? 5 : 4;
        ctx.save();
        ctx.fillStyle = color;
        ctx.globalAlpha = Math.sin(t * Math.PI) * (active ? 0.95 : 0.65);
        ctx.fillRect(px - sz / 2, py - sz / 2, sz, sz);
        ctx.restore();
      }
      ctx.fillStyle = active ? color : color + "aa";
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(endX - ux * 9 + uy * 4, endY - uy * 9 - ux * 4);
      ctx.lineTo(endX - ux * 9 - uy * 4, endY - uy * 9 + ux * 4);
      ctx.closePath();
      ctx.fill();
    } else {
      const mx = (startX + endX) / 2, my = (startY + endY) / 2;
      ctx.fillStyle = "#faedf0";
      ctx.strokeStyle = "#a61c3c";
      ctx.lineWidth = 1.5;
      ctx.fillRect(mx - 9, my - 9, 18, 18);
      ctx.strokeRect(mx - 9, my - 9, 18, 18);
      ctx.fillStyle = "#a61c3c";
      ctx.font = '700 10px "IBM Plex Mono", monospace';
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("✕", mx, my + 0.5);
    }
  });

  // Pass 2: Draw sharp rectangular sender nodes
  edges.forEach((e, i) => {
    const x = w * COORDS[i][0], y = h * COORDS[i][1];
    const active = e.source === state.selected;
    const hovered = e.source === state.hoveredSender;
    const severed = e.source === state.maskedSender;
    const color = e.weight >= 0 ? "#0f6b5e" : "#a61c3c";
    const left = Math.round(x - nodeW / 2) + 0.5;
    const top = Math.round(y - nodeH / 2) + 0.5;

    ctx.fillStyle = severed ? "#faedf0" : active ? "#faefe8" : "#fdfbf7";
    ctx.fillRect(left, top, nodeW, nodeH);

    ctx.fillStyle = severed ? "#a61c3c" : active ? "#b84a1e" : color;
    ctx.fillRect(left, top, 3.5, nodeH);

    ctx.strokeStyle = severed ? "#a61c3c" : active ? "#1a1613" : hovered ? "#584e43" : "#cfc4b2";
    ctx.lineWidth = active ? 1.8 : 1;
    ctx.strokeRect(left, top, nodeW, nodeH);

    ctx.font = `700 ${compact ? 11 : 13}px "IBM Plex Mono", monospace`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillStyle = "#1a1613";
    ctx.fillText(e.source, x + 1, y - 7);
    ctx.font = `600 ${compact ? 9 : 10}px "IBM Plex Mono", monospace`;
    ctx.fillStyle = severed ? "#a61c3c" : color;
    ctx.fillText(severed ? "MASKED" : signedEdgeMilli(e.weight), x + 1, y + 9);
  });

  if (state.maskedSender && now - state.shockStart < 1000) {
    const p = (now - state.shockStart) / 1000;
    const pad = p * 26;
    ctx.save();
    ctx.strokeStyle = "#a61c3c";
    ctx.globalAlpha = 1 - p;
    ctx.lineWidth = 1.8;
    ctx.strokeRect(cx - hubW / 2 - pad, cy - hubH / 2 - pad, hubW + pad * 2, hubH + pad * 2);
    ctx.restore();
  }

  // Pass 3: Sharp Target Hub Box
  const hLeft = Math.round(cx - hubW / 2) + 0.5;
  const hTop = Math.round(cy - hubH / 2) + 0.5;
  ctx.fillStyle = "#fdfbf7";
  ctx.fillRect(hLeft, hTop, hubW, hubH);
  ctx.fillStyle = "#1a1613";
  ctx.fillRect(hLeft, hTop, hubW, 17);
  ctx.strokeStyle = "#1a1613";
  ctx.lineWidth = 2;
  ctx.strokeRect(hLeft, hTop, hubW, hubH);

  ctx.font = '600 9px "IBM Plex Mono", monospace';
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillStyle = "#fdfbf7";
  ctx.fillText("TARGET", cx, hTop + 9);

  ctx.font = `700 ${compact ? 14 : 16}px "IBM Plex Mono", monospace`;
  ctx.fillStyle = "#1a1613";
  ctx.fillText(state.target, cx, hTop + 36);
}

/* Canvas pointer interaction */
function hitNode(ev) {
  const rect = $("network").getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  return state.nodes.find(n => Math.abs(n.x - x) <= n.w / 2 + 4 && Math.abs(n.y - y) <= n.h / 2 + 4);
}
$("network").addEventListener("mousemove", ev => {
  const hit = hitNode(ev);
  state.hoveredSender = hit ? hit.source : "";
  $("network").style.cursor = hit ? "pointer" : "default";
});
$("network").addEventListener("mouseleave", () => { state.hoveredSender = ""; });
$("network").addEventListener("click", ev => {
  const hit = hitNode(ev);
  if (hit) selectSender(hit.source);
});

async function move(by) {
  const i = state.dates.indexOf(state.date) + by;
  if (i < 0 || i >= state.dates.length) { stop(); return; }
  await load(state.dates[i], state.target);
}
async function play() {
  if (state.playing) { stop(); return; }
  state.playing = true;
  $("play").textContent = "Pause";
  while (state.playing && pane() === "replay" && state.day) {
    await move(1);
    await new Promise(r => setTimeout(r, 1250));
  }
}

let requestId = 0, activeTimer = 0;
function stopProgress() { if (activeTimer) { clearInterval(activeTimer); activeTimer = 0; } }
function startProgress(container, stages, hint) {
  stopProgress();
  const box = node("div", "", "progress-box");
  const head = node("div", "", "progress-head");
  const stageEl = node("span", stages[0][1], "progress-stage");
  const metaEl = node("span", "0% · 0.0s", "progress-meta");
  const track = node("div", "", "progress-track");
  track.setAttribute("role", "progressbar");
  const fill = node("div", "", "progress-fill");
  fill.style.width = "12%";
  track.append(fill);
  head.append(stageEl, metaEl);
  box.append(head, track);
  if (hint) box.append(node("div", hint, "progress-hint"));
  container.replaceChildren(box);
  const start = performance.now();
  activeTimer = setInterval(() => {
    const sec = (performance.now() - start) / 1000;
    const pct = Math.min(95, Math.round(12 + 83 * (1 - Math.exp(-sec / 4.5))));
    fill.style.width = pct + "%";
    metaEl.textContent = pct + "% · " + sec.toFixed(1) + "s";
    for (let i = stages.length - 1; i >= 0; i--) {
      if (sec >= stages[i][0]) { stageEl.textContent = stages[i][1]; break; }
    }
  }, 120);
  return {
    setStage(text) { stageEl.textContent = text; },
    finish() { stopProgress(); fill.style.width = "100%"; },
  };
}

function syncTaskPills() {
  const val = $("task").value;
  for (const b of $("task-pills").querySelectorAll(".task-pill")) {
    b.setAttribute("aria-selected", String(b.dataset.task === val));
  }
  $("pair-label").hidden = val !== "PAIRWISE_FORECAST_COT";
  if (val === "PAIRWISE_FORECAST_COT") renderPairChips(state.targets);
}

function taskChanged() {
  ++requestId;
  stopProgress();
  syncTaskPills();
  $("answer").replaceChildren();
}

function verdictTone(val) {
  const s = String(val).toUpperCase();
  if (["INCREASE", "FLOW_REINFORCED", "NET_SENDER", "HIGH_PROPAGATION", "STABLE", "A"].includes(s)) return "";
  if (["DECREASE", "FLOW_OFFSET", "NET_RECEIVER", "SHIFTING"].includes(s)) return "tone-red";
  if (["HIGH", "MULTI_DAY_RELAY", "MODERATE", "B"].includes(s)) return "tone-warm";
  return "tone-navy";
}

async function evaluate(action) {
  if (!state.day) return;
  const version = state.version, sender = state.selected;
  if (action === "mask") {
    $("task").value = "COUNTERFACTUAL_COT";
    taskChanged();
  }
  const id = ++requestId, task = $("task").value, req = payload();
  const current = () => id === requestId && version === state.version && sender === state.selected;
  const stages = action === "mask"
    ? [[0, "Masking edge & recomputing DRIFT score…"], [1.2, "Running Qwen3.5-4B on updated tokens…"]]
    : [[0, "Encoding 25 DRIFT tokens…"], [1.5, "Generating Qwen3.5-4B readout…"]];

  let progress = null;
  if (!offline) {
    progress = startProgress($("answer"), stages, "Running on GPU (typically 8–15s)…");
    $("ask").textContent = "Running…";
  }

  if (action === "mask") {
    try {
      const r = await request("intervene", { ...req, sender });
      if (!current()) return;
      state.maskedSender = sender;
      state.maskedDelta = r.delta;
      state.shockStart = performance.now();
      $("masked-key").hidden = false;
      $("canvas-hud").hidden = false;
      $("canvas-hud").textContent = `MASKED ${sender} → ${state.target}  |  Δscore ${signed(r.delta)}`;
      $("mask-result").textContent = "Seed-0 score " + signed(r.original_score) + " → " + signed(r.intervened_score) + " (Δ " + signed(r.delta) + ").";
      $("mask-box").classList.add("is-active");
      if (progress) progress.setStage("Running Qwen3.5-4B readout…");
    } catch (e) {
      if (current()) {
        stopProgress();
        $("ask").textContent = offline ? "Read analysis" : "Run model";
        $("mask-result").textContent = e.message;
        $("answer").replaceChildren();
      }
      return;
    }
  }

  try {
    const args = { ...req, task };
    if (task === "COUNTERFACTUAL_COT") args.sender = sender;
    if (task === "PAIRWISE_FORECAST_COT") args.other = $("pair").value.trim().toUpperCase();
    const r = await request("copilot", args);
    if (!current()) return;
    if (progress) progress.finish();

    const card = node("div", "", "cot-card");
    const head = node("div", "", "cot-header");
    head.append(node("strong", "DRIFT-TSLM v1.0"), node("span", task.replace("_COT", "").replaceAll("_", " "), "tag tag-navy"));
    const body = node("p", r.reasoning, "cot-reasoning");
    const list = node("dl", "", "conclusions");
    for (const [k, v] of Object.entries(r.answer)) {
      const wrap = node("div", "", "verdict-item");
      wrap.append(node("dt", title(k)), node("dd", String(v), verdictTone(v)));
      list.append(wrap);
    }
    const meta = node("small", r.cached ? "Archived output" : (r.inference_ms / 1000).toFixed(1) + " s", "cot-meta");
    card.append(head, body, list, meta);
    $("answer").replaceChildren(card);
  } catch (e) {
    if (current()) { stopProgress(); $("answer").replaceChildren(node("p", e.message, "notice")); }
  } finally {
    if (current()) { stopProgress(); $("ask").textContent = offline ? "Read analysis" : "Run model"; }
  }
}

/* Sharp Grouped Benchmark Bar Chart */
function buildBenchmarkChart(result) {
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("class", "bench-chart");
  const tasks = [
    ["ATTRIBUTION_COT", "Attribution"],
    ["COUNTERFACTUAL_COT", "Masked edge"],
    ["REGIME_ROLE_COT", "Role & regime"],
    ["PAIRWISE_FORECAST_COT", "Paired forecast"],
  ];
  const series = [
    { label: "DRIFT-TSLM v1.0", color: "#0f6b5e", get: k => result.variants.drift.real[k].accuracy },
    { label: "Returns-only baseline", color: "#29487d", get: k => result.variants.returns.real[k].accuracy },
    { label: "Same-day shuffled", color: "#b84a1e", get: k => result.variants.drift.shuffled[k].accuracy },
    { label: "Zeroed tokens", color: "#a61c3c", get: k => result.variants.drift.zero[k].accuracy },
  ];
  const w = 960, h = 252, left = 52, right = 24, top = 36, bottom = 42;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const mk = (tag, attrs, txt) => {
    const el = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
    if (txt !== undefined) el.textContent = txt;
    svg.append(el);
    return el;
  };
  series.forEach((s, idx) => {
    const lx = left + idx * 215;
    mk("rect", { x: lx, y: 9, width: 10, height: 10, fill: s.color });
    mk("text", { x: lx + 16, y: 18, fill: "#374151", "font-size": 12, "font-weight": 500, "font-family": "var(--sans)" }, s.label);
  });
  const bottom_y = h - bottom;
  const y = pct => bottom_y - pct * (bottom_y - top);
  for (const tick of [0, 0.25, 0.5, 0.75, 1.0]) {
    const ty = y(tick);
    mk("line", { x1: left, y1: ty, x2: w - right, y2: ty, stroke: tick === 0 ? "#12151a" : "#e7e5de" });
    mk("text", { x: left - 8, y: ty + 4, "text-anchor": "end", fill: "#6b7280", "font-size": 11, "font-family": "var(--mono)" }, `${Math.round(tick * 100)}%`);
  }
  const groupW = (w - left - right) / tasks.length;
  const barW = 28, gap = 6;
  tasks.forEach(([key, name], gi) => {
    const gx = left + gi * groupW + (groupW - (series.length * barW + (series.length - 1) * gap)) / 2;
    series.forEach((s, si) => {
      const val = s.get(key);
      const bx = gx + si * (barW + gap);
      const bh = Math.max(2, val * (bottom_y - top));
      const by = bottom_y - bh;
      mk("rect", { x: bx, y: by, width: barW, height: bh, fill: s.color, opacity: si === 0 ? 1 : 0.85 });
      if (si === 0) {
        mk("text", { x: bx + barW / 2, y: by - 6, "text-anchor": "middle", fill: "#0b6e64", "font-size": 11.5, "font-weight": 700, "font-family": "var(--mono)" }, `${(val * 100).toFixed(1)}%`);
      }
    });
    mk("text", { x: left + (gi + 0.5) * groupW, y: h - 14, "text-anchor": "middle", fill: "#12151a", "font-size": 13, "font-weight": 600, "font-family": "var(--sans)" }, name);
  });
  return svg;
}

function table(headers, rows, highlightIndex = -1) {
  const t = node("table", "", "model-table");
  const head = document.createElement("thead"), h = document.createElement("tr");
  for (const label of headers) h.append(node("th", label));
  head.append(h); t.append(head);
  const body = document.createElement("tbody");
  rows.forEach((row, idx) => {
    const tr = document.createElement("tr");
    if (idx === highlightIndex) tr.className = "highlight-row";
    for (const val of row) tr.append(node("td", String(val)));
    body.append(tr);
  });
  t.append(body);
  return t;
}

async function showModel() {
  const dest = $("model-content");
  const loading = node("div", "", "progress-box");
  const head = node("div", "", "progress-head");
  head.append(node("span", "Loading evaluation tables…", "progress-stage"));
  const track = node("div", "", "progress-track indeterminate");
  track.append(node("div", "", "progress-fill"));
  loading.append(head, track);
  dest.replaceChildren(loading);
  try {
    const data = await (await fetch("./model-card.json")).json();
    const paper = node("article", "", "panel sheet");
    paper.append(node("h2", data.heading), node("p", data.context));
    for (const t of data.tables) {
      const driftIdx = t.rows.findIndex(r => String(r[0]).toUpperCase().includes("DRIFT"));
      paper.append(node("h3", t.market), table(["Model", "IC", "RankIC", "Gross return"], t.rows, driftIdx));
    }
    paper.append(node("p", data.notes));
    dest.replaceChildren(paper);

    const result = await request("evaluation");
    const panel = node("article", "", "panel sheet");
    panel.append(node("h2", "Held-out test results (2024–2025)"), node("div", "Grounding test: " + result.gate, "gate-banner"));
    if (result.variants) {
      const keys = Object.keys(result.variants.drift.real);
      panel.append(node("p", "1,000 held-out questions per task across 500 trading days in 2024–2025. Exact accuracy checks every JSON field plus the 2–4 sentence explanation format. Prompts contain no target numbers."));
      panel.append(buildBenchmarkChart(result));
      const taskNames = { ATTRIBUTION_COT: "Attribution", COUNTERFACTUAL_COT: "Masked edge", REGIME_ROLE_COT: "Role & regime", PAIRWISE_FORECAST_COT: "Paired forecast" };
      const rows = [
        ["Text only", result.variants.text.text],
        ["Returns only", result.variants.returns.real],
        ["DRIFT", result.variants.drift.real],
        ["Same-day mismatch", result.variants.drift.shuffled],
        ["Zeroed", result.variants.drift.zero],
        ["Local only", result.variants.drift.local],
      ].map(([name, x]) => [name, ...keys.map(k => (100 * x[k].accuracy).toFixed(1) + "%")]);
      panel.append(table(["Input", ...keys.map(k => taskNames[k])], rows, 2));
      panel.append(node("p", "Each diagnostic task must drop by at least 15% under both shuffled and zeroed controls. Checkpoints were picked on 2023 validation."));
      const pairs = [["DRIFT-TSLM v1.0", "drift", "real"], ["Returns-only OpenTSLM + Qwen", "returns", "real"], ["Text-only Qwen", "text", "text"]];
      panel.append(node("h3", "Pairwise return ranking"),
        table(["Model", "Top/bottom quintile pairs", "Random whole-market pairs"], pairs.map(([name, variant, mode]) => {
          const metric = result.variants[variant][mode].PAIRWISE_FORECAST_COT;
          const random = result.random_pairs[variant][mode].PAIRWISE_FORECAST_COT;
          return [name, (100 * metric.fields.winner / metric.n).toFixed(1) + "%", (100 * random.fields.winner / random.n).toFixed(1) + "%"];
        }), 0));
    }
    if (result.pairwise_scores) {
      const pairs = [["Extreme pairs", "test"], ["Random pairs", "test_random_pairs"]];
      panel.append(node("h3", "Native DRIFT and winner-logit scores"),
        table(["Cohort", "Ensemble win", "Ensemble AUC", "Qwen margin win", "Qwen AUC"], pairs.map(([label, split]) => {
          const { native, logits } = result.pairwise_scores[split];
          return [label, (100 * native.win_rate).toFixed(2) + "%", native.auc.toFixed(4), (100 * logits.win_rate).toFixed(2) + "%", logits.auc.toFixed(4)];
        }), 1),
        node("p", "Qwen compares its A and B logits at the JSON winner field after generating its own evidence. A positive margin favors the first stock."));
    }
    for (const c of result.examples || []) {
      panel.append(node("h3", c.target + " on " + c.date + " (" + title(c.task.replace("_COT", "")) + ")"), node("p", c.question));
      const comparison = node("div", "", "case-comparison");
      for (const [name, key] of [["Returns-only baseline", "baseline"], ["DRIFT-TSLM v1.0", "drift"]]) {
        const item = node("article", "", "case-output" + (key === "drift" ? " is-drift" : ""));
        item.append(node("h4", name));
        const parts = c[key].match(/^<reasoning>([\s\S]*?)<\/reasoning>\s*<answer>([\s\S]*?)<\/answer>$/);
        if (parts) item.append(node("p", parts[1]), node("code", parts[2]));
        else item.append(node("p", c[key]));
        const check = c.evidence_check?.[key === "baseline" ? "returns" : "drift"];
        if (check) item.append(node("small", check.supported + " / " + check.total + " sentences match the reference evidence."));
        comparison.append(item);
      }
      panel.append(comparison, node("p", "Reference evidence: " + c.reference_evidence), node("code", JSON.stringify(c.gold)));
    }
    dest.prepend(panel);
  } catch (e) { notice(e.message); }
}

function route() {
  stop();
  const view = pane(), isStarmap = view === "starmap", isModel = view === "model", isMultiday = view === "multiday";
  for (const a of document.querySelectorAll("nav a")) {
    a.setAttribute("aria-current", a.hash === "#" + view ? "page" : "false");
  }
  $("starmap").hidden = !isStarmap;
  $("workspace").hidden = isStarmap || isModel || isMultiday;
  $("multiday").hidden = !isMultiday;
  $("model").hidden = !isModel;
  $("source-wrap").hidden = isMultiday;
  $("transport").hidden = view !== "replay";
  $("domain").textContent = {
    starmap: "01 — All 489 S&P 500 stocks on Jan 2, 2024",
    explorer: "02 — S&P 500 daily Alpha158 panel with Qwen3.5-4B",
    replay: "02 — Historical Alpha158 sessions with Qwen3.5-4B",
    multiday: "03 — Five-session edge masking across S&P 500 and CSI 300",
    model: "04 — Held-out 2024–2025 evaluation and paper benchmarks",
  }[view];
  $("page-title").textContent = {
    starmap: "Market Star Map",
    explorer: "Flow Explorer",
    replay: "Historical Replay",
    multiday: "Multi-Day Replay",
    model: "Model card",
  }[view];
  $("page-subtitle").textContent = {
    starmap: "View all 489 S&P 500 stocks from Jan 2, 2024 in a 3D sphere. Rotate, zoom, or search any ticker to trace its five strongest upstream links, then open it in Flow Explorer.",
    explorer: "Pick a stock and date to inspect its five strongest inbound channels, mask an edge, and read what Qwen3.5-4B infers from the 25 DRIFT tokens alone.",
    replay: "Step through 12 test dates in 2024–2025 to see how inbound channels and flow memory change from session to session.",
    multiday: "Mask an upstream edge across five sessions—today only, all five days, or four days masked then reopened—to see how much effect persists in DRIFT’s state.",
    model: "Held-out 2024–2025 accuracy, shuffled and zeroed token controls, and the paper’s stock-ranking tables.",
  }[view];
  document.title = "MarketFlow — " + $("page-title").textContent;
  notice("");
  if (isStarmap) {
    state.version++;
    showStarmap({
      onOpenExplorer: (date, ticker) => {
        state.date = date;
        state.target = ticker;
        location.hash = "#explorer";
        load(date, ticker);
      },
    });
  }
  else if (isMultiday) { state.version++; showMultiday(); }
  else if (isModel) { state.version++; showModel(); }
  else if (!state.day || state.busy) load(state.date, state.target);
  else startNetworkLoop();
}

$("filters").onsubmit = e => { e.preventDefault(); stop(); load($("date").value, $("ticker").value.trim().toUpperCase(), false); };
$("date").onchange = () => { stop(); load($("date").value, state.target); };
$("prev").onclick = () => { stop(); move(-1); };
$("next").onclick = () => { stop(); move(1); };
$("play").onclick = play;
$("ask").onclick = () => evaluate("ask");
$("mask").onclick = () => evaluate("mask");
$("task").onchange = () => { taskChanged(); if (offline) evaluate("ask"); };
$("pair").oninput = () => { taskChanged(); renderPairChips(state.targets); };
$("task-pills").addEventListener("click", ev => {
  const btn = ev.target.closest(".task-pill");
  if (!btn) return;
  $("task").value = btn.dataset.task;
  taskChanged();
  if (offline) evaluate("ask");
});

new ResizeObserver(() => { if (state.day && !$("workspace").hidden) startNetworkLoop(); }).observe($("network"));
window.addEventListener("hashchange", route);
route();
