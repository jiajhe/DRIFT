// Market Star Map: interactive single-date S&P 500 cross-sectional DRIFT constellation.
const $ = id => document.getElementById(id);

const FEATURED = ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM"];

const sm = {
  data: null,
  loading: null,
  nodes: [],
  byId: new Map(),
  downstream: [],
  hubRank: [],
  selectedIdx: -1,
  hoveredIdx: -1,
  layout: "constellation", // "constellation" | "role"
  colorBy: "score",        // "score" | "role"
  showWeb: true,
  showDownstream: false,
  // Camera in world space (world [-1, 1] fits inside min(w, h) * 0.44 * zoom)
  cam: { x: 0, y: 0, zoom: 1.0 },
  targetCam: null,
  // Layout interpolation
  pos: [],
  targetPos: [],
  layoutLerp: 1,
  // Pointer state
  dragging: false,
  dragMoved: false,
  dragStartX: 0,
  dragStartY: 0,
  camStartX: 0,
  camStartY: 0,
  // Animation & callbacks
  animFrame: 0,
  lastNow: 0,
  onOpenExplorer: null,
  archivedTargets: new Set(FEATURED),
  initialized: false,
};

const fmt = n => Number(n).toFixed(3);
const signed = n => (n > 0 ? "+" : "") + Number(n).toFixed(4);
const signedShort = n => (n > 0 ? "+" : "") + Number(n).toFixed(2);
const signedEdgeMilli = w => {
  const v = Number(w) * 1000;
  return (v > 0 ? "+" : "") + v.toFixed(2);
};

// Fuzzy ticker search: exact (0) > prefix (1) > substring (2) > subsequence (3)
function fuzzyMatch(query, ticker) {
  if (!query) return { rank: 4, indices: [] };
  if (ticker === query) return { rank: 0, indices: [...Array(ticker.length).keys()] };
  if (ticker.startsWith(query)) return { rank: 1, indices: [...Array(query.length).keys()] };
  const sub = ticker.indexOf(query);
  if (sub >= 0) return { rank: 2, indices: Array.from({ length: query.length }, (_, k) => sub + k) };
  // Subsequence match
  let qi = 0;
  const indices = [];
  for (let ti = 0; ti < ticker.length && qi < query.length; ti++) {
    if (ticker[ti] === query[qi]) {
      indices.push(ti);
      qi++;
    }
  }
  if (qi === query.length) return { rank: 3, indices };
  return null;
}

let suggestItems = [];
let suggestActive = -1;

function closeSuggest() {
  const box = $("sm-suggest");
  if (!box) return;
  box.hidden = true;
  $("sm-search").setAttribute("aria-expanded", "false");
  suggestItems = [];
  suggestActive = -1;
}

function renderSuggest(raw) {
  const box = $("sm-suggest");
  if (!box || !sm.nodes.length) return;
  const q = (raw || "").trim().toUpperCase();
  let matches = [];
  if (!q) {
    const picks = [...new Set([...FEATURED, ...sm.hubRank.slice(0, 6).map(i => sm.nodes[i].id)])];
    matches = picks
      .filter(id => sm.byId.has(id))
      .slice(0, 8)
      .map(id => ({ idx: sm.byId.get(id), rank: 4, indices: [] }));
  } else {
    for (let i = 0; i < sm.nodes.length; i++) {
      const m = fuzzyMatch(q, sm.nodes[i].id);
      if (m) matches.push({ idx: i, ...m });
    }
    matches.sort((a, b) =>
      a.rank - b.rank ||
      sm.nodes[b.idx].out - sm.nodes[a.idx].out ||
      sm.nodes[a.idx].id.localeCompare(sm.nodes[b.idx].id)
    );
    matches = matches.slice(0, 8);
  }
  suggestItems = matches;
  suggestActive = matches.length ? 0 : -1;
  if (!matches.length) {
    box.replaceChildren(nodeEl("div", `No ticker matching "${q}"`, "sm-suggest-empty"));
    box.hidden = false;
    $("sm-search").setAttribute("aria-expanded", "true");
    return;
  }
  box.replaceChildren(...matches.map((m, pos) => {
    const n = sm.nodes[m.idx];
    const row = document.createElement("button");
    row.type = "button";
    row.className = "sm-suggest-item" + (pos === suggestActive ? " is-active" : "");
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", String(pos === suggestActive));

    const nameSpan = document.createElement("span");
    nameSpan.className = "sm-suggest-ticker";
    const hi = new Set(m.indices);
    for (let k = 0; k < n.id.length; k++) {
      if (hi.has(k)) nameSpan.append(nodeEl("mark", n.id[k]));
      else nameSpan.append(document.createTextNode(n.id[k]));
    }

    row.append(
      nameSpan,
      nodeEl("span", `${n.out} down`, "sm-suggest-meta"),
      nodeEl("span", signedShort(n.s), n.s >= 0 ? "pos" : "neg"),
    );
    row.addEventListener("mousedown", e => {
      e.preventDefault();
      $("sm-status").hidden = true;
      closeSuggest();
      selectNode(m.idx, { fly: true });
    });
    return row;
  }));
  box.hidden = false;
  $("sm-search").setAttribute("aria-expanded", "true");
}

function updateSuggestHighlight() {
  const box = $("sm-suggest");
  if (!box) return;
  Array.from(box.querySelectorAll(".sm-suggest-item")).forEach((el, i) => {
    el.classList.toggle("is-active", i === suggestActive);
    el.setAttribute("aria-selected", String(i === suggestActive));
  });
}

function nodeEl(tag, text = "", cls = "") {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined && text !== null) el.textContent = text;
  return el;
}

function getScale(w, h) {
  return Math.min(w, h) * 0.45 * sm.cam.zoom;
}

function worldToScreen(wx, wy, w, h) {
  const s = getScale(w, h);
  return [
    w * 0.5 + (wx - sm.cam.x) * s,
    h * 0.5 - (wy - sm.cam.y) * s,
  ];
}

function screenToWorld(sx, sy, w, h) {
  const s = getScale(w, h);
  return [
    sm.cam.x + (sx - w * 0.5) / s,
    sm.cam.y - (sy - h * 0.5) / s,
  ];
}

function nodeRadius(n) {
  // Base star radius scales subtly with out-degree (market hubness)
  return 2.4 + Math.min(4.8, Math.sqrt(n.out) * 0.95);
}

function nodeColor(n) {
  if (sm.colorBy === "role") {
    const net = n.gs - n.gr;
    if (net > 0.04) return "#2dd4bf";  // Net sender (bright teal)
    if (net < -0.04) return "#f59e0b"; // Net receiver (amber)
    return "#94a3b8";                  // Balanced (slate)
  }
  if (n.s >= 0.35) return "#14b8a6";   // Strong positive DRIFT score
  if (n.s >= 0.0) return "#5eead4";    // Mild positive
  if (n.s <= -0.35) return "#f43f5e";  // Strong negative
  return "#fda4af";                    // Mild negative
}

function flyTo(idx, maxZoom = 4.2) {
  if (idx < 0 || idx >= sm.nodes.length) return;
  const canvas = $("sm-canvas");
  const w = canvas?.clientWidth || 800, h = canvas?.clientHeight || 590;
  const pts = [sm.targetPos[idx] || sm.pos[idx]];
  for (const [srcIdx] of sm.nodes[idx].in_edges) {
    pts.push(sm.targetPos[srcIdx] || sm.pos[srcIdx]);
  }
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [px, py] of pts) {
    if (px < minX) minX = px;
    if (px > maxX) maxX = px;
    if (py < minY) minY = py;
    if (py > maxY) maxY = py;
  }
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const spanX = Math.max(0.18, maxX - minX);
  const spanY = Math.max(0.18, maxY - minY);
  const baseScale = Math.min(w, h) * 0.45;
  const fitZoom = Math.min((w * 0.72) / (spanX * baseScale), (h * 0.72) / (spanY * baseScale));
  const zoom = Math.max(1.2, Math.min(maxZoom, fitZoom));
  sm.targetCam = {
    x: cx,
    y: cy,
    zoom,
    startX: sm.cam.x,
    startY: sm.cam.y,
    startZoom: sm.cam.zoom,
    startTime: performance.now(),
    duration: 420,
  };
  ensureLoop();
}

function resetCamera() {
  sm.targetCam = {
    x: 0,
    y: 0,
    zoom: 1.0,
    startX: sm.cam.x,
    startY: sm.cam.y,
    startZoom: sm.cam.zoom,
    startTime: performance.now(),
    duration: 360,
  };
  ensureLoop();
}

function selectNode(idx, { fly = true, zoom = 4.2 } = {}) {
  sm.selectedIdx = idx;
  if (idx >= 0) {
    const n = sm.nodes[idx];
    $("sm-search").value = n.id;
    renderInspector();
    renderQuickChips();
    if (fly) flyTo(idx, zoom);
  } else {
    $("sm-search").value = "";
    renderInspector();
    renderQuickChips();
  }
  ensureLoop();
}

function setLayout(mode) {
  if (sm.layout === mode) return;
  sm.layout = mode;
  for (const b of $("sm-layout-switch").querySelectorAll("button")) {
    b.setAttribute("aria-pressed", String(b.dataset.layout === mode));
  }
  sm.targetPos = sm.nodes.map(n => mode === "role" ? [n.rx, n.ry] : [n.x, n.y]);
  sm.layoutLerp = 0;
  if (sm.selectedIdx >= 0) {
    const p = sm.targetPos[sm.selectedIdx];
    sm.targetCam = {
      x: p[0],
      y: p[1],
      zoom: sm.cam.zoom,
      startX: sm.cam.x,
      startY: sm.cam.y,
      startZoom: sm.cam.zoom,
      startTime: performance.now(),
      duration: 420,
    };
  }
  ensureLoop();
}

function setColorBy(mode) {
  sm.colorBy = mode;
  for (const b of $("sm-color-switch").querySelectorAll("button")) {
    b.setAttribute("aria-pressed", String(b.dataset.color === mode));
  }
  updateLegend();
  ensureLoop();
}

function updateLegend() {
  const leg = $("sm-legend");
  if (!leg) return;
  if (sm.colorBy === "role") {
    leg.innerHTML =
      `<span class="sm-dot-pos">Net Sender</span>` +
      `<span class="sm-dot-amb">Net Receiver</span>` +
      `<span class="sm-dot-neu">Balanced</span>`;
  } else {
    leg.innerHTML =
      `<span class="sm-dot-pos">Positive DRIFT</span>` +
      `<span class="sm-dot-neg">Negative DRIFT</span>` +
      `<span class="sm-dot-gold">Selected</span>`;
  }
}

function renderQuickChips() {
  const wrap = $("sm-quick-chips");
  if (!wrap) return;
  const topHubs = sm.hubRank.slice(0, 4).map(i => sm.nodes[i].id);
  const picks = [...new Set([...FEATURED.slice(0, 6), ...topHubs])].filter(id => sm.byId.has(id)).slice(0, 10);
  const current = sm.selectedIdx >= 0 ? sm.nodes[sm.selectedIdx].id : "";
  wrap.replaceChildren(...picks.map(sym => {
    const b = nodeEl("button", sym, "ticker-pill");
    b.type = "button";
    b.setAttribute("aria-pressed", String(sym === current));
    b.onclick = () => selectNode(sm.byId.get(sym), { fly: true, zoom: 4.2 });
    return b;
  }));
}

function renderInspector() {
  const empty = $("sm-inspector-empty");
  const body = $("sm-inspector-body");
  if (sm.selectedIdx < 0) {
    empty.hidden = false;
    body.hidden = true;
    // Populate top hubs list in empty state
    const hubs = $("sm-top-hubs");
    hubs.replaceChildren(...sm.hubRank.slice(0, 8).map(idx => {
      const n = sm.nodes[idx];
      const b = nodeEl("button", "", "sm-hub-row");
      b.type = "button";
      b.append(
        nodeEl("strong", n.id),
        nodeEl("span", `${n.out} downstream`, "sm-hub-meta"),
        nodeEl("span", signedShort(n.s), n.s >= 0 ? "pos" : "neg"),
      );
      b.onclick = () => selectNode(idx, { fly: true, zoom: 4.2 });
      return b;
    }));
    return;
  }

  empty.hidden = true;
  body.hidden = false;
  const n = sm.nodes[sm.selectedIdx];
  $("sm-sel-ticker").textContent = n.id;
  const netRole = n.gs - n.gr;
  const roleTag = $("sm-sel-role");
  roleTag.textContent = netRole > 0.04 ? "NET SENDER" : netRole < -0.04 ? "NET RECEIVER" : "BALANCED";
  roleTag.className = "tag " + (netRole > 0.04 ? "tag-teal" : netRole < -0.04 ? "tag-rust" : "tag-navy");

  const scoreEl = $("sm-sel-score");
  scoreEl.textContent = signed(n.s);
  scoreEl.style.color = n.s >= 0 ? "var(--teal)" : "var(--crimson)";

  $("sm-sel-local").textContent = signedShort(n.loc);
  $("sm-sel-flow").textContent = signedShort(n.flw);
  $("sm-sel-role-nums").innerHTML = `<span class="math"><var>g</var><sub><var>s</var></sub></span> = ${n.gs.toFixed(2)}, <span class="math"><var>g</var><sub><var>r</var></sub></span> = ${n.gr.toFixed(2)}, ${n.out} downstream`;

  // Upstream top-5 list
  const maxW = Math.max(...n.in_edges.map(e => Math.abs(e[1]))) || 1;
  $("sm-up-list").replaceChildren(...n.in_edges.map(([srcIdx, w]) => {
    const src = sm.nodes[srcIdx];
    const b = nodeEl("button", "", "row sm-trace-row");
    b.type = "button";
    const bar = nodeEl("div", "", "row-bar " + (w >= 0 ? "pos-bar" : "neg-bar"));
    bar.style.width = (100 * Math.abs(w) / maxW).toFixed(1) + "%";
    b.append(
      bar,
      nodeEl("span", `${src.id} → ${n.id}`),
      nodeEl("span", signedEdgeMilli(w), w >= 0 ? "pos" : "neg"),
    );
    b.title = `K[${n.id}, ${src.id}] = ${signedEdgeMilli(w)} × 10⁻³ (click to focus ${src.id})`;
    b.onclick = () => selectNode(srcIdx, { fly: true, zoom: Math.max(sm.cam.zoom, 4.2) });
    return b;
  }));

  // Downstream list
  const down = sm.downstream[sm.selectedIdx] || [];
  $("sm-down-count").textContent = `${down.length} stocks`;
  const downList = $("sm-down-list");
  if (!down.length) {
    downList.replaceChildren(nodeEl("p", "No S&P 500 stocks list this ticker in their top-5 inbound channels on this date.", "sm-empty-note"));
  } else {
    downList.replaceChildren(...down.slice(0, 8).map(([dstIdx, w]) => {
      const dst = sm.nodes[dstIdx];
      const b = nodeEl("button", "", "sm-down-chip");
      b.type = "button";
      b.append(nodeEl("span", dst.id), nodeEl("small", signedEdgeMilli(w), w >= 0 ? "pos" : "neg"));
      b.title = `K[${dst.id}, ${n.id}] = ${signedEdgeMilli(w)} × 10⁻³ (click to focus ${dst.id})`;
      b.onclick = () => selectNode(dstIdx, { fly: true, zoom: Math.max(sm.cam.zoom, 4.2) });
      return b;
    }));
  }

  // Deep link to Flow Explorer
  const openBtn = $("sm-open-explorer");
  const canOpen = sm.archivedTargets.has(n.id);
  openBtn.textContent = canOpen
    ? `Open ${n.id} in Flow Explorer →`
    : `Open Flow Explorer (${n.id} requires live API)`;
  openBtn.disabled = false;
  openBtn.onclick = () => {
    if (sm.onOpenExplorer) sm.onOpenExplorer(sm.data.date, canOpen ? n.id : "AAPL");
  };
}

function drawFrame(now) {
  const canvas = $("sm-canvas");
  if (!canvas || $("starmap").hidden) { sm.animFrame = 0; return; }
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;

  // Step camera animation
  let animating = false;
  if (sm.targetCam) {
    const t = Math.min(1, (now - sm.targetCam.startTime) / sm.targetCam.duration);
    const ease = 1 - Math.pow(1 - t, 3);
    sm.cam.x = sm.targetCam.startX + (sm.targetCam.x - sm.targetCam.startX) * ease;
    sm.cam.y = sm.targetCam.startY + (sm.targetCam.y - sm.targetCam.startY) * ease;
    sm.cam.zoom = sm.targetCam.startZoom + (sm.targetCam.zoom - sm.targetCam.startZoom) * ease;
    if (t >= 1) sm.targetCam = null;
    else animating = true;
  }

  // Step layout interpolation
  if (sm.layoutLerp < 1) {
    sm.layoutLerp = Math.min(1, sm.layoutLerp + 0.08);
    const e = 1 - Math.pow(1 - sm.layoutLerp, 3);
    for (let i = 0; i < sm.pos.length; i++) {
      sm.pos[i][0] += (sm.targetPos[i][0] - sm.pos[i][0]) * Math.max(0.18, e);
      sm.pos[i][1] += (sm.targetPos[i][1] - sm.pos[i][1]) * Math.max(0.18, e);
    }
    if (sm.layoutLerp >= 1) {
      for (let i = 0; i < sm.pos.length; i++) {
        sm.pos[i][0] = sm.targetPos[i][0];
        sm.pos[i][1] = sm.targetPos[i][1];
      }
    } else {
      animating = true;
    }
  }

  const dpr = devicePixelRatio || 1;
  const tw = Math.round(w * dpr), th = Math.round(h * dpr);
  if (canvas.width !== tw || canvas.height !== th) { canvas.width = tw; canvas.height = th; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Deep observatory dark background with subtle radial vignette
  const bg = ctx.createRadialGradient(w * 0.5, h * 0.5, 20, w * 0.5, h * 0.5, Math.max(w, h) * 0.72);
  bg.addColorStop(0, "#111827");
  bg.addColorStop(1, "#080c14");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  // Subtle celestial coordinate rings / axes
  const [ox, oy] = worldToScreen(0, 0, w, h);
  const scale = getScale(w, h);
  ctx.save();
  ctx.strokeStyle = "rgba(148, 163, 184, 0.08)";
  ctx.lineWidth = 1;
  if (sm.layout === "constellation") {
    for (const r of [0.33, 0.66, 0.98]) {
      ctx.beginPath();
      ctx.arc(ox, oy, r * scale, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(ox - scale, oy); ctx.lineTo(ox + scale, oy);
    ctx.moveTo(ox, oy - scale); ctx.lineTo(ox, oy + scale);
    ctx.stroke();
  } else {
    // Role x Score quadrant axes
    ctx.beginPath();
    ctx.moveTo(0, oy); ctx.lineTo(w, oy);
    ctx.moveTo(ox, 0); ctx.lineTo(ox, h);
    ctx.stroke();
    ctx.fillStyle = "rgba(148, 163, 184, 0.42)";
    ctx.font = '500 10px "IBM Plex Mono", monospace';
    ctx.textAlign = "right";
    ctx.fillText("NET SENDER →", w - 14, oy - 6);
    ctx.textAlign = "left";
    ctx.fillText("← NET RECEIVER", 14, oy - 6);
    ctx.fillText("↑ POSITIVE DRIFT", ox + 8, 18);
    ctx.fillText("↓ NEGATIVE DRIFT", ox + 8, h - 12);
  }
  ctx.restore();

  // Compute screen positions once per frame
  const sx = new Float32Array(sm.nodes.length);
  const sy = new Float32Array(sm.nodes.length);
  for (let i = 0; i < sm.nodes.length; i++) {
    const [px, py] = worldToScreen(sm.pos[i][0], sm.pos[i][1], w, h);
    sx[i] = px;
    sy[i] = py;
  }

  const focusIdx = sm.selectedIdx >= 0 ? sm.selectedIdx : sm.hoveredIdx;
  const hasFocus = focusIdx >= 0;
  const upstreamSet = new Set();
  const downstreamSet = new Set();
  if (hasFocus) {
    for (const [srcIdx] of sm.nodes[focusIdx].in_edges) upstreamSet.add(srcIdx);
    if (sm.showDownstream) {
      for (const [dstIdx] of sm.downstream[focusIdx] || []) downstreamSet.add(dstIdx);
    }
  }

  // Layer 1: Faint global constellation skeleton edges
  if (sm.showWeb && sm.data?.skeleton) {
    ctx.save();
    ctx.lineWidth = 0.8;
    for (const [u, v, wt] of sm.data.skeleton) {
      ctx.strokeStyle = hasFocus
        ? "rgba(148, 163, 184, 0.045)"
        : (wt >= 0 ? "rgba(45, 212, 191, 0.13)" : "rgba(244, 63, 94, 0.12)");
      ctx.beginPath();
      ctx.moveTo(sx[u], sy[u]);
      ctx.lineTo(sx[v], sy[v]);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Layer 2: Focused stock's downstream edges (optional)
  if (hasFocus && sm.showDownstream) {
    const down = sm.downstream[focusIdx] || [];
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.25;
    for (const [dstIdx, wt] of down) {
      ctx.strokeStyle = wt >= 0 ? "rgba(251, 191, 36, 0.55)" : "rgba(251, 146, 60, 0.48)";
      ctx.beginPath();
      ctx.moveTo(sx[focusIdx], sy[focusIdx]);
      ctx.lineTo(sx[dstIdx], sy[dstIdx]);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Layer 3: Focused stock's Top-5 upstream edges (with directional glow & traveling pulses)
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (hasFocus) {
    const fNode = sm.nodes[focusIdx];
    const maxW = Math.max(...fNode.in_edges.map(e => Math.abs(e[1]))) || 1;
    const tx = sx[focusIdx], ty = sy[focusIdx];
    fNode.in_edges.forEach(([srcIdx, wt], rank) => {
      const fx = sx[srcIdx], fy = sy[srcIdx];
      const dx = tx - fx, dy = ty - fy;
      const dist = Math.hypot(dx, dy) || 1;
      // Subtle quadratic curve offset so overlapping edges separate cleanly
      const curve = (rank % 2 === 0 ? 1 : -1) * Math.min(34, dist * 0.14);
      const mx = (fx + tx) * 0.5 - (dy / dist) * curve;
      const my = (fy + ty) * 0.5 + (dx / dist) * curve;
      const col = wt >= 0 ? "#2dd4bf" : "#fb7185";
      const strength = Math.abs(wt) / maxW;

      ctx.save();
      ctx.strokeStyle = col;
      ctx.globalAlpha = 0.45 + 0.5 * strength;
      ctx.lineWidth = 1.2 + 1.8 * strength;
      ctx.beginPath();
      ctx.moveTo(fx, fy);
      ctx.quadraticCurveTo(mx, my, tx, ty);
      ctx.stroke();

      // Traveling light pulse along the quadratic curve (sender -> target)
      if (!reducedMotion) {
        const phase = ((now * 0.00048) + rank * 0.19) % 1;
        const u = 1 - phase;
        const px = u * u * fx + 2 * u * phase * mx + phase * phase * tx;
        const py = u * u * fy + 2 * u * phase * my + phase * phase * ty;
        ctx.globalAlpha = 0.95;
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(px, py, 2.3, 0, Math.PI * 2);
        ctx.fill();
      }

      // Edge weight badge near curve midpoint when zoomed in or selected (scaled ×10⁻³)
      if (sm.selectedIdx === focusIdx) {
        ctx.globalAlpha = 0.92;
        ctx.font = '600 9.5px "IBM Plex Mono", monospace';
        const lbl = signedEdgeMilli(wt);
        const lw = ctx.measureText(lbl).width + 8;
        const bx = 0.25 * fx + 0.5 * mx + 0.25 * tx;
        const by = 0.25 * fy + 0.5 * my + 0.25 * ty;
        ctx.fillStyle = "rgba(8, 12, 20, 0.86)";
        ctx.fillRect(bx - lw * 0.5, by - 7.5, lw, 15);
        ctx.strokeStyle = col;
        ctx.lineWidth = 0.8;
        ctx.strokeRect(bx - lw * 0.5, by - 7.5, lw, 15);
        ctx.fillStyle = col;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(lbl, bx, by + 0.5);
      }
      ctx.restore();
    });
    if (!reducedMotion) animating = true;
  }

  // Layer 4: All ~489 Star Nodes
  for (let i = 0; i < sm.nodes.length; i++) {
    const px = sx[i], py = sy[i];
    if (px < -20 || px > w + 20 || py < -20 || py > h + 20) continue;
    const n = sm.nodes[i];
    const isFocus = i === focusIdx;
    const isUp = upstreamSet.has(i);
    const isDown = downstreamSet.has(i);
    const baseR = nodeRadius(n) * Math.min(1.6, 0.85 + sm.cam.zoom * 0.14);

    ctx.save();
    if (hasFocus && !isFocus && !isUp && !isDown) {
      ctx.globalAlpha = 0.22;
    } else {
      ctx.globalAlpha = 0.95;
    }

    // Soft starlight halo for hubs, focus, and connected neighbors
    if (isFocus || isUp || isDown || n.out >= 10) {
      const haloR = isFocus ? baseR * 3.4 : baseR * 2.2;
      const g = ctx.createRadialGradient(px, py, 0.5, px, py, haloR);
      const hc = isFocus ? "rgba(250, 204, 21, 0.45)" : isUp ? "rgba(45, 212, 191, 0.35)" : isDown ? "rgba(251, 191, 36, 0.28)" : "rgba(148, 163, 184, 0.18)";
      g.addColorStop(0, hc);
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(px, py, haloR, 0, Math.PI * 2);
      ctx.fill();
    }

    // Star core
    ctx.fillStyle = isFocus ? "#facc15" : nodeColor(n);
    ctx.beginPath();
    ctx.arc(px, py, isFocus ? baseR * 1.45 : isUp ? baseR * 1.2 : baseR, 0, Math.PI * 2);
    ctx.fill();

    if (isFocus || isUp) {
      ctx.strokeStyle = isFocus ? "#fef08a" : "#ffffff";
      ctx.lineWidth = isFocus ? 1.8 : 1.1;
      ctx.stroke();
    }
    ctx.restore();
  }

  // Layer 5: LOD Star Labels with simple screen-space collision avoidance
  const placed = [];
  const tryLabel = (idx, priority = false) => {
    const px = sx[idx], py = sy[idx];
    if (px < 18 || px > w - 18 || py < 14 || py > h - 14) return;
    const text = sm.nodes[idx].id;
    const r = nodeRadius(sm.nodes[idx]) * Math.min(1.6, 0.85 + sm.cam.zoom * 0.14);
    const lx = px, ly = py - r - 8;
    const boxW = text.length * 6.8 + 8, boxH = 14;
    if (!priority) {
      for (const [bx, by, bw, bh] of placed) {
        if (Math.abs(lx - bx) < (boxW + bw) * 0.52 && Math.abs(ly - by) < (boxH + bh) * 0.55) return;
      }
    }
    placed.push([lx, ly, boxW, boxH]);
    ctx.save();
    ctx.font = priority ? '700 11px "IBM Plex Mono", monospace' : '500 10px "IBM Plex Mono", monospace';
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    if (priority) {
      ctx.fillStyle = "rgba(8, 12, 20, 0.88)";
      ctx.fillRect(lx - boxW * 0.5, ly - 7.5, boxW, 15);
      ctx.fillStyle = idx === focusIdx ? "#fde047" : "#f8fafc";
    } else {
      ctx.fillStyle = hasFocus ? "rgba(226, 232, 240, 0.45)" : "rgba(226, 232, 240, 0.82)";
    }
    ctx.fillText(text, lx, ly + 0.5);
    ctx.restore();
  };

  if (hasFocus) {
    tryLabel(focusIdx, true);
    for (const [srcIdx] of sm.nodes[focusIdx].in_edges) tryLabel(srcIdx, true);
    if (sm.showDownstream) {
      for (const [dstIdx] of (sm.downstream[focusIdx] || []).slice(0, 10)) tryLabel(dstIdx, false);
    }
  }
  // Ambient labels scale with zoom level
  if (sm.cam.zoom >= 2.2) {
    for (const idx of sm.hubRank) tryLabel(idx, false);
  } else {
    for (const sym of FEATURED) {
      if (sm.byId.has(sym)) tryLabel(sm.byId.get(sym), false);
    }
    for (const idx of sm.hubRank.slice(0, 14)) tryLabel(idx, false);
  }

  // Update zoom HUD readout
  const hud = $("sm-zoom-readout");
  if (hud) hud.textContent = `${sm.cam.zoom.toFixed(1)}×`;

  if (animating) {
    sm.animFrame = requestAnimationFrame(drawFrame);
  } else {
    sm.animFrame = 0;
  }
}

function ensureLoop() {
  if (!sm.animFrame && sm.data && !$("starmap").hidden) {
    sm.animFrame = requestAnimationFrame(drawFrame);
  }
}

function hitTest(clientX, clientY) {
  const canvas = $("sm-canvas");
  const rect = canvas.getBoundingClientRect();
  const mx = clientX - rect.left, my = clientY - rect.top;
  const w = rect.width, h = rect.height;
  let best = -1, bestDist = 14;
  for (let i = 0; i < sm.nodes.length; i++) {
    const [px, py] = worldToScreen(sm.pos[i][0], sm.pos[i][1], w, h);
    const d = Math.hypot(mx - px, my - py);
    if (d < bestDist) { bestDist = d; best = i; }
  }
  return best;
}

function bindEvents() {
  if (sm.initialized) return;
  sm.initialized = true;
  const canvas = $("sm-canvas");

  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    sm.targetCam = null;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const [wxBefore, wyBefore] = screenToWorld(mx, my, rect.width, rect.height);
    const factor = e.deltaY < 0 ? 1.18 : 1 / 1.18;
    sm.cam.zoom = Math.max(0.75, Math.min(16, sm.cam.zoom * factor));
    const [wxAfter, wyAfter] = screenToWorld(mx, my, rect.width, rect.height);
    sm.cam.x += wxBefore - wxAfter;
    sm.cam.y += wyBefore - wyAfter;
    ensureLoop();
  }, { passive: false });

  canvas.addEventListener("pointerdown", e => {
    if (e.button !== 0) return;
    sm.dragging = true;
    sm.dragMoved = false;
    sm.targetCam = null;
    sm.dragStartX = e.clientX;
    sm.dragStartY = e.clientY;
    sm.camStartX = sm.cam.x;
    sm.camStartY = sm.cam.y;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = "grabbing";
  });

  canvas.addEventListener("pointermove", e => {
    const rect = canvas.getBoundingClientRect();
    if (sm.dragging) {
      const dx = e.clientX - sm.dragStartX, dy = e.clientY - sm.dragStartY;
      if (Math.hypot(dx, dy) > 4) sm.dragMoved = true;
      const s = getScale(rect.width, rect.height);
      sm.cam.x = sm.camStartX - dx / s;
      sm.cam.y = sm.camStartY + dy / s;
      ensureLoop();
      return;
    }
    const hit = hitTest(e.clientX, e.clientY);
    if (hit !== sm.hoveredIdx) {
      sm.hoveredIdx = hit;
      canvas.style.cursor = hit >= 0 ? "pointer" : "grab";
      const tip = $("sm-tooltip");
      if (hit >= 0) {
        const n = sm.nodes[hit];
        tip.textContent = `${n.id}: score ${signedShort(n.s)}, ${n.out} downstream`;
        tip.hidden = false;
      } else {
        tip.hidden = true;
      }
      ensureLoop();
    }
    if (hit >= 0) {
      const tip = $("sm-tooltip");
      tip.style.left = Math.min(rect.width - 180, Math.max(10, e.clientX - rect.left + 12)) + "px";
      tip.style.top = Math.max(10, e.clientY - rect.top - 30) + "px";
    }
  });

  canvas.addEventListener("pointerup", e => {
    if (!sm.dragging) return;
    sm.dragging = false;
    canvas.releasePointerCapture(e.pointerId);
    canvas.style.cursor = sm.hoveredIdx >= 0 ? "pointer" : "grab";
    if (!sm.dragMoved) {
      const hit = hitTest(e.clientX, e.clientY);
      if (hit >= 0) selectNode(hit, { fly: true, zoom: Math.max(sm.cam.zoom, 3.8) });
      else selectNode(-1, { fly: false });
    }
  });

  canvas.addEventListener("pointerleave", () => {
    if (sm.hoveredIdx >= 0) {
      sm.hoveredIdx = -1;
      $("sm-tooltip").hidden = true;
      ensureLoop();
    }
  });

  const searchInput = $("sm-search");
  searchInput.addEventListener("focus", () => renderSuggest(searchInput.value));
  searchInput.addEventListener("input", () => renderSuggest(searchInput.value));
  searchInput.addEventListener("blur", () => setTimeout(closeSuggest, 120));
  searchInput.addEventListener("keydown", e => {
    if ($("sm-suggest").hidden || !suggestItems.length) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      suggestActive = (suggestActive + 1) % suggestItems.length;
      updateSuggestHighlight();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      suggestActive = (suggestActive - 1 + suggestItems.length) % suggestItems.length;
      updateSuggestHighlight();
    } else if (e.key === "Escape") {
      closeSuggest();
    } else if (e.key === "Enter" && suggestActive >= 0) {
      e.preventDefault();
      const picked = suggestItems[suggestActive];
      $("sm-status").hidden = true;
      closeSuggest();
      selectNode(picked.idx, { fly: true });
    }
  });

  $("sm-search-form").addEventListener("submit", e => {
    e.preventDefault();
    closeSuggest();
    const raw = searchInput.value.trim().toUpperCase();
    if (!raw) { selectNode(-1, { fly: false }); resetCamera(); return; }
    let idx = sm.byId.get(raw);
    if (idx === undefined) {
      let best = null;
      for (let i = 0; i < sm.nodes.length; i++) {
        const m = fuzzyMatch(raw, sm.nodes[i].id);
        if (m && (!best || m.rank < best.rank || (m.rank === best.rank && sm.nodes[i].out > sm.nodes[best.idx].out))) {
          best = { idx: i, rank: m.rank };
        }
      }
      if (best) idx = best.idx;
    }
    if (idx !== undefined) {
      $("sm-status").hidden = true;
      selectNode(idx, { fly: true });
    } else {
      $("sm-status").textContent = `Ticker "${raw}" is not in the ${sm.data.date} S&P 500 universe.`;
      $("sm-status").hidden = false;
    }
  });

  $("sm-clear").addEventListener("click", () => {
    $("sm-status").hidden = true;
    selectNode(-1, { fly: false });
    resetCamera();
  });

  $("sm-zoom-in").addEventListener("click", () => {
    sm.targetCam = null;
    sm.cam.zoom = Math.min(16, sm.cam.zoom * 1.45);
    ensureLoop();
  });
  $("sm-zoom-out").addEventListener("click", () => {
    sm.targetCam = null;
    sm.cam.zoom = Math.max(0.75, sm.cam.zoom / 1.45);
    ensureLoop();
  });
  $("sm-zoom-reset").addEventListener("click", resetCamera);

  $("sm-layout-switch").addEventListener("click", e => {
    const b = e.target.closest("button[data-layout]");
    if (b) setLayout(b.dataset.layout);
  });
  $("sm-color-switch").addEventListener("click", e => {
    const b = e.target.closest("button[data-color]");
    if (b) setColorBy(b.dataset.color);
  });
  $("sm-toggle-web").addEventListener("change", e => {
    sm.showWeb = e.target.checked;
    ensureLoop();
  });
  $("sm-toggle-down").addEventListener("change", e => {
    sm.showDownstream = e.target.checked;
    ensureLoop();
  });

  new ResizeObserver(() => ensureLoop()).observe(canvas);
}

export async function showStarmap({ onOpenExplorer } = {}) {
  if (onOpenExplorer) sm.onOpenExplorer = onOpenExplorer;
  bindEvents();
  try {
    if (!sm.data) {
      $("sm-status").textContent = "Loading cross-sectional star map…";
      $("sm-status").hidden = false;
      sm.loading ||= fetch("./starmap-demo.json").then(r => {
        if (!r.ok) throw Error("Could not load starmap-demo.json");
        return r.json();
      });
      sm.data = await sm.loading;
      sm.nodes = sm.data.nodes;
      sm.byId = new Map(sm.nodes.map((n, i) => [n.id, i]));
      sm.downstream = Array.from({ length: sm.nodes.length }, () => []);
      sm.nodes.forEach((n, dstIdx) => {
        for (const [srcIdx, w] of n.in_edges) {
          sm.downstream[srcIdx].push([dstIdx, w]);
        }
      });
      for (const list of sm.downstream) list.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
      sm.hubRank = sm.nodes.map((_, i) => i).sort((a, b) => sm.nodes[b].out - sm.nodes[a].out);
      sm.pos = sm.nodes.map(n => [n.x, n.y]);
      sm.targetPos = sm.nodes.map(n => [n.x, n.y]);

      $("sm-date-badge").textContent = `Jan 2, 2024 — ${sm.data.universe_size} stocks, ${sm.data.edge_count.toLocaleString("en-US")} directed links`;
      updateLegend();
      renderQuickChips();
      // Start in full-market overview with AAPL gently highlighted (no zoom yet so user sees the whole galaxy)
      const aapl = sm.byId.get("AAPL");
      if (aapl !== undefined) {
        sm.selectedIdx = aapl;
        $("sm-search").value = "AAPL";
      }
      renderInspector();
      renderQuickChips();
    }
    $("sm-status").hidden = true;
    ensureLoop();
  } catch (err) {
    sm.loading = null;
    $("sm-status").textContent = err.message;
    $("sm-status").hidden = false;
  }
}
