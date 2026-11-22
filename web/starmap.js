// Market Star Map: interactive 3D volumetric sphere of S&P 500 DRIFT flows.
const $ = id => document.getElementById(id);

const FEATURED = ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM"];
const DEFAULT_PITCH = 0.24; // gentle 3D camera elevation
const DEFAULT_YAW = -0.42;
const CAM_DIST = 3.15;

const sm = {
  data: null,
  loading: null,
  nodes: [],
  byId: new Map(),
  downstream: [],
  hubRank: [],
  selectedIdx: -1,
  hoveredIdx: -1,
  layout: "constellation", // "constellation" (3D sphere volume) | "role" (flat 2D quadrant)
  colorBy: "score",        // "score" | "role"
  showWeb: true,
  showDownstream: false,
  autoSpin: true,
  spinResumeAt: 0,
  // 3D spherical orbit camera around focal target (tx, ty, tz)
  cam: { tx: 0, ty: 0, tz: 0, yaw: DEFAULT_YAW, pitch: DEFAULT_PITCH, zoom: 1.04 },
  targetCam: null,
  // Angular momentum for silky drag inertia
  velYaw: 0,
  velPitch: 0,
  // 3D layout interpolation: [x, y, z] per stock
  pos: [],
  targetPos: [],
  layoutLerp: 1,
  // Cached screen projection for crisp hit-testing
  screenX: new Float32Array(0),
  screenY: new Float32Array(0),
  screenZ: new Float32Array(0),
  screenP: new Float32Array(0),
  // Pointer state
  dragging: false,
  dragPan: false,
  dragMoved: false,
  lastPointerX: 0,
  lastPointerY: 0,
  // Animation & callbacks
  animFrame: 0,
  lastNow: 0,
  onOpenExplorer: null,
  archivedTargets: new Set(FEATURED),
  initialized: false,
};

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
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
  return Math.min(w, h) * 0.42 * sm.cam.zoom;
}

// Standard 3D spherical orbit projection: yaw around vertical Y, pitch around X.
function project3D(wx, wy, wz, w, h) {
  const dx = wx - sm.cam.tx;
  const dy = wy - sm.cam.ty;
  const dz = wz - sm.cam.tz;
  const cy = Math.cos(sm.cam.yaw), sy = Math.sin(sm.cam.yaw);
  const cp = Math.cos(sm.cam.pitch), sp = Math.sin(sm.cam.pitch);

  // Rotate around vertical Y axis
  const x1 = dx * cy + dz * sy;
  const y1 = dy;
  const z1 = -dx * sy + dz * cy;

  // Tilt around horizontal X axis
  const x2 = x1;
  const y2 = y1 * cp - z1 * sp;
  const z2 = y1 * sp + z1 * cp;

  const flat2D = sm.layout === "role" && Math.abs(sm.cam.pitch) < 0.01 && Math.abs(sm.cam.yaw) < 0.01;
  const persp = flat2D ? 1.0 : CAM_DIST / Math.max(0.6, CAM_DIST - z2);
  const s = getScale(w, h) * persp;
  return [w * 0.5 + x2 * s, h * 0.5 - y2 * s, z2, persp];
}

function nodeRadius(n) {
  return 2.3 + Math.min(4.6, Math.sqrt(n.out) * 0.90);
}

function nodeColor(n) {
  if (sm.colorBy === "role") {
    const net = n.gs - n.gr;
    if (net > 0.04) return "#2dd4bf";
    if (net < -0.04) return "#f59e0b";
    return "#94a3b8";
  }
  if (n.s >= 0.35) return "#14b8a6";
  if (n.s >= 0.0) return "#5eead4";
  if (n.s <= -0.35) return "#f43f5e";
  return "#fda4af";
}

function flyTo(idx, maxZoom = 2.6) {
  if (idx < 0 || idx >= sm.nodes.length) return;
  const canvas = $("sm-canvas");
  const w = canvas?.clientWidth || 800, h = canvas?.clientHeight || 590;
  const pts = [sm.targetPos[idx] || sm.pos[idx]];
  for (const [srcIdx] of sm.nodes[idx].in_edges) {
    pts.push(sm.targetPos[srcIdx] || sm.pos[srcIdx]);
  }
  let sx = 0, sy = 0, sz = 0;
  for (const [px, py, pz] of pts) { sx += px; sy += py; sz += pz || 0; }
  const cx = sx / pts.length, cy = sy / pts.length, cz = sz / pts.length;
  let maxR = 0.16;
  for (const [px, py, pz] of pts) {
    const d = Math.hypot(px - cx, py - cy, (pz || 0) - cz);
    if (d > maxR) maxR = d;
  }
  const baseScale = Math.min(w, h) * 0.42;
  const fitZoom = (Math.min(w, h) * 0.36) / (maxR * baseScale);
  const zoom = clamp(fitZoom, 1.15, maxZoom);
  sm.velYaw = 0;
  sm.velPitch = 0;
  sm.targetCam = {
    tx: cx, ty: cy, tz: cz,
    yaw: sm.layout === "role" ? 0 : sm.cam.yaw,
    pitch: sm.layout === "role" ? 0 : sm.cam.pitch,
    zoom,
    startTx: sm.cam.tx, startTy: sm.cam.ty, startTz: sm.cam.tz,
    startYaw: sm.cam.yaw, startPitch: sm.cam.pitch, startZoom: sm.cam.zoom,
    startTime: performance.now(),
    duration: 440,
  };
  sm.spinResumeAt = performance.now() + 900;
  ensureLoop();
}

function resetCamera() {
  sm.velYaw = 0;
  sm.velPitch = 0;
  const isRole = sm.layout === "role";
  sm.targetCam = {
    tx: 0, ty: 0, tz: 0,
    yaw: isRole ? 0 : sm.cam.yaw,
    pitch: isRole ? 0 : DEFAULT_PITCH,
    zoom: isRole ? 1.0 : 1.04,
    startTx: sm.cam.tx, startTy: sm.cam.ty, startTz: sm.cam.tz,
    startYaw: sm.cam.yaw, startPitch: sm.cam.pitch, startZoom: sm.cam.zoom,
    startTime: performance.now(),
    duration: 420,
  };
  ensureLoop();
}

function selectNode(idx, { fly = true, zoom = 2.6 } = {}) {
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
  sm.targetPos = sm.nodes.map(n =>
    mode === "role" ? [n.rx, n.ry, 0] : [n.x, n.y, n.z || 0]
  );
  sm.layoutLerp = 0;
  sm.velYaw = 0;
  sm.velPitch = 0;
  const focus = sm.selectedIdx >= 0 ? sm.targetPos[sm.selectedIdx] : [0, 0, 0];
  sm.targetCam = {
    tx: sm.selectedIdx >= 0 ? focus[0] : 0,
    ty: sm.selectedIdx >= 0 ? focus[1] : 0,
    tz: sm.selectedIdx >= 0 ? focus[2] : 0,
    yaw: mode === "role" ? 0 : DEFAULT_YAW,
    pitch: mode === "role" ? 0 : DEFAULT_PITCH,
    zoom: sm.selectedIdx >= 0 ? Math.min(sm.cam.zoom, 2.2) : (mode === "role" ? 1.0 : 1.04),
    startTx: sm.cam.tx, startTy: sm.cam.ty, startTz: sm.cam.tz,
    startYaw: sm.cam.yaw, startPitch: sm.cam.pitch, startZoom: sm.cam.zoom,
    startTime: performance.now(),
    duration: 460,
  };
  updateSpinButton();
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

function updateSpinButton() {
  const btn = $("sm-spin-toggle");
  if (!btn) return;
  const active = sm.autoSpin && sm.layout === "constellation";
  btn.textContent = active ? "Pause" : "Spin";
  btn.setAttribute("aria-pressed", String(active));
  btn.disabled = sm.layout === "role";
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
    b.onclick = () => selectNode(sm.byId.get(sym), { fly: true, zoom: 2.4 });
    return b;
  }));
}

function renderInspector() {
  const empty = $("sm-inspector-empty");
  const body = $("sm-inspector-body");
  if (sm.selectedIdx < 0) {
    empty.hidden = false;
    body.hidden = true;
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
      b.onclick = () => selectNode(idx, { fly: true, zoom: 2.4 });
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
    b.onclick = () => selectNode(srcIdx, { fly: true, zoom: Math.max(sm.cam.zoom, 2.4) });
    return b;
  }));

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
      b.onclick = () => selectNode(dstIdx, { fly: true, zoom: Math.max(sm.cam.zoom, 2.4) });
      return b;
    }));
  }

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

// Draw subtle 3D volumetric bounding sphere guides (outer rim + 3 faint great circles).
function drawSphereVolumeGuides(ctx, w, h) {
  const [ox, oy, oz, opersp] = project3D(0, 0, 0, w, h);
  const R = 1.0;
  // Soft volumetric glow inside the sphere core
  const dCenter = Math.max(0.8, CAM_DIST - oz);
  const apparentR = getScale(w, h) * (CAM_DIST * R / Math.sqrt(Math.max(0.25, dCenter * dCenter - R * R * 0.35)));
  ctx.save();
  const volGlow = ctx.createRadialGradient(ox, oy, 4, ox, oy, apparentR);
  volGlow.addColorStop(0, "rgba(45, 212, 191, 0.09)");
  volGlow.addColorStop(0.55, "rgba(56, 189, 248, 0.04)");
  volGlow.addColorStop(0.92, "rgba(148, 163, 184, 0.02)");
  volGlow.addColorStop(1, "rgba(8, 12, 20, 0)");
  ctx.fillStyle = volGlow;
  ctx.beginPath();
  ctx.arc(ox, oy, apparentR, 0, Math.PI * 2);
  ctx.fill();

  // Outer spherical boundary silhouette
  ctx.strokeStyle = "rgba(148, 163, 184, 0.14)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(ox, oy, apparentR, 0, Math.PI * 2);
  ctx.stroke();

  // Three faint orthogonal great circles (equator + two meridians) rotating with the 3D volume
  const segs = 80;
  const circles = [
    a => [Math.cos(a), 0, Math.sin(a)], // XZ equator
    a => [Math.cos(a), Math.sin(a), 0], // XY meridian
    a => [0, Math.sin(a), Math.cos(a)], // YZ meridian
  ];
  ctx.strokeStyle = "rgba(148, 163, 184, 0.075)";
  for (const fn of circles) {
    ctx.beginPath();
    for (let k = 0; k <= segs; k++) {
      const a = (k / segs) * Math.PI * 2;
      const [gx, gy, gz] = fn(a);
      const [px, py] = project3D(gx * R, gy * R, gz * R, w, h);
      if (k === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.restore();
  void opersp;
}

function drawFrame(now) {
  const canvas = $("sm-canvas");
  if (!canvas || $("starmap").hidden) { sm.animFrame = 0; return; }
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;

  const dt = sm.lastNow ? Math.min(48, now - sm.lastNow) : 16;
  sm.lastNow = now;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let animating = false;

  // Step camera fly animation
  if (sm.targetCam) {
    const t = Math.min(1, (now - sm.targetCam.startTime) / sm.targetCam.duration);
    const ease = 1 - Math.pow(1 - t, 3);
    sm.cam.tx = sm.targetCam.startTx + (sm.targetCam.tx - sm.targetCam.startTx) * ease;
    sm.cam.ty = sm.targetCam.startTy + (sm.targetCam.ty - sm.targetCam.startTy) * ease;
    sm.cam.tz = sm.targetCam.startTz + (sm.targetCam.tz - sm.targetCam.startTz) * ease;
    sm.cam.yaw = sm.targetCam.startYaw + (sm.targetCam.yaw - sm.targetCam.startYaw) * ease;
    sm.cam.pitch = sm.targetCam.startPitch + (sm.targetCam.pitch - sm.targetCam.startPitch) * ease;
    sm.cam.zoom = sm.targetCam.startZoom + (sm.targetCam.zoom - sm.targetCam.startZoom) * ease;
    if (t >= 1) sm.targetCam = null;
    else animating = true;
  } else if (!sm.dragging && sm.layout === "constellation") {
    // Apply smooth drag inertia
    if (Math.abs(sm.velYaw) > 1e-5 || Math.abs(sm.velPitch) > 1e-5) {
      sm.cam.yaw += sm.velYaw;
      sm.cam.pitch = clamp(sm.cam.pitch + sm.velPitch, -1.25, 1.25);
      sm.velYaw *= 0.92;
      sm.velPitch *= 0.90;
      animating = true;
    }
    // Gentle continuous 3D sphere rotation
    if (sm.autoSpin && !reducedMotion && now >= sm.spinResumeAt) {
      const spinRate = sm.selectedIdx >= 0 && sm.cam.zoom > 1.6 ? 0.000095 : 0.00018;
      sm.cam.yaw += dt * spinRate;
      animating = true;
    }
  }

  // Step 3D layout interpolation
  if (sm.layoutLerp < 1) {
    sm.layoutLerp = Math.min(1, sm.layoutLerp + 0.08);
    const e = 1 - Math.pow(1 - sm.layoutLerp, 3);
    for (let i = 0; i < sm.pos.length; i++) {
      sm.pos[i][0] += (sm.targetPos[i][0] - sm.pos[i][0]) * Math.max(0.18, e);
      sm.pos[i][1] += (sm.targetPos[i][1] - sm.pos[i][1]) * Math.max(0.18, e);
      sm.pos[i][2] += (sm.targetPos[i][2] - sm.pos[i][2]) * Math.max(0.18, e);
    }
    if (sm.layoutLerp >= 1) {
      for (let i = 0; i < sm.pos.length; i++) {
        sm.pos[i][0] = sm.targetPos[i][0];
        sm.pos[i][1] = sm.targetPos[i][1];
        sm.pos[i][2] = sm.targetPos[i][2];
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

  // Deep observatory dark backdrop
  const bg = ctx.createRadialGradient(w * 0.5, h * 0.5, 20, w * 0.5, h * 0.5, Math.max(w, h) * 0.74);
  bg.addColorStop(0, "#101726");
  bg.addColorStop(1, "#070b13");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  if (sm.layout === "constellation") {
    drawSphereVolumeGuides(ctx, w, h);
  } else {
    const [ox, oy] = project3D(0, 0, 0, w, h);
    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.14)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, oy); ctx.lineTo(w, oy);
    ctx.moveTo(ox, 0); ctx.lineTo(ox, h);
    ctx.stroke();
    ctx.fillStyle = "rgba(148, 163, 184, 0.45)";
    ctx.font = '500 10px "IBM Plex Mono", monospace';
    ctx.textAlign = "right";
    ctx.fillText("NET SENDER →", w - 14, oy - 6);
    ctx.textAlign = "left";
    ctx.fillText("← NET RECEIVER", 14, oy - 6);
    ctx.fillText("↑ POSITIVE DRIFT", ox + 8, 18);
    ctx.fillText("↓ NEGATIVE DRIFT", ox + 8, h - 12);
    ctx.restore();
  }

  // Project all 489 stars once per frame
  const count = sm.nodes.length;
  if (sm.screenX.length !== count) {
    sm.screenX = new Float32Array(count);
    sm.screenY = new Float32Array(count);
    sm.screenZ = new Float32Array(count);
    sm.screenP = new Float32Array(count);
  }
  const sx = sm.screenX, sy = sm.screenY, sz = sm.screenZ, sp = sm.screenP;
  for (let i = 0; i < count; i++) {
    const [px, py, pz, persp] = project3D(sm.pos[i][0], sm.pos[i][1], sm.pos[i][2], w, h);
    sx[i] = px; sy[i] = py; sz[i] = pz; sp[i] = persp;
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
      const depthAlpha = clamp(0.45 + 0.55 * ((sp[u] + sp[v]) * 0.5 - 0.65), 0.25, 1.0);
      ctx.strokeStyle = hasFocus
        ? `rgba(148, 163, 184, ${(0.042 * depthAlpha).toFixed(3)})`
        : (wt >= 0
          ? `rgba(45, 212, 191, ${(0.14 * depthAlpha).toFixed(3)})`
          : `rgba(244, 63, 94, ${(0.13 * depthAlpha).toFixed(3)})`);
      ctx.beginPath();
      ctx.moveTo(sx[u], sy[u]);
      ctx.lineTo(sx[v], sy[v]);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Layer 2: Optional downstream edges
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

  // Layer 3: Focused stock's Top-5 upstream 3D links (near-straight so 3D length stays faithful)
  const badgeBoxes = [];
  if (hasFocus) {
    const fNode = sm.nodes[focusIdx];
    const maxW = Math.max(...fNode.in_edges.map(e => Math.abs(e[1]))) || 1;
    const tx = sx[focusIdx], ty = sy[focusIdx];
    const badgeTs = [0.42, 0.56, 0.34, 0.64, 0.48];
    fNode.in_edges.forEach(([srcIdx, wt], rank) => {
      const fx = sx[srcIdx], fy = sy[srcIdx];
      const dx = tx - fx, dy = ty - fy;
      const dist = Math.hypot(dx, dy) || 1;
      // Very subtle curve just to separate co-linear links while preserving true distance perception
      const curve = (rank % 2 === 0 ? 1 : -1) * Math.min(14, dist * 0.06);
      const mx = (fx + tx) * 0.5 - (dy / dist) * curve;
      const my = (fy + ty) * 0.5 + (dx / dist) * curve;

      const col = wt >= 0 ? "#2dd4bf" : "#fb7185";
      const strength = Math.abs(wt) / maxW;

      ctx.save();
      ctx.strokeStyle = col;
      ctx.globalAlpha = 0.52 + 0.46 * strength;
      ctx.lineWidth = 1.25 + 1.75 * strength;
      ctx.beginPath();
      ctx.moveTo(fx, fy);
      ctx.quadraticCurveTo(mx, my, tx, ty);
      ctx.stroke();

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

      if (sm.selectedIdx === focusIdx) {
        ctx.globalAlpha = 0.94;
        ctx.font = '600 9.5px "IBM Plex Mono", monospace';
        const lbl = signedEdgeMilli(wt);
        const lw = ctx.measureText(lbl).width + 8;
        const lh = 15;
        const bt = badgeTs[rank % badgeTs.length];
        const bu = 1 - bt;
        let bx = bu * bu * fx + 2 * bu * bt * mx + bt * bt * tx;
        let by = bu * bu * fy + 2 * bu * bt * my + bt * bt * ty;
        for (let tries = 0; tries < 6; tries++) {
          const hit = badgeBoxes.some(([ox, oy, ow, oh]) =>
            Math.abs(bx - ox) < (lw + ow) * 0.54 && Math.abs(by - oy) < (lh + oh) * 0.56
          );
          if (!hit) break;
          by += (rank % 2 === 0 ? -1 : 1) * 16;
        }
        badgeBoxes.push([bx, by, lw, lh]);
        ctx.fillStyle = "rgba(8, 12, 20, 0.90)";
        ctx.fillRect(bx - lw * 0.5, by - lh * 0.5, lw, lh);
        ctx.strokeStyle = col;
        ctx.lineWidth = 0.8;
        ctx.strokeRect(bx - lw * 0.5, by - lh * 0.5, lw, lh);
        ctx.fillStyle = col;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(lbl, bx, by + 0.5);
      }
      ctx.restore();
    });
    if (!reducedMotion) animating = true;
  }

  // Layer 4: Depth-sorted 3D Volumetric Star Nodes (back-to-front painter's order)
  const order = Array.from({ length: count }, (_, i) => i).sort((a, b) => sz[a] - sz[b]);
  for (let k = 0; k < count; k++) {
    const i = order[k];
    const px = sx[i], py = sy[i], persp = sp[i];
    if (px < -24 || px > w + 24 || py < -24 || py > h + 24) continue;
    const n = sm.nodes[i];
    const isFocus = i === focusIdx;
    const isUp = upstreamSet.has(i);
    const isDown = downstreamSet.has(i);
    const baseR = nodeRadius(n) * Math.min(1.65, 0.84 + sm.cam.zoom * 0.14) * clamp(persp, 0.58, 1.55);

    ctx.save();
    const depthFade = clamp(0.32 + 0.68 * ((persp - 0.65) / 0.62), 0.28, 1.0);
    if (hasFocus && !isFocus && !isUp && !isDown) {
      ctx.globalAlpha = 0.24 * depthFade;
    } else {
      ctx.globalAlpha = isFocus || isUp ? 0.98 : 0.94 * depthFade;
    }

    if (isFocus || isUp || isDown || n.out >= 10) {
      const haloR = isFocus ? baseR * 3.3 : baseR * 2.2;
      const g = ctx.createRadialGradient(px, py, 0.5, px, py, haloR);
      const hc = isFocus ? "rgba(250, 204, 21, 0.45)" : isUp ? "rgba(45, 212, 191, 0.35)" : isDown ? "rgba(251, 191, 36, 0.28)" : "rgba(148, 163, 184, 0.18)";
      g.addColorStop(0, hc);
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(px, py, haloR, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = isFocus ? "#facc15" : nodeColor(n);
    ctx.beginPath();
    ctx.arc(px, py, isFocus ? baseR * 1.42 : isUp ? baseR * 1.18 : baseR, 0, Math.PI * 2);
    ctx.fill();

    if (isFocus || isUp) {
      ctx.strokeStyle = isFocus ? "#fef08a" : "#ffffff";
      ctx.lineWidth = isFocus ? 1.8 : 1.1;
      ctx.stroke();
    }
    ctx.restore();
  }

  // Layer 5: LOD Star Labels with screen-space collision avoidance
  const placed = [];
  const tryLabel = (idx, priority = false) => {
    const px = sx[idx], py = sy[idx], persp = sp[idx];
    if (px < 18 || px > w - 18 || py < 14 || py > h - 14) return;
    if (!priority && persp < 0.78) return; // keep far back-side labels quiet for clear 3D depth
    const text = sm.nodes[idx].id;
    const r = nodeRadius(sm.nodes[idx]) * Math.min(1.65, 0.84 + sm.cam.zoom * 0.14) * clamp(persp, 0.62, 1.5);
    const lx = px;
    let ly = py - r - 8;
    const boxW = text.length * 6.8 + 8, boxH = 14;
    if (!priority) {
      for (const [bx, by, bw, bh] of placed) {
        if (Math.abs(lx - bx) < (boxW + bw) * 0.52 && Math.abs(ly - by) < (boxH + bh) * 0.55) return;
      }
    } else {
      for (let tries = 0; tries < 4; tries++) {
        const hit = placed.some(([bx, by, bw, bh]) =>
          Math.abs(lx - bx) < (boxW + bw) * 0.52 && Math.abs(ly - by) < (boxH + bh) * 0.58
        );
        if (!hit) break;
        ly = tries % 2 === 0 ? py + r + 11 : ly - 15;
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
      ctx.fillStyle = hasFocus ? "rgba(226, 232, 240, 0.42)" : "rgba(226, 232, 240, 0.80)";
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
  if (sm.cam.zoom >= 2.0) {
    for (const sym of FEATURED) {
      if (sm.byId.has(sym)) tryLabel(sm.byId.get(sym), false);
    }
    for (const idx of sm.hubRank.slice(0, hasFocus ? 20 : 32)) tryLabel(idx, false);
  } else {
    for (const sym of FEATURED) {
      if (sm.byId.has(sym)) tryLabel(sm.byId.get(sym), false);
    }
    for (const idx of sm.hubRank.slice(0, 14)) tryLabel(idx, false);
  }

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
    sm.lastNow = 0;
    sm.animFrame = requestAnimationFrame(drawFrame);
  }
}

function hitTest(clientX, clientY) {
  const canvas = $("sm-canvas");
  const rect = canvas.getBoundingClientRect();
  const mx = clientX - rect.left, my = clientY - rect.top;
  let best = -1, bestDist = 15, bestZ = -Infinity;
  for (let i = 0; i < sm.nodes.length; i++) {
    const d = Math.hypot(mx - sm.screenX[i], my - sm.screenY[i]);
    if (d < bestDist || (d < 15 && sm.screenZ[i] > bestZ && d < bestDist + 3)) {
      bestDist = d;
      bestZ = sm.screenZ[i];
      best = i;
    }
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
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    sm.cam.zoom = clamp(sm.cam.zoom * factor, 0.65, 10);
    ensureLoop();
  }, { passive: false });

  canvas.addEventListener("contextmenu", e => e.preventDefault());

  canvas.addEventListener("pointerdown", e => {
    if (e.button !== 0 && e.button !== 2) return;
    sm.dragging = true;
    sm.dragPan = sm.layout === "role" || e.shiftKey || e.button === 2;
    sm.dragMoved = false;
    sm.targetCam = null;
    sm.velYaw = 0;
    sm.velPitch = 0;
    sm.lastPointerX = e.clientX;
    sm.lastPointerY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = "grabbing";
  });

  canvas.addEventListener("pointermove", e => {
    const rect = canvas.getBoundingClientRect();
    if (sm.dragging) {
      const dx = e.clientX - sm.lastPointerX;
      const dy = e.clientY - sm.lastPointerY;
      if (Math.hypot(dx, dy) > 2) sm.dragMoved = true;
      sm.lastPointerX = e.clientX;
      sm.lastPointerY = e.clientY;

      if (sm.dragPan) {
        const s = getScale(rect.width, rect.height);
        const cy = Math.cos(sm.cam.yaw), sy = Math.sin(sm.cam.yaw);
        const cp = Math.cos(sm.cam.pitch), sp = Math.sin(sm.cam.pitch);
        const vx = -dx / s, vy = dy / s;
        // Inverse camera rotation for smooth screen-plane panning
        sm.cam.tx += vx * cy - vy * sp * sy;
        sm.cam.ty += vy * cp;
        sm.cam.tz += vx * sy + vy * sp * cy;
      } else {
        // Drag to freely orbit the 3D sphere
        const dYaw = dx * 0.0064;
        const dPitch = dy * 0.0056;
        sm.cam.yaw += dYaw;
        sm.cam.pitch = clamp(sm.cam.pitch + dPitch, -1.25, 1.25);
        sm.velYaw = sm.velYaw * 0.4 + dYaw * 0.6;
        sm.velPitch = sm.velPitch * 0.4 + dPitch * 0.6;
      }
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
      if (hit >= 0) selectNode(hit, { fly: true, zoom: Math.max(sm.cam.zoom, 2.4) });
      else selectNode(-1, { fly: false });
    } else {
      sm.spinResumeAt = performance.now() + 1500;
      ensureLoop();
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
    sm.cam.zoom = clamp(sm.cam.zoom * 1.4, 0.65, 10);
    ensureLoop();
  });
  $("sm-zoom-out").addEventListener("click", () => {
    sm.targetCam = null;
    sm.cam.zoom = clamp(sm.cam.zoom / 1.4, 0.65, 10);
    ensureLoop();
  });
  $("sm-zoom-reset").addEventListener("click", resetCamera);

  const spinBtn = $("sm-spin-toggle");
  if (spinBtn) {
    spinBtn.addEventListener("click", () => {
      sm.autoSpin = !sm.autoSpin;
      sm.spinResumeAt = 0;
      updateSpinButton();
      ensureLoop();
    });
  }

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
      sm.pos = sm.nodes.map(n => [n.x, n.y, n.z || 0]);
      sm.targetPos = sm.nodes.map(n => [n.x, n.y, n.z || 0]);

      $("sm-date-badge").textContent = `Jan 2, 2024 — ${sm.data.universe_size} stocks, ${sm.data.edge_count.toLocaleString("en-US")} directed links`;
      updateLegend();
      updateSpinButton();
      renderQuickChips();
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
