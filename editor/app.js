/* Photo Prod — a small layered photo editor with a Photoshop-style UI */
'use strict';

/* ---------------------------------- state ---------------------------------- */

const state = {
  doc: { width: 1280, height: 800, name: 'Untitled' },
  layers: [],            // { id, name, canvas, ctx, x, y, visible, opacity, blend }
  active: 0,             // index into layers
  tool: 'brush',
  color: '#4da3ff',
  brush: { size: 24, opacity: 100 },
  fill: { tolerance: 32 },
  zoom: 1,
  panX: 0,
  panY: 0,
  selection: null,       // { x, y, w, h } in doc pixels
  shapePreview: null,    // live selection-shape drag preview
  adjust: { brightness: 100, contrast: 100, saturate: 100, hue: 0, blur: 0 },
  history: [],
  future: [],
};

let layerSeq = 1;

const BLEND_MODES = [
  ['normal', 'source-over'], ['multiply', 'multiply'], ['screen', 'screen'],
  ['overlay', 'overlay'], ['darken', 'darken'], ['lighten', 'lighten'],
  ['soft light', 'soft-light'], ['difference', 'difference'],
];

const $ = (sel) => document.querySelector(sel);
const docCanvas = $('#doc');
const docCtx = docCanvas.getContext('2d');
const overlay = $('#overlay');
const overlayCtx = overlay.getContext('2d');
const stage = $('#stage');
const viewport = $('#viewport');

/* --------------------------------- layers ---------------------------------- */

function makeLayer(name, opts = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = state.doc.width;
  canvas.height = state.doc.height;
  return {
    id: layerSeq++,
    name,
    canvas,
    ctx: canvas.getContext('2d', { willReadFrequently: true }),
    x: 0, y: 0,
    visible: true,
    opacity: 100,
    blend: 'source-over',
    edgeFeather: 0,
    edgeBase: null,    // lazily captured snapshot at feather 0
    variations: null,  // [Image] when this is an AI variation layer
    variationIndex: 0,
    ...opts,
  };
}

// Soften only the layer's alpha edge (keeps RGB sharp): redraw the unfeathered
// base, then intersect its alpha with a blurred copy so the boundary fades
// inward without blurring the content or adding an outward glow.
function featherLayerEdge(layer, px) {
  if (!layer.edgeBase) {
    layer.edgeBase = document.createElement('canvas');
    layer.edgeBase.width = layer.canvas.width;
    layer.edgeBase.height = layer.canvas.height;
    layer.edgeBase.getContext('2d').drawImage(layer.canvas, 0, 0);
  }
  layer.edgeFeather = px;
  const ctx = layer.ctx;
  ctx.clearRect(0, 0, layer.canvas.width, layer.canvas.height);
  ctx.drawImage(layer.edgeBase, 0, 0);
  if (px > 0) {
    const tmp = document.createElement('canvas');
    tmp.width = layer.canvas.width; tmp.height = layer.canvas.height;
    const tctx = tmp.getContext('2d');
    tctx.filter = `blur(${px}px)`;
    tctx.drawImage(layer.edgeBase, 0, 0);
    ctx.save();
    ctx.globalCompositeOperation = 'destination-in';   // alpha *= blurred alpha
    ctx.drawImage(tmp, 0, 0);
    ctx.restore();
  }
  render();
  renderLayerList();
}

function activeLayer() { return state.layers[state.active]; }

function addLayer(name = `Layer ${layerSeq}`) {
  pushHistory();
  state.layers.splice(state.active + 1, 0, makeLayer(name));
  state.active += 1;
  refresh();
}

function duplicateLayer() {
  const src = activeLayer();
  if (!src) return;
  pushHistory();
  const copy = makeLayer(`${src.name} copy`, {
    x: src.x, y: src.y, opacity: src.opacity, blend: src.blend,
  });
  copy.ctx.drawImage(src.canvas, 0, 0);
  state.layers.splice(state.active + 1, 0, copy);
  state.active += 1;
  refresh();
}

function deleteLayer() {
  if (state.layers.length <= 1) return;
  pushHistory();
  state.layers.splice(state.active, 1);
  state.active = Math.max(0, state.active - 1);
  refresh();
}

function moveLayer(dir) {
  const i = state.active, j = i + dir;
  if (j < 0 || j >= state.layers.length) return;
  pushHistory();
  [state.layers[i], state.layers[j]] = [state.layers[j], state.layers[i]];
  state.active = j;
  refresh();
}

/* --------------------------------- history --------------------------------- */

const HISTORY_LIMIT = 12;

function snapshot() {
  return {
    doc: { ...state.doc },
    active: state.active,
    selection: state.selection ? { ...state.selection } : null,
    layers: state.layers.map((l) => {
      const c = document.createElement('canvas');
      c.width = l.canvas.width; c.height = l.canvas.height;
      c.getContext('2d').drawImage(l.canvas, 0, 0);
      return { ...l, canvas: c, ctx: null };
    }),
  };
}

function restore(snap) {
  state.doc = { ...snap.doc };
  state.active = snap.active;
  state.selection = snap.selection ? { ...snap.selection } : null;
  state.layers = snap.layers.map((l) => {
    const layer = makeLayer(l.name, {
      id: l.id, x: l.x, y: l.y, visible: l.visible, opacity: l.opacity, blend: l.blend,
      edgeFeather: l.edgeFeather || 0,
      variations: l.variations || null, variationIndex: l.variationIndex || 0,
    });
    layer.canvas.width = l.canvas.width;
    layer.canvas.height = l.canvas.height;
    layer.ctx.drawImage(l.canvas, 0, 0);
    return layer;
  });
  resizeDocCanvas();
  refresh();
}

function pushHistory() {
  state.history.push(snapshot());
  if (state.history.length > HISTORY_LIMIT) state.history.shift();
  state.future.length = 0;
}

function undo() {
  if (!state.history.length) return;
  state.future.push(snapshot());
  restore(state.history.pop());
}

function redo() {
  if (!state.future.length) return;
  state.history.push(snapshot());
  restore(state.future.pop());
}

/* -------------------------------- rendering -------------------------------- */

function resizeDocCanvas() {
  docCanvas.width = state.doc.width;
  docCanvas.height = state.doc.height;
  overlay.width = state.doc.width;
  overlay.height = state.doc.height;
  stage.style.width = `${state.doc.width}px`;
  stage.style.height = `${state.doc.height}px`;
}

function adjustFilterString() {
  const a = state.adjust;
  const parts = [];
  if (a.brightness !== 100) parts.push(`brightness(${a.brightness}%)`);
  if (a.contrast !== 100) parts.push(`contrast(${a.contrast}%)`);
  if (a.saturate !== 100) parts.push(`saturate(${a.saturate}%)`);
  if (a.hue !== 0) parts.push(`hue-rotate(${a.hue}deg)`);
  if (a.blur !== 0) parts.push(`blur(${a.blur}px)`);
  return parts.join(' ');
}

function render() {
  docCtx.clearRect(0, 0, docCanvas.width, docCanvas.height);
  const previewFilter = adjustFilterString();
  state.layers.forEach((l, i) => {
    if (!l.visible) return;
    docCtx.save();
    docCtx.globalAlpha = l.opacity / 100;
    docCtx.globalCompositeOperation = l.blend;
    if (previewFilter && i === state.active) docCtx.filter = previewFilter;
    docCtx.drawImage(l.canvas, l.x, l.y);
    docCtx.restore();
  });
}

function applyStageTransform() {
  stage.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
  docCanvas.style.imageRendering = state.zoom >= 2 ? 'pixelated' : 'auto';
  $('#status-zoom').textContent = `${Math.round(state.zoom * 100)}%`;
}

function centerDocument() {
  const vw = viewport.clientWidth, vh = viewport.clientHeight;
  state.panX = (vw - state.doc.width * state.zoom) / 2;
  state.panY = (vh - state.doc.height * state.zoom) / 2;
  applyStageTransform();
}

function fitToView() {
  const vw = viewport.clientWidth - 60, vh = viewport.clientHeight - 60;
  state.zoom = Math.min(vw / state.doc.width, vh / state.doc.height, 1);
  centerDocument();
}

/* marching ants — animation loop runs only while a selection exists */
let antsPhase = 0;
let antsRunning = false;
function antsStroke(pathFn) {
  overlayCtx.lineWidth = 1 / state.zoom;
  overlayCtx.setLineDash([5 / state.zoom, 4 / state.zoom]);
  overlayCtx.strokeStyle = '#000';
  overlayCtx.lineDashOffset = -antsPhase;
  pathFn();
  overlayCtx.stroke();
  overlayCtx.strokeStyle = '#fff';
  overlayCtx.lineDashOffset = -antsPhase + 4.5 / state.zoom;
  pathFn();
  overlayCtx.stroke();
}

function drawOverlay() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  const sel = state.selection, pre = state.shapePreview;
  if (!sel && !pre) { antsRunning = false; return; }

  if (sel) {
    if (sel.mask) {
      overlayCtx.save();
      overlayCtx.globalAlpha = 0.18;
      overlayCtx.drawImage(sel.mask, 0, 0);
      overlayCtx.restore();
    }
    if (sel.contour) {
      overlayCtx.lineWidth = 1 / state.zoom;
      overlayCtx.setLineDash([5 / state.zoom, 4 / state.zoom]);
      overlayCtx.strokeStyle = '#000';
      overlayCtx.lineDashOffset = -antsPhase;
      overlayCtx.stroke(sel.contour);
      overlayCtx.strokeStyle = '#fff';
      overlayCtx.lineDashOffset = -antsPhase + 4.5 / state.zoom;
      overlayCtx.stroke(sel.contour);
    } else {
      antsStroke(() => {
        overlayCtx.beginPath();
        overlayCtx.rect(sel.x + 0.5, sel.y + 0.5, sel.w, sel.h);
      });
    }
  }

  if (pre) {
    antsStroke(() => {
      overlayCtx.beginPath();
      if (pre.type === 'rect') {
        overlayCtx.rect(pre.x + 0.5, pre.y + 0.5, pre.w, pre.h);
      } else if (pre.type === 'ellipse') {
        overlayCtx.ellipse(pre.x + pre.w / 2, pre.y + pre.h / 2, pre.w / 2, pre.h / 2, 0, 0, Math.PI * 2);
      } else if (pre.type === 'lasso' && pre.points.length > 1) {
        overlayCtx.moveTo(pre.points[0].x, pre.points[0].y);
        for (const p of pre.points) overlayCtx.lineTo(p.x, p.y);
      }
    });
  }

  antsPhase = (antsPhase + 0.15) % 9;
  requestAnimationFrame(drawOverlay);
}

function ensureAnts() {
  if (!antsRunning && (state.selection || state.shapePreview)) {
    antsRunning = true;
    requestAnimationFrame(drawOverlay);
  } else if (!state.selection && !state.shapePreview) {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  }
}

/* ------------------------------ selection system ----------------------------
   Every committed selection is a pixel mask (white-on-transparent canvas) plus
   a bounding box. Rect/ellipse/lasso rasterize their shape; AI tools return
   masks directly. Boolean combine: Shift = add, Alt = subtract.              */

function docSizedCanvas() {
  const c = document.createElement('canvas');
  c.width = state.doc.width; c.height = state.doc.height;
  return c;
}

function maskBBox(mask) {
  const w = mask.width, h = mask.height;
  const d = mask.getContext('2d').getImageData(0, 0, w, h).data;
  let minX = w, minY = h, maxX = -1, maxY = -1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      if (d[(row + x) * 4 + 3] > 8) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  return maxX < 0 ? null : { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
}

// Marching squares: trace the mask's alpha boundary into a Path2D (doc space)
// so the selection outline hugs the actual shape instead of its bounding box.
// Computed once per selection change and cached on the selection.
function computeContour(mask) {
  const w = mask.width, h = mask.height;
  const data = mask.getContext('2d').getImageData(0, 0, w, h).data;
  const step = Math.max(1, Math.round(Math.max(w, h) / 1400));
  const at = (x, y) =>
    (x < 0 || y < 0 || x >= w || y >= h) ? 0 : (data[(y * w + x) * 4 + 3] > 127 ? 1 : 0);
  const path = new Path2D();
  const seg = (a, c) => { path.moveTo(a[0], a[1]); path.lineTo(c[0], c[1]); };
  for (let y = 0; y < h; y += step) {
    for (let x = 0; x < w; x += step) {
      const code = (at(x, y) << 3) | (at(x + step, y) << 2) |
                   (at(x + step, y + step) << 1) | at(x, y + step);
      if (code === 0 || code === 15) continue;
      const t = [x + step / 2, y], r = [x + step, y + step / 2];
      const b = [x + step / 2, y + step], l = [x, y + step / 2];
      switch (code) {
        case 1: case 14: seg(l, b); break;
        case 2: case 13: seg(b, r); break;
        case 3: case 12: seg(l, r); break;
        case 4: case 11: seg(t, r); break;
        case 6: case 9:  seg(t, b); break;
        case 7: case 8:  seg(l, t); break;
        case 5: seg(l, t); seg(b, r); break;
        case 10: seg(t, r); seg(l, b); break;
      }
    }
  }
  return path;
}

function setMaskSelection(mask, base = mask) {
  const b = mask && maskBBox(mask);
  state.selection = b ? { ...b, mask, base, contour: computeContour(mask) } : null;
  hideExpandTip();   // a stale recommendation shouldn't outlive the selection it was for
  ensureAnts();
}

function currentMaskCanvas() {
  // existing selection as a mask canvas (builds one for legacy rect selections)
  if (!state.selection) return null;
  if (state.selection.mask) return state.selection.mask;
  const c = docSizedCanvas();
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  const { x, y, w, h } = state.selection;
  ctx.fillRect(x, y, w, h);
  return c;
}

function combineSelection(shapeMask, mode) {
  if (mode === 'replace' || !state.selection) {
    if (mode === 'subtract' && !state.selection) return;   // nothing to cut from
    setMaskSelection(shapeMask);
    resetFeatherSlider();
    return;
  }
  const base = docSizedCanvas();
  const ctx = base.getContext('2d');
  ctx.drawImage(currentMaskCanvas(), 0, 0);
  if (mode === 'subtract') ctx.globalCompositeOperation = 'destination-out';
  ctx.drawImage(shapeMask, 0, 0);
  setMaskSelection(base);
  resetFeatherSlider();
}

function shapeMask(draw) {
  const c = docSizedCanvas();
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  draw(ctx);
  return c;
}

function selectAll() {
  setMaskSelection(shapeMask((ctx) => ctx.fillRect(0, 0, state.doc.width, state.doc.height)));
}

function deselect() {
  state.selection = null;
  state.shapePreview = null;
  ensureAnts();
}

function invertSelection() {
  const prev = currentMaskCanvas();
  const c = docSizedCanvas();
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, c.width, c.height);
  if (prev) {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.drawImage(prev, 0, 0);
  }
  setMaskSelection(c);
}

// Grow a hard mask strictly OUTWARD by ~px (dilation). Blur spreads the edge
// both ways; thresholding at a low alpha keeps everything from the original
// interior out to ~px beyond the edge — so the object is never eroded inward.
function dilateMask(src, px) {
  if (!px) return src;
  const c = docSizedCanvas();
  const ctx = c.getContext('2d');
  ctx.filter = `blur(${px}px)`;
  ctx.drawImage(src, 0, 0);
  ctx.filter = 'none';
  const d = ctx.getImageData(0, 0, c.width, c.height);
  const a = d.data;
  for (let i = 3; i < a.length; i += 4) {
    if (a[i] > 38) { a[i] = 255; a[i - 1] = 255; a[i - 2] = 255; a[i - 3] = 255; }
    else { a[i] = 0; }
  }
  ctx.putImageData(d, 0, 0);
  return c;
}

// Feather OUTWARD only: soften the edge outside the mask while keeping the
// interior fully selected. Blur both ways, then stamp the hard mask back on
// top so the inward half of the ramp is restored to solid — no creep inward.
function featherOutward(src, px) {
  if (!px) return src;
  const c = docSizedCanvas();
  const ctx = c.getContext('2d');
  ctx.filter = `blur(${px}px)`;
  ctx.drawImage(src, 0, 0);
  ctx.filter = 'none';
  ctx.drawImage(src, 0, 0);
  return c;
}

// Rebuild the selection from its original base, expanded then feathered — both
// strictly outward, so the intended object is never eaten into.
function applySelectionEdge() {
  const base = state.selection?.base;
  if (!base) return;
  const expand = +($('#sel-expand')?.value || 0);
  const feather = +($('#sel-feather')?.value || 0);
  let m = dilateMask(base, expand);
  m = featherOutward(m, feather);
  setMaskSelection(m, base);
}

function resetFeatherSlider() {
  for (const id of ['#sel-feather', '#sel-expand']) {
    const s = $(id);
    if (s) { s.value = 0; $(`${id}-val`).textContent = '0px'; }
  }
}

/* ----------------------------------- UI ------------------------------------ */

const TOOLS = {
  move:    { key: 'v', label: 'Move',
    icon: '<path d="M9 2v14M2 9h14M9 2l-2.5 2.5M9 2l2.5 2.5M9 16l-2.5-2.5M9 16l2.5-2.5M2 9l2.5-2.5M2 9l2.5 2.5M16 9l-2.5-2.5M16 9l-2.5 2.5"/>' },
  marquee: { key: 'm', label: 'Rectangular Marquee (M toggles shape)',
    icon: '<rect x="2.5" y="3.5" width="13" height="11" stroke-dasharray="3 2"/>' },
  ellipse: { key: 'm', label: 'Elliptical Marquee (M toggles shape)',
    icon: '<ellipse cx="9" cy="9" rx="6.5" ry="5.5" stroke-dasharray="3 2"/>' },
  lasso:   { key: 'l', label: 'Lasso',
    icon: '<path d="M9 3c3.6 0 6.5 1.8 6.5 4.2S12.6 11.5 9 11.5 2.5 9.6 2.5 7.2 5.4 3 9 3z" stroke-dasharray="3 2"/><path d="M5.5 11.5c-.6 1.2-.5 2.5-2 3.5 1.8.4 3-.3 3.6-1.6"/>' },
  objselect: { key: 'w', label: 'Object Select — click an object, AI masks it',
    icon: '<path d="M11 7l4.5 4.5-2 2L9 9z"/><path d="M4 2.5v3M2.5 4h3M13 2l.6 1.6L15 4.2l-1.4.6L13 6.5l-.6-1.7L11 4.2l1.4-.6zM4.5 12l.5 1.3 1.3.5-1.3.5-.5 1.2-.5-1.2-1.2-.5 1.2-.5z"/>' },
  brush:   { key: 'b', label: 'Brush',
    icon: '<path d="M14.5 2.5l1 1c.6.6.6 1.5 0 2.1L9 12l-3 1 1-3 6.4-6.5c.6-.6 1.5-.6 2.1 0zM5.5 13.5c-1 1-3 2-3.5 1.5S5 12.5 4.5 12"/>' },
  eraser:  { key: 'e', label: 'Eraser',
    icon: '<path d="M7 15h8M3.4 11.6l7.2-7.2c.6-.6 1.5-.6 2.1 0l2.9 2.9c.6.6.6 1.5 0 2.1L10.4 15H7.6l-4.2-4.2a1 1 0 010-1.4z"/><path d="M7.5 7.5l4 4"/>' },
  fill:    { key: 'g', label: 'Fill (paint bucket)',
    icon: '<path d="M8.5 2.5l5 5-5.4 5.4c-.6.6-1.6.6-2.2 0l-2.8-2.8c-.6-.6-.6-1.6 0-2.2L8.5 2.5zM3 8.4h10M15.5 11.5s1 1.6 1 2.4a1 1 0 11-2 0c0-.8 1-2.4 1-2.4z"/>' },
  picker:  { key: 'i', label: 'Eyedropper',
    icon: '<path d="M11.5 6.5l-7 7-1 2.5 2.5-1 7-7M10 5l3-3 3 3-3 3M10 5l3 3"/>' },
  zoom:    { key: 'z', label: 'Zoom',
    icon: '<circle cx="8" cy="8" r="5.5"/><path d="M12 12l4 4M6 8h4M8 6v4"/>' },
};

function buildToolbar() {
  const bar = $('#toolbar');
  for (const [name, t] of Object.entries(TOOLS)) {
    const btn = document.createElement('button');
    btn.className = 'tool-btn';
    btn.dataset.tool = name;
    btn.title = `${t.label} (${t.key.toUpperCase()})`;
    btn.innerHTML = `<svg viewBox="0 0 18 18">${t.icon}</svg>`;
    btn.addEventListener('click', () => setTool(name));
    bar.appendChild(btn);
  }
}

function setTool(name) {
  state.tool = name;
  document.querySelectorAll('.tool-btn').forEach((b) =>
    b.classList.toggle('active', b.dataset.tool === name));
  $('#status-tool').textContent = TOOLS[name].label;
  buildOptionsBar();
  viewport.style.cursor = { move: 'move', zoom: 'zoom-in', picker: 'crosshair', marquee: 'crosshair' }[name] || 'crosshair';
}

function buildOptionsBar() {
  const bar = $('#optionsbar');
  bar.innerHTML = '';
  const add = (html) => { bar.insertAdjacentHTML('beforeend', html); };

  add(`<div class="opt"><span>Color</span><input type="color" id="opt-color" value="${state.color}"></div>`);

  if (state.tool === 'brush' || state.tool === 'eraser') {
    add(`<div class="opt"><span>Size</span><input type="range" id="opt-size" min="1" max="200" value="${state.brush.size}"><span class="opt-val" id="opt-size-val">${state.brush.size}px</span></div>`);
    add(`<div class="opt"><span>Opacity</span><input type="range" id="opt-opacity" min="1" max="100" value="${state.brush.opacity}"><span class="opt-val" id="opt-opacity-val">${state.brush.opacity}%</span></div>`);
    $('#opt-size').addEventListener('input', (e) => {
      state.brush.size = +e.target.value;
      $('#opt-size-val').textContent = `${state.brush.size}px`;
    });
    $('#opt-opacity').addEventListener('input', (e) => {
      state.brush.opacity = +e.target.value;
      $('#opt-opacity-val').textContent = `${state.brush.opacity}%`;
    });
  }

  if (state.tool === 'fill') {
    add(`<div class="opt"><span>Tolerance</span><input type="range" id="opt-tol" min="0" max="128" value="${state.fill.tolerance}"><span class="opt-val" id="opt-tol-val">${state.fill.tolerance}</span></div>`);
    $('#opt-tol').addEventListener('input', (e) => {
      state.fill.tolerance = +e.target.value;
      $('#opt-tol-val').textContent = state.fill.tolerance;
    });
  }

  if (state.tool === 'zoom') {
    add('<div class="opt"><span>Click to zoom in · Alt-click to zoom out · Ctrl+0 fit</span></div>');
  }
  if (state.tool === 'marquee' || state.tool === 'ellipse' || state.tool === 'lasso') {
    add('<div class="opt"><span>Drag to select · Shift adds · Alt subtracts · M toggles rect/ellipse · Esc clears</span></div>');
  }
  if (state.tool === 'objselect') {
    add('<div class="opt"><span>Click an object — AI masks it · Shift adds · Alt subtracts</span></div>');
  }
  if (state.tool === 'move') {
    add('<div class="opt"><span>Drag to move the active layer</span></div>');
  }

  $('#opt-color').addEventListener('input', (e) => { state.color = e.target.value; });
}

/* menus */
const MENUS = {
  File: [
    ['New…', 'Ctrl+N', () => newDocDialog()],
    ['Open…', 'Ctrl+O', () => openFile('open')],
    ['Place as Layer…', '', () => openFile('place')],
    null,
    ['Export PNG', 'Ctrl+S', exportPNG],
  ],
  Edit: [
    ['Undo', 'Ctrl+Z', undo],
    ['Redo', 'Ctrl+Y', redo],
    null,
    ['Cut', 'Ctrl+X', () => copySelection(true)],
    ['Copy', 'Ctrl+C', () => copySelection(false)],
    ['Paste', 'Ctrl+V', pasteClipboard],
    ['Layer via Copy', 'Ctrl+J', layerViaCopy],
  ],
  Select: [
    ['Select All', 'Ctrl+A', selectAll],
    ['Deselect', 'Ctrl+D', deselect],
    ['Inverse', 'Ctrl+Shift+I', invertSelection],
  ],
  Image: [
    ['Crop to Selection', '', cropToSelection],
    ['Expand Canvas…', '', expandCanvasDialog],
    ['Flatten Image', '', flattenImage],
  ],
  Filter: [
    ['Invert', '', () => bakeFilter('invert(1)')],
    ['Grayscale', '', () => bakeFilter('grayscale(1)')],
    ['Sepia', '', () => bakeFilter('sepia(1)')],
    null,
    ['Blur (5px)', '', () => bakeFilter('blur(5px)')],
    ['Sharpen (contrast)', '', () => bakeFilter('contrast(130%) saturate(110%)')],
  ],
  View: [
    ['Zoom In', 'Ctrl + +', () => zoomBy(1.25)],
    ['Zoom Out', 'Ctrl + -', () => zoomBy(0.8)],
    ['Fit on Screen', 'Ctrl+0', fitToView],
    ['100%', 'Ctrl+1', () => { state.zoom = 1; centerDocument(); }],
  ],
};

function buildMenus() {
  const nav = $('#menus');
  for (const [title, items] of Object.entries(MENUS)) {
    const item = document.createElement('div');
    item.className = 'menu-item';
    const btn = document.createElement('button');
    btn.textContent = title;
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const wasOpen = item.classList.contains('open');
      closeMenus();
      if (!wasOpen) item.classList.add('open');
    });
    const drop = document.createElement('div');
    drop.className = 'menu-drop';
    for (const entry of items) {
      if (!entry) { drop.insertAdjacentHTML('beforeend', '<hr>'); continue; }
      const [label, shortcut, fn] = entry;
      const b = document.createElement('button');
      b.innerHTML = `<span>${label}</span><span class="shortcut">${shortcut}</span>`;
      b.addEventListener('click', () => { closeMenus(); fn(); refresh(); });
      drop.appendChild(b);
    }
    item.append(btn, drop);
    nav.appendChild(item);
  }
  document.addEventListener('click', closeMenus);
}

function closeMenus() {
  document.querySelectorAll('.menu-item.open').forEach((m) => m.classList.remove('open'));
}

/* adjustments panel */
const ADJUSTMENTS = [
  ['brightness', 'Brightness', 0, 200, 100, '%'],
  ['contrast', 'Contrast', 0, 200, 100, '%'],
  ['saturate', 'Saturation', 0, 200, 100, '%'],
  ['hue', 'Hue', -180, 180, 0, '°'],
  ['blur', 'Blur', 0, 20, 0, 'px'],
];

function buildAdjustPanel() {
  const body = $('#adjust-body');
  for (const [key, label, min, max, def, unit] of ADJUSTMENTS) {
    const row = document.createElement('div');
    row.className = 'adj-row';
    row.innerHTML = `<label>${label}</label>
      <input type="range" id="adj-${key}" min="${min}" max="${max}" value="${def}">
      <span class="adj-val" id="adj-${key}-val">${def}${unit}</span>`;
    body.appendChild(row);
    row.querySelector('input').addEventListener('input', (e) => {
      state.adjust[key] = +e.target.value;
      $(`#adj-${key}-val`).textContent = `${e.target.value}${unit}`;
      render();
    });
  }
  const actions = document.createElement('div');
  actions.className = 'adj-actions';
  actions.innerHTML = '<button id="adj-reset">Reset</button><button id="adj-apply" class="primary">Apply</button>';
  body.appendChild(actions);
  $('#adj-reset').addEventListener('click', () => { resetAdjustments(); render(); });
  $('#adj-apply').addEventListener('click', applyAdjustments);
}

function resetAdjustments() {
  for (const [key, , , , def, unit] of ADJUSTMENTS) {
    state.adjust[key] = def;
    $(`#adj-${key}`).value = def;
    $(`#adj-${key}-val`).textContent = `${def}${unit}`;
  }
}

function applyAdjustments() {
  const filter = adjustFilterString();
  if (!filter) return;
  bakeFilter(filter);
  resetAdjustments();
  refresh();
}

function bakeFilter(filter) {
  const l = activeLayer();
  if (!l) return;
  pushHistory();
  const tmp = document.createElement('canvas');
  tmp.width = l.canvas.width; tmp.height = l.canvas.height;
  const tctx = tmp.getContext('2d');
  tctx.filter = filter;
  tctx.drawImage(l.canvas, 0, 0);
  l.ctx.clearRect(0, 0, l.canvas.width, l.canvas.height);
  l.ctx.drawImage(tmp, 0, 0);
  refresh();
}

/* layers panel */
function buildLayersPanel() {
  const blend = $('#blend-select');
  for (const [label, value] of BLEND_MODES) {
    const opt = document.createElement('option');
    opt.value = value; opt.textContent = label;
    blend.appendChild(opt);
  }
  blend.addEventListener('change', () => {
    const l = activeLayer();
    if (l) { l.blend = blend.value; refresh(); }
  });
  $('#layer-opacity').addEventListener('input', (e) => {
    const l = activeLayer();
    if (l) {
      l.opacity = +e.target.value;
      $('#layer-opacity-val').textContent = `${l.opacity}%`;
      render();
    }
  });
  const edge = $('#layer-edge');
  edge.addEventListener('pointerdown', () => { if (activeLayer()) pushHistory(); });
  edge.addEventListener('input', (e) => {
    const l = activeLayer();
    if (!l) return;
    const px = +e.target.value;
    $('#layer-edge-val').textContent = `${px}px`;
    featherLayerEdge(l, px);
  });
  $('#layer-add').addEventListener('click', () => addLayer());
  $('#layer-dup').addEventListener('click', duplicateLayer);
  $('#layer-del').addEventListener('click', deleteLayer);
  $('#layer-up').addEventListener('click', () => moveLayer(1));
  $('#layer-down').addEventListener('click', () => moveLayer(-1));
}

function renderLayerList() {
  const list = $('#layer-list');
  list.innerHTML = '';
  // top layer first, Photoshop style
  for (let i = state.layers.length - 1; i >= 0; i--) {
    const l = state.layers[i];
    const row = document.createElement('div');
    row.className = `layer-row${i === state.active ? ' active' : ''}`;

    const eye = document.createElement('button');
    eye.className = `layer-eye${l.visible ? '' : ' hidden-layer'}`;
    eye.textContent = '👁';
    eye.title = 'Toggle visibility';
    eye.addEventListener('click', (e) => {
      e.stopPropagation();
      l.visible = !l.visible;
      refresh();
    });

    const thumb = document.createElement('canvas');
    thumb.className = 'layer-thumb';
    thumb.width = 40; thumb.height = 28;
    const scale = Math.min(40 / l.canvas.width, 28 / l.canvas.height);
    thumb.getContext('2d').drawImage(
      l.canvas, (40 - l.canvas.width * scale) / 2, (28 - l.canvas.height * scale) / 2,
      l.canvas.width * scale, l.canvas.height * scale);

    const name = document.createElement('span');
    name.className = 'layer-name';
    name.textContent = l.name;
    name.title = 'Double-click to rename';
    name.addEventListener('dblclick', () => {
      const input = document.createElement('input');
      input.value = l.name;
      name.replaceChildren(input);
      input.focus(); input.select();
      const commit = () => { l.name = input.value.trim() || l.name; renderLayerList(); };
      input.addEventListener('blur', commit);
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') commit();
        e.stopPropagation();
      });
    });

    row.append(eye, thumb, name);
    row.addEventListener('click', () => { state.active = i; refresh(); });
    list.appendChild(row);
  }
  const l = activeLayer();
  if (l) {
    $('#blend-select').value = l.blend;
    $('#layer-opacity').value = l.opacity;
    $('#layer-opacity-val').textContent = `${l.opacity}%`;
    $('#layer-edge').value = l.edgeFeather;
    $('#layer-edge-val').textContent = `${l.edgeFeather}px`;
    const vary = $('#layer-variations');
    if (l.variations) {
      vary.hidden = false;
      $('#vary-counter').textContent = `${l.variationIndex + 1} / ${l.variations.length}`;
    } else {
      vary.hidden = true;
    }
  }
}

function refresh() {
  render();
  renderLayerList();
  ensureAnts();
  $('#status-size').textContent = `${state.doc.width} × ${state.doc.height}px`;
  $('#titlebar-doc').textContent = `${state.doc.name} @ ${Math.round(state.zoom * 100)}%`;
}

/* ------------------------------ pointer / tools ----------------------------- */

function docCoords(e) {
  const r = stage.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) / state.zoom,
    y: (e.clientY - r.top) / state.zoom,
  };
}

let pointer = null;   // active drag info
let spaceDown = false;

function capturePointer(e) {
  try { viewport.setPointerCapture(e.pointerId); } catch { /* synthetic or already-released pointer */ }
}

viewport.addEventListener('pointerdown', (e) => {
  if (e.button === 1 || spaceDown) {
    pointer = { mode: 'pan', sx: e.clientX, sy: e.clientY, px: state.panX, py: state.panY };
    capturePointer(e);
    return;
  }
  if (e.button !== 0) return;
  const p = docCoords(e);
  const l = activeLayer();
  capturePointer(e);

  switch (state.tool) {
    case 'brush':
    case 'eraser':
      if (!l) return;
      pushHistory();
      pointer = { mode: state.tool, last: p };
      drawStroke(l, p, p);
      break;
    case 'move':
      if (!l) return;
      pushHistory();
      pointer = { mode: 'move', sx: p.x, sy: p.y, lx: l.x, ly: l.y };
      break;
    case 'marquee':
    case 'ellipse':
    case 'lasso':
      pointer = {
        mode: 'shape',
        shape: state.tool === 'marquee' ? 'rect' : state.tool,
        sx: p.x, sy: p.y,
        points: [p],
        combine: e.shiftKey ? 'add' : e.altKey ? 'subtract' : 'replace',
        moved: false,
      };
      break;
    case 'objselect':
      selectObjectAt(p, e.shiftKey ? 'add' : e.altKey ? 'subtract' : 'replace');
      break;
    case 'picker':
      pickColor(p);
      break;
    case 'fill':
      if (!l) return;
      pushHistory();
      floodFill(l, p);
      refresh();
      break;
    case 'zoom':
      zoomBy(e.altKey ? 0.8 : 1.25, e.clientX, e.clientY);
      break;
  }
});

viewport.addEventListener('pointermove', (e) => {
  const p = docCoords(e);
  $('#status-pos').textContent = `${Math.round(p.x)}, ${Math.round(p.y)}`;
  if (!pointer) return;

  switch (pointer.mode) {
    case 'pan':
      state.panX = pointer.px + e.clientX - pointer.sx;
      state.panY = pointer.py + e.clientY - pointer.sy;
      applyStageTransform();
      break;
    case 'brush':
    case 'eraser':
      drawStroke(activeLayer(), pointer.last, p);
      pointer.last = p;
      break;
    case 'move': {
      const l = activeLayer();
      l.x = Math.round(pointer.lx + p.x - pointer.sx);
      l.y = Math.round(pointer.ly + p.y - pointer.sy);
      render();
      break;
    }
    case 'shape': {
      pointer.moved = true;
      if (pointer.shape === 'lasso') {
        pointer.points.push(p);
        state.shapePreview = { type: 'lasso', points: pointer.points };
      } else {
        state.shapePreview = {
          type: pointer.shape,
          x: Math.round(Math.min(pointer.sx, p.x)),
          y: Math.round(Math.min(pointer.sy, p.y)),
          w: Math.round(Math.abs(p.x - pointer.sx)),
          h: Math.round(Math.abs(p.y - pointer.sy)),
        };
      }
      ensureAnts();
      break;
    }
  }
});

viewport.addEventListener('pointerup', () => {
  if (pointer?.mode === 'shape') {
    const pre = state.shapePreview;
    state.shapePreview = null;
    if (!pointer.moved) {
      if (pointer.combine === 'replace') deselect();   // plain click clears
    } else if (pointer.shape === 'lasso') {
      const pts = pointer.points;
      if (pts.length > 2) {
        combineSelection(shapeMask((ctx) => {
          ctx.beginPath();
          ctx.moveTo(pts[0].x, pts[0].y);
          for (const q of pts) ctx.lineTo(q.x, q.y);
          ctx.closePath();
          ctx.fill();
        }), pointer.combine);
      }
    } else if (pre && pre.w > 1 && pre.h > 1) {
      combineSelection(shapeMask((ctx) => {
        if (pointer.shape === 'ellipse') {
          ctx.beginPath();
          ctx.ellipse(pre.x + pre.w / 2, pre.y + pre.h / 2, pre.w / 2, pre.h / 2, 0, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.fillRect(pre.x, pre.y, pre.w, pre.h);
        }
      }), pointer.combine);
    }
    ensureAnts();
  }
  if (pointer?.mode === 'brush' || pointer?.mode === 'eraser') renderLayerList();
  if (pointer?.mode === 'move') renderLayerList();
  pointer = null;
});

let scratchCanvas = null;
function getScratch() {
  if (!scratchCanvas) scratchCanvas = document.createElement('canvas');
  if (scratchCanvas.width !== state.doc.width || scratchCanvas.height !== state.doc.height) {
    scratchCanvas.width = state.doc.width;
    scratchCanvas.height = state.doc.height;
  }
  return scratchCanvas;
}

function drawStroke(layer, from, to) {
  const mask = state.selection?.mask;
  if (mask) {
    // stroke in doc space, clipped through the AI mask, then blitted to the layer
    const s = getScratch();
    const sctx = s.getContext('2d');
    sctx.clearRect(0, 0, s.width, s.height);
    sctx.strokeStyle = state.color;
    sctx.lineWidth = state.brush.size;
    sctx.lineCap = 'round';
    sctx.lineJoin = 'round';
    sctx.beginPath();
    sctx.moveTo(from.x, from.y);
    sctx.lineTo(to.x + 0.01, to.y + 0.01);
    sctx.stroke();
    sctx.globalCompositeOperation = 'destination-in';
    sctx.drawImage(mask, 0, 0);
    sctx.globalCompositeOperation = 'source-over';
    const ctx = layer.ctx;
    ctx.save();
    ctx.globalAlpha = state.brush.opacity / 100;
    ctx.globalCompositeOperation = state.tool === 'eraser' ? 'destination-out' : 'source-over';
    ctx.drawImage(s, -layer.x, -layer.y);
    ctx.restore();
    render();
    return;
  }
  const ctx = layer.ctx;
  ctx.save();
  if (state.selection) {
    const { x, y, w, h } = state.selection;
    ctx.beginPath();
    ctx.rect(x - layer.x, y - layer.y, w, h);
    ctx.clip();
  }
  ctx.globalAlpha = state.brush.opacity / 100;
  ctx.globalCompositeOperation = state.tool === 'eraser' ? 'destination-out' : 'source-over';
  ctx.strokeStyle = state.color;
  ctx.lineWidth = state.brush.size;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(from.x - layer.x, from.y - layer.y);
  ctx.lineTo(to.x - layer.x + 0.01, to.y - layer.y + 0.01);
  ctx.stroke();
  ctx.restore();
  render();
}

function pickColor(p) {
  const x = Math.floor(p.x), y = Math.floor(p.y);
  if (x < 0 || y < 0 || x >= docCanvas.width || y >= docCanvas.height) return;
  const [r, g, b, a] = docCtx.getImageData(x, y, 1, 1).data;
  if (a === 0) return;
  state.color = '#' + [r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('');
  const colorInput = $('#opt-color');
  if (colorInput) colorInput.value = state.color;
}

function floodFill(layer, p) {
  const lx = Math.floor(p.x - layer.x), ly = Math.floor(p.y - layer.y);
  const { width: w, height: h } = layer.canvas;
  if (lx < 0 || ly < 0 || lx >= w || ly >= h) return;

  const img = layer.ctx.getImageData(0, 0, w, h);
  const data = new Uint32Array(img.data.buffer);
  const bytes = img.data;

  const target = data[ly * w + lx];
  const tr = bytes[(ly * w + lx) * 4], tg = bytes[(ly * w + lx) * 4 + 1],
        tb = bytes[(ly * w + lx) * 4 + 2], ta = bytes[(ly * w + lx) * 4 + 3];

  const hex = state.color;
  const fr = parseInt(hex.slice(1, 3), 16), fg = parseInt(hex.slice(3, 5), 16), fb = parseInt(hex.slice(5, 7), 16);
  const fillVal = (255 << 24) | (fb << 16) | (fg << 8) | fr;
  if (target === fillVal) return;

  const tol = state.fill.tolerance;
  // selection bounds in layer space
  let bx0 = 0, by0 = 0, bx1 = w - 1, by1 = h - 1;
  if (state.selection) {
    bx0 = Math.max(bx0, Math.floor(state.selection.x - layer.x));
    by0 = Math.max(by0, Math.floor(state.selection.y - layer.y));
    bx1 = Math.min(bx1, Math.ceil(state.selection.x + state.selection.w - layer.x) - 1);
    by1 = Math.min(by1, Math.ceil(state.selection.y + state.selection.h - layer.y) - 1);
    if (lx < bx0 || lx > bx1 || ly < by0 || ly > by1) return;
  }

  // per-pixel selection-mask test (doc space → layer space offset)
  let selAlpha = null;
  if (state.selection?.mask) {
    selAlpha = state.selection.mask.getContext('2d')
      .getImageData(0, 0, state.doc.width, state.doc.height).data;
  }
  const docW = state.doc.width, docH = state.doc.height;
  const inSel = (x, y) => {
    if (!selAlpha) return true;
    const dx = x + layer.x, dy = y + layer.y;
    if (dx < 0 || dy < 0 || dx >= docW || dy >= docH) return false;
    return selAlpha[(dy * docW + dx) * 4 + 3] > 127;
  };
  if (!inSel(lx, ly)) return;

  const matches = (i) => {
    const o = i * 4;
    return Math.abs(bytes[o] - tr) <= tol && Math.abs(bytes[o + 1] - tg) <= tol &&
           Math.abs(bytes[o + 2] - tb) <= tol && Math.abs(bytes[o + 3] - ta) <= tol &&
           inSel(i % w, (i / w) | 0);
  };

  const visited = new Uint8Array(w * h);
  const stack = [ly * w + lx];
  visited[ly * w + lx] = 1;
  while (stack.length) {
    const i = stack.pop();
    data[i] = fillVal;
    const x = i % w, y = (i / w) | 0;
    if (x > bx0 && !visited[i - 1] && matches(i - 1)) { visited[i - 1] = 1; stack.push(i - 1); }
    if (x < bx1 && !visited[i + 1] && matches(i + 1)) { visited[i + 1] = 1; stack.push(i + 1); }
    if (y > by0 && !visited[i - w] && matches(i - w)) { visited[i - w] = 1; stack.push(i - w); }
    if (y < by1 && !visited[i + w] && matches(i + w)) { visited[i + w] = 1; stack.push(i + w); }
  }
  layer.ctx.putImageData(img, 0, 0);
}

/* ------------------------------- zoom & keys -------------------------------- */

function zoomBy(factor, cx, cy) {
  const rect = viewport.getBoundingClientRect();
  const px = (cx ?? rect.left + rect.width / 2) - rect.left;
  const py = (cy ?? rect.top + rect.height / 2) - rect.top;
  const newZoom = Math.min(32, Math.max(0.05, state.zoom * factor));
  // keep point under cursor fixed
  state.panX = px - (px - state.panX) * (newZoom / state.zoom);
  state.panY = py - (py - state.panY) * (newZoom / state.zoom);
  state.zoom = newZoom;
  applyStageTransform();
  $('#titlebar-doc').textContent = `${state.doc.name} @ ${Math.round(state.zoom * 100)}%`;
}

viewport.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (e.ctrlKey || state.tool === 'zoom') {
    zoomBy(e.deltaY < 0 ? 1.15 : 0.87, e.clientX, e.clientY);
  } else {
    state.panX -= e.deltaX;
    state.panY -= e.deltaY;
    applyStageTransform();
  }
}, { passive: false });

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

  // arrows cycle the active layer's AI variations, if it has any
  if (activeLayer()?.variations) {
    if (e.key === 'ArrowLeft') { e.preventDefault(); setLayerVariation(activeLayer(), activeLayer().variationIndex - 1); return; }
    if (e.key === 'ArrowRight') { e.preventDefault(); setLayerVariation(activeLayer(), activeLayer().variationIndex + 1); return; }
  }

  if (e.code === 'Space') { spaceDown = true; viewport.style.cursor = 'grab'; e.preventDefault(); return; }

  if (e.ctrlKey || e.metaKey) {
    const k = e.key.toLowerCase();
    if (k === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); }
    else if (k === 'y') { e.preventDefault(); redo(); }
    else if (k === 'a') { e.preventDefault(); selectAll(); }
    else if (k === 'd') { e.preventDefault(); deselect(); }
    else if (k === 'i' && e.shiftKey) { e.preventDefault(); invertSelection(); }
    else if (k === 'c') { e.preventDefault(); copySelection(false); }
    else if (k === 'x') { e.preventDefault(); copySelection(true); }
    else if (k === 'v') { e.preventDefault(); pasteClipboard(); }
    else if (k === 'j') { e.preventDefault(); layerViaCopy(); }
    else if (k === 's') { e.preventDefault(); exportPNG(); }
    else if (k === 'n') { e.preventDefault(); newDocDialog(); }
    else if (k === 'o') { e.preventDefault(); openFile('open'); }
    else if (k === '0') { e.preventDefault(); fitToView(); }
    else if (k === '1') { e.preventDefault(); state.zoom = 1; centerDocument(); }
    else if (k === '=' || k === '+') { e.preventDefault(); zoomBy(1.25); }
    else if (k === '-') { e.preventDefault(); zoomBy(0.8); }
    return;
  }

  if (e.key === 'Escape') { deselect(); return; }
  if (e.key === '[') { state.brush.size = Math.max(1, state.brush.size - 4); buildOptionsBar(); return; }
  if (e.key === ']') { state.brush.size = Math.min(200, state.brush.size + 4); buildOptionsBar(); return; }
  if (e.key === 'Delete' && state.selection) {
    const l = activeLayer();
    if (!l) return;
    pushHistory();
    if (state.selection.mask) {
      l.ctx.save();
      l.ctx.globalCompositeOperation = 'destination-out';
      l.ctx.drawImage(state.selection.mask, -l.x, -l.y);
      l.ctx.restore();
    } else {
      const { x, y, w, h } = state.selection;
      l.ctx.clearRect(x - l.x, y - l.y, w, h);
    }
    refresh();
    return;
  }

  if (e.key.toLowerCase() === 'm') {
    setTool(state.tool === 'marquee' ? 'ellipse' : 'marquee');   // M cycles shapes
    return;
  }
  for (const [name, t] of Object.entries(TOOLS)) {
    if (e.key.toLowerCase() === t.key) { setTool(name); return; }
  }
});

document.addEventListener('keyup', (e) => {
  if (e.code === 'Space') { spaceDown = false; setTool(state.tool); }
});

/* ------------------------------ file handling ------------------------------- */

let fileMode = 'open';

function openFile(mode) {
  fileMode = mode;
  $('#file-input').value = '';
  $('#file-input').click();
}

$('#file-input').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    URL.revokeObjectURL(url);
    if (fileMode === 'open') {
      loadAsDocument(img, file.name.replace(/\.[^.]+$/, ''));
    } else {
      placeAsLayer(img, file.name.replace(/\.[^.]+$/, ''));
    }
  };
  img.src = url;
});

/* drag & drop places as a layer */
viewport.addEventListener('dragover', (e) => e.preventDefault());
viewport.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = [...e.dataTransfer.files].find((f) => f.type.startsWith('image/'));
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    URL.revokeObjectURL(url);
    placeAsLayer(img, file.name.replace(/\.[^.]+$/, ''));
  };
  img.src = url;
});

function loadAsDocument(img, name) {
  pushHistory();
  state.doc = { width: img.naturalWidth, height: img.naturalHeight, name };
  state.selection = null;
  resetAdjustments();
  const layer = makeLayer(name);
  layer.ctx.drawImage(img, 0, 0);
  state.layers = [layer];
  state.active = 0;
  resizeDocCanvas();
  fitToView();
  refresh();
}

function placeAsLayer(img, name) {
  pushHistory();
  const layer = makeLayer(name);
  const scale = Math.min(1, state.doc.width / img.naturalWidth, state.doc.height / img.naturalHeight);
  const w = Math.round(img.naturalWidth * scale), h = Math.round(img.naturalHeight * scale);
  layer.ctx.drawImage(img, (state.doc.width - w) / 2, (state.doc.height - h) / 2, w, h);
  state.layers.splice(state.active + 1, 0, layer);
  state.active += 1;
  refresh();
}

function exportPNG() {
  render();
  const a = document.createElement('a');
  a.download = `${state.doc.name}.png`;
  a.href = docCanvas.toDataURL('image/png');
  a.click();
}

function cropToSelection() {
  if (!state.selection) return;
  pushHistory();
  const { x, y, w, h } = state.selection;
  for (const l of state.layers) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    c.getContext('2d').drawImage(l.canvas, l.x - x, l.y - y);
    l.canvas.width = w; l.canvas.height = h;
    l.ctx.drawImage(c, 0, 0);
    l.x = 0; l.y = 0;
  }
  state.doc.width = w;
  state.doc.height = h;
  state.selection = null;
  resizeDocCanvas();
  fitToView();
  refresh();
}

function flattenImage() {
  pushHistory();
  render();
  const flat = makeLayer('Background');
  flat.ctx.drawImage(docCanvas, 0, 0);
  state.layers = [flat];
  state.active = 0;
  refresh();
}

/* --------------------------------- new doc ---------------------------------- */

function newDocDialog() {
  const root = $('#modal-root');
  root.hidden = false;
  root.innerHTML = `
    <div class="modal">
      <h2>New Document</h2>
      <div class="field"><label>Name</label><input id="nd-name" value="Untitled"></div>
      <div class="field"><label>Width</label><input id="nd-w" type="number" value="1280" min="1" max="8192"></div>
      <div class="field"><label>Height</label><input id="nd-h" type="number" value="800" min="1" max="8192"></div>
      <div class="field"><label>Background</label>
        <select id="nd-bg"><option value="white">White</option><option value="transparent">Transparent</option><option value="black">Black</option></select>
      </div>
      <div class="modal-actions">
        <button id="nd-cancel">Cancel</button>
        <button id="nd-create" class="primary">Create</button>
      </div>
    </div>`;
  $('#nd-cancel').addEventListener('click', () => { root.hidden = true; });
  $('#nd-create').addEventListener('click', () => {
    const w = Math.max(1, Math.min(8192, +$('#nd-w').value || 1280));
    const h = Math.max(1, Math.min(8192, +$('#nd-h').value || 800));
    createDocument($('#nd-name').value.trim() || 'Untitled', w, h, $('#nd-bg').value);
    root.hidden = true;
  });
}

function createDocument(name, w, h, bg) {
  state.doc = { width: w, height: h, name };
  state.selection = null;
  state.history.length = 0;
  state.future.length = 0;
  resetAdjustments();
  const layer = makeLayer('Background');
  if (bg !== 'transparent') {
    layer.ctx.fillStyle = bg;
    layer.ctx.fillRect(0, 0, w, h);
  }
  state.layers = [layer];
  state.active = 0;
  resizeDocCanvas();
  fitToView();
  refresh();
}

/* ------------------------------ clipboard / lift ---------------------------- */

let clipboard = null;   // { canvas, x, y } — masked pixels in doc space

// pull the active layer's pixels within a mask into a doc-space canvas
function maskedExtract(layer, maskCanvas) {
  const tmp = docSizedCanvas();
  const ctx = tmp.getContext('2d');
  ctx.drawImage(layer.canvas, layer.x, layer.y);
  ctx.globalCompositeOperation = 'destination-in';
  ctx.drawImage(maskCanvas, 0, 0);   // soft alpha preserved → feathered edges survive
  return tmp;
}

function copySelection(cut) {
  const l = activeLayer();
  if (!l) return;
  const mask = currentMaskCanvas();
  let src, bx, by, bw, bh;
  if (mask) {
    src = maskedExtract(l, mask);
    ({ x: bx, y: by, w: bw, h: bh } = state.selection);
  } else {
    src = docSizedCanvas();
    src.getContext('2d').drawImage(l.canvas, l.x, l.y);
    bx = 0; by = 0; bw = state.doc.width; bh = state.doc.height;
  }
  const clip = document.createElement('canvas');
  clip.width = bw; clip.height = bh;
  clip.getContext('2d').drawImage(src, bx, by, bw, bh, 0, 0, bw, bh);
  clipboard = { canvas: clip, x: bx, y: by };

  if (cut) {
    pushHistory();
    if (mask) {
      l.ctx.save();
      l.ctx.globalCompositeOperation = 'destination-out';
      l.ctx.drawImage(mask, -l.x, -l.y);
      l.ctx.restore();
    } else {
      l.ctx.clearRect(0, 0, l.canvas.width, l.canvas.height);
    }
    refresh();
  }
}

function pasteClipboard() {
  if (!clipboard) return;
  pushHistory();
  const layer = makeLayer('Pasted');
  layer.ctx.drawImage(clipboard.canvas, clipboard.x, clipboard.y);
  state.layers.splice(state.active + 1, 0, layer);
  state.active += 1;
  setTool('move');
  refresh();
}

// Ctrl+J — isolate the current selection onto its own transparent layer, in place
function layerViaCopy() {
  const l = activeLayer();
  if (!l) return;
  const mask = currentMaskCanvas();
  pushHistory();
  const layer = makeLayer(mask ? `${l.name} (isolated)` : `${l.name} copy`);
  layer.ctx.drawImage(mask ? maskedExtract(l, mask) : l.canvas, mask ? 0 : l.x, mask ? 0 : l.y);
  state.layers.splice(state.active + 1, 0, layer);
  state.active += 1;
  refresh();
}

/* ------------------------------ generative fill ----------------------------- */

// AI service runs on the same machine that serves the editor, so reuse the
// hostname the page was loaded from (works for localhost and LAN access alike)
const GEN_SERVICE = `http://${location.hostname}:8765`;

function compositeClean() {
  // full composite without the live adjustment preview filter
  const c = document.createElement('canvas');
  c.width = state.doc.width; c.height = state.doc.height;
  const ctx = c.getContext('2d');
  for (const l of state.layers) {
    if (!l.visible) continue;
    ctx.save();
    ctx.globalAlpha = l.opacity / 100;
    ctx.globalCompositeOperation = l.blend;
    ctx.drawImage(l.canvas, l.x, l.y);
    ctx.restore();
  }
  return c;
}

function selectionMask() {
  const c = document.createElement('canvas');
  c.width = state.doc.width; c.height = state.doc.height;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, c.width, c.height);
  if (state.selection.mask) {
    ctx.drawImage(state.selection.mask, 0, 0);   // white-on-transparent over black
  } else {
    const { x, y, w, h } = state.selection;
    ctx.fillStyle = '#fff';
    ctx.fillRect(x, y, w, h);
  }
  return c;
}

function genStatus(msg, cls = '') {
  const el = $('#gen-status');
  el.textContent = msg;
  el.className = cls;
}

// After a fill, glow the seam-refinement controls so users discover them.
function pulseRefineControls() {
  const vary = $('#layer-variations');
  const targets = [
    vary && !vary.hidden ? vary : null,   // surface the option cycler first if present
    $('#edge-ctl'), $('#opacity-ctl'),
  ].filter(Boolean);
  if (!targets.length) return;
  targets[0].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  const clear = () => targets.forEach((t) => t.classList.remove('attn'));
  targets.forEach((t) => { t.classList.remove('attn'); void t.offsetWidth; t.classList.add('attn'); });
  setTimeout(clear, 5500);
  // stop nagging the moment they engage any of them
  ['#layer-edge', '#layer-opacity', '#vary-next', '#vary-prev'].forEach((s) =>
    $(s).addEventListener('click', clear, { once: true }));
  ['#layer-edge', '#layer-opacity'].forEach((s) =>
    $(s).addEventListener('input', clear, { once: true }));
}

let fillExpandOverride = false;   // set once the user opts out for the session

function showExpandTip() { const t = $('#gen-expand-tip'); if (t) t.hidden = false; }
function hideExpandTip() { const t = $('#gen-expand-tip'); if (t) t.hidden = true; }

async function generativeFill() {
  const prompt = $('#gen-prompt').value.trim();
  if (!state.selection) { genStatus('Make a selection first (M).', 'error'); return; }
  if (!prompt) { genStatus('Describe what to generate.', 'error'); return; }

  // A tight (un-expanded) selection often leaves a faint silhouette seam.
  // Recommend a small expand first — the user can override for the session.
  const expand = +($('#sel-expand')?.value || 0);
  if (expand === 0 && !fillExpandOverride) { showExpandTip(); return; }
  hideExpandTip();

  const btn = $('#gen-go');
  btn.disabled = true;
  const sel = { ...state.selection };
  // clamp selection to the canvas
  sel.x = Math.max(0, sel.x); sel.y = Math.max(0, sel.y);
  sel.w = Math.min(state.doc.width - sel.x, sel.w);
  sel.h = Math.min(state.doc.height - sel.y, sel.h);

  try {
    genStatus('Checking service…', 'busy');
    const health = await fetch(`${GEN_SERVICE}/health`).then((r) => r.json());
    genStatus(health.loaded
      ? 'Generating…'
      : 'Loading FLUX.1 Fill onto the GPU — first run takes a few minutes…', 'busy');

    const t0 = performance.now();
    const res = await fetch(`${GEN_SERVICE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: compositeClean().toDataURL('image/png'),
        mask: selectionMask().toDataURL('image/png'),
        prompt,
        count: 3,
      }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `service error ${res.status}`);
    }
    const out = await res.json();

    // a soft clip mask in the selection shape — each option keeps just the fill
    const clip = document.createElement('canvas');
    clip.width = state.doc.width; clip.height = state.doc.height;
    const cctx = clip.getContext('2d');
    if (sel.mask) {
      cctx.filter = 'blur(3px)';
      cctx.drawImage(sel.mask, 0, 0);
    } else {
      cctx.fillStyle = '#fff';
      cctx.fillRect(sel.x, sel.y, sel.w, sel.h);
    }

    // turn each returned full-image into a clipped, full-canvas variation
    const variations = await Promise.all(out.images.map((url) => new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => {
        const v = document.createElement('canvas');
        v.width = state.doc.width; v.height = state.doc.height;
        const vc = v.getContext('2d');
        vc.drawImage(im, 0, 0);
        vc.globalCompositeOperation = 'destination-in';
        vc.drawImage(clip, 0, 0);
        resolve(v);
      };
      im.onerror = reject;
      im.src = url;
    })));

    pushHistory();
    const layer = makeLayer(`Fill: ${prompt.slice(0, 24)}`);
    layer.variations = variations;
    layer.variationIndex = 0;
    layer.ctx.drawImage(variations[0], 0, 0);
    state.layers.splice(state.active + 1, 0, layer);
    state.active += 1;
    refresh();
    const n = variations.length;
    genStatus(n > 1
      ? `${n} options in ${((performance.now() - t0) / 1000).toFixed(1)}s — cycle with ‹ › in Layers, then tweak Edge / Opacity ↓`
      : `Done in ${((performance.now() - t0) / 1000).toFixed(1)}s — tweak Edge / Opacity to refine the seam ↓`);
    pulseRefineControls();
  } catch (err) {
    genStatus(
      err.message.includes('fetch')
        ? 'Service not running — start genai-service/server.py first.'
        : `Error: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
  }
}

/* -------------------------------- reimagine -------------------------------- */

function rmgStatus(msg, cls = '') {
  const el = $('#rmg-status');
  el.textContent = msg;
  el.className = cls;
}

async function reimagine() {
  const prompt = $('#rmg-prompt').value.trim();
  if (!prompt) { rmgStatus('Describe a variation first.', 'error'); return; }
  const btn = $('#rmg-go');
  btn.disabled = true;
  try {
    rmgStatus('Checking service…', 'busy');
    const health = await fetch(`${GEN_SERVICE}/health`).then((r) => r.json());
    rmgStatus(health.img2img_loaded === false || !health.img2img_loaded
      ? 'Loading the variation model onto the GPU (first run)…'
      : 'Imagining 3 variations…', 'busy');

    const t0 = performance.now();
    const res = await fetch(`${GEN_SERVICE}/reimagine`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: compositeClean().toDataURL('image/png'),
        prompt,
        likeness: +$('#rmg-likeness').value,
        count: 3,
      }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `service error ${res.status}`);
    }
    const out = await res.json();
    const imgs = await Promise.all(out.images.map((url) => new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = reject;
      im.src = url;
    })));
    // drop a layer that holds ALL variations — switch between them anytime later
    pushHistory();
    const layer = makeLayer(`Reimagine: ${prompt.slice(0, 22)}`);
    layer.variations = imgs;
    layer.variationIndex = 0;
    layer.ctx.drawImage(imgs[0], 0, 0, state.doc.width, state.doc.height);
    state.layers.splice(state.active + 1, 0, layer);
    state.active += 1;
    refresh();
    rmgStatus(`${imgs.length} variations in ${((performance.now() - t0) / 1000).toFixed(1)}s — cycle them with ‹ › in Layers.`);
  } catch (err) {
    rmgStatus(
      err.message.includes('fetch')
        ? 'Service not running — start genai-service/server.py first.'
        : `Error: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
  }
}

// switch which variation an AI-variation layer shows (non-destructive, revisitable)
function setLayerVariation(layer, index) {
  if (!layer || !layer.variations) return;
  const n = layer.variations.length;
  layer.variationIndex = (index + n) % n;
  layer.ctx.clearRect(0, 0, layer.canvas.width, layer.canvas.height);
  layer.ctx.drawImage(layer.variations[layer.variationIndex], 0, 0, layer.canvas.width, layer.canvas.height);
  layer.edgeBase = null;           // content changed; re-base edge feather
  render();
  renderLayerList();
}

/* ------------------------------ select subject ------------------------------ */

function selStatus(msg, cls = '') {
  const el = $('#sel-status');
  el.textContent = msg;
  el.className = cls;
}

function maskUrlToCanvas(dataUrl) {
  // white-on-black mask PNG → white-on-transparent alpha canvas
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const m = docSizedCanvas();
      const mctx = m.getContext('2d');
      mctx.drawImage(img, 0, 0, m.width, m.height);
      const d = mctx.getImageData(0, 0, m.width, m.height);
      for (let i = 0; i < d.data.length; i += 4) {
        d.data[i + 3] = d.data[i];                    // alpha from luminance
        d.data[i] = d.data[i + 1] = d.data[i + 2] = 255;
      }
      mctx.putImageData(d, 0, 0);
      resolve(m);
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

async function postSegment(path, body) {
  const res = await fetch(`${GEN_SERVICE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: compositeClean().toDataURL('image/png'), ...body }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `service error ${res.status}`);
  }
  return res.json();
}

function segmentError(err) {
  selStatus(
    err.message.includes('fetch')
      ? 'Service not running — start genai-service/server.py first.'
      : `Error: ${err.message}`, 'error');
}

async function selectSubject() {
  const text = $('#sel-text').value.trim();
  if (!text) { selStatus('Type what to select.', 'error'); return; }
  const btn = $('#sel-go');
  btn.disabled = true;
  try {
    selStatus('Finding it…', 'busy');
    const t0 = performance.now();
    const out = await postSegment('/segment', { text });
    combineSelection(await maskUrlToCanvas(out.mask), 'replace');
    const s = ((performance.now() - t0) / 1000).toFixed(1);
    selStatus(`Selected ${out.count} instance${out.count > 1 ? 's' : ''} in ${s}s.`);
  } catch (err) {
    segmentError(err);
  } finally {
    btn.disabled = false;
  }
}

async function selectObjectAt(p, combine) {
  try {
    selStatus('Masking object…', 'busy');
    const t0 = performance.now();
    const out = await postSegment('/segment-point', { points: [[p.x, p.y]] });
    combineSelection(await maskUrlToCanvas(out.mask), combine);
    const verb = { replace: 'Selected', add: 'Added', subtract: 'Subtracted' }[combine];
    selStatus(`${verb} object (score ${out.score}) in ${((performance.now() - t0) / 1000).toFixed(1)}s.`);
  } catch (err) {
    segmentError(err);
  }
}

/* --------------------------------- expand ---------------------------------- */

function expandCanvasDialog() {
  const root = $('#modal-root');
  root.hidden = false;
  root.innerHTML = `
    <div class="modal">
      <h2>Expand Canvas</h2>
      <div class="field"><label>Left</label><input id="ex-l" type="number" value="0" min="0" max="4096"></div>
      <div class="field"><label>Right</label><input id="ex-r" type="number" value="256" min="0" max="4096"></div>
      <div class="field"><label>Top</label><input id="ex-t" type="number" value="0" min="0" max="4096"></div>
      <div class="field"><label>Bottom</label><input id="ex-b" type="number" value="0" min="0" max="4096"></div>
      <div class="modal-actions">
        <button id="ex-cancel">Cancel</button>
        <button id="ex-ok" class="primary">Expand</button>
      </div>
    </div>`;
  $('#ex-cancel').addEventListener('click', () => { root.hidden = true; });
  $('#ex-ok').addEventListener('click', () => {
    const v = (id) => Math.max(0, Math.min(4096, +$(id).value || 0));
    expandCanvas(v('#ex-l'), v('#ex-t'), v('#ex-r'), v('#ex-b'));
    root.hidden = true;
  });
}

function expandCanvas(left, top, right, bottom) {
  if (!left && !top && !right && !bottom) return;
  pushHistory();
  state.doc.width += left + right;
  state.doc.height += top + bottom;
  for (const l of state.layers) {
    l.x += left;
    l.y += top;
  }
  // select the fresh border area on the largest edge to make outpainting one
  // step: expand → prompt → generate
  let r;
  if (right >= Math.max(left, top, bottom)) {
    r = { x: state.doc.width - right - 24, y: 0, w: right + 24, h: state.doc.height };
  } else if (left >= Math.max(top, bottom)) {
    r = { x: 0, y: 0, w: left + 24, h: state.doc.height };
  } else if (bottom >= top) {
    r = { x: 0, y: state.doc.height - bottom - 24, w: state.doc.width, h: bottom + 24 };
  } else {
    r = { x: 0, y: 0, w: state.doc.width, h: top + 24 };
  }
  resizeDocCanvas();
  setMaskSelection(shapeMask((ctx) => ctx.fillRect(r.x, r.y, r.w, r.h)));
  fitToView();
  refresh();
  genStatus('New area selected — describe the fill and Generate to outpaint.');
}

/* demo content so the first impression isn't a blank page */
function demoContent() {
  const bg = activeLayer();
  const g = bg.ctx.createLinearGradient(0, 0, state.doc.width, state.doc.height);
  g.addColorStop(0, '#1c2742');
  g.addColorStop(0.55, '#3c5a8f');
  g.addColorStop(1, '#e98f5f');
  bg.ctx.fillStyle = g;
  bg.ctx.fillRect(0, 0, state.doc.width, state.doc.height);
  bg.name = 'Sky';

  const sun = makeLayer('Sun');
  sun.ctx.fillStyle = '#ffd98a';
  sun.ctx.beginPath();
  sun.ctx.arc(state.doc.width * 0.68, state.doc.height * 0.42, 90, 0, Math.PI * 2);
  sun.ctx.fill();
  sun.ctx.filter = 'blur(2px)';

  const hills = makeLayer('Hills');
  hills.ctx.fillStyle = '#10182b';
  hills.ctx.beginPath();
  hills.ctx.moveTo(0, state.doc.height);
  for (let x = 0; x <= state.doc.width; x += 8) {
    const y = state.doc.height * 0.72
      + Math.sin(x / 140) * 46
      + Math.sin(x / 47 + 2) * 18;
    hills.ctx.lineTo(x, y);
  }
  hills.ctx.lineTo(state.doc.width, state.doc.height);
  hills.ctx.closePath();
  hills.ctx.fill();

  state.layers.push(sun, hills);
  state.active = state.layers.length - 1;
}

/* ----------------------------------- init ----------------------------------- */

// Collapsible panel sections + a draggable width handle so users can manage
// the (tall) right panel and surface whatever section they need.
function setupPanels() {
  document.querySelectorAll('.panel > h3').forEach((h) => {
    const panel = h.parentElement;
    const key = `pp-collapsed-${panel.id}`;
    if (localStorage.getItem(key) === '1') panel.classList.add('collapsed');
    h.addEventListener('click', () => {
      panel.classList.toggle('collapsed');
      localStorage.setItem(key, panel.classList.contains('collapsed') ? '1' : '0');
    });
  });

  const savedW = localStorage.getItem('pp-panel-w');
  if (savedW) document.documentElement.style.setProperty('--panel-w', `${savedW}px`);

  const resizer = $('#panel-resizer');
  let dragging = false;
  resizer.addEventListener('pointerdown', (e) => {
    dragging = true;
    resizer.classList.add('dragging');
    resizer.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  resizer.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const w = Math.max(220, Math.min(560, window.innerWidth - e.clientX));
    document.documentElement.style.setProperty('--panel-w', `${w}px`);
  });
  const stop = () => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove('dragging');
    const w = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--panel-w'), 10);
    if (w) localStorage.setItem('pp-panel-w', w);
    applyStageTransform();
  };
  resizer.addEventListener('pointerup', stop);
  resizer.addEventListener('lostpointercapture', stop);
}

function init() {
  buildMenus();
  buildToolbar();
  buildAdjustPanel();
  buildLayersPanel();
  $('#gen-go').addEventListener('click', generativeFill);
  $('#gen-expand-apply').addEventListener('click', () => {
    $('#sel-expand').value = 5;
    $('#sel-expand-val').textContent = '5px';
    applySelectionEdge();              // grow the selection outward by 5px
    hideExpandTip();
    generativeFill();                  // expand is now 5 → proceeds
  });
  $('#gen-expand-skip').addEventListener('click', () => {
    fillExpandOverride = true;         // don't recommend again this session
    hideExpandTip();
    generativeFill();
  });
  $('#rmg-go').addEventListener('click', reimagine);
  $('#rmg-likeness').addEventListener('input', (e) => {
    $('#rmg-likeness-val').textContent = e.target.value;
  });
  $('#rmg-prompt').addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); reimagine(); }
  });
  setupPanels();
  $('#vary-prev').addEventListener('click', () => setLayerVariation(activeLayer(), activeLayer().variationIndex - 1));
  $('#vary-next').addEventListener('click', () => setLayerVariation(activeLayer(), activeLayer().variationIndex + 1));
  $('#sel-go').addEventListener('click', selectSubject);
  $('#sel-text').addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Enter') selectSubject();
  });
  $('#sel-feather').addEventListener('input', (e) => {
    $('#sel-feather-val').textContent = `${e.target.value}px`;
    applySelectionEdge();
  });
  $('#sel-expand').addEventListener('input', (e) => {
    $('#sel-expand-val').textContent = `${e.target.value}px`;
    applySelectionEdge();
  });
  $('#sel-invert').addEventListener('click', invertSelection);
  $('#sel-all').addEventListener('click', selectAll);
  $('#sel-none').addEventListener('click', deselect);
  $('#gen-prompt').addEventListener('keydown', (e) => {
    e.stopPropagation();   // don't trigger tool shortcuts while typing
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); generativeFill(); }
  });
  createDocument('Untitled', 1280, 800, 'transparent');
  demoContent();
  state.history.length = 0;
  setTool('brush');
  fitToView();
  refresh();
  window.addEventListener('resize', () => applyStageTransform());
}

init();
