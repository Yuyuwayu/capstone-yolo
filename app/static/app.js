/* ═══════════════════════════════════════════════════
   FishWatch — Frontend Application Logic
   Vanilla JS SPA · No dependencies
   ═══════════════════════════════════════════════════ */

const API = '';
let currentPage = 'monitor';
let envOk = false;

// ── State ─────────────────────────────────────────
let dsState = { dataset: '', split: 'train', filter: 'all', selected: new Set() };
let monitorInterval = null;
let trainingInterval = null;
let trLossTrend = [];
let trLastEpoch = 0;
let trendData = [];

// ══════════════════════════════════════════════════
// ROUTER
// ══════════════════════════════════════════════════

function navigate(page) {
  currentPage = page;
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));

  const el = document.getElementById('page-' + page);
  if (el) el.classList.add('active');
  const tab = document.querySelector(`.nav-tab[data-page="${page}"]`);
  if (tab) tab.classList.add('active');

  // Lifecycle
  stopMonitorPolling();
  stopTrainingPolling();

  if (page === 'monitor') initMonitor();
  if (page === 'dataset') initDataset();
  if (page === 'training') initTraining();
  if (page === 'models') initModels();
}

document.querySelectorAll('.nav-tab').forEach(tab => {
  tab.addEventListener('click', () => navigate(tab.dataset.page));
});

// ══════════════════════════════════════════════════
// ENVIRONMENT CHECK
// ══════════════════════════════════════════════════

async function checkEnv() {
  try {
    const res = await fetch(API + '/api/env/check');
    const data = await res.json();
    renderEnv(data);
    envOk = data.all_ok;
    updateEnvDot(data.all_ok);

    if (!data.all_ok) {
      navigate('setup');
      document.getElementById('page-setup').classList.add('active');
    } else {
      navigate('monitor');
    }
  } catch {
    updateEnvDot(false);
    navigate('monitor');
  }
}

function renderEnv(data) {
  const pkgEl = document.getElementById('setup-packages');
  pkgEl.innerHTML = '<p class="t-label" style="margin-bottom:var(--sp-sm)">Packages</p>' +
    data.packages.map(p => `
      <div class="check-item">
        <span class="check-icon">${p.installed ? '✓' : '✗'}</span>
        <span style="color:${p.installed ? 'var(--primary)' : 'var(--danger)'}">${p.name}</span>
        <span class="t-secondary" style="margin-left:auto">${p.version || 'not installed'}</span>
      </div>
    `).join('');

  // Python
  pkgEl.innerHTML = `
    <div class="check-item">
      <span class="check-icon">${data.python.ok ? '✓' : '✗'}</span>
      <span style="color:${data.python.ok ? 'var(--primary)' : 'var(--danger)'}">Python</span>
      <span class="t-secondary" style="margin-left:auto">${data.python.version}</span>
    </div>` + pkgEl.innerHTML;

  const dirEl = document.getElementById('setup-dirs');
  dirEl.innerHTML = data.directories.map(d => {
    const icon = !d.exists ? '✗' : d.empty ? '⚠' : '✓';
    const color = !d.exists ? 'var(--danger)' : d.empty ? 'var(--warning)' : 'var(--primary)';
    const note = !d.exists ? 'missing' : d.empty ? 'empty' : 'ok';
    return `
      <div class="check-item">
        <span class="check-icon">${icon}</span>
        <span style="color:${color}">${d.path}</span>
        <span class="t-secondary" style="margin-left:auto">${note}</span>
      </div>`;
  }).join('');
}

function updateEnvDot(ok) {
  const dot = document.getElementById('env-dot');
  const label = document.getElementById('env-label');
  dot.className = 'dot ' + (ok ? 'dot-ok' : 'dot-error');
  label.textContent = ok ? 'Ready' : 'Setup needed';
  label.style.color = ok ? 'var(--tertiary)' : 'var(--danger)';
}

document.getElementById('btn-install').addEventListener('click', async () => {
  await fetch(API + '/api/env/install', { method: 'POST' });
  document.getElementById('install-log-wrap').style.display = 'block';
  const logEl = document.getElementById('install-log');
  const poll = setInterval(async () => {
    const res = await fetch(API + '/api/env/install/status');
    const data = await res.json();
    logEl.textContent = data.logs.join('\n');
    logEl.scrollTop = logEl.scrollHeight;
    if (!data.running) {
      clearInterval(poll);
      checkEnv();
    }
  }, 1000);
});

document.getElementById('btn-recheck').addEventListener('click', checkEnv);

// ══════════════════════════════════════════════════
// MONITOR
// ══════════════════════════════════════════════════

function initMonitor() {
  loadMonitorModels();
  loadSourceOptions();
  startMonitorPolling();
}

async function loadMonitorModels() {
  try {
    const res = await fetch(API + '/api/models');
    const models = await res.json();
    const sel = document.getElementById('monitor-model');
    sel.innerHTML = models.map(m =>
      `<option value="${m.name}" ${m.is_active ? 'selected' : ''}>${m.name}${m.is_active ? ' (active)' : ''}</option>`
    ).join('');
  } catch {}
}

// ── Auto-detect cameras ──────────────────────────

async function loadSourceOptions() {
  const camGroup = document.getElementById('cam-optgroup');
  camGroup.label = 'Detecting webcams…';
  try {
    const res = await fetch(API + '/api/monitor/cameras');
    const cameras = await res.json();
    if (cameras.length) {
      camGroup.label = 'Webcams';
      camGroup.innerHTML = cameras.map((c, i) =>
        `<option value="webcam-${c.index}" ${i === 0 ? 'selected' : ''}>${c.name}</option>`
      ).join('');
    } else {
      camGroup.label = 'No webcams found';
    }
  } catch {
    camGroup.label = 'Webcams (detection failed)';
  }
}

document.getElementById('monitor-model').addEventListener('change', async (e) => {
  await fetch(API + '/api/models/' + e.target.value + '/activate', { method: 'POST' });
  loadMonitorModels();
});

function startMonitorPolling() {
  pollMonitor();
  monitorInterval = setInterval(pollMonitor, 1000);
}

function stopMonitorPolling() {
  if (monitorInterval) clearInterval(monitorInterval);
  monitorInterval = null;
}

async function pollMonitor() {
  // Skip polling when browser camera is active (it updates stats itself)
  if (browserCamActive) return;
  try {
    const res = await fetch(API + '/api/monitor/status');
    const d = await res.json();

    const statusEl = document.getElementById('status-text');
    statusEl.textContent = d.status;
    statusEl.className = 'status-text ' + (d.status === 'Lapar' ? 'status-lapar' : 'status-ok');

    document.getElementById('stat-count').textContent = d.fish_count;
    document.getElementById('stat-dist').textContent = d.avg_distance + ' px';
    document.getElementById('stat-time').textContent = d.timestamp || '—';

    if (d.has_frame) {
      trendData.push(d.avg_distance);
      if (trendData.length > 60) trendData.shift();
      drawTrendChart();
    }

    updateDetectionLog(d);
  } catch {}
}

function updateDetectionLog(d) {
  if (!d.has_frame && !d.timestamp) return;
  const log = document.getElementById('detection-log');
  const entry = document.createElement('div');
  entry.className = 'entry';
  entry.innerHTML = `<span class="t-mono">${d.timestamp}</span> — <span style="color:${d.status === 'Lapar' ? 'var(--danger)' : 'var(--tertiary)'}">${d.status}</span> <span class="t-secondary">(${d.avg_distance} px, ${d.fish_count} fish)</span>`;
  if (log.querySelector('.empty-state')) log.innerHTML = '';
  log.prepend(entry);
  while (log.children.length > 50) log.removeChild(log.lastChild);
}

// ── Source type switching ─────────────────────────

document.getElementById('monitor-source-type').addEventListener('change', (e) => {
  const val = e.target.value;
  document.getElementById('source-extra-wrap').style.display = 'none';
  document.getElementById('video-upload-wrap').style.display = 'none';

  if (val === 'droidcam') {
    document.getElementById('source-extra-wrap').style.display = 'flex';
    document.getElementById('source-extra-label').textContent = 'IP Address';
    document.getElementById('monitor-source-extra').placeholder = '192.168.1.7:4747';
  } else if (val === 'video-file') {
    document.getElementById('video-upload-wrap').style.display = 'flex';
  }
});

let uploadedVideoFilename = null;

function getSourceValue() {
  const type = document.getElementById('monitor-source-type').value;
  const extra = document.getElementById('monitor-source-extra').value;
  if (type === 'droidcam' && extra) return 'http://' + extra + '/video';
  if (type === 'video-file' && uploadedVideoFilename) return uploadedVideoFilename;
  if (type.startsWith('webcam-')) return type.split('-')[1];
  return 'webcam';
}

// ── Browser Camera (getUserMedia) ─────────────────

let browserCamActive = false;
let browserCamStream = null;
let browserCamInterval = null;

async function startBrowserCamera() {
  try {
    const constraints = { video: { facingMode: 'environment', width: 640, height: 480 } };
    browserCamStream = await navigator.mediaDevices.getUserMedia(constraints);
    const videoEl = document.getElementById('browser-cam-video');
    videoEl.srcObject = browserCamStream;
    videoEl.style.display = 'none'; // keep hidden, we show annotated frames

    browserCamActive = true;
    // Capture and send frames at ~4 fps
    browserCamInterval = setInterval(captureAndSendFrame, 250);
  } catch (err) {
    alert('Camera access failed: ' + err.message);
  }
}

function stopBrowserCamera() {
  browserCamActive = false;
  if (browserCamInterval) clearInterval(browserCamInterval);
  browserCamInterval = null;
  if (browserCamStream) {
    browserCamStream.getTracks().forEach(t => t.stop());
    browserCamStream = null;
  }
  const videoEl = document.getElementById('browser-cam-video');
  videoEl.srcObject = null;
  videoEl.style.display = 'none';
}

async function captureAndSendFrame() {
  if (!browserCamActive) return;
  const videoEl = document.getElementById('browser-cam-video');
  const canvas = document.getElementById('browser-cam-canvas');
  if (videoEl.readyState < 2) return; // not ready yet

  canvas.width = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoEl, 0, 0);

  canvas.toBlob(async (blob) => {
    if (!blob || !browserCamActive) return;
    const form = new FormData();
    form.append('file', blob, 'frame.jpg');

    try {
      const res = await fetch(API + '/api/monitor/process_frame', { method: 'POST', body: form });
      const d = await res.json();

      // Show annotated frame
      document.getElementById('video-feed').src = 'data:image/jpeg;base64,' + d.image;

      // Update stats
      const statusEl = document.getElementById('status-text');
      statusEl.textContent = d.status;
      statusEl.className = 'status-text ' + (d.status === 'Lapar' ? 'status-lapar' : 'status-ok');

      document.getElementById('stat-count').textContent = d.fish_count;
      document.getElementById('stat-dist').textContent = d.avg_distance + ' px';
      document.getElementById('stat-time').textContent = d.timestamp;

      trendData.push(d.avg_distance);
      if (trendData.length > 60) trendData.shift();
      drawTrendChart();

      updateDetectionLog(d);
    } catch {}
  }, 'image/jpeg', 0.75);
}

// ── Video file upload via browse dialog ───────────

document.getElementById('btn-video-browse').addEventListener('click', () => {
  document.getElementById('video-upload-input').click();
});

document.getElementById('video-upload-input').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  document.getElementById('video-file-name').textContent = 'Uploading…';
  const form = new FormData();
  form.append('file', file);
  try {
    const res = await fetch(API + '/api/monitor/upload_video', { method: 'POST', body: form });
    const data = await res.json();
    if (data.success) {
      uploadedVideoFilename = data.filename;
      document.getElementById('video-file-name').textContent = data.filename;
    }
  } catch {
    document.getElementById('video-file-name').textContent = 'Upload failed';
  }
  e.target.value = '';
});

// ── Start / Stop Stream ───────────────────────────

document.getElementById('btn-stream-start').addEventListener('click', async () => {
  const type = document.getElementById('monitor-source-type').value;

  // Stop any existing browser camera
  stopBrowserCamera();

  // Hide placeholder, show video feed
  document.getElementById('video-placeholder').style.display = 'none';
  const feed = document.getElementById('video-feed');
  feed.style.display = 'block';

  if (type === 'browser-cam') {
    await startBrowserCamera();
    return;
  }

  const source = getSourceValue();
  await fetch(API + '/api/monitor/start?source=' + encodeURIComponent(source), { method: 'POST' });
  feed.src = API + '/api/monitor/video_feed?' + Date.now();
});

document.getElementById('btn-stream-stop').addEventListener('click', async () => {
  stopBrowserCamera();
  await fetch(API + '/api/monitor/stop', { method: 'POST' });
  document.getElementById('video-feed').src = '';
  document.getElementById('video-feed').style.display = 'none';
  document.getElementById('video-placeholder').style.display = 'flex';
});

// Settings
document.getElementById('slider-threshold').addEventListener('input', (e) => {
  document.getElementById('threshold-val').textContent = e.target.value;
});
document.getElementById('slider-threshold').addEventListener('change', (e) => {
  fetch(API + '/api/monitor/config', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ distance_threshold: Number(e.target.value) })
  });
});

document.getElementById('slider-conf').addEventListener('input', (e) => {
  document.getElementById('conf-val').textContent = (e.target.value / 100).toFixed(2);
});
document.getElementById('slider-conf').addEventListener('change', (e) => {
  fetch(API + '/api/monitor/config', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ confidence_threshold: e.target.value / 100 })
  });
});

// ══════════════════════════════════════════════════
// CHARTS (Canvas)
// ══════════════════════════════════════════════════

function drawTrendChart() {
  drawLineChart('chart-trend', trendData, 'var(--tertiary)');
}

function drawLineChart(canvasId, data, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !data.length) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;

  // Reset canvas size to 0 before measuring parent — prevents feedback loop
  canvas.style.width = '100%';
  canvas.width = 0;
  const rect = canvas.parentElement.getBoundingClientRect();
  const w = rect.width - 32; // subtract padding
  const h = 160;

  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  ctx.scale(dpr, dpr);

  const pad = 30;

  ctx.clearRect(0, 0, w, h);

  const max = Math.max(...data) * 1.1 || 1;
  const min = Math.min(...data) * 0.9 || 0;
  const range = max - min || 1;

  // Grid lines
  ctx.strokeStyle = 'rgba(143,165,181,0.1)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const y = pad + (h - pad * 2) * i / 3;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - 10, y); ctx.stroke();
    ctx.fillStyle = '#8FA5B5';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(Math.round(max - range * i / 3), 0, y + 4);
  }

  // Line
  ctx.strokeStyle = color.startsWith('var') ? '#3CBAB2' : color;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  data.forEach((val, i) => {
    const x = pad + (w - pad - 10) * i / (data.length - 1 || 1);
    const y = pad + (h - pad * 2) * (1 - (val - min) / range);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ══════════════════════════════════════════════════
// DATASET
// ══════════════════════════════════════════════════

async function initDataset() {
  // Load all dataset names (always complete list)
  try {
    const namesRes = await fetch(API + '/api/dataset/list');
    const names = await namesRes.json();
    const dsSelect = document.getElementById('ds-select');
    dsSelect.innerHTML = names.map(d => `<option value="${d}">${d}</option>`).join('');
    if (!dsState.dataset || !names.includes(dsState.dataset)) {
      dsState.dataset = names[0] || '';
    }
    dsSelect.value = dsState.dataset;
  } catch {}

  // Load stats for current dataset
  if (dsState.dataset) {
    try {
      const res = await fetch(API + '/api/dataset/stats');
      const stats = await res.json();
      updateSplitOptions(stats);
    } catch {
      // Show defaults if stats fail
      document.getElementById('ds-stats').innerHTML = '0 total';
    }
  }

  loadImages();
  loadMergeCheckboxes();
  if (dsState.dataset) {
    loadClasses();
  }
}

function updateSplitOptions(stats) {
  const ds = stats[dsState.dataset];
  const splitSel = document.getElementById('ds-split');
  const statsEl = document.getElementById('ds-stats');

  if (!ds || !ds.splits) {
    // Dataset exists but has no data yet — show defaults
    splitSel.innerHTML = '<option value="train">train</option><option value="val">val</option>';
    if (!dsState.split) dsState.split = 'train';
    splitSel.value = dsState.split;
    statsEl.innerHTML = '0 total · <span style="color:var(--tertiary)">0 annotated</span> · <span style="color:var(--danger)">0 missing</span>';
    return;
  }

  const splits = Object.keys(ds.splits);
  splitSel.innerHTML = splits.map(s => `<option value="${s}">${s}</option>`).join('');
  if (!splits.includes(dsState.split)) dsState.split = splits[0] || 'train';
  splitSel.value = dsState.split;

  const sp = ds.splits[dsState.split];
  if (sp) {
    statsEl.innerHTML =
      `${sp.total} total · <span style="color:var(--tertiary)">${sp.annotated} annotated</span> · <span style="color:var(--danger)">${sp.unannotated} missing</span>`;
  } else {
    statsEl.innerHTML = '0 total';
  }
}

document.getElementById('ds-select').addEventListener('change', (e) => {
  dsState.dataset = e.target.value;
  dsState.selected.clear();
  initDataset();
});

document.getElementById('btn-ds-new').addEventListener('click', async () => {
  const name = prompt("Enter new dataset name (alphanumeric and underscores only):");
  if (!name) return;
  try {
    const res = await fetch(API + `/api/dataset/${name}/create`, { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      dsState.dataset = name;
      initDataset();
    } else {
      alert("Error: " + data.error);
    }
  } catch (err) {
    alert("Failed to create dataset.");
  }
});

document.getElementById('btn-ds-delete-all').addEventListener('click', () => {
  if (!dsState.dataset) return;
  showConfirm(`Delete entire dataset '${dsState.dataset}'? This cannot be undone!`, async () => {
    try {
      const res = await fetch(API + `/api/dataset/${dsState.dataset}`, { method: 'DELETE' });
      const data = await res.json();
      if (data.success) {
        dsState.dataset = '';
        initDataset();
      } else {
        alert("Error: " + data.error);
      }
    } catch {
      alert("Failed to delete dataset.");
    }
  });
});
document.getElementById('ds-split').addEventListener('change', (e) => {
  dsState.split = e.target.value;
  dsState.selected.clear();
  loadImages();
});
document.getElementById('ds-filter').addEventListener('change', (e) => {
  dsState.filter = e.target.value;
  loadImages();
});

async function loadImages() {
  const gallery = document.getElementById('ds-gallery');
  gallery.innerHTML = '<div class="skeleton" style="height:200px"></div>';
  try {
    const res = await fetch(API + `/api/dataset/${dsState.dataset}/images?split=${dsState.split}&filter=${dsState.filter}`);
    const images = await res.json();

    if (!images.length) {
      gallery.innerHTML = '<div class="empty-state"><div class="icon">📁</div><p>No images found.</p></div>';
      return;
    }

    gallery.innerHTML = images.map(img => `
      <div class="gallery-item ${dsState.selected.has(img.filename) ? 'selected' : ''}" data-name="${img.filename}">
        <input type="checkbox" class="checkbox gallery-check" ${dsState.selected.has(img.filename) ? 'checked' : ''} />
        <button class="gallery-delete" title="Delete">✗</button>
        <img loading="lazy" src="${API}/api/dataset/${dsState.dataset}/image/${dsState.split}/${img.filename}" alt="${img.filename}" />
        <span class="gallery-badge ${img.has_label ? 'badge-annotated' : 'badge-missing'}">${img.has_label ? '✓' : '✗'}</span>
      </div>
    `).join('');

    // Click handlers
    gallery.querySelectorAll('.gallery-item').forEach(item => {
      // Click image → preview
      item.addEventListener('click', (e) => {
        if (e.target.classList.contains('gallery-check') || e.target.classList.contains('gallery-delete')) return;
        previewImage(item.dataset.name);
      });

      // Checkbox → select
      item.querySelector('.gallery-check').addEventListener('change', (e) => {
        e.stopPropagation();
        toggleSelect(item.dataset.name, e.target.checked);
      });

      // Per-image delete button
      item.querySelector('.gallery-delete').addEventListener('click', (e) => {
        e.stopPropagation();
        const fname = item.dataset.name;
        showConfirm(`Delete "${fname}"?`, async () => {
          await fetch(API + `/api/dataset/${dsState.dataset}/image/${dsState.split}/${fname}`, { method: 'DELETE' });
          dsState.selected.delete(fname);
          loadImages();
          initDataset();
        });
      });
    });
  } catch {
    gallery.innerHTML = '<div class="empty-state"><p>Failed to load images.</p></div>';
  }
  updateDeleteBtn();
}

function toggleSelect(filename, checked) {
  if (checked) dsState.selected.add(filename);
  else dsState.selected.delete(filename);
  const item = document.querySelector(`.gallery-item[data-name="${filename}"]`);
  if (item) item.classList.toggle('selected', checked);
  updateDeleteBtn();
}

function updateDeleteBtn() {
  const btn = document.getElementById('btn-ds-delete');
  const count = dsState.selected.size;
  btn.style.display = count > 0 ? 'inline-flex' : 'none';
  document.getElementById('ds-sel-count').textContent = count;
}

let currentPreviewFile = null;

// ── Annotation Canvas Logic ───────────────────────
let annotBoxes = [];
let annotClasses = {};

const canvas = document.getElementById('annot-canvas');
const ctx = canvas.getContext('2d');
const imgEl = document.getElementById('ds-preview-orig');
let isDrawing = false;
let isDraggingBox = false;
let dragOffsetX = 0; let dragOffsetY = 0;
let startX = 0; let startY = 0;
let currentRect = null;
let selectedBoxIndex = -1;

const COLORS = ['#b2ba3c', '#3cba5d', '#3c8eba', '#ba3c3c', '#ba3c9e', '#e6a822', '#3cba9b', '#9bba3c'];

async function loadClasses() {
  try {
    const res = await fetch(API + `/api/dataset/${dsState.dataset}/classes`);
    const data = await res.json();
    annotClasses = {};
    const select = document.getElementById('annot-class-select');
    select.innerHTML = '';
    const classNames = [];
    data.classes.forEach(c => {
      annotClasses[c.id] = c.name;
      classNames.push(c.name);
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = `${c.id}: ${c.name}`;
      select.appendChild(opt);
    });
    document.getElementById('ds-classes-input').value = classNames.join(', ');
  } catch {}
}

document.getElementById('btn-classes-save').addEventListener('click', async () => {
  const val = document.getElementById('ds-classes-input').value;
  const classes = val.split(',').map(s => s.trim()).filter(s => s);
  const status = document.getElementById('classes-status');
  try {
    await fetch(API + `/api/dataset/${dsState.dataset}/classes`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({classes})
    });
    status.textContent = 'Saved!';
    setTimeout(() => status.textContent='', 2000);
    loadClasses();
  } catch {
    status.textContent = 'Error';
  }
});

async function previewImage(filename) {
  currentPreviewFile = filename;
  document.getElementById('ds-preview-section').style.display = 'block';
  document.getElementById('ds-preview-name').textContent = filename;
  
  // Load image
  imgEl.src = `${API}/api/dataset/${dsState.dataset}/image/${dsState.split}/${filename}`;
  
  // Load labels
  const res = await fetch(API + `/api/dataset/${dsState.dataset}/labels/${dsState.split}/${filename}`);
  const data = await res.json();
  annotBoxes = data.labels || []; // [[id, xc, yc, w, h]]
  selectedBoxIndex = -1;
  updateDeleteBoxBtn();
}

function updateDeleteBoxBtn() {
  const btn = document.getElementById('btn-annot-delete-box');
  if (selectedBoxIndex !== -1) {
    btn.style.display = 'inline-block';
  } else {
    btn.style.display = 'none';
  }
}

// Draw loop
function drawCanvas() {
  if (!imgEl.complete || imgEl.naturalWidth === 0) return;
  canvas.width = imgEl.naturalWidth;
  canvas.height = imgEl.naturalHeight;
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);
  
  // Draw existing boxes
  annotBoxes.forEach((box, i) => {
    const [cid, xc, yc, w, h] = box;
    const px = (xc - w/2) * canvas.width;
    const py = (yc - h/2) * canvas.height;
    const pw = w * canvas.width;
    const ph = h * canvas.height;
    
    ctx.strokeStyle = i === selectedBoxIndex ? '#fff' : (COLORS[cid % COLORS.length] || '#00ff00');
    ctx.lineWidth = i === selectedBoxIndex ? 4 : 2;
    ctx.strokeRect(px, py, pw, ph);
    
    // Label bg
    ctx.fillStyle = ctx.strokeStyle;
    const name = annotClasses[cid] || `cls${cid}`;
    ctx.font = '24px monospace';
    const txtWidth = ctx.measureText(name).width;
    ctx.fillRect(px, py - 30, txtWidth + 10, 30);
    ctx.fillStyle = '#000';
    ctx.fillText(name, px + 5, py - 6);
  });
  
  // Draw current rect
  if (isDrawing && currentRect) {
    const activeCid = parseInt(document.getElementById('annot-class-select').value) || 0;
    ctx.strokeStyle = COLORS[activeCid % COLORS.length] || '#00ff00';
    ctx.lineWidth = 2;
    ctx.strokeRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h);
  }
}

imgEl.onload = drawCanvas;
window.addEventListener('resize', drawCanvas);

function getCanvasPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

canvas.addEventListener('mousedown', (e) => {
  const pos = getCanvasPos(e);
  
  // Check if clicking existing box
  selectedBoxIndex = -1;
  for (let i = annotBoxes.length - 1; i >= 0; i--) {
    const [cid, xc, yc, w, h] = annotBoxes[i];
    const px = (xc - w/2) * canvas.width;
    const py = (yc - h/2) * canvas.height;
    const pw = w * canvas.width;
    const ph = h * canvas.height;
    if (pos.x >= px && pos.x <= px+pw && pos.y >= py && pos.y <= py+ph) {
      selectedBoxIndex = i;
      isDraggingBox = true;
      dragOffsetX = pos.x - (xc * canvas.width);
      dragOffsetY = pos.y - (yc * canvas.height);
      updateDeleteBoxBtn();
      drawCanvas();
      return;
    }
  }

  // Otherwise start drawing new box
  selectedBoxIndex = -1;
  updateDeleteBoxBtn();
  isDrawing = true;
  startX = pos.x;
  startY = pos.y;
  currentRect = {x: startX, y: startY, w: 0, h: 0};
  drawCanvas();
});

canvas.addEventListener('mousemove', (e) => {
  const pos = getCanvasPos(e);

  if (isDraggingBox && selectedBoxIndex !== -1) {
    const box = annotBoxes[selectedBoxIndex];
    let newXc = (pos.x - dragOffsetX) / canvas.width;
    let newYc = (pos.y - dragOffsetY) / canvas.height;
    
    // Clamp to boundaries
    newXc = Math.max(box[3]/2, Math.min(1 - box[3]/2, newXc));
    newYc = Math.max(box[4]/2, Math.min(1 - box[4]/2, newYc));
    
    annotBoxes[selectedBoxIndex][1] = newXc;
    annotBoxes[selectedBoxIndex][2] = newYc;
    drawCanvas();
    return;
  }

  if (!isDrawing) return;
  currentRect.x = Math.min(startX, pos.x);
  currentRect.y = Math.min(startY, pos.y);
  currentRect.w = Math.abs(pos.x - startX);
  currentRect.h = Math.abs(pos.y - startY);
  drawCanvas();
});

canvas.addEventListener('mouseup', () => {
  if (isDraggingBox) {
    isDraggingBox = false;
    return;
  }
  
  if (isDrawing && currentRect.w > 5 && currentRect.h > 5) {
    // Convert to YOLO format
    const cid = parseInt(document.getElementById('annot-class-select').value) || 0;
    const xc = (currentRect.x + currentRect.w/2) / canvas.width;
    const yc = (currentRect.y + currentRect.h/2) / canvas.height;
    const w = currentRect.w / canvas.width;
    const h = currentRect.h / canvas.height;
    annotBoxes.push([cid, xc, yc, w, h]);
  }
  isDrawing = false;
  currentRect = null;
  drawCanvas();
});

document.addEventListener('keydown', (e) => {
  if ((e.key === 'Delete' || e.key === 'Backspace') && selectedBoxIndex !== -1) {
    // Make sure we are not focused on an input field
    if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    annotBoxes.splice(selectedBoxIndex, 1);
    selectedBoxIndex = -1;
    updateDeleteBoxBtn();
    drawCanvas();
  }
});

document.getElementById('btn-annot-delete-box').addEventListener('click', () => {
  if (selectedBoxIndex !== -1) {
    annotBoxes.splice(selectedBoxIndex, 1);
    selectedBoxIndex = -1;
    updateDeleteBoxBtn();
    drawCanvas();
  }
});

document.getElementById('btn-annot-clear').addEventListener('click', () => {
  annotBoxes = [];
  selectedBoxIndex = -1;
  updateDeleteBoxBtn();
  drawCanvas();
});

document.getElementById('btn-annot-save').addEventListener('click', async () => {
  const btn = document.getElementById('btn-annot-save');
  btn.textContent = 'Saving...';
  try {
    await fetch(API + `/api/dataset/${dsState.dataset}/labels/${dsState.split}/${currentPreviewFile}`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({labels: annotBoxes})
    });
    // refresh stats & classes
    loadImages();
    initDataset();
  } catch {}
  btn.textContent = 'Save';
});

document.getElementById('btn-preview-close').addEventListener('click', () => {
  document.getElementById('ds-preview-section').style.display = 'none';
  currentPreviewFile = null;
});
// Preview delete button
document.getElementById('btn-preview-delete').addEventListener('click', () => {
  if (!currentPreviewFile) return;
  showConfirm(`Delete "${currentPreviewFile}"?`, async () => {
    await fetch(API + `/api/dataset/${dsState.dataset}/image/${dsState.split}/${currentPreviewFile}`, { method: 'DELETE' });
    document.getElementById('ds-preview-section').style.display = 'none';
    currentPreviewFile = null;
    loadImages();
    initDataset();
  });
});

// Import — shared upload function (auto-splits into train/val)
async function uploadImages(fileList) {
  const status = document.getElementById('ds-import-status');
  // Filter to image files only (folder browse includes all files)
  const images = [...fileList].filter(f =>
    /\.(jpe?g|png|bmp|webp|tiff?)$/i.test(f.name)
  );
  if (!images.length) { status.textContent = 'No images found'; return; }

  const ratio = document.getElementById('ds-split-ratio').value / 100;
  status.textContent = `Uploading ${images.length} image(s)…`;
  const form = new FormData();
  for (const f of images) form.append('files', f);

  try {
    const res = await fetch(API + `/api/dataset/${dsState.dataset}/upload-images?train_ratio=${ratio}`, {
      method: 'POST', body: form
    });
    const data = await res.json();
    if (data.success) {
      status.textContent = `✓ ${data.imported} imported (train: ${data.train}, val: ${data.val})`;
      initDataset();
    } else {
      status.textContent = data.error || 'Failed';
    }
  } catch {
    status.textContent = 'Upload failed';
  }
}

// Browse files
document.getElementById('btn-import').addEventListener('click', () => {
  document.getElementById('ds-import-files').click();
});
document.getElementById('ds-import-files').addEventListener('change', (e) => {
  uploadImages(e.target.files);
  e.target.value = '';
});

// Browse folder
document.getElementById('btn-import-folder').addEventListener('click', () => {
  document.getElementById('ds-import-folder').click();
});
document.getElementById('ds-import-folder').addEventListener('change', (e) => {
  uploadImages(e.target.files);
  e.target.value = '';
});

// Split ratio slider (used during import)
document.getElementById('ds-split-ratio').addEventListener('input', (e) => {
  const v = e.target.value;
  document.getElementById('ds-ratio-val').textContent = `${v}/${100 - v}`;
});

// Bulk delete selected
document.getElementById('btn-ds-delete').addEventListener('click', () => {
  showConfirm(`Delete ${dsState.selected.size} image(s)?`, async () => {
    await fetch(API + `/api/dataset/${dsState.dataset}/delete-bulk/${dsState.split}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filenames: [...dsState.selected] })
    });
    dsState.selected.clear();
    loadImages();
    initDataset();
  });
});

// Delete entire split
document.getElementById('btn-ds-delete-split').addEventListener('click', () => {
  showConfirm(`Delete ALL images in "${dsState.dataset} / ${dsState.split}"? This cannot be undone.`, async () => {
    const res = await fetch(API + `/api/dataset/${dsState.dataset}/images?split=${dsState.split}&filter=all`);
    const images = await res.json();
    const filenames = images.map(i => i.filename);
    if (!filenames.length) return;
    await fetch(API + `/api/dataset/${dsState.dataset}/delete-bulk/${dsState.split}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filenames })
    });
    dsState.selected.clear();
    loadImages();
    initDataset();
  });
});

// ── Merge Datasets ───────────────────────────────

async function loadMergeCheckboxes() {
  try {
    const res = await fetch(API + '/api/dataset/list');
    const names = await res.json();
    const container = document.getElementById('merge-checkboxes');
    container.innerHTML = names.map(name =>
      `<label class="chip-checkbox">
        <input type="checkbox" class="merge-cb" value="${name}" /> ${name}
      </label>`
    ).join('');
  } catch {}
}

document.getElementById('btn-merge').addEventListener('click', async () => {
  const checked = [...document.querySelectorAll('.merge-cb:checked')].map(cb => cb.value);
  const target = document.getElementById('merge-name').value.trim();
  const status = document.getElementById('merge-status');

  if (checked.length < 2) { status.textContent = 'Select at least 2 datasets'; return; }
  if (!target) { status.textContent = 'Enter a name'; return; }

  status.textContent = 'Merging…';
  try {
    const res = await fetch(API + '/api/dataset/merge', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sources: checked, target })
    });
    const data = await res.json();
    if (data.success) {
      status.textContent = `✓ Merged ${data.total_images} images → "${data.dataset}"`;
      // Refresh dataset dropdown & merge checkboxes
      const dsSel = document.getElementById('ds-select');
      dsSel.innerHTML += `<option value="${data.dataset}">${data.dataset}</option>`;
      dsSel.value = data.dataset;
      dsState.dataset = data.dataset;
      initDataset();
      loadMergeCheckboxes();
    } else {
      status.textContent = data.error || 'Failed';
    }
  } catch {
    status.textContent = 'Merge failed';
  }
});

// ══════════════════════════════════════════════════
// TRAINING
// ══════════════════════════════════════════════════

async function initTraining() {
  // Load datasets into training dropdown
  try {
    const res = await fetch(API + '/api/dataset/list');
    const names = await res.json();
    const sel = document.getElementById('tr-dataset');
    sel.innerHTML = names.map(name =>
      `<option value="${name}">${name}</option>`
    ).join('');
    if (sel.value) loadTrainingClasses(sel.value);
  } catch {}

  await pollTraining();
}

async function loadTrainingClasses(dataset) {
  const el = document.getElementById('tr-classes');
  el.textContent = 'Scanning labels…';
  try {
    const res = await fetch(API + `/api/dataset/${dataset}/classes`);
    const data = await res.json();
    if (data.classes.length) {
      el.innerHTML = data.classes.map(c =>
        `<span style="display:inline-block;background:var(--surface);border:1px solid var(--tertiary);border-radius:var(--rounded-sm);padding:2px 8px;margin:2px">${c.id}: ${c.name}</span>`
      ).join(' ');
    } else {
      el.textContent = 'No labels found — annotate your dataset first';
    }
  } catch {
    el.textContent = 'Failed to scan';
  }
}

document.getElementById('tr-dataset').addEventListener('change', (e) => {
  loadTrainingClasses(e.target.value);
});

document.getElementById('btn-train-start').addEventListener('click', async () => {
  const cfg = {
    model: document.getElementById('tr-model').value,
    dataset: document.getElementById('tr-dataset').value,
    epochs: Number(document.getElementById('tr-epochs').value),
    batch: Number(document.getElementById('tr-batch').value),
    imgsz: Number(document.getElementById('tr-imgsz').value),
    device: document.getElementById('tr-device').value,
  };
  document.getElementById('tr-metrics-grid').style.display = 'none';
  document.getElementById('tr-trend-container').style.display = 'none';
  trLossTrend = [];
  trLastEpoch = 0;
  
  ['box', 'cls', 'dfl', 'map50', 'map50-95'].forEach(id => {
    document.getElementById('tr-metric-' + id).textContent = '—';
  });
  
  await fetch(API + '/api/training/start', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg)
  });
  startTrainingPolling();
});

document.getElementById('btn-train-stop').addEventListener('click', async () => {
  const res = await fetch(API + '/api/training/stop', { method: 'POST' });
  const data = await res.json();
  if (!data.success) {
    document.getElementById('training-progress').style.display = 'none';
  }
});

function startTrainingPolling() {
  pollTraining();
  trainingInterval = setInterval(pollTraining, 2000);
}

function stopTrainingPolling() {
  if (trainingInterval) clearInterval(trainingInterval);
  trainingInterval = null;
}

async function pollTraining() {
  try {
    const res = await fetch(API + '/api/training/status');
    const d = await res.json();

    const progressEl = document.getElementById('training-progress');

    if (d.is_training || d.current_epoch > 0 || d.logs.length > 0) {
      // Don't override display if it was hidden by the stop button and is no longer training
      if (d.is_training) {
        progressEl.style.display = 'block';
      } else if (progressEl.style.display !== 'none' && d.logs.length > 0) {
        progressEl.style.display = 'block';
      }

      document.getElementById('tr-epoch-label').textContent = `${d.current_epoch} / ${d.total_epochs}`;
      document.getElementById('tr-progress-bar').style.width = d.progress + '%';

      if (d.metrics && Object.keys(d.metrics).length > 0) {
        document.getElementById('tr-metrics-grid').style.display = 'grid';
        if (d.metrics.box_loss) document.getElementById('tr-metric-box').textContent = d.metrics.box_loss;
        if (d.metrics.cls_loss) document.getElementById('tr-metric-cls').textContent = d.metrics.cls_loss;
        if (d.metrics.dfl_loss) document.getElementById('tr-metric-dfl').textContent = d.metrics.dfl_loss;
        if (d.metrics.map50) document.getElementById('tr-metric-map50').textContent = d.metrics.map50;
        if (d.metrics.map50_95) document.getElementById('tr-metric-map50-95').textContent = d.metrics.map50_95;

        // Update Trend
        if (d.current_epoch > trLastEpoch && d.metrics.box_loss) {
          trLastEpoch = d.current_epoch;
          trLossTrend.push(parseFloat(d.metrics.box_loss));
          document.getElementById('tr-trend-container').style.display = 'block';
          drawLineChart('chart-tr-trend', trLossTrend, 'var(--tertiary)');
        }
      }

      const logEl = document.getElementById('tr-log');
      logEl.innerHTML = d.logs.map(l => {
        let cls = '';
        if (l.startsWith('[OK]')) cls = 'log-ok';
        else if (l.startsWith('[ERROR]')) cls = 'log-error';
        else if (l.startsWith('[WARN]')) cls = 'log-warn';
        return `<div class="${cls}">${escHtml(l)}</div>`;
      }).join('');
      logEl.scrollTop = logEl.scrollHeight;

      if (d.is_training && !trainingInterval) startTrainingPolling();
    }

    if (!d.is_training && trainingInterval) {
      stopTrainingPolling();
      if (d.new_model_available) {
        document.getElementById('tr-progress-bar').style.width = '100%';
      }
    }
  } catch {}
}

// ══════════════════════════════════════════════════
// MODELS
// ══════════════════════════════════════════════════

async function initModels() {
  try {
    const res = await fetch(API + '/api/models');
    const models = await res.json();

    const listEl = document.getElementById('models-list');
    if (!models.length) {
      listEl.innerHTML = '<div class="empty-state"><div class="icon">🧠</div><p>No trained models found.</p></div>';
      return;
    }

    listEl.innerHTML = models.map(m => {
      const metricsText = Object.entries(m.metrics).map(([k, v]) => `${k}: ${v}`).join(' · ');
      return `
        <div class="model-card">
          <div class="model-info">
            <span class="model-name">${m.name}</span>
            <span class="model-meta">${m.date} · ${m.size_mb} MB · ${m.epochs} epochs${metricsText ? ' · ' + metricsText : ''}</span>
          </div>
          <div class="model-actions">
            ${m.is_active
              ? '<span class="badge-active"><span class="dot dot-ok"></span> Active</span>'
              : `<button class="btn btn-outline btn-sm" onclick="activateModel('${m.name}')">Activate</button>`
            }
            <button class="btn-ghost" title="Delete" onclick="deleteModel('${m.name}')" ${m.is_active ? 'disabled' : ''}>🗑</button>
            <button class="btn-ghost" title="View curves" onclick="viewCurves('${m.name}')">📈</button>
          </div>
        </div>`;
    }).join('');
  } catch {}
}

window.activateModel = async function(name) {
  await fetch(API + '/api/models/' + name + '/activate', { method: 'POST' });
  initModels();
  loadMonitorModels();
};

window.deleteModel = function(name) {
  showConfirm(`Delete model "${name}" and all its weights?`, async () => {
    const res = await fetch(API + '/api/models/' + name, { method: 'DELETE' });
    const data = await res.json();
    if (!data.success) alert(data.error || 'Failed to delete.');
    initModels();
  });
};

window.viewCurves = async function(name) {
  try {
    const res = await fetch(API + '/api/models/' + name + '/curves');
    const data = await res.json();
    if (!data || !data.length) return;

    document.getElementById('curves-section').style.display = 'block';
    document.getElementById('curves-model-name').textContent = name;

    const mapData = data.map(d => d.mAP50).filter(v => v !== undefined);
    const lossData = data.map(d => d.box_loss).filter(v => v !== undefined);

    setTimeout(() => {
      drawLineChart('chart-map', mapData, '#3CBAB2');
      drawLineChart('chart-loss', lossData, '#8FA5B5');
    }, 100);
  } catch {}
};

// ══════════════════════════════════════════════════
// CONFIRM DIALOG
// ══════════════════════════════════════════════════

let confirmCallback = null;

function showConfirm(text, onOk) {
  document.getElementById('confirm-text').textContent = text;
  document.getElementById('confirm-dialog').style.display = 'flex';
  confirmCallback = onOk;
}

document.getElementById('confirm-cancel').addEventListener('click', () => {
  document.getElementById('confirm-dialog').style.display = 'none';
  confirmCallback = null;
});

document.getElementById('confirm-ok').addEventListener('click', () => {
  document.getElementById('confirm-dialog').style.display = 'none';
  if (confirmCallback) confirmCallback();
  confirmCallback = null;
});

// ══════════════════════════════════════════════════
// UTILS
// ══════════════════════════════════════════════════

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ══════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════

checkEnv();
