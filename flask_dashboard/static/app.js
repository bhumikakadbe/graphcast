/**
 * app.js — GraphCast Rainfall Prediction UI Logic
 * ================================================
 * - Animated star canvas background
 * - Tab switching (Forecast / Model Upgrader)
 * - Leaflet map selector for region bounds
 * - Checkpoint loader from /api/checkpoints
 * - Job submission, polling, results rendering
 * - Canvas-based precipitation map with time-step animation
 * - Tooltip on hover, stats cards, NetCDF download
 * - Model Upgrader: progressive training, loss chart, log, checkpoint manager
 */

// ─────────────────────────────────────────────────────────────────────────────
// 1. Stars Canvas
// ─────────────────────────────────────────────────────────────────────────────
(function initStars() {
  const canvas = document.getElementById("starsCanvas");
  const ctx    = canvas.getContext("2d");
  let stars    = [];
  let w, h;

  function resize() {
    w = canvas.width  = window.innerWidth;
    h = canvas.height = window.innerHeight;
    stars = Array.from({ length: 260 }, () => ({
      x:    Math.random() * w,
      y:    Math.random() * h,
      r:    Math.random() * 1.2 + 0.3,
      a:    Math.random(),
      da:   (Math.random() * 0.008 + 0.002) * (Math.random() < 0.5 ? 1 : -1),
      vx:   (Math.random() - 0.5) * 0.12,
      vy:   (Math.random() - 0.5) * 0.12,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    for (const s of stars) {
      s.a = Math.max(0.05, Math.min(1, s.a + s.da));
      if (s.a <= 0.05 || s.a >= 1) s.da *= -1;
      s.x = (s.x + s.vx + w) % w;
      s.y = (s.y + s.vy + h) % h;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(180, 210, 255, ${s.a})`;
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  resize();
  draw();
})();

// ─────────────────────────────────────────────────────────────────────────────
// 2. Tab Switching
// ─────────────────────────────────────────────────────────────────────────────
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));

  const activeTab = document.getElementById(tabId);
  if (activeTab) activeTab.classList.add('active');

  const btnId = tabId === 'forecast-tab' ? 'tab-btn-forecast' : 'tab-btn-upgrade';
  const btn = document.getElementById(btnId);
  if (btn) btn.classList.add('active');

  // After switching to forecast tab, resize leaflet map
  if (tabId === 'forecast-tab' && _leafletMap) {
    setTimeout(() => _leafletMap.invalidateSize(), 100);
  }

  // After switching to upgrade tab, redraw loss chart
  if (tabId === 'upgrade-tab') {
    setTimeout(() => drawLossChart(_lossHistory), 100);
  }

  window.dispatchEvent(new Event('resize'));
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Initialization
// ─────────────────────────────────────────────────────────────────────────────
let _leafletMap = null;
let _leafletMarker = null;
let _leafletRect = null;

document.addEventListener("DOMContentLoaded", () => {
  // Default forecast date = 5 days ago
  const d = new Date();
  d.setDate(d.getDate() - 5);
  const maxDateStr = d.toISOString().slice(0, 10);

  const dateInput = document.getElementById("forecastDate");
  dateInput.value = maxDateStr;
  dateInput.max = maxDateStr;

  loadCheckpoints();
  setupSlider();
  setupRegionToggle();
  setupTimeSlider();
  setupMapCanvas();
  setupLeafletMap();
  setupEpochSlider();
  buildTimelinePreview();
  fetchUpgradeCheckpoints();
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Leaflet Map Selector
// ─────────────────────────────────────────────────────────────────────────────
function setupLeafletMap() {
  _leafletMap = L.map('region-map-container', {
    center: [21.1458, 79.0882],
    zoom: 5,
    zoomControl: false,
    attributionControl: false,
  });

  // Dark themed tiles
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 18,
  }).addTo(_leafletMap);

  // Add zoom control bottom-right
  L.control.zoom({ position: 'bottomright' }).addTo(_leafletMap);

  // Marker for Nagpur default
  _leafletMarker = L.marker([21.1458, 79.0882]).addTo(_leafletMap);

  // Draw rectangle for bounding box
  _leafletRect = L.rectangle([[20.0, 78.0], [22.0, 80.0]], {
    color: '#60a5fa',
    weight: 2,
    fillOpacity: 0.12,
    dashArray: '6 4',
  }).addTo(_leafletMap);

  // Click on map to update coordinates (only for Custom mode)
  _leafletMap.on('click', function(e) {
    const isCustom = document.getElementById('regionCustom').checked;
    if (!isCustom) return;

    const lat = e.latlng.lat;
    const lon = e.latlng.lng;

    // Set as center of a 2° bounding box
    document.getElementById('latMin').value = (lat - 1).toFixed(1);
    document.getElementById('latMax').value = (lat + 1).toFixed(1);
    document.getElementById('lonMin').value = (lon - 1).toFixed(1);
    document.getElementById('lonMax').value = (lon + 1).toFixed(1);

    updateLeafletFromInputs();
  });

  // Sync marker when inputs change
  ['latMin', 'latMax', 'lonMin', 'lonMax'].forEach(id => {
    document.getElementById(id).addEventListener('change', updateLeafletFromInputs);
  });
}

function updateLeafletFromInputs() {
  const latMin = parseFloat(document.getElementById('latMin').value);
  const latMax = parseFloat(document.getElementById('latMax').value);
  const lonMin = parseFloat(document.getElementById('lonMin').value);
  const lonMax = parseFloat(document.getElementById('lonMax').value);

  if (isNaN(latMin) || isNaN(latMax) || isNaN(lonMin) || isNaN(lonMax)) return;

  const centerLat = (latMin + latMax) / 2;
  const centerLon = (lonMin + lonMax) / 2;

  _leafletMarker.setLatLng([centerLat, centerLon]);
  _leafletRect.setBounds([[latMin, lonMin], [latMax, lonMax]]);
  _leafletMap.fitBounds([[latMin, lonMin], [latMax, lonMax]], { padding: [20, 20] });
}

function setRegionPreset(lat, lon, zoom, label) {
  _leafletMarker.setLatLng([lat, lon]);
  _leafletMap.setView([lat, lon], zoom);

  // Update rect for Nagpur preset
  if (label === 'nagpur') {
    _leafletRect.setBounds([[20.0, 78.0], [22.0, 80.0]]);
  } else if (label === 'global') {
    _leafletRect.setBounds([[-85, -180], [85, 180]]);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Checkpoint Loader
// ─────────────────────────────────────────────────────────────────────────────
async function loadCheckpoints() {
  const sel  = document.getElementById("checkpoint");
  const hint = document.getElementById("checkpointHint");
  try {
    const res  = await fetch("/api/checkpoints");
    const data = await res.json();
    const list = data.checkpoints || [];
    sel.innerHTML = list
      .map(c => `<option value="${c}">${c.replace(".npz", "").replace(/GraphCast(?:_[\w]+)?\s*-\s*/i, "")}</option>`)
      .join("");

    const small = list.find(c => c.toLowerCase().includes("small"));
    if (small) sel.value = small;

    hint.textContent = `${list.length} checkpoint(s) available from GCS`;
  } catch (err) {
    sel.innerHTML = `<option value="">⚠ Could not fetch checkpoints</option>`;
    hint.textContent = "GCS unreachable — enter a checkpoint name manually";
    console.warn("Checkpoint fetch failed:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. UI Controls Setup
// ─────────────────────────────────────────────────────────────────────────────
function setupSlider() {
  const slider = document.getElementById("forecastSteps");
  const label  = document.getElementById("stepsLabel");
  function update() {
    const v = parseInt(slider.value);
    label.textContent = `${v} step${v > 1 ? "s" : ""} (${v * 6}h)`;
  }
  slider.addEventListener("input", update);
  update();
}

function setupEpochSlider() {
  const slider = document.getElementById("epochsPerYear");
  const label  = document.getElementById("epochLabel");
  if (!slider || !label) return;
  function update() {
    const v = parseInt(slider.value);
    label.textContent = `${v} epoch${v > 1 ? "s" : ""}`;
  }
  slider.addEventListener("input", update);
  update();
}

function setupRegionToggle() {
  document.querySelectorAll('input[name="region"]').forEach(r => {
    r.addEventListener("change", () => {
      const isCustom = document.getElementById("regionCustom").checked;
      const isGlobal = document.getElementById("regionGlobal").checked;
      document.getElementById("customBounds").classList.toggle("hidden", !isCustom);

      if (isGlobal) {
        setRegionPreset(20.0, 0.0, 2, 'global');
      } else if (!isCustom) {
        setRegionPreset(21.1458, 79.0882, 5, 'nagpur');
      } else {
        updateLeafletFromInputs();
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. State
// ─────────────────────────────────────────────────────────────────────────────
let _jobId       = null;
let _pollTimer   = null;
let _result      = null;
let _currentStep = 0;
let _playing     = true;
let _playTimer   = null;

// ─────────────────────────────────────────────────────────────────────────────
// 8. Start Forecast
// ─────────────────────────────────────────────────────────────────────────────
async function startForecast() {
  const btn  = document.getElementById("runBtn");
  btn.disabled = true;

  // Enforce ERA5 5-day availability constraint
  const dateStr = document.getElementById("forecastDate").value;
  const selectedDate = new Date(dateStr);
  const maxAllowed = new Date();
  maxAllowed.setDate(maxAllowed.getDate() - 5);

  selectedDate.setHours(0,0,0,0);
  maxAllowed.setHours(0,0,0,0);

  if (selectedDate > maxAllowed) {
    showStatus(`❌ Selected date must be on or before ${maxAllowed.toISOString().slice(0, 10)} (ERA5 has a 5-day release delay).`, true);
    btn.disabled = false;
    return;
  }

  const isCustom = document.getElementById("regionCustom").checked;
  const isGlobal = document.getElementById("regionGlobal").checked;
  const body = {
    date:            document.getElementById("forecastDate").value,
    n_steps:         parseInt(document.getElementById("forecastSteps").value),
    resolution:      parseFloat(document.getElementById("resolution").value),
    pressure_levels: parseInt(document.getElementById("pressureLevels").value),
    checkpoint:      document.getElementById("checkpoint").value,
    is_global:       isGlobal,
    lat_min:         isCustom ? parseFloat(document.getElementById("latMin").value) : 20.0,
    lat_max:         isCustom ? parseFloat(document.getElementById("latMax").value) : 22.0,
    lon_min:         isCustom ? parseFloat(document.getElementById("lonMin").value) : 78.0,
    lon_max:         isCustom ? parseFloat(document.getElementById("lonMax").value) : 80.0,
  };

  let regionLabel = "Nagpur Region";
  if (isGlobal) regionLabel = "Global Forecast";
  else if (isCustom) regionLabel = `${body.lat_min}–${body.lat_max}°N, ${body.lon_min}–${body.lon_max}°E`;

  document.getElementById("vizRegionLabel").textContent = regionLabel;
  showStatus("⏳ Submitting job…", false);

  try {
    const res  = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Submission failed");

    _jobId = data.job_id;
    console.log("Job started:", _jobId);
    startPolling();
  } catch (err) {
    showStatus(`❌ ${err.message}`, true);
    btn.disabled = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Polling
// ─────────────────────────────────────────────────────────────────────────────
function startPolling() {
  if (_pollTimer) clearInterval(_pollTimer);
  _pollTimer = setInterval(pollStatus, 3000);
  pollStatus();
}

async function pollStatus() {
  if (!_jobId) return;
  try {
    const res  = await fetch(`/api/status/${_jobId}`);
    const data = await res.json();
    showStatus(data.progress || data.status, data.status === "error");
    if (data.status === "done") {
      clearInterval(_pollTimer);
      document.getElementById("runBtn").disabled = false;
      handleResult(data.result);
    } else if (data.status === "error") {
      clearInterval(_pollTimer);
      document.getElementById("runBtn").disabled = false;
    }
  } catch (err) {
    console.warn("Poll error:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Status Bar
// ─────────────────────────────────────────────────────────────────────────────
function showStatus(msg, isError) {
  const bar  = document.getElementById("statusBar");
  const dot  = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  const spin = document.getElementById("statusSpinner");

  bar.classList.remove("hidden");
  text.textContent = msg;
  if (isError) {
    dot.classList.add("error");
    spin.classList.add("hidden");
  } else {
    dot.classList.remove("error");
    spin.classList.toggle("hidden", msg.startsWith("✅"));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Results Rendering
// ─────────────────────────────────────────────────────────────────────────────
function handleResult(result) {
  _result      = result;
  _currentStep = 0;
  _playing     = true;
  document.getElementById("idleState").classList.add("hidden");
  document.getElementById("resultsState").classList.remove("hidden");
  buildTimePills(result.times);
  buildStatsCards(result);
  updateColorbarLabels(result.data);
  renderMapStep(0);
  startPlayback();
  renderInsights();
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Time Pills
// ─────────────────────────────────────────────────────────────────────────────
function buildTimePills(times) {
  const container = document.getElementById("timeSteps");
  container.innerHTML = times
    .map((t, i) => `<div class="step-pill${i === 0 ? " active" : ""}"
         id="pill-${i}" onclick="goToStep(${i})">+${t}</div>`)
    .join("");
  const slider = document.getElementById("timeSlider");
  slider.max = times.length - 1;
  slider.value = 0;
}

function goToStep(idx) {
  _currentStep = idx;
  renderMapStep(idx);
  updateTimePills(idx);
  document.getElementById("timeSlider").value = idx;
  updateTimeLabel(idx);
}

function updateTimePills(idx) {
  document.querySelectorAll(".step-pill").forEach((p, i) => {
    p.classList.toggle("active", i === idx);
  });
}

function updateTimeLabel(idx) {
  const t = _result ? _result.times[idx] : "";
  document.getElementById("currentTimeLabel").textContent = `Step ${idx + 1} — +${t}`;
}

function setupTimeSlider() {
  const slider = document.getElementById("timeSlider");
  slider.addEventListener("input", () => {
    const idx = parseInt(slider.value);
    _currentStep = idx;
    renderMapStep(idx);
    updateTimePills(idx);
    updateTimeLabel(idx);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Playback
// ─────────────────────────────────────────────────────────────────────────────
function startPlayback() {
  if (_playTimer) clearInterval(_playTimer);
  _playing = true;
  document.getElementById("playPauseBtn").textContent = "⏸ Pause";
  _playTimer = setInterval(() => {
    if (!_result) return;
    _currentStep = (_currentStep + 1) % _result.times.length;
    goToStep(_currentStep);
  }, 900);
}

function stopPlayback() {
  clearInterval(_playTimer);
  _playing = false;
  document.getElementById("playPauseBtn").textContent = "▶ Play";
}

function togglePlayback() {
  _playing ? stopPlayback() : startPlayback();
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. Canvas Map Renderer
// ─────────────────────────────────────────────────────────────────────────────
let _mapCtx    = null;
let _mapCanvas = null;

function setupMapCanvas() {
  _mapCanvas = document.getElementById("mapCanvas");
  _mapCtx    = _mapCanvas.getContext("2d");
  _mapCanvas.addEventListener("mousemove", onMapHover);
  _mapCanvas.addEventListener("mouseleave", () => {
    document.getElementById("mapTooltip").classList.add("hidden");
  });
}

function precipToColor(mm, maxMm) {
  const t = Math.min(mm / Math.max(maxMm, 1), 1);
  const stops = [
    [0,   [13,  71, 161]],
    [0.1, [21, 101, 192]],
    [0.2, [25, 118, 210]],
    [0.3, [66, 165, 245]],
    [0.4, [128, 222, 234]],
    [0.5, [165, 214, 167]],
    [0.65,[255, 241, 118]],
    [0.78,[255, 183,  77]],
    [0.9, [239,  83,  80]],
    [1.0, [183,  28,  28]],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + f * (c1[0] - c0[0])),
        Math.round(c0[1] + f * (c1[1] - c0[1])),
        Math.round(c0[2] + f * (c1[2] - c0[2])),
      ];
    }
  }
  return [183, 28, 28];
}

function renderMapStep(stepIdx) {
  if (!_result || !_mapCtx) return;
  const data = _result.data[stepIdx];
  if (!data || !data.length) return;
  const nLat = data.length;
  const nLon = data[0].length;
  const rect = _mapCanvas.getBoundingClientRect();
  _mapCanvas.width  = rect.width  * devicePixelRatio;
  _mapCanvas.height = rect.height * devicePixelRatio;
  const cw = _mapCanvas.width;
  const ch = _mapCanvas.height;

  let maxVal = 0;
  for (const row of data) for (const v of row) if (v > maxVal) maxVal = v;

  const imgData = _mapCtx.createImageData(cw, ch);
  const pixels  = imgData.data;
  for (let r = 0; r < ch; r++) {
    const latIdx = Math.floor((r / ch) * nLat);
    for (let c = 0; c < cw; c++) {
      const lonIdx = Math.floor((c / cw) * nLon);
      const mm = (data[latIdx] && data[latIdx][lonIdx]) || 0;
      const [R, G, B] = precipToColor(mm, maxVal || 1);
      const idx = (r * cw + c) * 4;
      pixels[idx]     = R;
      pixels[idx + 1] = G;
      pixels[idx + 2] = B;
      pixels[idx + 3] = 230;
    }
  }
  _mapCtx.putImageData(imgData, 0, 0);

  _mapCtx.strokeStyle = "rgba(255,255,255,0.07)";
  _mapCtx.lineWidth   = 1;
  for (let r = 0; r < nLat; r++) {
    const y = (r / nLat) * ch;
    _mapCtx.beginPath(); _mapCtx.moveTo(0, y); _mapCtx.lineTo(cw, y); _mapCtx.stroke();
  }
  for (let c = 0; c < nLon; c++) {
    const x = (c / nLon) * cw;
    _mapCtx.beginPath(); _mapCtx.moveTo(x, 0); _mapCtx.lineTo(x, ch); _mapCtx.stroke();
  }
  buildAxisLabels(_result.lats, _result.lons);
  updateTimePills(stepIdx);
  updateColorbarLabels([[maxVal]]);
}

function buildAxisLabels(lats, lons) {
  const latDiv = document.getElementById("latLabels");
  const lonDiv = document.getElementById("lonLabels");
  latDiv.innerHTML = [...lats].reverse()
    .map(l => `<span>${l.toFixed(1)}°N</span>`).join("");
  lonDiv.innerHTML = lons
    .map(l => `<span>${l.toFixed(1)}°E</span>`).join("");
}

function updateColorbarLabels(data) {
  let maxVal = 0;
  if (Array.isArray(data)) {
    for (const step of data) {
      if (!Array.isArray(step)) continue;
      for (const row of step) {
        if (!Array.isArray(row)) continue;
        for (const v of row) if (v > maxVal) maxVal = v;
      }
    }
  }
  if (!isNaN(data?.[0]?.[0])) maxVal = data[0][0];
  maxVal = Math.max(maxVal, 1);
  document.getElementById("cbMax").textContent = `${maxVal.toFixed(1)} mm`;
  document.getElementById("cbMid").textContent = `${(maxVal / 2).toFixed(1)} mm`;
  document.getElementById("cbMin").textContent = `0 mm`;
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. Map Tooltip
// ─────────────────────────────────────────────────────────────────────────────
function onMapHover(evt) {
  if (!_result) return;
  const rect  = _mapCanvas.getBoundingClientRect();
  const px    = (evt.clientX - rect.left) / rect.width;
  const py    = (evt.clientY - rect.top)  / rect.height;
  const data  = _result.data[_currentStep];
  if (!data) return;
  const nLat  = data.length;
  const nLon  = data[0].length;
  const latIdx = Math.min(Math.floor(py * nLat), nLat - 1);
  const lonIdx = Math.min(Math.floor(px * nLon), nLon - 1);
  const mm   = data[latIdx]?.[lonIdx] ?? 0;
  const lat  = _result.lats[latIdx];
  const lon  = _result.lons[lonIdx];
  const tooltip = document.getElementById("mapTooltip");
  tooltip.textContent = `${lat?.toFixed(1)}°N, ${lon?.toFixed(1)}°E — ${mm.toFixed(2)} mm`;
  tooltip.style.left  = `${evt.clientX - rect.left + 12}px`;
  tooltip.style.top   = `${evt.clientY - rect.top  - 20}px`;
  tooltip.classList.remove("hidden");
}

// ─────────────────────────────────────────────────────────────────────────────
// 16. Stats Cards
// ─────────────────────────────────────────────────────────────────────────────
function buildStatsCards(result) {
  const allVals = result.data.flat(2).filter(v => !isNaN(v));
  const maxV    = Math.max(...allVals);
  const mean    = allVals.reduce((a, b) => a + b, 0) / allVals.length;
  const steps   = result.times.length;
  const cards = [
    { label: "Peak Precipitation", value: maxV.toFixed(2), unit: "mm / 6h" },
    { label: "Mean (Region)",       value: mean.toFixed(3), unit: "mm / 6h" },
    { label: "Forecast Steps",      value: steps,            unit: "× 6h"   },
    { label: "Grid Points",
      value: result.lats.length * result.lons.length,
      unit: "cells" },
  ];
  document.getElementById("statsRow").innerHTML = cards
    .map(c => `
      <div class="stat-card">
        <div class="stat-label">${c.label}</div>
        <div class="stat-value">${c.value}</div>
        <div class="stat-unit">${c.unit}</div>
      </div>`).join("");
}

// ─────────────────────────────────────────────────────────────────────────────
// 17. Download NetCDF
// ─────────────────────────────────────────────────────────────────────────────
function downloadNetCDF() {
  if (!_jobId) return;
  window.location.href = `/api/download/${_jobId}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// 18. Written Insights
// ─────────────────────────────────────────────────────────────────────────────
function renderInsights() {
  const panel = document.getElementById("insightsPanel");
  const content = document.getElementById("insightsContent");
  if (!_result || !_result.data) return;
  panel.classList.remove("hidden");
  const baseDate = document.getElementById("forecastDate").value;
  const regionText = document.getElementById("vizRegionLabel").textContent;
  const nSteps = parseInt(document.getElementById("forecastSteps").value);
  let maxPrecip = 0, maxLat = 0, maxLon = 0, maxTimeIdx = 0, totalPrecip = 0, pointCount = 0;
  for (let t = 0; t < _result.data.length; t++) {
    const stepData = _result.data[t];
    for (let i = 0; i < stepData.length; i++) {
      for (let j = 0; j < stepData[i].length; j++) {
        const val = stepData[i][j];
        if (val > maxPrecip) { maxPrecip = val; maxLat = _result.lats[i]; maxLon = _result.lons[j]; maxTimeIdx = t; }
        totalPrecip += val; pointCount++;
      }
    }
  }
  const avgPrecip = (totalPrecip / Math.max(1, pointCount)).toFixed(2);
  const timeOffset = _result.times[maxTimeIdx] || "+0:00:00";
  content.innerHTML = `
    <div class="insight-summary">
      <p>📝 <span>Forecast Base Date</span> <strong>${baseDate}</strong></p>
      <p>🔍 <span>Region Analyzed</span> <strong>${regionText}</strong></p>
      <p>⏱️ <span>Timeline Projected</span> <strong>${nSteps * 6} Hours into the future</strong></p>
      <p>🌧️ <span>Maximum Rainfall Event</span> <strong style="color: #f87171;">${maxPrecip.toFixed(1)} mm</strong></p>
      <p>📍 <span>Peak Location Coordinates</span> <strong>${maxLat.toFixed(2)}°N, ${maxLon.toFixed(2)}°E</strong></p>
      <p>⏰ <span>Estimated Peak Timing</span> <strong>Arriving at ${timeOffset}</strong></p>
      <p>🌍 <span>Average Region Precipitation</span> <strong>${avgPrecip} mm</strong></p>
    </div>`;
}

// ═════════════════════════════════════════════════════════════════════════════
// MODEL UPGRADER
// ═════════════════════════════════════════════════════════════════════════════

let _lossHistory = [];
let _upgradePolling = null;

// ─────────────────────────────────────────────────────────────────────────────
// 19. Timeline Preview
// ─────────────────────────────────────────────────────────────────────────────
function buildTimelinePreview() {
  const startYear = parseInt(document.getElementById('startYear').value);
  const endYear   = parseInt(document.getElementById('endYear').value);
  const container = document.getElementById('timelineContainer');

  if (endYear <= startYear) {
    container.innerHTML = `<div class="timeline-node">
      <div class="timeline-dot pending">!</div>
      <span class="timeline-year">End year must be after start year</span>
    </div>`;
    return;
  }

  let html = '';
  for (let y = startYear; y <= endYear; y++) {
    html += `<div class="timeline-node">
      <div class="timeline-dot pending" id="tl-dot-${y}">—</div>
      <span class="timeline-year" id="tl-year-${y}">${y}</span>
    </div>`;
  }
  container.innerHTML = html;
}

// Re-build timeline whenever years change
document.addEventListener('DOMContentLoaded', () => {
  ['startYear', 'endYear'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', buildTimelinePreview);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 20. Start Upgrade
// ─────────────────────────────────────────────────────────────────────────────
async function startUpgrade() {
  const btn = document.getElementById('upgradeBtn');
  btn.disabled = true;

  const startYear   = parseInt(document.getElementById('startYear').value);
  const endYear     = parseInt(document.getElementById('endYear').value);
  const epochs      = parseInt(document.getElementById('epochsPerYear').value);
  const lr          = parseFloat(document.getElementById('learningRate').value);
  const useSimulated = document.getElementById('dataSimulated').checked;

  if (endYear <= startYear) {
    addUpgradeLog('End year must be after start year', 'error');
    btn.disabled = false;
    return;
  }

  // Update status
  const pill = document.getElementById('upgradeStatusPill');
  pill.className = 'upgrade-status-pill running';
  pill.textContent = '● Training…';

  // Build timeline with pending dots
  buildTimelinePreview();

  // Clear loss history and log
  _lossHistory = [];
  drawLossChart([]);
  document.getElementById('upgradeLog').innerHTML = '';
  addUpgradeLog(`Starting progressive training: ${startYear} → ${endYear}`, 'info');
  addUpgradeLog(`Epochs/year: ${epochs} | LR: ${lr} | Data: ${useSimulated ? 'Simulated' : 'CDS Real'}`, 'info');

  try {
    const res = await fetch('/api/upgrade/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start_year: startYear,
        end_year: endYear,
        epochs_per_year: epochs,
        learning_rate: lr,
        use_simulated: useSimulated,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || data.detail || 'Failed to start');

    addUpgradeLog(`Job accepted: ${data.message || 'Training started'}`, 'info');
    startUpgradePolling();
  } catch (err) {
    addUpgradeLog(`Error: ${err.message}`, 'error');
    pill.className = 'upgrade-status-pill idle';
    pill.textContent = '● Error';
    btn.disabled = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 21. Upgrade Polling
// ─────────────────────────────────────────────────────────────────────────────
function startUpgradePolling() {
  if (_upgradePolling) clearInterval(_upgradePolling);
  _upgradePolling = setInterval(pollUpgradeStatus, 2000);
  pollUpgradeStatus();
}

async function pollUpgradeStatus() {
  try {
    const res = await fetch('/api/upgrade/status');
    const data = await res.json();

    // Update timeline dots
    if (data.completed_years) {
      data.completed_years.forEach(y => {
        const dot = document.getElementById(`tl-dot-${y}`);
        const yearEl = document.getElementById(`tl-year-${y}`);
        if (dot) { dot.className = 'timeline-dot done'; dot.textContent = '✓'; }
        if (yearEl) yearEl.className = 'timeline-year done';
      });
    }
    if (data.current_year) {
      const dot = document.getElementById(`tl-dot-${data.current_year}`);
      const yearEl = document.getElementById(`tl-year-${data.current_year}`);
      if (dot && !dot.classList.contains('done')) {
        dot.className = 'timeline-dot active';
        dot.textContent = '⟳';
      }
      if (yearEl && !yearEl.classList.contains('done')) yearEl.className = 'timeline-year active';
    }

    // Update loss chart
    if (data.loss_history && data.loss_history.length > 0) {
      _lossHistory = data.loss_history;
      drawLossChart(_lossHistory);
      const chartStatus = document.getElementById('lossChartStatus');
      chartStatus.className = 'upgrade-status-pill running';
      chartStatus.textContent = `${_lossHistory.length} points`;
    }

    // Update log
    if (data.log_lines) {
      data.log_lines.forEach(line => {
        if (!document.querySelector(`[data-log-key="${line}"]`)) {
          addUpgradeLog(line, 'info', line);
        }
      });
    }

    // Check done/error
    const pill = document.getElementById('upgradeStatusPill');
    if (data.status === 'done') {
      clearInterval(_upgradePolling);
      pill.className = 'upgrade-status-pill done';
      pill.textContent = '● Complete';
      document.getElementById('upgradeBtn').disabled = false;
      addUpgradeLog('✅ Progressive training complete!', 'info');
      const chartStatus = document.getElementById('lossChartStatus');
      chartStatus.className = 'upgrade-status-pill done';
      chartStatus.textContent = 'Done';
      fetchUpgradeCheckpoints();
    } else if (data.status === 'error') {
      clearInterval(_upgradePolling);
      pill.className = 'upgrade-status-pill idle';
      pill.textContent = '● Error';
      document.getElementById('upgradeBtn').disabled = false;
      addUpgradeLog(`❌ ${data.error || 'Training failed'}`, 'error');
    }
  } catch (err) {
    console.warn('Upgrade poll error:', err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 22. Loss Chart (Canvas)
// ─────────────────────────────────────────────────────────────────────────────
function drawLossChart(history) {
  const canvas = document.getElementById('lossCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();

  if (rect.width === 0) return;

  canvas.width  = rect.width * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;
  const W = canvas.width;
  const H = canvas.height;
  const pad = { top: 20, right: 20, bottom: 30, left: 50 };

  ctx.clearRect(0, 0, W, H);

  // Background grid
  ctx.strokeStyle = 'rgba(96,165,250,0.08)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * (H - pad.top - pad.bottom);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
  }

  if (!history || history.length < 2) {
    ctx.fillStyle = 'rgba(148,163,184,0.5)';
    ctx.font = `${12 * devicePixelRatio}px Inter, sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText('No loss data yet', W / 2, H / 2);
    return;
  }

  const maxLoss = Math.max(...history);
  const minLoss = Math.min(...history);
  const range = maxLoss - minLoss || 1;

  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;

  // Gradient fill
  const grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
  grad.addColorStop(0, 'rgba(96,165,250,0.25)');
  grad.addColorStop(1, 'rgba(96,165,250,0.02)');

  ctx.beginPath();
  ctx.moveTo(pad.left, H - pad.bottom);
  history.forEach((v, i) => {
    const x = pad.left + (i / (history.length - 1)) * chartW;
    const y = pad.top + (1 - (v - minLoss) / range) * chartH;
    ctx.lineTo(x, y);
  });
  ctx.lineTo(pad.left + chartW, H - pad.bottom);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  history.forEach((v, i) => {
    const x = pad.left + (i / (history.length - 1)) * chartW;
    const y = pad.top + (1 - (v - minLoss) / range) * chartH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#60a5fa';
  ctx.lineWidth = 2 * devicePixelRatio;
  ctx.stroke();

  // Dot on last point
  const lastX = pad.left + chartW;
  const lastY = pad.top + (1 - (history[history.length - 1] - minLoss) / range) * chartH;
  ctx.beginPath();
  ctx.arc(lastX, lastY, 4 * devicePixelRatio, 0, Math.PI * 2);
  ctx.fillStyle = '#60a5fa';
  ctx.fill();

  // Y-axis labels
  ctx.fillStyle = 'rgba(148,163,184,0.7)';
  ctx.font = `${10 * devicePixelRatio}px Inter, sans-serif`;
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = maxLoss - (i / 4) * range;
    const y = pad.top + (i / 4) * chartH;
    ctx.fillText(val.toFixed(3), pad.left - 8, y + 4);
  }

  // X-axis label
  ctx.textAlign = 'center';
  ctx.fillText('Epoch', W / 2, H - 4);
}

// ─────────────────────────────────────────────────────────────────────────────
// 23. Log Console Helper
// ─────────────────────────────────────────────────────────────────────────────
function addUpgradeLog(message, level = 'info', key = null) {
  const logEl = document.getElementById('upgradeLog');
  const now = new Date();
  const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
  const levelClass = level === 'error' ? 'error' : level === 'warn' ? 'warn' : '';

  const line = document.createElement('div');
  line.className = 'log-line';
  if (key) line.dataset.logKey = key;
  line.innerHTML = `<span class="log-time">[${timeStr}]</span><span class="log-msg ${levelClass}">${message}</span>`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

// ─────────────────────────────────────────────────────────────────────────────
// 24. Checkpoint Manager
// ─────────────────────────────────────────────────────────────────────────────
async function fetchUpgradeCheckpoints() {
  try {
    const res = await fetch('/api/upgrade/checkpoints');
    const data = await res.json();
    const body = document.getElementById('ckptTableBody');

    if (!data.checkpoints || data.checkpoints.length === 0) {
      body.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted); padding:24px;">
        No checkpoints yet. Run progressive training to generate models.
      </td></tr>`;
      return;
    }

    body.innerHTML = data.checkpoints.map(ck => `
      <tr>
        <td><strong>${ck.year}</strong></td>
        <td style="font-family:monospace; font-size:0.75rem;">${ck.filename}</td>
        <td>${ck.size || '—'}</td>
        <td><span class="ckpt-badge ${ck.active ? 'active' : 'idle'}">${ck.active ? '● Active' : '○ Idle'}</span></td>
        <td>
          <button class="ckpt-action-btn promote" onclick="promoteCheckpoint('${ck.filename}')">⬆ Promote</button>
          <button class="ckpt-action-btn delete" onclick="deleteCheckpoint('${ck.filename}')">🗑 Delete</button>
        </td>
      </tr>`).join('');
  } catch (err) {
    console.warn('Checkpoint fetch failed:', err);
  }
}

async function promoteCheckpoint(filename) {
  try {
    const res = await fetch('/api/upgrade/promote', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename }),
    });
    const data = await res.json();
    addUpgradeLog(`Promoted: ${filename}`, 'info');
    fetchUpgradeCheckpoints();
  } catch (err) {
    addUpgradeLog(`Promote failed: ${err.message}`, 'error');
  }
}

async function deleteCheckpoint(filename) {
  if (!confirm(`Delete checkpoint ${filename}?`)) return;
  try {
    const res = await fetch('/api/upgrade/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename }),
    });
    const data = await res.json();
    addUpgradeLog(`Deleted: ${filename}`, 'warn');
    fetchUpgradeCheckpoints();
  } catch (err) {
    addUpgradeLog(`Delete failed: ${err.message}`, 'error');
  }
}
