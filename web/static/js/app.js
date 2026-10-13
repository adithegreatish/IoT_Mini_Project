// Smart Parking & HSRP Technical Dashboard Controller

const state = {
  activeTab: 'overview',
  camera: {},
  slots: [],
  occupancyEvents: [],
  plateEvents: [],
  maskPlates: true,
  calibrationMode: false,
  activeSlotId: null,
  currentPolygonPoints: [],
  referenceSize: { width: 1280, height: 720 },
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initPolling();
  initStreamControls();
  initCalibrationCanvas();
  initDiagnostics();
  initAllocationForm();
  initQuickAllotmentForm();
});

// --- Navigation ---
function initNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.target;
      switchTab(target);
    });
  });
}

function switchTab(targetTabId) {
  state.activeTab = targetTabId;
  document.querySelectorAll('.nav-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.target === targetTabId);
  });
  document.querySelectorAll('.tab-pane').forEach(p => {
    p.classList.toggle('active', p.id === `pane-${targetTabId}`);
  });

  // Hardware stream optimization: Only activate MJPEG stream on the visible video element
  const overviewFeed = document.getElementById('overview-video-feed');
  const monitorFeed = document.getElementById('monitor-video-feed');

  if (targetTabId === 'overview') {
    if (overviewFeed && (!overviewFeed.src || !overviewFeed.src.includes('/api/stream/video_feed'))) {
      overviewFeed.src = '/api/stream/video_feed';
    }
    if (monitorFeed) monitorFeed.src = '';
  } else if (targetTabId === 'monitor') {
    if (monitorFeed && (!monitorFeed.src || !monitorFeed.src.includes('/api/stream/video_feed'))) {
      monitorFeed.src = '/api/stream/video_feed';
    }
    if (overviewFeed) overviewFeed.src = '';
  } else {
    // Non-video tabs: stop stream completely to release 100% video decoding CPU on Raspberry Pi
    if (overviewFeed) overviewFeed.src = '';
    if (monitorFeed) monitorFeed.src = '';
  }

  if (targetTabId === 'calibration') {
    renderCalibrationCanvas();
  } else if (targetTabId === 'allocation' || targetTabId === 'overview') {
    fetchAvailability(true);
  } else if (targetTabId === 'events') {
    fetchEvents();
  } else if (targetTabId === 'models') {
    fetchModels();
  } else if (targetTabId === 'camera-settings') {
    fetchCameraDevices();
  }
}

// --- Polling & State Sync ---
function initPolling() {
  fetchHealth();
  fetchSlots();
  fetchAvailability(false);

  // Throttled 3.0s polling to prevent CPU spikes and keep browser snappy
  setInterval(() => {
    if (!document.hidden) {
      fetchHealth();
    }
  }, 3000);

  setInterval(() => {
    if (!document.hidden) {
      fetchSlots();
      fetchAvailability(false);
    }
  }, 3000);
}

async function fetchHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    state.camera = data.camera || {};

    // Update Header Badges
    const badge = document.getElementById('camera-status-badge');
    const text = document.getElementById('camera-status-text');
    if (badge && text) {
      text.textContent = state.camera.state || 'DISCONNECTED';
      badge.className = `badge ${state.camera.is_stale ? 'badge-stale' : (state.camera.is_mock ? 'badge-mock' : 'badge-live')}`;
    }

    // Update Metric Displays
    const fpsElem = document.getElementById('metric-fps');
    if (fpsElem) fpsElem.textContent = `${state.camera.observed_fps || 0.0} FPS`;

    const resElem = document.getElementById('metric-res');
    if (resElem) resElem.textContent = state.camera.negotiated_resolution || '0x0';

    const uptimeElem = document.getElementById('metric-uptime');
    if (uptimeElem) {
      const sec = data.uptime_sec || 0;
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      uptimeElem.textContent = `${m}m ${s}s`;
    }

    // Update Pause Preview Button state
    const pauseBtn = document.getElementById('btn-pause-preview');
    if (pauseBtn) {
      pauseBtn.textContent = state.camera.preview_paused ? 'Resume Preview' : 'Pause Preview';
    }

    const procBtn = document.getElementById('btn-toggle-proc');
    if (procBtn) {
      procBtn.textContent = state.camera.processing_active ? 'Stop Processing' : 'Start Processing';
      procBtn.className = state.camera.processing_active ? 'btn btn-danger' : 'btn btn-primary';
    }
  } catch (err) {
    console.debug('Health poll failed:', err);
  }
}

async function fetchSlots() {
  try {
    const res = await fetch('/api/slots');
    const data = await res.json();
    state.slots = data.slots || [];
    state.referenceSize = {
      width: data.reference_width || 1280,
      height: data.reference_height || 720,
    };
    updateOverviewCounters();
    if (state.activeTab === 'calibration') {
      renderCalibrationList();
      renderCalibrationCanvas();
    }
  } catch (err) {
    console.debug('Slot fetch failed:', err);
  }
}

function updateOverviewCounters() {
  const total = state.slots.length;
  let occupied = 0;
  let free = 0;
  let unknown = 0;

  state.slots.forEach(s => {
    if (s.state === 'OCCUPIED') occupied++;
    else if (s.state === 'FREE') free++;
    else unknown++;
  });

  const elTotal = document.getElementById('counter-total');
  const elFree = document.getElementById('counter-free');
  const elOcc = document.getElementById('counter-occupied');
  const elUnk = document.getElementById('counter-unknown');

  if (elTotal) elTotal.textContent = total;
  if (elFree) elFree.textContent = free;
  if (elOcc) elOcc.textContent = occupied;
  if (elUnk) elUnk.textContent = unknown;
}

// --- Stream Controls ---
function initStreamControls() {
  const btnPause = document.getElementById('btn-pause-preview');
  if (btnPause) {
    btnPause.addEventListener('click', async () => {
      const endpoint = state.camera.preview_paused ? '/api/camera/resume_preview' : '/api/camera/pause_preview';
      await fetch(endpoint, { method: 'POST' });
      fetchHealth();
    });
  }

  const btnProc = document.getElementById('btn-toggle-proc');
  if (btnProc) {
    btnProc.addEventListener('click', async () => {
      await fetch('/api/camera/toggle_processing', { method: 'POST' });
      fetchHealth();
    });
  }

  const selectScenario = document.getElementById('select-scenario');
  if (selectScenario) {
    selectScenario.addEventListener('change', async (e) => {
      await fetch('/api/camera/set_scenario', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: e.target.value }),
      });
      showToast(`Simulation scenario set to: ${e.target.value.toUpperCase()}`);
    });
  }
}

// --- Calibration Canvas ---
let canvas, ctx;

function initCalibrationCanvas() {
  canvas = document.getElementById('calibration-canvas');
  if (!canvas) return;
  ctx = canvas.getContext('2d');

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / canvas.width;
    const y = (e.clientY - rect.top) / canvas.height;

    state.currentPolygonPoints.push([parseFloat(x.toFixed(3)), parseFloat(y.toFixed(3))]);
    renderCalibrationCanvas();
  });

  const btnAdd = document.getElementById('btn-add-slot');
  if (btnAdd) {
    btnAdd.addEventListener('click', () => {
      const id = document.getElementById('input-slot-id').value.trim() || `S0${state.slots.length + 1}`;
      const label = document.getElementById('input-slot-label').value.trim() || `Bay ${id}`;
      const type = document.getElementById('select-slot-type').value;

      if (state.currentPolygonPoints.length < 3) {
        showToast('Please click at least 3 points on the canvas to form a bay polygon.');
        return;
      }

      state.slots.push({
        id: id,
        label: label,
        slot_type: type,
        enabled: true,
        points: [...state.currentPolygonPoints],
      });

      state.currentPolygonPoints = [];
      saveSlotsToBackend();
      renderCalibrationList();
      renderCalibrationCanvas();
    });
  }

  const btnClearCurrent = document.getElementById('btn-clear-drawing');
  if (btnClearCurrent) {
    btnClearCurrent.addEventListener('click', () => {
      state.currentPolygonPoints = [];
      renderCalibrationCanvas();
    });
  }
}

function renderCalibrationCanvas() {
  if (!canvas || !ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw configured slots
  state.slots.forEach((slot, index) => {
    if (!slot.points || slot.points.length < 3) return;

    ctx.beginPath();
    const pts = slot.points;
    ctx.moveTo(pts[0][0] * canvas.width, pts[0][1] * canvas.height);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i][0] * canvas.width, pts[i][1] * canvas.height);
    }
    ctx.closePath();

    ctx.fillStyle = 'rgba(0, 240, 255, 0.15)';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#00f0ff';
    ctx.stroke();

    // Label
    const cx = pts[0][0] * canvas.width + 10;
    const cy = pts[0][1] * canvas.height + 25;
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 13px sans-serif';
    ctx.fillText(`${slot.id} (${slot.slot_type})`, cx, cy);
  });

  // Draw current active drawing points
  if (state.currentPolygonPoints.length > 0) {
    ctx.beginPath();
    const pts = state.currentPolygonPoints;
    ctx.moveTo(pts[0][0] * canvas.width, pts[0][1] * canvas.height);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i][0] * canvas.width, pts[i][1] * canvas.height);
    }

    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#f59e0b';
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw vertex handles
    pts.forEach(p => {
      ctx.beginPath();
      ctx.arc(p[0] * canvas.width, p[1] * canvas.height, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#f59e0b';
      ctx.fill();
      ctx.stroke();
    });
  }
}

function renderCalibrationList() {
  const container = document.getElementById('calibration-slots-list');
  if (!container) return;
  container.innerHTML = '';

  state.slots.forEach((slot, idx) => {
    const item = document.createElement('div');
    item.style.padding = '10px';
    item.style.borderBottom = '1px solid rgba(255, 255, 255, 0.06)';
    item.style.display = 'flex';
    item.style.justifyContent = 'space-between';
    item.style.alignItems = 'center';

    item.innerHTML = `
      <div>
        <strong style="color: var(--accent-cyan);">${slot.id}</strong>: ${slot.label} 
        <span style="font-size: 11px; color: var(--text-muted);">[${slot.slot_type}]</span>
      </div>
      <button class="btn btn-danger" style="padding: 4px 8px; font-size: 11px;" onclick="deleteSlot(${idx})">Delete</button>
    `;
    container.appendChild(item);
  });
}

window.deleteSlot = (idx) => {
  state.slots.splice(idx, 1);
  saveSlotsToBackend();
  renderCalibrationList();
  renderCalibrationCanvas();
};

async function saveSlotsToBackend() {
  try {
    const res = await fetch('/api/slots', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        reference_width: state.referenceSize.width,
        reference_height: state.referenceSize.height,
        slots: state.slots,
      }),
    });
    if (res.ok) {
      showToast('Calibration saved successfully.');
    }
  } catch (err) {
    showToast('Failed to save calibration.');
  }
}

// --- Events ---
async function fetchEvents() {
  try {
    const resOcc = await fetch('/api/events/occupancy?limit=25');
    const dataOcc = await resOcc.json();
    const tbodyOcc = document.getElementById('table-occupancy-events');
    if (tbodyOcc) {
      tbodyOcc.innerHTML = '';
      (dataOcc.events || []).forEach(e => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td><code>${e.timestamp}</code></td>
          <td><strong>${e.slot_id}</strong></td>
          <td><span class="badge ${e.previous_state === 'FREE' ? 'badge-live' : 'badge-stale'}">${e.previous_state}</span></td>
          <td><span class="badge ${e.new_state === 'FREE' ? 'badge-live' : 'badge-stale'}">${e.new_state}</span></td>
          <td>${(e.confidence * 100).toFixed(1)}%</td>
        `;
        tbodyOcc.appendChild(row);
      });
    }

    const resPlt = await fetch('/api/events/plates?limit=25');
    const dataPlt = await resPlt.json();
    const tbodyPlt = document.getElementById('table-plate-events');
    if (tbodyPlt) {
      tbodyPlt.innerHTML = '';
      (dataPlt.events || []).forEach(p => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td><code>${p.timestamp}</code></td>
          <td><strong>${p.slot_id || 'N/A'}</strong></td>
          <td><code>${p.masked_plate}</code></td>
          <td><span class="badge badge-mock">${p.hsrp_outcome}</span></td>
          <td>${(p.confidence * 100).toFixed(1)}%</td>
        `;
        tbodyPlt.appendChild(row);
      });
    }
  } catch (err) {
    console.debug('Failed to load events:', err);
  }
}

// --- Diagnostics ---
function initDiagnostics() {
  const btnQuick = document.getElementById('btn-run-diag-quick');
  const btnFull = document.getElementById('btn-run-diag-full');

  if (btnQuick) {
    btnQuick.addEventListener('click', () => runDiagnostics('quick'));
  }
  if (btnFull) {
    btnFull.addEventListener('click', () => runDiagnostics('full'));
  }
}

async function runDiagnostics(type) {
  const progressBar = document.getElementById('diag-progress-bar');
  const resultsTable = document.getElementById('diag-results-tbody');
  const summaryBox = document.getElementById('diag-summary-box');

  if (progressBar) progressBar.style.width = '20%';
  if (summaryBox) summaryBox.textContent = `Running ${type.toUpperCase()} self-diagnostics...`;
  if (resultsTable) resultsTable.innerHTML = '';

  try {
    const res = await fetch(`/api/diagnostics/run?run_type=${type}`, { method: 'POST' });
    const report = await res.json();

    if (progressBar) progressBar.style.width = '100%';
    if (summaryBox) {
      summaryBox.innerHTML = `
        <strong>Result: ${report.overall_status}</strong> | Total: ${report.summary.total} | 
        Passed: <span style="color: var(--status-free);">${report.summary.pass}</span> | 
        Warnings: <span style="color: var(--status-unknown);">${report.summary.warn}</span> | 
        Failed: <span style="color: var(--status-occupied);">${report.summary.fail}</span>
        (${report.total_duration_sec}s)
      `;
    }

    if (resultsTable) {
      resultsTable.innerHTML = '';
      (report.results || []).forEach(r => {
        const row = document.createElement('tr');
        const badgeClass = r.status === 'PASS' ? 'badge-live' : (r.status === 'WARN' ? 'badge-stale' : 'badge-danger');
        row.innerHTML = `
          <td><code>${r.test_id}</code></td>
          <td><strong>${r.name}</strong></td>
          <td><span class="badge ${badgeClass}">${r.status}</span></td>
          <td>${r.duration_ms} ms</td>
          <td>${r.message}</td>
        `;
        resultsTable.appendChild(row);
      });
    }
  } catch (err) {
    if (summaryBox) summaryBox.textContent = `Diagnostics run failed: ${err.message}`;
  }
}

// --- Camera Settings & Models ---
async function fetchCameraDevices() {
  try {
    const res = await fetch('/api/camera/devices');
    const data = await res.json();
    const container = document.getElementById('camera-devices-list');
    if (container) {
      container.innerHTML = '';
      (data.devices || []).forEach(d => {
        const div = document.createElement('div');
        div.style.padding = '10px';
        div.style.borderBottom = '1px solid rgba(255, 255, 255, 0.06)';
        div.innerHTML = `
          <strong>Device #${d.device_index}</strong>: ${d.display_name} 
          <span style="color: var(--text-muted);">(${d.default_width}x${d.default_height} @ ${d.reported_fps} FPS)</span>
        `;
        container.appendChild(div);
      });
    }
  } catch (err) {
    console.debug('Failed to fetch devices:', err);
  }
}

async function fetchModels() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    const tbody = document.getElementById('table-models-tbody');
    if (tbody) {
      tbody.innerHTML = '';
      (data.models || []).forEach(m => {
        const row = document.createElement('tr');
        const isReady = m.file_exists === true;
        const badgeClass = isReady ? 'badge-live' : 'badge-stale';
        const statusText = isReady ? 'READY' : 'NOT TRAINED';
        row.innerHTML = `
          <td><strong>${m.model_id}</strong></td>
          <td>${m.architecture}</td>
          <td><code>${m.runtime}</code></td>
          <td>${m.purpose}</td>
          <td><span class="badge ${badgeClass}">${statusText}</span></td>
        `;
        tbody.appendChild(row);
      });
    }
  } catch (err) {
    console.debug('Failed to fetch models:', err);
  }
}

// --- Toast Alerts ---
function showToast(msg) {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  container.appendChild(t);
  setTimeout(() => {
    t.remove();
  }, 3500);
}

// =========================================================================
// Virtual Parking Space Allocation & Concurrency Client Logic (Phase 7)
// =========================================================================

function initAllocationForm() {
  const form = document.getElementById('form-allocate-slot');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('btn-submit-allocation');
    btn.disabled = true;
    btn.textContent = 'Allocating (Securing Lock)...';

    const payload = {
      user_id: document.getElementById('alloc-input-user').value.trim(),
      vehicle_plate: document.getElementById('alloc-input-plate').value.trim() || null,
      slot_type: document.getElementById('alloc-input-type').value,
      zone: document.getElementById('alloc-input-zone').value || null,
      duration_minutes: parseInt(document.getElementById('alloc-input-duration').value, 10),
    };

    try {
      const res = await fetch('/api/parking/allocate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Allocation failed');
      }

      showToast(`Slot ${data.slot_id} Allocated! Reservation Code: ${data.reservation_code}`);

      // Display receipt
      const receiptBox = document.getElementById('alloc-receipt-box');
      const details = document.getElementById('alloc-receipt-details');
      if (receiptBox && details) {
        receiptBox.style.display = 'block';
        details.innerHTML = `
          <strong>Assigned Bay:</strong> ${data.slot_id} (${data.slot_label})<br>
          <strong>Zone / Level:</strong> ${data.zone} / ${data.level} (${data.slot_type.toUpperCase()})<br>
          <strong>Reservation Code:</strong> <code>${data.reservation_code}</code><br>
          <strong>Expires At:</strong> ${new Date(data.expires_at).toLocaleTimeString()}<br>
          <span style="color: var(--accent-cyan); font-weight: 600;">Double-Allocation Prevention: Enforced via SQLite IMMEDIATE lock</span>
        `;
      }

      fetchAvailability(true);
    } catch (err) {
      showToast(`Allocation Error: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Allocate Parking Slot';
    }
  });
}

async function fetchAvailability(refreshAudit = false) {
  try {
    const res = await fetch('/api/parking/availability');
    if (!res.ok) return;
    const data = await res.json();

    // Update Counts
    const elVacant = document.getElementById('alloc-count-vacant');
    const elReserved = document.getElementById('alloc-count-reserved');
    const elOcc = document.getElementById('alloc-count-occupied');
    const elUnk = document.getElementById('alloc-count-unknown');

    if (elVacant) elVacant.textContent = data.reservation_counts?.allocatable_vacant ?? 0;
    if (elReserved) elReserved.textContent = data.reservation_counts?.active_reservations ?? 0;
    if (elOcc) elOcc.textContent = data.physical_counts?.occupied ?? 0;
    if (elUnk) elUnk.textContent = data.physical_counts?.unknown ?? 0;

    // Render Breakdown Chips
    const chipsContainer = document.getElementById('alloc-breakdown-chips');
    if (chipsContainer) {
      let chipsHtml = '';
      if (data.by_zone) {
        for (const [zName, zStats] of Object.entries(data.by_zone)) {
          chipsHtml += `
            <div style="background: var(--bg-surface); padding: 8px 14px; border-radius: 6px; border: 1px solid var(--border-color); font-size: 12px;">
              <strong>${zName}:</strong> ${zStats.allocatable} Free / ${zStats.total} Total
            </div>
          `;
        }
      }
      if (data.by_slot_type) {
        for (const [tName, tStats] of Object.entries(data.by_slot_type)) {
          chipsHtml += `
            <div style="background: var(--bg-surface); padding: 8px 14px; border-radius: 6px; border: 1px solid var(--border-color); font-size: 12px;">
              <strong>${tName.toUpperCase()}:</strong> ${tStats.allocatable} Free / ${tStats.total} Total
            </div>
          `;
        }
      }
      chipsContainer.innerHTML = chipsHtml;
    }

    // Render Slot Registry Table (Virtual Allocation Tab)
    const tbody = document.getElementById('tbody-allocation-slots');
    if (tbody && data.slots) {
      tbody.innerHTML = '';
      data.slots.forEach(s => {
        const row = document.createElement('tr');
        const pBadge = s.physical_state === 'FREE' ? 'badge-live' : (s.physical_state === 'OCCUPIED' ? 'badge-danger' : 'badge-stale');
        const rBadge = s.reservation_status === 'AVAILABLE' ? 'badge-live' : (s.reservation_status === 'RESERVED' ? 'badge-mock' : 'badge-danger');

        let actionCell = '<span style="color: var(--text-muted); font-size: 11px;">--</span>';
        if (s.active_reservation_code) {
          actionCell = `<button class="btn btn-sm btn-danger" onclick="releaseReservation('${s.active_reservation_code}')">Release</button>`;
        } else if (s.reservation_status === 'AVAILABLE' && s.physical_state === 'FREE') {
          actionCell = `<button class="btn btn-sm btn-accent" onclick="prefillAllocationForm('${s.slot_id}', '${s.slot_type}')">Allot Slot</button>`;
        }

        row.innerHTML = `
          <td><strong>${s.slot_id}</strong></td>
          <td>${s.zone} (${s.level})</td>
          <td><code>${s.slot_type}</code></td>
          <td><span class="badge ${pBadge}">${s.physical_state}</span></td>
          <td><span class="badge ${rBadge}">${s.reservation_status}</span></td>
          <td><code>${s.active_reservation_code || '--'}</code></td>
          <td>${actionCell}</td>
        `;
        tbody.appendChild(row);
      });
    }

    // Render Overview Active Allotments Table & Quick Release
    const overviewTbody = document.getElementById('tbody-overview-allotments');
    if (overviewTbody && data.slots) {
      overviewTbody.innerHTML = '';
      if (data.slots.length === 0) {
        overviewTbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No calibrated bays available.</td></tr>';
      } else {
        data.slots.forEach(s => {
          const row = document.createElement('tr');
          const pBadge = s.physical_state === 'FREE' ? 'badge-live' : (s.physical_state === 'OCCUPIED' ? 'badge-danger' : 'badge-stale');
          const rBadge = s.reservation_status === 'AVAILABLE' ? 'badge-live' : (s.reservation_status === 'RESERVED' ? 'badge-mock' : 'badge-danger');
          const userInfo = s.active_plate ? `${s.active_plate} (${s.active_user_id || 'Driver'})` : (s.active_user_id || '--');

          let actionBtn = '<span style="color: var(--text-muted); font-size: 11px;">--</span>';
          if (s.active_reservation_code) {
            actionBtn = `<button class="btn btn-sm btn-danger" onclick="releaseReservation('${s.active_reservation_code}')">Release</button>`;
          } else if (s.reservation_status === 'AVAILABLE' && s.physical_state === 'FREE') {
            actionBtn = `<button class="btn btn-sm btn-accent" onclick="prefillQuickAllot('${s.slot_id}', '${s.slot_type}')">Allot Bay</button>`;
          }

          row.innerHTML = `
            <td><strong>${s.slot_id}</strong></td>
            <td><code>${s.slot_type}</code></td>
            <td><span class="badge ${pBadge}">${s.physical_state}</span></td>
            <td><span class="badge ${rBadge}">${s.reservation_status}</span></td>
            <td>${userInfo}</td>
            <td><code>${s.active_reservation_code || '--'}</code></td>
            <td>${actionBtn}</td>
          `;
          overviewTbody.appendChild(row);
        });
      }
    }

    // Populate Overview Quick-Allot Dropdown with free bays
    const quickSlotSelect = document.getElementById('quick-alloc-slot');
    if (quickSlotSelect && data.slots) {
      const currentVal = quickSlotSelect.value;
      let optionsHtml = '<option value="">⚡ Auto-Assign Best Available Bay</option>';
      data.slots.forEach(s => {
        if (s.reservation_status === 'AVAILABLE' && s.physical_state === 'FREE') {
          const isSelected = s.slot_id === currentVal ? 'selected' : '';
          optionsHtml += `<option value="${s.slot_id}" ${isSelected}>Bay ${s.slot_id} (${s.zone} · ${s.slot_type.toUpperCase()})</option>`;
        }
      });
      quickSlotSelect.innerHTML = optionsHtml;
    }

    if (refreshAudit) {
      fetchAllocationAudit();
    }
  } catch (err) {
    console.debug('Failed to fetch availability:', err);
  }
}

async function releaseReservation(code) {
  try {
    const res = await fetch('/api/parking/release', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reservation_code: code }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Release failed');
    showToast(`Reservation ${code} Released! Bay is now free.`);
    fetchAvailability(true);
    fetchSlots();
  } catch (err) {
    showToast(`Release Error: ${err.message}`);
  }
}

async function fetchAllocationAudit() {
  try {
    const res = await fetch('/api/parking/audit?limit=25');
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById('tbody-allocation-audit');
    if (tbody && data.audit_trail) {
      tbody.innerHTML = '';
      data.audit_trail.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td><strong>${item.event_type}</strong></td>
          <td>${item.slot_id}</td>
          <td><code>${item.reservation_code || '--'}</code></td>
          <td>${item.physical_state || '--'}</td>
          <td style="font-size: 12px;">${item.details || ''}</td>
          <td style="font-size: 11px; color: var(--text-muted);">${new Date(item.timestamp).toLocaleTimeString()}</td>
        `;
        tbody.appendChild(row);
      });
    }
  } catch (err) {
    console.debug('Failed to fetch audit:', err);
  }
}

// --- Overview Quick Allotment Form Controller ---
function initQuickAllotmentForm() {
  const form = document.getElementById('form-quick-allot');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('btn-quick-allot');
    btn.disabled = true;
    btn.textContent = 'Allocating (Securing ACID Lock)...';

    const slotVal = document.getElementById('quick-alloc-slot').value;
    const payload = {
      user_id: document.getElementById('quick-alloc-user').value.trim() || 'driver_01',
      vehicle_plate: document.getElementById('quick-alloc-plate').value.trim() || null,
      slot_id: slotVal || null,
      slot_type: document.getElementById('quick-alloc-type').value,
      duration_minutes: parseInt(document.getElementById('quick-alloc-duration').value, 10),
    };

    try {
      const res = await fetch('/api/parking/allocate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Allocation failed');
      }

      showToast(`Bay ${data.slot_id} Allotted! Code: ${data.reservation_code}`);

      const receipt = document.getElementById('quick-alloc-receipt');
      const body = document.getElementById('quick-receipt-body');
      if (receipt && body) {
        receipt.style.display = 'block';
        body.innerHTML = `
          <strong>Assigned Bay:</strong> <span style="color: var(--accent-cyan); font-weight: 700;">${data.slot_id}</span> (${data.slot_label})<br>
          <strong>Reservation Code:</strong> <code>${data.reservation_code}</code><br>
          <strong>Driver:</strong> ${data.user_id} ${data.vehicle_plate ? `· Plate: <strong>${data.vehicle_plate}</strong>` : ''}<br>
          <strong>Expires At:</strong> ${new Date(data.expires_at).toLocaleTimeString()}<br>
          <button class="btn btn-sm btn-danger" style="margin-top: 8px;" onclick="releaseReservation('${data.reservation_code}')">Release This Bay Now</button>
        `;
      }

      // Reset plate
      document.getElementById('quick-alloc-plate').value = '';

      fetchAvailability(true);
      fetchSlots();
    } catch (err) {
      showToast(`Allocation Error: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = '⚡ Allot Parking Bay Now';
    }
  });
}

function prefillQuickAllot(slotId, slotType) {
  const slotSelect = document.getElementById('quick-alloc-slot');
  if (slotSelect) slotSelect.value = slotId;
  const typeSelect = document.getElementById('quick-alloc-type');
  if (typeSelect && slotType) typeSelect.value = slotType;
  const form = document.getElementById('form-quick-allot');
  if (form) form.scrollIntoView({ behavior: 'smooth', block: 'center' });
  showToast(`Selected Bay ${slotId} for allotment`);
}

function prefillAllocationForm(slotId, slotType) {
  const typeSelect = document.getElementById('alloc-input-type');
  if (typeSelect && slotType) typeSelect.value = slotType;
  const form = document.getElementById('form-allocate-slot');
  if (form) form.scrollIntoView({ behavior: 'smooth', block: 'center' });
  showToast(`Ready to allocate Bay ${slotId}`);
}

