/* ================================================================
   SurveillanceAI — Frontend Controller (Black Screen Fix)
   ================================================================ */

// ── State ─────────────────────────────────────────────────────────
let ws = null;
let frameLoopId = null;
let frameLoopActive = false;
let waitingForAck = false;
let statsInterval = null;
let alertCount = 0;

// Off-screen image used for flicker-free canvas draws
const _offscreen = new Image();
let _pendingFrame = null; // latest b64 frame waiting to be drawn
let _drawScheduled = false;

const WS_HOST = window.location.host;
const WS_PROTOCOL = window.location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${WS_PROTOCOL}://${WS_HOST}/ws/stream`);

// ── DOM helpers ───────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const rawVideo = () => $("raw-video");
const hiddenCanvas = () => $("hidden-canvas");
const processedFeed = () => $("processed-feed"); // <canvas> element
const videoIdle = () => $("video-idle");
const feedOverlay = () => $("feed-overlay");
const threatFlash = () => $("threat-flash");
const alertList = () => $("alert-list");
const stripTags = () => $("strip-tags");
const stripEmpty = () => $("strip-empty");

// ── Clock ─────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  $("clock").textContent = now.toTimeString().slice(0, 8);
  $("date-display").textContent = now
    .toLocaleDateString("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
    })
    .toUpperCase();
}
setInterval(updateClock, 1000);
updateClock();

// ── Stats ─────────────────────────────────────────────────────────
async function fetchStats() {
  try {
    const data = await fetch("/api/stats").then((r) => r.json());
    $("stat-total").textContent = data.total_alerts ?? 0;
    $("stat-high").textContent = data.by_severity?.HIGH ?? 0;
    $("stat-med").textContent = data.by_severity?.MEDIUM ?? 0;
    $("alert-count-badge").textContent = alertCount;
  } catch (_) {}
}

// ── Webcam ────────────────────────────────────────────────────────
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    const vid = rawVideo();
    vid.srcObject = stream;
    vid.style.display = "block"; // show raw feed immediately
    videoIdle().style.display = "none";
    processedFeed().style.display = "none"; // hidden until first processed frame arrives
    await vid.play();
    openWebSocket();
  } catch (err) {
    alert(`Webcam access denied: ${err.message}`);
  }
}

// ── Video file ────────────────────────────────────────────────────
function loadVideoFile(event) {
  const file = event.target.files[0];
  if (!file) return;

  // Reset the input so the same file can be re-selected after Stop
  event.target.value = "";

  // Always stop any active stream cleanly before starting a new one
  if (ws || frameLoopActive) stopStream();

  const vid = rawVideo();
  // Revoke any previous object URL to free memory
  if (vid._objectUrl) {
    URL.revokeObjectURL(vid._objectUrl);
    vid._objectUrl = null;
  }

  const objectUrl = URL.createObjectURL(file);
  vid._objectUrl = objectUrl;
  vid.src = objectUrl;
  vid.style.display = "block";
  videoIdle().style.display = "none";
  processedFeed().style.display = "none";

  // Wait for video metadata before opening the WebSocket — ensures
  // videoWidth/videoHeight are populated when the first frame is sent
  vid.onloadeddata = () => {
    vid.onloadeddata = null;
    vid.play().then(openWebSocket).catch(console.error);
  };

  // Fallback: if loadeddata already fired (cached), play immediately
  if (vid.readyState >= 2) {
    vid.onloadeddata = null;
    vid.play().then(openWebSocket).catch(console.error);
  }
}

// ── WebSocket ─────────────────────────────────────────────────────
function openWebSocket() {
  // Guard against OPEN *or* CONNECTING — both mean a socket is already in-flight
  if (
    ws &&
    (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)
  )
    return;

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setStatus("online");
    $("btn-stop").disabled = false;
    feedOverlay().style.display = "block";
    // Set waitingForAck=true immediately so the rAF loop never sends a
    // frame before the first response arrives. Cleared by first onmessage.
    waitingForAck = true;
    frameLoopActive = true;
    frameLoopId = requestAnimationFrame(frameLoop);
    statsInterval = setInterval(fetchStats, 3000);
    // Send the very first frame manually to kick off the ping-pong cycle
    const vid = rawVideo();
    if (vid.videoWidth) {
      waitingForAck = true;
      sendFrame(vid);
    } else {
      // Video not ready yet — wait for metadata then send
      vid.addEventListener(
        "loadeddata",
        () => {
          waitingForAck = true;
          sendFrame(vid);
        },
        { once: true },
      );
    }
  };

  ws.onmessage = (e) => {
    waitingForAck = false; // backend responded — allow next frame
    try {
      handleResult(JSON.parse(e.data));
    } catch (_) {}
  };

  ws.onerror = (e) => console.error("[WS] Error:", e);

  ws.onclose = () => {
    frameLoopActive = false;
    if (frameLoopId) cancelAnimationFrame(frameLoopId);
    clearInterval(statsInterval);
    setStatus("offline");
    feedOverlay().style.display = "none";
    $("btn-stop").disabled = true;
  };
}

// Minimum ms between frames sent to backend.
// At 200ms (5 FPS) on CPU the backend can keep up; raise to 300ms if still lagging.
const MIN_FRAME_INTERVAL_MS = 200;
let _lastSentAt = 0;

// ── Frame capture loop (rAF-gated) ───────────────────────────────
function frameLoop() {
  if (!frameLoopActive) return;
  frameLoopId = requestAnimationFrame(frameLoop);

  // Hard gate: don't send if still waiting for previous ack
  if (waitingForAck) return;

  // Soft gate: don't send faster than MIN_FRAME_INTERVAL_MS even if ack arrived
  const now = performance.now();
  if (now - _lastSentAt < MIN_FRAME_INTERVAL_MS) return;

  const vid = rawVideo();
  if (!vid.videoWidth || vid.paused || vid.ended) return;

  _lastSentAt = now;
  waitingForAck = true;
  sendFrame(vid);
}

function sendFrame(vid) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    waitingForAck = false;
    return;
  }
  const capture = hiddenCanvas();
  // Cap at 640px wide — matches INFERENCE_WIDTH in detector.py
  // Sending larger frames wastes bandwidth with no detection benefit
  const scale = Math.min(1, 640 / (vid.videoWidth || 640));
  const w = Math.round((vid.videoWidth || 640) * scale);
  const h = Math.round((vid.videoHeight || 480) * scale);

  if (capture.width !== w) capture.width = w;
  if (capture.height !== h) capture.height = h;

  capture.getContext("2d").drawImage(vid, 0, 0, w, h);
  // Quality 0.55 — good enough for YOLO, meaningfully smaller payload
  const b64 = capture.toDataURL("image/jpeg", 0.55).split(",")[1];
  ws.send(JSON.stringify({ frame: b64 }));
}

// ── Handle inference result ───────────────────────────────────────
function handleResult(data) {
  if (data.error) return;

  if (data.frame) {
    scheduleFrameDraw(data.frame);
  }

  // Detection strip
  if (data.detections?.length) {
    stripEmpty().style.display = "none";
    stripTags().innerHTML = data.detections
      .map(
        (d) => `<span class="tag ${d.suspicious ? "tag-danger" : "tag-safe"}">
                   ${d.label.toUpperCase()} ${d.confidence}
                 </span>`,
      )
      .join("");
  } else {
    stripEmpty().style.display = "inline";
    stripTags().innerHTML = "";
  }

  if (data.alert) {
    setStatus("threat");
    triggerFlash();
  } else {
    // Don't override "threat" state — let its timer handle reset
    if (!$("status-pill").className.includes("threat")) setStatus("online");
  }

  if (data.alert_data) {
    addAlertCard(data.alert_data);
    fetchStats();
  }
}

// ── Flicker-free canvas rendering ────────────────────────────────
//
// Fix explanation:
//   - Use a single shared _offscreen Image so the browser reuses the decode buffer
//   - Queue at most one pending draw per rAF tick (_drawScheduled gate)
//   - Only assign canvas.width / canvas.height when they differ — assignment
//     always clears the canvas, causing the black flash
//   - raw video stays hidden-but-playing; only toggled once on first processed frame
//
function scheduleFrameDraw(b64) {
  _pendingFrame = b64; // always keep latest, discard stale intermediate frames
  if (_drawScheduled) return; // already a draw queued this tick
  _drawScheduled = true;

  requestAnimationFrame(() => {
    _drawScheduled = false;
    if (!_pendingFrame) return;

    const src = "data:image/jpeg;base64," + _pendingFrame;
    _pendingFrame = null;

    // Decode off-screen first, then draw atomically
    _offscreen.onload = () => {
      const canvas = processedFeed();
      const ctx = canvas.getContext("2d");

      if (canvas.width !== _offscreen.naturalWidth)
        canvas.width = _offscreen.naturalWidth;
      if (canvas.height !== _offscreen.naturalHeight)
        canvas.height = _offscreen.naturalHeight;

      ctx.drawImage(_offscreen, 0, 0);

      // First processed frame: swap raw video out, canvas in
      if (canvas.style.display === "none") {
        canvas.style.display = "block";
        rawVideo().style.display = "none";
      }
    };

    _offscreen.src = src;
  });
}

// ── Stop ──────────────────────────────────────────────────────────
function stopStream() {
  frameLoopActive = false;
  if (frameLoopId) {
    cancelAnimationFrame(frameLoopId);
    frameLoopId = null;
  }
  clearInterval(statsInterval);
  statsInterval = null;
  waitingForAck = false;
  _pendingFrame = null;
  _drawScheduled = false;
  _lastSentAt = 0;

  if (ws) {
    ws.close();
    ws = null;
  }

  const vid = rawVideo();
  // Stop webcam tracks if active
  if (vid.srcObject) {
    vid.srcObject.getTracks().forEach((t) => t.stop());
    vid.srcObject = null;
  }
  // Revoke object URL and fully reset the video element
  if (vid._objectUrl) {
    URL.revokeObjectURL(vid._objectUrl);
    vid._objectUrl = null;
  }
  vid.removeAttribute("src");
  vid.load(); // resets internal decoder state — critical for sequential uploads
  vid.style.display = "none";

  const canvas = processedFeed();
  canvas.style.display = "none";
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

  videoIdle().style.display = "flex";
  stripTags().innerHTML = "";
  stripEmpty().style.display = "inline";
  setStatus("offline");
  $("btn-stop").disabled = true;
}

// ── Clear alerts ──────────────────────────────────────────────────
async function clearAlerts() {
  await fetch("/api/alerts", { method: "DELETE" });
  alertCount = 0;
  $("alert-count-badge").textContent = 0;
  alertList().innerHTML = `
    <div class="alert-empty" id="alert-empty">
      <div class="empty-icon">◎</div>
      <div>No alerts yet</div>
      <div class="empty-sub">System monitoring in standby</div>
    </div>`;
  fetchStats();
}

// ── Status pill ───────────────────────────────────────────────────
let _statusResetTimer = null;
function setStatus(state) {
  const pill = $("status-pill");
  const text = $("status-text");
  if (state === "threat") {
    pill.className = "status-pill threat";
    text.textContent = "THREAT";
    clearTimeout(_statusResetTimer);
    _statusResetTimer = setTimeout(() => {
      pill.className = "status-pill online";
      text.textContent = "LIVE";
    }, 2500);
  } else if (state === "online") {
    pill.className = "status-pill online";
    text.textContent = "LIVE";
  } else {
    clearTimeout(_statusResetTimer);
    pill.className = "status-pill";
    text.textContent = "OFFLINE";
  }
}

// ── Threat flash ──────────────────────────────────────────────────
function triggerFlash() {
  const el = threatFlash();
  el.classList.remove("active");
  void el.offsetWidth;
  el.classList.add("active");
  setTimeout(() => el.classList.remove("active"), 500);
}

// ── Snapshot modal ────────────────────────────────────────────────
function openModal(snapshotUrl, alert) {
  $("modal-img").src = snapshotUrl;
  $("modal-meta").innerHTML = `
    ID: ${alert.id}<br>
    Time: ${alert.timestamp}<br>
    Severity: ${alert.severity}<br>
    Detections: ${alert.detections.map((d) => d.label).join(", ")}
    ${alert.face_match ? `<br>Match: ${alert.face_match.name}` : ""}
  `;
  $("modal-backdrop").classList.add("open");
}
function closeModal() {
  $("modal-backdrop").classList.remove("open");
}
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});

// ── Alert card ────────────────────────────────────────────────────
function addAlertCard(alert) {
  const list = alertList();
  const empty = $("alert-empty");
  if (empty) empty.remove();

  alertCount++;
  $("alert-count-badge").textContent = alertCount;

  const detText = alert.detections
    .map((d) => `⚠ ${d.label.toUpperCase()} (${d.confidence})`)
    .join("  ·  ");

  const faceHtml = alert.face_match
    ? `<div class="alert-face">🎯 MATCH: ${alert.face_match.name.toUpperCase()} — ${alert.face_match.record}</div>`
    : "";
  const snapHint = alert.snapshot_url
    ? `<div class="alert-snap-hint">🖼 Click to view snapshot</div>`
    : "";

  const card = document.createElement("div");
  card.className = `alert-card sev-${alert.severity}`;
  card.innerHTML = `
    <div class="alert-top">
      <span class="alert-sev sev-${alert.severity}">${alert.severity}</span>
      <span class="alert-time">${alert.timestamp}</span>
    </div>
    <div class="alert-detections">${detText}</div>
    ${faceHtml}${snapHint}
  `;
  if (alert.snapshot_url)
    card.addEventListener("click", () => openModal(alert.snapshot_url, alert));

  list.insertBefore(card, list.firstChild);
}

// ── Boot ──────────────────────────────────────────────────────────
(async () => {
  await fetchStats();
  try {
    const alerts = await fetch("/api/alerts").then((r) => r.json());
    alerts.forEach(addAlertCard);
  } catch (_) {}
})();
