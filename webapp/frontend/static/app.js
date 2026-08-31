/* Fruit & Vegetable Detection - front end.
 *
 * Two tabs, two transports, one model behind both:
 *
 *   Upload  POST /api/detect   multipart -> JSON with a server-drawn JPEG
 *   Live    WS   /ws/detect    JPEG frames -> JSON coordinates, drawn here
 *
 * The live tab gets coordinates rather than a picture because a frame is
 * replaced 30 times a second: re-encoding each one server-side would double
 * the work and the bandwidth for something nobody looks at twice. Drawing on a
 * canvas over the <video> costs nothing and keeps the video itself crisp.
 *
 * Class colours are fetched from /api/meta rather than written here, so the
 * overlay matches the server-drawn image and the figures in the report.
 */

'use strict';

const $ = (id) => document.getElementById(id);

let META = null;          // classes, colours, defaults - filled by boot()
let confidence = 0.35;

/* ------------------------------------------------------------------ boot */

async function boot() {
  try {
    const res = await fetch('/api/meta');
    if (!res.ok) throw new Error(res.statusText);
    META = await res.json();
  } catch (err) {
    $('model-badge').innerHTML = '<span class="sep">model failed to load</span>';
    console.error('could not read /api/meta', err);
    return;
  }

  confidence = META.default_threshold;
  $('conf').value = confidence;
  $('conf-value').textContent = confidence.toFixed(2);

  const m = META.model;
  $('model-badge').innerHTML =
    `<b>${m.name}</b><span class="sep">&middot;</span>${m.params_millions} M params` +
    `<span class="sep">&middot;</span>${m.imgsz}px` +
    `<span class="sep">&middot;</span>${m.device}` +
    `<span class="sep">&middot;</span>${m.test_mAP_50_95.toFixed(4)} mAP`;

  renderCounts($('upload-counts'), emptyCounts());
  renderCounts($('live-counts'), emptyCounts());
  renderCounts($('session-counts'), emptyCounts());
  buildSamples();
  restoreSavedVisibility();
  refreshSessions();

  if (!window.isSecureContext) {
    // getUserMedia is gated on a secure context, and http://192.168.x.x is not
    // one. Say so up front rather than letting the browser fail silently.
    const note = $('live-note');
    note.hidden = false;
    note.textContent =
      'This page is not in a secure context, so the browser will refuse camera ' +
      'access. Open it as http://localhost:' + location.port + ' instead.';
  }
}

const emptyCounts = () => Object.fromEntries(META.classes.map((c) => [c, 0]));
const colourOf = (name) => (META && META.colors[name]) || '#666';

/* ------------------------------------------------------------------ tabs */

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('is-active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('is-active'));
    tab.classList.add('is-active');
    $('panel-' + tab.dataset.tab).classList.add('is-active');
    // A hidden panel has no width, so anything measured while it was closed
    // was measured as zero. Re-fit now that it is on screen.
    if (tab.dataset.tab === 'upload') fitStage();
  });
});

/* ------------------------------------------------------- shared threshold */

$('conf').addEventListener('input', (e) => {
  confidence = parseFloat(e.target.value);
  $('conf-value').textContent = confidence.toFixed(2);
  // The socket is cheap to tell and takes effect on the very next frame.
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ conf: confidence }));
  }
});

// Re-running the upload means another forward pass, so wait for the drag to
// finish rather than firing on every step of the slider.
$('conf').addEventListener('change', () => { if (lastFile) runUpload(lastFile); });

/* --------------------------------------------------------- shared render */

function renderCounts(table, counts) {
  table.innerHTML = META.classes.map((name) => {
    const n = counts[name] || 0;
    return `<tr class="${n ? '' : 'zero'}">
      <td><span class="swatch" style="background:${colourOf(name)}"></span>${name}</td>
      <td class="n">${n}</td>
    </tr>`;
  }).join('');
}

function renderDetections(list, detections) {
  if (!detections.length) {
    list.innerHTML = '<li class="muted">No objects above the threshold.</li>';
    return;
  }
  list.innerHTML = detections.map((d) => {
    const pct = Math.round(d.confidence * 100);
    const colour = colourOf(d.label);
    return `<li>
      <span class="swatch" style="background:${colour}"></span>
      <span class="name">${d.label}</span>
      <span class="bar"><i style="width:${pct}%;background:${colour}"></i></span>
      <span class="pct">${pct}%</span>
    </li>`;
  }).join('');
}

/* ----------------------------------------------------- fitting the stage */

/* Both frames are sized to the result's own aspect ratio rather than to half
 * the panel. Left on `1fr 1fr` a portrait photo sits in a landscape box with a
 * wide white margin down either side, and CSS alone cannot fix that: capping
 * the image with `max-height` changes how tall it *renders* but the frame is
 * still laid out from the image's intrinsic width. The response already
 * carries the true dimensions, so the width is just arithmetic. */

const STAGE_GAP = 20;            // must match `gap` on .stage in style.css
const STAGE_MAX_VH = 0.62;       // tallest an image may get, as a share of the viewport
const NARROW_BREAKPOINT = 900;   // where style.css collapses .stage to one column

let shownSize = null;            // {w, h} of the image currently on screen

function fitStage() {
  const stage = $('upload-stage');
  if (!shownSize) {
    stage.classList.remove('is-fitted');
    stage.style.gridTemplateColumns = '';
    stage.style.justifyContent = '';
    return;
  }

  stage.classList.add('is-fitted');

  // Below the breakpoint the stage is a single full-width column already, so
  // there is no margin to reclaim - let the image fill it.
  if (window.innerWidth <= NARROW_BREAKPOINT) {
    stage.style.gridTemplateColumns = '';
    stage.style.justifyContent = '';
    return;
  }

  const available = stage.clientWidth;
  if (!available) return;   // panel is hidden - nothing to measure yet

  const perColumn = (available - STAGE_GAP) / 2;
  const scale = Math.min(perColumn / shownSize.w,
                         (window.innerHeight * STAGE_MAX_VH) / shownSize.h);
  const width = Math.max(180, Math.round(shownSize.w * scale));

  stage.style.gridTemplateColumns = `${width}px ${width}px`;
  stage.style.justifyContent = 'center';
}

window.addEventListener('resize', fitStage);

/* ---------------------------------------------------------------- upload */

let lastFile = null;

const dropzone = $('dropzone');
const fileInput = $('file-input');

dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

['dragenter', 'dragover'].forEach((type) =>
  dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    dropzone.classList.add('is-over');
  }));

['dragleave', 'drop'].forEach((type) =>
  dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    dropzone.classList.remove('is-over');
  }));

dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

function buildSamples() {
  $('sample-list').innerHTML = META.samples
    .map((name, i) => `<button data-sample="${name}">Sample ${i + 1}</button>`)
    .join('');
  $('sample-list').querySelectorAll('button').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const name = btn.dataset.sample;
      const blob = await (await fetch('/samples/' + name)).blob();
      handleFile(new File([blob], name, { type: blob.type }));
    });
  });
}

function handleFile(file) {
  lastFile = file;
  const img = $('original');
  if (img.src.startsWith('blob:')) URL.revokeObjectURL(img.src);
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  $('dropzone-empty').hidden = true;
  runUpload(file);
}

async function runUpload(file) {
  $('upload-busy').hidden = false;

  const body = new FormData();
  body.append('file', file);
  body.append('conf', confidence);

  const started = performance.now();
  let data;
  try {
    const res = await fetch('/api/detect', { method: 'POST', body });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(detail.detail || res.statusText);
    }
    data = await res.json();
  } catch (err) {
    $('upload-busy').hidden = true;
    $('output-empty').hidden = false;
    $('output-empty').innerHTML = `<p class="muted">${err.message}</p>`;
    return;
  }
  const rtt = performance.now() - started;

  $('annotated').src = data.image_data_url;
  $('annotated').hidden = false;
  $('output-empty').hidden = true;
  $('upload-busy').hidden = true;

  renderCounts($('upload-counts'), data.counts);
  renderDetections($('upload-detections'), data.detections);
  $('upload-total').textContent = data.total;
  $('upload-ms').textContent = data.inference_ms.toFixed(1) + ' ms';
  $('upload-rtt').textContent = rtt.toFixed(0) + ' ms';
  $('upload-size').textContent = `${data.width} x ${data.height}`;

  shownSize = { w: data.width, h: data.height };
  fitStage();

  const link = $('download');
  link.hidden = false;
  link.onclick = () => {
    const a = document.createElement('a');
    a.href = data.image_data_url;
    a.download = 'detected-' + (data.filename || 'image').replace(/\.[^.]+$/, '') + '.jpg';
    a.click();
  };
}

/* -------------------------------------------------------------- sessions */

/* A session runs from Start camera to Stop camera and is owned by the SERVER -
 * the socket opening is what starts it. That is deliberate: the files have to
 * be written somewhere, and a session that ends because the tab was closed
 * still has to be recorded, which a browser-side tally could not guarantee.
 *
 * The number shown is distinct tracker ids, not a sum of per-frame counts.
 * Summing frames would make one apple held up for ten seconds read as 300. */

function renderSession(state) {
  renderCounts($('session-counts'), state.counts);
  $('session-total').textContent = state.total;
  const started = new Date(state.started_at).toLocaleTimeString();
  $('session-meta').textContent =
    `Started ${started} · ${state.duration_seconds.toFixed(0)} s · ${state.frames} frames`;
}

function resetSessionCard() {
  renderCounts($('session-counts'), emptyCounts());
  $('session-total').textContent = '0';
  $('session-meta').textContent = 'Press start to begin a session.';
}

function countChips(counts) {
  const present = Object.entries(counts).filter(([, n]) => n > 0);
  if (!present.length) return '<span class="muted">nothing detected</span>';
  return '<div class="chips">' + present.map(([name, n]) =>
    `<span class="chip"><i style="background:${colourOf(name)}"></i>${n} ${name}</span>`
  ).join('') + '</div>';
}

/* Collapsing the list is remembered across reloads. It is a per-browser
 * convenience, not state anyone else depends on, so localStorage is the right
 * home for it - and it is wrapped because a private window can refuse. */
const SAVED_OPEN_KEY = 'savedSessionsOpen';

function setSavedVisible(visible) {
  $('saved-body').hidden = !visible;
  $('saved-toggle').textContent = visible ? 'Hide' : 'Show';
  $('saved-toggle').setAttribute('aria-expanded', String(visible));
  try {
    localStorage.setItem(SAVED_OPEN_KEY, visible ? '1' : '0');
  } catch (err) {
    /* storage unavailable - the toggle still works for this page view */
  }
}

function restoreSavedVisibility() {
  let open = true;
  try {
    open = localStorage.getItem(SAVED_OPEN_KEY) !== '0';
  } catch (err) {
    /* default to open */
  }
  setSavedVisible(open);
}

$('saved-toggle').addEventListener('click', () => setSavedVisible($('saved-body').hidden));

async function refreshSessions() {
  let data;
  try {
    data = await (await fetch('/api/sessions')).json();
  } catch (err) {
    console.error('could not list sessions', err);
    return;
  }

  $('csv-link').hidden = !data.csv_exists;

  const table = $('session-list');
  if (!data.sessions.length) {
    table.innerHTML =
      '<tr><td class="empty">No sessions recorded yet. Start the camera to make one.</td></tr>';
    return;
  }

  table.innerHTML =
    '<tr><th>Started</th><th>Duration</th><th>Frames</th><th>Counted</th>' +
    '<th>Total</th><th>File</th></tr>' +
    data.sessions.map((s) => `<tr>
      <td>${new Date(s.started_at).toLocaleString()}</td>
      <td class="num">${s.duration_seconds.toFixed(0)} s</td>
      <td class="num">${s.frames}</td>
      <td>${countChips(s.counts)}</td>
      <td class="num"><strong>${s.total}</strong></td>
      <td><a href="/sessions/session_${s.session_id}.json" download>JSON</a></td>
    </tr>`).join('');
}

/* ------------------------------------------------------------------ live */

/* Frames are captured at this width regardless of what the camera gives us.
 * The model resizes to 640 anyway, so sending 1280px frames would only pay
 * for a bigger JPEG encode and a bigger upload. Boxes come back in this
 * coordinate space, and the overlay canvas is the same size, so nothing has
 * to be rescaled when drawing. */
const CAPTURE_WIDTH = 640;

const video = $('video');
const overlay = $('overlay');
const overlayCtx = overlay.getContext('2d');
const capture = document.createElement('canvas');
const captureCtx = capture.getContext('2d', { willReadFrequently: false });

let socket = null;
let stream = null;
let running = false;

/* A webcam hands over the true image, but every video-call app shows you a
 * mirror, so the raw feed reads as "backwards" - text is reversed and moving
 * your hand right moves it right instead of left. Mirroring is therefore on by
 * default, and it is purely cosmetic: the frame sent for detection is taken
 * with drawImage, which ignores CSS transforms, so the model always sees the
 * true image and the counts are identical either way. */
let mirrored = true;
let framesDone = 0;
let sentAt = 0;
let lastReplyAt = 0;
const intervals = [];   // recent gaps between results, for the FPS average

$('live-toggle').addEventListener('click', () => (running ? stopLive() : startLive()));

$('mirror').addEventListener('change', (e) => {
  mirrored = e.target.checked;
  applyMirror();
});

function applyMirror() {
  document.querySelector('.video-stage').classList.toggle('is-mirrored', mirrored);
}

function setStatus(text, kind) {
  const pill = $('live-status');
  pill.textContent = text;
  pill.className = 'pill' + (kind ? ' is-' + kind : '');
}

async function startLive() {
  setStatus('requesting camera', null);
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    setStatus(cameraError(err), 'error');
    return;
  }

  video.srcObject = stream;
  await video.play().catch(() => {});
  if (!video.videoWidth) {
    await new Promise((resolve) => video.addEventListener('loadedmetadata', resolve, { once: true }));
  }

  const scale = CAPTURE_WIDTH / video.videoWidth;
  capture.width = CAPTURE_WIDTH;
  capture.height = Math.round(video.videoHeight * scale);
  overlay.width = capture.width;
  overlay.height = capture.height;

  applyMirror();
  $('video-wrap').classList.add('is-on');
  $('live-toggle').textContent = 'Stop camera';
  $('live-toggle').classList.add('is-stop');
  running = true;
  framesDone = 0;
  lastReplyAt = 0;
  intervals.length = 0;
  resetSessionCard();

  openSocket();
}

function cameraError(err) {
  if (err.name === 'NotAllowedError') return 'permission denied';
  if (err.name === 'NotFoundError') return 'no camera found';
  if (!navigator.mediaDevices) return 'camera blocked - needs localhost or HTTPS';
  return err.name || 'camera unavailable';
}

function openSocket() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${scheme}://${location.host}/ws/detect`);
  socket.binaryType = 'arraybuffer';

  socket.onopen = () => {
    setStatus('live', 'live');
    socket.send(JSON.stringify({ conf: confidence }));
    pump();
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const now = performance.now();

    if (data.error === 'busy') {
      // The server allows one camera session at a time, because the tracker
      // lives on the model and two streams would corrupt each other's ids.
      setStatus(data.detail || 'already running elsewhere', 'error');
      stopLive();
      return;
    }

    if (data.session) renderSession(data.session);

    drawOverlay(data.detections);
    renderCounts($('live-counts'), data.counts);
    $('live-total').textContent = data.total;
    $('live-ms').textContent = data.inference_ms.toFixed(1) + ' ms';
    $('live-rtt').textContent = (now - sentAt).toFixed(0) + ' ms';
    $('live-frames').textContent = ++framesDone;

    // Gap between consecutive results, not the round trip. They are not the
    // same number: a background tab throttles canvas.toBlob to about once a
    // second while each round trip still takes 80 ms, and reporting the round
    // trip there would claim 12 fps for a stream actually running at 1.
    // Rolling mean over 30, because a single gap jitters too much to read and
    // a lifetime average would hide a slowdown.
    if (lastReplyAt) {
      intervals.push(now - lastReplyAt);
      if (intervals.length > 30) intervals.shift();
      const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      $('live-fps').textContent = (1000 / mean).toFixed(1);
    }
    lastReplyAt = now;

    pump();   // the next frame is only grabbed now, so none ever queue up
  };

  socket.onerror = () => setStatus('connection error', 'error');
  socket.onclose = () => { if (running) stopLive(); };
}

function pump() {
  if (!running || !socket || socket.readyState !== WebSocket.OPEN) return;
  if (!video.videoWidth) { requestAnimationFrame(pump); return; }

  captureCtx.drawImage(video, 0, 0, capture.width, capture.height);
  capture.toBlob((blob) => {
    if (!blob || !running || socket.readyState !== WebSocket.OPEN) return;
    sentAt = performance.now();
    socket.send(blob);
  }, 'image/jpeg', 0.7);
}

function drawOverlay(detections) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  const lineWidth = Math.max(2, Math.round(overlay.width * 0.005));
  const fontSize = Math.max(11, Math.round(overlay.width * 0.028));
  overlayCtx.lineWidth = lineWidth;
  overlayCtx.font = `600 ${fontSize}px "Segoe UI", system-ui, sans-serif`;
  overlayCtx.textBaseline = 'top';

  for (const det of detections) {
    let [x0, y0, x1, y1] = det.box;
    // The picture is mirrored in CSS but the coordinates describe the true
    // frame, so reflect them here. Doing it per box rather than by flipping
    // the whole canvas keeps the labels the right way round.
    if (mirrored) {
      [x0, x1] = [overlay.width - x1, overlay.width - x0];
    }
    const colour = colourOf(det.label);

    overlayCtx.strokeStyle = colour;
    overlayCtx.strokeRect(x0, y0, x1 - x0, y1 - y0);

    const text = `${det.label} ${det.confidence.toFixed(2)}`;
    const pad = 3;
    const width = overlayCtx.measureText(text).width + pad * 2;
    const height = fontSize + pad * 2;
    // Above the box, or tucked inside when it is already at the top edge.
    const plateY = y0 - height < 0 ? y0 : y0 - height;

    overlayCtx.fillStyle = colour;
    overlayCtx.fillRect(x0, plateY, width, height);
    overlayCtx.fillStyle = '#fff';
    overlayCtx.fillText(text, x0 + pad, plateY + pad);
  }
}

function finishSession(record) {
  // `null` means the session saw no frames, so nothing was written - starting
  // the camera and stopping it straight away should not leave a file behind.
  if (!record) {
    $('session-meta').textContent = 'Session ended with no frames - nothing saved.';
    return;
  }
  renderSession(record);
  $('session-meta').textContent =
    `Saved as session_${record.session_id}.json · ${record.frames} frames · ` +
    `${record.duration_seconds.toFixed(0)} s`;
}

function stopLive() {
  running = false;

  const sock = socket;
  socket = null;
  if (sock) {
    sock.onclose = null;
    if (sock.readyState === WebSocket.OPEN) {
      // The polite ending: ask the server to close the session and hand back
      // its summary before the socket goes. This is a nicety, not the
      // mechanism - the server writes the session on disconnect either way, so
      // a crashed tab or a pulled cable still leaves a record.
      sock.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if ('session_closed' in data) finishSession(data.session_closed);
        sock.close();
        refreshSessions();
      };
      try {
        sock.send(JSON.stringify({ action: 'end' }));
      } catch (err) {
        sock.close();   // already closing from the server's side
      }
      // If the summary never arrives, do not hold the socket open forever.
      setTimeout(() => {
        if (sock.readyState < WebSocket.CLOSING) sock.close();
        refreshSessions();
      }, 1200);
    } else {
      sock.close();
      setTimeout(refreshSessions, 400);
    }
  }

  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  video.srcObject = null;

  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  $('video-wrap').classList.remove('is-on');
  $('live-toggle').textContent = 'Start camera';
  $('live-toggle').classList.remove('is-stop');
  setStatus('idle', null);
}

// Leaving the tab open with the camera running would keep the GPU busy for a
// page nobody is looking at.
window.addEventListener('pagehide', () => { if (running) stopLive(); });

boot();
