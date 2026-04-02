/* HSIDetect — app.js */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initUpload();
  initCamera();
  fetchStatus();
});

/* ── Status ──────────────────────────────────────────────── */
function fetchStatus() {
  fetch("/api/status")
    .then(r => r.json())
    .then(d => {
      const dot = document.getElementById("statusDot");
      if (dot) dot.style.background = d.status === "online" ? "#10B981" : "#EF4444";
    })
    .catch(() => {});
}

/* ── Tabs ────────────────────────────────────────────────── */
function initTabs() {
  const tabUpload   = document.getElementById("tabUpload");
  const tabCamera   = document.getElementById("tabCamera");
  const uploadPanel = document.getElementById("uploadPanel");
  const cameraPanel = document.getElementById("cameraPanel");

  tabUpload.addEventListener("click", () => {
    tabUpload.classList.add("active");
    tabCamera.classList.remove("active");
    uploadPanel.style.display = "block";
    cameraPanel.style.display = "none";
  });

  tabCamera.addEventListener("click", () => {
    tabCamera.classList.add("active");
    tabUpload.classList.remove("active");
    cameraPanel.style.display = "block";
    uploadPanel.style.display = "none";
  });
}

/* ── Upload ──────────────────────────────────────────────── */
function initUpload() {
  const dropZone   = document.getElementById("dropZone");
  const fileInput  = document.getElementById("fileInput");
  const previewImg = document.getElementById("previewImg");
  const dropHint   = document.getElementById("dropHint");
  const clearBtn   = document.getElementById("clearBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");

  /* Click on drop zone → open file picker */
  dropZone.addEventListener("click", (e) => {
    if (e.target === clearBtn) return;
    fileInput.click();
  });

  /* File selected via picker */
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) showPreview(fileInput.files[0]);
  });

  /* Drag & drop */
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) showPreview(file);
  });

  /* Clear button */
  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    previewImg.src          = "";
    previewImg.style.display = "none";
    clearBtn.style.display   = "none";
    dropHint.style.display   = "block";
    fileInput.value          = "";
    window._uploadFile       = null;
    document.getElementById("resultsSection").style.display = "none";
  });

  /* Analyse button */
  analyzeBtn.addEventListener("click", () => {
    if (!window._uploadFile) {
      alert("Please select an image first.");
      return;
    }
    const fd = new FormData();
    fd.append("image", window._uploadFile);
    runAnalysis(fd, analyzeBtn);
  });

  function showPreview(file) {
    window._uploadFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src           = e.target.result;
      previewImg.style.display = "block";
      dropHint.style.display   = "none";
      clearBtn.style.display   = "block";
    };
    reader.readAsDataURL(file);
  }
}

/* ── Camera ──────────────────────────────────────────────── */
function initCamera() {
  const startBtn     = document.getElementById("startCameraBtn");
  const captureBtn   = document.getElementById("captureBtn");
  const analyzeBtn   = document.getElementById("analyzeCameraBtn");
  const video        = document.getElementById("cameraVideo");
  const canvas       = document.getElementById("cameraCanvas");
  const camPreview   = document.getElementById("cameraPreview");

  startBtn.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject     = stream;
      video.style.display = "block";
      captureBtn.style.display = "inline-flex";
      startBtn.style.display   = "none";
    } catch (e) {
      alert("Camera unavailable: " + e.message);
    }
  });

  captureBtn.addEventListener("click", () => {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    camPreview.src           = canvas.toDataURL("image/jpeg");
    camPreview.style.display = "block";
    canvas.toBlob((blob) => {
      window._cameraBlob = blob;
      analyzeBtn.style.display = "flex";
    }, "image/jpeg");
  });

  analyzeBtn.addEventListener("click", () => {
    if (!window._cameraBlob) { alert("Capture an image first."); return; }
    const fd = new FormData();
    fd.append("image", window._cameraBlob, "capture.jpg");
    runAnalysis(fd, analyzeBtn);
  });
}

/* ── Core Analysis ───────────────────────────────────────── */
function runAnalysis(formData, btn) {
  setLoading(true, btn);
  document.getElementById("resultsSection").style.display = "none";

  fetch("/predict", { method: "POST", body: formData })
    .then((r) => {
      if (!r.ok) return r.json().then((e) => Promise.reject(e.error || "Server error " + r.status));
      return r.json();
    })
    .then((data) => {
      if (data.error) throw new Error(data.error);
      setLoading(false, btn);
      displayResults(data);
    })
    .catch((err) => {
      setLoading(false, btn);
      alert("Analysis failed: " + err);
      console.error(err);
    });
}

function setLoading(on, btn) {
  btn.disabled = on;
  btn.innerHTML = on
    ? '<span class="spinner"></span> Analysing...'
    : '<span class="btn-dot"></span> Analyse Image';
}

/* ── Display Results ─────────────────────────────────────── */
function displayResults(d) {
  console.log("Result:", d);

  const s = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  const c = (id, v) => { const el = document.getElementById(id); if (el) el.className    = v; };
  const i = (id, v) => { const el = document.getElementById(id); if (el) el.src          = v; };

  if (d.is_anomaly) {
    c("resultBanner", "result-banner anomaly");
    c("verdictIcon",  "verdict-icon anomaly");
    document.getElementById("verdictIcon").textContent = "⚠";
    s("verdictText", "ANOMALY DETECTED");
    s("verdictSub",  `Foreign material found in ${d.anomaly_pct}% of grain area.`);
  } else {
    c("resultBanner", "result-banner safe");
    c("verdictIcon",  "verdict-icon safe");
    document.getElementById("verdictIcon").textContent = "✓";
    s("verdictText", "SAFE");
    s("verdictSub",  "No significant foreign material anomalies detected.");
  }

  s("statCoverage",    d.anomaly_pct + "%");
  s("statMaxConf",     d.max_conf    + "%");
  s("statMeanConf",    d.mean_conf   + "%");
  s("statThreshold",   "z > " + d.threshold);
  s("statSensitivity", d.sensitivity || "z > " + d.threshold);

  i("imgOriginal", d.original);
  i("imgHeatmap",  d.heatmap);
  i("imgOverlay",  d.overlay);

  const section = document.getElementById("resultsSection");
  section.style.display = "block";
  section.scrollIntoView({ behavior: "smooth" });
}
