/**
 * Bi-Temporal Vegetation & Remote Sensing Change Analysis Client Application.
 * Handles interactive layer switching, pan/zoom, region vector overlays,
 * sample pair loading, execution progress tracking, and dynamic UI updates.
 */

// State
const state = {
  currentMode: "sentinel2_multispectral",
  t1File: null,
  t2File: null,
  t1Url: null,
  t2Url: null,
  overlayUrl: null,
  t1VegUrl: null,
  t2VegUrl: null,
  lossUrl: null,
  gainUrl: null,
  activeTab: "changes",
  activeRegionFilter: "all",
  regions: [],
  currentZoom: 1,
  panX: 0,
  panY: 0,
  isPanning: false,
  startX: 0,
  startY: 0,
  selectedSample: "sentinel2_demo",
};

// DOM Elements
const el = {
  btnLoadS2: document.getElementById("btnLoadS2"),
  btnLoadCDVQA: document.getElementById("btnLoadCDVQA"),
  currentModeTag: document.getElementById("currentModeTag"),
  fileT1: document.getElementById("fileT1"),
  fileT2: document.getElementById("fileT2"),
  labelT1: document.getElementById("labelT1"),
  labelT2: document.getElementById("labelT2"),
  thumbT1: document.getElementById("thumbT1"),
  thumbT2: document.getElementById("thumbT2"),
  queryInput: document.getElementById("queryInput"),
  btnRunAnalysis: document.getElementById("btnRunAnalysis"),
  chips: document.querySelectorAll(".chip"),
  
  // Progress & Error
  progressBanner: document.getElementById("progressBanner"),
  progressStatusText: document.getElementById("progressStatusText"),
  errorBanner: document.getElementById("errorBanner"),
  errorMessage: document.getElementById("errorMessage"),
  btnCloseError: document.getElementById("btnCloseError"),
  
  // Viewer
  tabT1: document.getElementById("tabT1"),
  tabT2: document.getElementById("tabT2"),
  tabChanges: document.getElementById("tabChanges"),
  tabVegT1: document.getElementById("tabVegT1"),
  tabVegT2: document.getElementById("tabVegT2"),
  tabLoss: document.getElementById("tabLoss"),
  tabGain: document.getElementById("tabGain"),
  filterBtns: document.querySelectorAll(".btn-filter"),
  filterAll: document.getElementById("filterAll"),
  filterLoss: document.getElementById("filterLoss"),
  filterGain: document.getElementById("filterGain"),
  filterSurface: document.getElementById("filterSurface"),
  filterNone: document.getElementById("filterNone"),
  sliderOpacity: document.getElementById("sliderOpacity"),
  labelOpacity: document.getElementById("labelOpacity"),
  viewport: document.getElementById("viewport"),
  imageStage: document.getElementById("imageStage"),
  viewImgBase: document.getElementById("viewImgBase"),
  viewImgOverlay: document.getElementById("viewImgOverlay"),
  regionSvgOverlay: document.getElementById("regionSvgOverlay"),
  btnZoomIn: document.getElementById("btnZoomIn"),
  btnZoomOut: document.getElementById("btnZoomOut"),
  btnResetZoom: document.getElementById("btnResetZoom"),
  
  // Region Popup
  regionPopup: document.getElementById("regionPopup"),
  popTitle: document.getElementById("popTitle"),
  popType: document.getElementById("popType"),
  popArea: document.getElementById("popArea"),
  popPixels: document.getElementById("popPixels"),
  popT1: document.getElementById("popT1"),
  popT2: document.getElementById("popT2"),
  popCentroid: document.getElementById("popCentroid"),
  btnClosePopup: document.getElementById("btnClosePopup"),
  
  // Legends
  legLoss: document.getElementById("legLoss"),
  legGain: document.getElementById("legGain"),
  legSurface: document.getElementById("legSurface"),
  
  // Statistics
  statsUnitBadge: document.getElementById("statsUnitBadge"),
  statT1Veg: document.getElementById("statT1Veg"),
  statT1Sub: document.getElementById("statT1Sub"),
  statT2Veg: document.getElementById("statT2Veg"),
  statT2Sub: document.getElementById("statT2Sub"),
  statLossArea: document.getElementById("statLossArea"),
  statLossCount: document.getElementById("statLossCount"),
  statGainArea: document.getElementById("statGainArea"),
  statGainCount: document.getElementById("statGainCount"),
  statNetChange: document.getElementById("statNetChange"),
  statNetSub: document.getElementById("statNetSub"),
  statPctChange: document.getElementById("statPctChange"),
  
  // Explanation & Artifacts
  explanationText: document.getElementById("explanationText"),
  intentBadge: document.getElementById("intentBadge"),
  linkGeoJson: document.getElementById("linkGeoJson"),
  linkAnalysisJson: document.getElementById("linkAnalysisJson"),
  linkChangeMask: document.getElementById("linkChangeMask"),
  
  // Evidence & Confidence
  confTierBadge: document.getElementById("confTierBadge"),
  confMeterBar: document.getElementById("confMeterBar"),
  confExplanation: document.getElementById("confExplanation"),
  evidenceList: document.getElementById("evidenceList"),
};

// Initial Setup
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  loadSampleMode("sentinel2_demo");
});

function initEventListeners() {
  // Mode selection buttons
  el.btnLoadS2.addEventListener("click", () => loadSampleMode("sentinel2_demo"));
  el.btnLoadCDVQA.addEventListener("click", () => loadSampleMode("cdvqa_demo"));

  // File Upload inputs
  el.fileT1.addEventListener("change", (e) => handleFileSelect(e, "t1"));
  el.fileT2.addEventListener("change", (e) => handleFileSelect(e, "t2"));

  // Query chips
  el.chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      el.chips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      el.queryInput.value = chip.getAttribute("data-query");
    });
  });

  // Run analysis button
  el.btnRunAnalysis.addEventListener("click", runAnalysis);

  // Viewer tabs
  el.tabT1.addEventListener("click", () => switchViewerTab("t1"));
  el.tabT2.addEventListener("click", () => switchViewerTab("t2"));
  el.tabChanges.addEventListener("click", () => switchViewerTab("changes"));
  el.tabVegT1.addEventListener("click", () => switchViewerTab("veg_t1"));
  el.tabVegT2.addEventListener("click", () => switchViewerTab("veg_t2"));
  el.tabLoss.addEventListener("click", () => switchViewerTab("loss"));
  el.tabGain.addEventListener("click", () => switchViewerTab("gain"));

  // Region Bounding Box Filter Buttons
  el.filterBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const filter = btn.getAttribute("data-filter");
      setRegionFilter(filter);
    });
  });

  // Opacity Slider
  el.sliderOpacity.addEventListener("input", (e) => {
    const val = e.target.value;
    el.labelOpacity.innerText = `${val}%`;
    el.viewImgOverlay.style.opacity = val / 100;
  });

  // Zoom and Pan
  el.btnZoomIn.addEventListener("click", () => adjustZoom(0.2));
  el.btnZoomOut.addEventListener("click", () => adjustZoom(-0.2));
  el.btnResetZoom.addEventListener("click", resetView);

  el.viewport.addEventListener("mousedown", startPan);
  window.addEventListener("mousemove", doPan);
  window.addEventListener("mouseup", endPan);
  el.viewport.addEventListener("wheel", handleWheel, { passive: false });

  // Popups & Banners
  el.btnClosePopup.addEventListener("click", () => el.regionPopup.classList.add("hidden"));
  el.btnCloseError.addEventListener("click", () => el.errorBanner.classList.add("hidden"));
}

// Sample Loader
async function loadSampleMode(sampleId) {
  state.selectedSample = sampleId;
  el.errorBanner.classList.add("hidden");

  if (sampleId === "sentinel2_demo") {
    el.btnLoadS2.classList.add("active");
    el.btnLoadCDVQA.classList.remove("active");
    el.currentModeTag.innerText = "Mode: Sentinel-2 Multispectral";
    state.currentMode = "sentinel2_multispectral";
    el.labelT1.innerText = "sentinel2_t1.tif";
    el.labelT2.innerText = "sentinel2_t2.tif";
    el.legLoss.classList.remove("hidden");
    el.legGain.classList.remove("hidden");
    el.legSurface.classList.remove("hidden");

    // Fetch existing sample blobs
    try {
      const b1 = await fetch("/data/sentinel2_t1.tif").then((r) => r.blob()).catch(() => null);
      const b2 = await fetch("/data/sentinel2_t2.tif").then((r) => r.blob()).catch(() => null);
      if (b1) state.t1File = new File([b1], "sentinel2_t1.tif", { type: "image/tiff" });
      if (b2) state.t2File = new File([b2], "sentinel2_t2.tif", { type: "image/tiff" });
    } catch (e) {}

    // Auto-run analysis for instant demonstration
    runAnalysis();
  } else {
    el.btnLoadCDVQA.classList.add("active");
    el.btnLoadS2.classList.remove("active");
    el.currentModeTag.innerText = "Mode: CDVQA Optical (ChangeFormerV6)";
    state.currentMode = "optical_rgb";
    el.labelT1.innerText = "02180.png (T1)";
    el.labelT2.innerText = "02180.png (T2)";
    el.queryInput.value = "How much vegetation changed?";
    el.legLoss.classList.remove("hidden");
    el.legGain.classList.remove("hidden");
    el.legSurface.classList.remove("hidden");

    try {
      const b1 = await fetch("/data/cdvqa_pair1_t1.png").then((r) => r.blob()).catch(() => null);
      const b2 = await fetch("/data/cdvqa_pair1_t2.png").then((r) => r.blob()).catch(() => null);
      if (b1) state.t1File = new File([b1], "02180.png", { type: "image/png" });
      if (b2) state.t2File = new File([b2], "02180.png", { type: "image/png" });
    } catch (e) {}

    runAnalysis();
  }
}

function handleFileSelect(e, target) {
  const file = e.target.files[0];
  if (!file) return;

  if (target === "t1") {
    state.t1File = file;
    el.labelT1.innerText = file.name;
    if (file.type.startsWith("image/")) {
      el.thumbT1.src = URL.createObjectURL(file);
      el.thumbT1.classList.remove("hidden");
    }
  } else {
    state.t2File = file;
    el.labelT2.innerText = file.name;
    if (file.type.startsWith("image/")) {
      el.thumbT2.src = URL.createObjectURL(file);
      el.thumbT2.classList.remove("hidden");
    }
  }
}

// Viewer Tab Switching
function switchViewerTab(tab) {
  state.activeTab = tab;
  el.tabT1.classList.toggle("active", tab === "t1");
  el.tabT2.classList.toggle("active", tab === "t2");
  el.tabChanges.classList.toggle("active", tab === "changes");
  el.tabVegT1.classList.toggle("active", tab === "veg_t1");
  el.tabVegT2.classList.toggle("active", tab === "veg_t2");
  el.tabLoss.classList.toggle("active", tab === "loss");
  el.tabGain.classList.toggle("active", tab === "gain");

  if (tab === "t1") {
    el.viewImgBase.src = state.t1Url || "";
    el.viewImgOverlay.classList.add("hidden");
    el.regionSvgOverlay.classList.add("hidden");
  } else if (tab === "t2") {
    el.viewImgBase.src = state.t2Url || "";
    el.viewImgOverlay.classList.add("hidden");
    el.regionSvgOverlay.classList.add("hidden");
  } else if (tab === "veg_t1") {
    el.viewImgBase.src = state.t1VegUrl || state.t1Url || "";
    el.viewImgOverlay.classList.add("hidden");
    el.regionSvgOverlay.classList.add("hidden");
  } else if (tab === "veg_t2") {
    el.viewImgBase.src = state.t2VegUrl || state.t2Url || "";
    el.viewImgOverlay.classList.add("hidden");
    el.regionSvgOverlay.classList.add("hidden");
  } else if (tab === "loss") {
    el.viewImgBase.src = state.lossUrl || state.overlayUrl || "";
    el.viewImgOverlay.classList.add("hidden");
    el.regionSvgOverlay.classList.remove("hidden");
    renderVectorOverlay(state.regions, "loss");
  } else if (tab === "gain") {
    el.viewImgBase.src = state.gainUrl || state.overlayUrl || "";
    el.viewImgOverlay.classList.add("hidden");
    el.regionSvgOverlay.classList.remove("hidden");
    renderVectorOverlay(state.regions, "gain");
  } else {
    // Detected changes tab
    el.viewImgBase.src = state.t2Url || state.t1Url || "";
    el.viewImgOverlay.src = state.overlayUrl || "";
    el.viewImgOverlay.classList.remove("hidden");
    el.regionSvgOverlay.classList.remove("hidden");
    renderVectorOverlay(state.regions, state.activeRegionFilter);
  }
}

function setRegionFilter(filter) {
  state.activeRegionFilter = filter;
  el.filterBtns.forEach((b) => b.classList.toggle("active", b.getAttribute("data-filter") === filter));
  renderVectorOverlay(state.regions, filter);
}

// Pan & Zoom
function adjustZoom(delta) {
  state.currentZoom = Math.min(4, Math.max(0.6, state.currentZoom + delta));
  applyTransform();
}

function resetView() {
  state.currentZoom = 1;
  state.panX = 0;
  state.panY = 0;
  applyTransform();
}

function applyTransform() {
  el.imageStage.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.currentZoom})`;
}

function startPan(e) {
  state.isPanning = true;
  state.startX = e.clientX - state.panX;
  state.startY = e.clientY - state.panY;
}

function doPan(e) {
  if (!state.isPanning) return;
  state.panX = e.clientX - state.startX;
  state.panY = e.clientY - state.startY;
  applyTransform();
}

function endPan() {
  state.isPanning = false;
}

function handleWheel(e) {
  e.preventDefault();
  const zoomFactor = e.deltaY < 0 ? 0.1 : -0.1;
  adjustZoom(zoomFactor);
}

// Main Execution
async function runAnalysis() {
  const query = el.queryInput.value.trim();
  if (!query) {
    showError("Please enter a question or select a query.");
    return;
  }

  el.errorBanner.classList.add("hidden");
  el.progressBanner.classList.remove("hidden");
  el.regionPopup.classList.add("hidden");
  el.btnRunAnalysis.disabled = true;

  // Animate processing steps sequence
  const stepIds = [
    "step-validate",
    "step-query",
    "step-cf",
    "step-ndvi",
    "step-regions",
    "step-gainloss",
    "step-stats",
    "step-visuals",
    "step-explain",
  ];
  stepIds.forEach((id) => {
    const s = document.getElementById(id);
    s.classList.remove("active", "completed");
  });

  let currentStepIdx = 0;
  const stepInterval = setInterval(() => {
    if (currentStepIdx < stepIds.length) {
      if (currentStepIdx > 0) {
        document.getElementById(stepIds[currentStepIdx - 1]).classList.remove("active");
        document.getElementById(stepIds[currentStepIdx - 1]).classList.add("completed");
      }
      const activeEl = document.getElementById(stepIds[currentStepIdx]);
      activeEl.classList.add("active");
      el.progressStatusText.innerText = activeEl.innerText;
      currentStepIdx++;
    }
  }, 220);

  const formData = new FormData();
  formData.append("query", query);

  // If files selected from disk, use them; otherwise fetch bundled sample files
  if (state.t1File) {
    formData.append("image_t1", state.t1File);
  } else {
    const sPath = state.selectedSample === "sentinel2_demo" ? "/data/sentinel2_t1.tif" : "/data/cdvqa_pair1_t1.png";
    const b = await fetch(sPath).then((r) => r.blob());
    formData.append("image_t1", b, "sample_t1");
  }

  if (state.t2File) {
    formData.append("image_t2", state.t2File);
  } else {
    const sPath = state.selectedSample === "sentinel2_demo" ? "/data/sentinel2_t2.tif" : "/data/cdvqa_pair1_t2.png";
    const b = await fetch(sPath).then((r) => r.blob());
    formData.append("image_t2", b, "sample_t2");
  }

  try {
    const resp = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });

    clearInterval(stepInterval);
    stepIds.forEach((id) => document.getElementById(id).classList.add("completed"));
    setTimeout(() => el.progressBanner.classList.add("hidden"), 500);

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      const msg = errData.detail?.validation_errors?.join(" ") || errData.detail?.reason || "Analysis failed.";
      showError(msg);
      el.btnRunAnalysis.disabled = false;
      return;
    }

    const data = await resp.json();
    renderAnalysisResults(data);
  } catch (err) {
    clearInterval(stepInterval);
    el.progressBanner.classList.add("hidden");
    showError("Network or server error occurred: " + err.message);
  } finally {
    el.btnRunAnalysis.disabled = false;
  }
}

function showError(msg) {
  el.errorMessage.innerText = msg;
  el.errorBanner.classList.remove("hidden");
}

// Render Results
function renderAnalysisResults(data) {
  const stats = data.statistics;
  const isMultispectral = data.mode === "sentinel2_multispectral";

  // Visuals URLs
  state.t1Url = `/${data.visual_outputs.t1_original}`;
  state.t2Url = `/${data.visual_outputs.t2_original}`;
  state.overlayUrl = `/${data.visual_outputs.change_highlighted}`;
  state.t1VegUrl = data.visual_outputs.t1_vegetation ? `/${data.visual_outputs.t1_vegetation}` : null;
  state.t2VegUrl = data.visual_outputs.t2_vegetation ? `/${data.visual_outputs.t2_vegetation}` : null;
  state.lossUrl = data.visual_outputs.vegetation_loss_preview ? `/${data.visual_outputs.vegetation_loss_preview}` : null;
  state.gainUrl = data.visual_outputs.vegetation_gain_preview ? `/${data.visual_outputs.vegetation_gain_preview}` : null;
  state.regions = data.regions || [];

  // Toggle layer button visibility depending on artifact availability
  el.tabVegT1.style.display = state.t1VegUrl ? "inline-block" : "none";
  el.tabVegT2.style.display = state.t2VegUrl ? "inline-block" : "none";
  el.tabLoss.style.display = state.lossUrl ? "inline-block" : "none";
  el.tabGain.style.display = state.gainUrl ? "inline-block" : "none";

  // Set active filter based on query intent
  if (data.intent && data.intent.intent === "VEGETATION_LOSS") {
    state.activeRegionFilter = "loss";
  } else if (data.intent && data.intent.intent === "VEGETATION_GAIN") {
    state.activeRegionFilter = "gain";
  } else {
    state.activeRegionFilter = "all";
  }
  el.filterBtns.forEach((b) =>
    b.classList.toggle("active", b.getAttribute("data-filter") === state.activeRegionFilter)
  );

  switchViewerTab("changes");

  // Render SVG Vector Overlays
  renderVectorOverlay(data.regions, state.activeRegionFilter);

  // Statistics KPI updates
  el.statsUnitBadge.innerText = stats.area_unit || "pixels (% of image)";
  const unit = stats.area_unit || "ha";

  if (stats.t1_vegetation_pixels !== null) {
    el.statT1Veg.innerText = `${stats.t1_vegetation_area !== null ? stats.t1_vegetation_area : (stats.t1_vegetation_pixels || 0)} ${unit}`;
    el.statT1Sub.innerText = `${(stats.t1_vegetation_pixels || 0).toLocaleString()} px`;

    el.statT2Veg.innerText = `${stats.t2_vegetation_area !== null ? stats.t2_vegetation_area : (stats.t2_vegetation_pixels || 0)} ${unit}`;
    el.statT2Sub.innerText = `${(stats.t2_vegetation_pixels || 0).toLocaleString()} px`;

    el.statLossArea.innerText = `${stats.vegetation_loss_area !== null ? stats.vegetation_loss_area : (stats.vegetation_loss_pixels || 0)} ${unit}`;
    el.statLossCount.innerText = `${stats.loss_region_count || 0} regions`;

    el.statGainArea.innerText = `${stats.vegetation_gain_area !== null ? stats.vegetation_gain_area : (stats.vegetation_gain_pixels || 0)} ${unit}`;
    el.statGainCount.innerText = `${stats.gain_region_count || 0} regions`;

    const net = stats.net_vegetation_change_area !== null ? stats.net_vegetation_change_area : (stats.net_vegetation_change_pixels || 0);
    el.statNetChange.innerText = `${net >= 0 ? "+" : ""}${net} ${unit}`;
    el.statNetChange.className = `kpi-val ${net < 0 ? "text-loss" : "text-gain"}`;

    const pct = stats.percentage_change || 0;
    el.statPctChange.innerText = `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
    el.statPctChange.className = `kpi-val ${pct < 0 ? "text-loss" : "text-gain"}`;
  } else {
    // Fallback mode without vegetation analysis
    el.statT1Veg.innerText = "N/A";
    el.statT1Sub.innerText = "Optical RGB Mode";
    el.statT2Veg.innerText = "N/A";
    el.statT2Sub.innerText = "Optical RGB Mode";

    el.statLossArea.innerText = `${stats.total_changed_area || 0} ${unit}`;
    el.statLossCount.innerText = `${stats.total_regions_count} change regions`;

    el.statGainArea.innerText = "N/A";
    el.statGainCount.innerText = "No Vegetation Change";

    el.statNetChange.innerText = `${stats.total_change_percentage}%`;
    el.statNetChange.className = "kpi-val text-loss";
    el.statNetSub.innerText = "Total surface change";

    el.statPctChange.innerText = `${(stats.total_changed_pixels || 0).toLocaleString()} px`;
  }

  // Explanation & Intent
  el.intentBadge.innerText = `Intent: ${data.intent.intent} (${Math.round(data.intent.confidence * 100)}%)`;
  el.explanationText.innerText = data.explanation;

  // Artifact links
  if (data.visual_outputs.change_regions_geojson) {
    el.linkGeoJson.href = `/${data.visual_outputs.change_regions_geojson}`;
  }
  if (data.visual_outputs.analysis_json) {
    el.linkAnalysisJson.href = `/${data.visual_outputs.analysis_json}`;
  }
  if (data.visual_outputs.change_mask) {
    el.linkChangeMask.href = `/${data.visual_outputs.change_mask}`;
  }

  // Evidence & Confidence
  const conf = data.confidence;
  el.confTierBadge.innerText = `${conf.confidence_tier} (${Math.round(conf.overall_confidence * 100)}%)`;
  el.confMeterBar.style.width = `${Math.round(conf.overall_confidence * 100)}%`;
  el.confExplanation.innerText = conf.explanation;

  // Evidence items
  el.evidenceList.innerHTML = "";
  (data.evidence.items || []).forEach((item) => {
    const div = document.createElement("div");
    div.className = "evidence-item";
    div.innerHTML = `<span class="ev-key">${item.label}:</span><span class="ev-val">${item.value}</span>`;
    el.evidenceList.appendChild(div);
  });
}

// Render Vector Overlays (SVG)
function renderVectorOverlay(regions, filter) {
  el.regionSvgOverlay.innerHTML = "";
  if (!regions || regions.length === 0) return;

  const currentFilter = filter !== undefined ? filter : state.activeRegionFilter;
  if (currentFilter === "none") return;

  // Dynamically compute viewBox
  let w = 256;
  let h = 256;
  if (el.viewImgBase.naturalWidth && el.viewImgBase.naturalWidth > 0) {
    w = el.viewImgBase.naturalWidth;
    h = el.viewImgBase.naturalHeight;
  } else if (regions.length > 0) {
    const maxX = Math.max(...regions.map((r) => (r.bounds ? r.bounds[2] : 256)));
    const maxY = Math.max(...regions.map((r) => (r.bounds ? r.bounds[3] : 256)));
    w = Math.max(w, maxX);
    h = Math.max(h, maxY);
  }
  el.regionSvgOverlay.setAttribute("viewBox", `0 0 ${w} ${h}`);

  const filteredRegions = regions.filter((r) => {
    if (currentFilter === "loss") return r.region_type === "VEGETATION_LOSS";
    if (currentFilter === "gain") return r.region_type === "VEGETATION_GAIN";
    if (currentFilter === "surface") return r.region_type === "SURFACE_CHANGE";
    return true; // "all"
  });

  filteredRegions.forEach((r) => {
    const [minX, minY, maxX, maxY] = r.bounds;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", minX);
    rect.setAttribute("y", minY);
    rect.setAttribute("width", maxX - minX);
    rect.setAttribute("height", maxY - minY);

    let cls = "region-polygon";
    if (r.region_type === "VEGETATION_GAIN") cls += " gain";
    else if (r.region_type === "SURFACE_CHANGE") cls += " surface";
    rect.setAttribute("class", cls);

    rect.addEventListener("click", (e) => {
      e.stopPropagation();
      openRegionPopup(r);
    });

    el.regionSvgOverlay.appendChild(rect);
  });
}

function openRegionPopup(region) {
  el.popTitle.innerText = `Region #${region.region_id}`;
  el.popType.innerText = region.region_type;
  el.popArea.innerText = region.area_physical ? `${region.area_physical} ${region.area_unit}` : `${region.area_pixels} pixels`;
  el.popPixels.innerText = `${region.area_pixels.toLocaleString()} px`;
  el.popT1.innerText = region.t1_vegetation_status;
  el.popT2.innerText = region.t2_vegetation_status;
  el.popCentroid.innerText = `(${region.centroid[0]}, ${region.centroid[1]})`;
  el.regionPopup.classList.remove("hidden");
}
