const summaryGrid = document.getElementById("summary-grid");
const summaryNote = document.getElementById("summary-note");
const actionResult = document.getElementById("action-result");
const monitorGrid = document.getElementById("monitor-grid");
const monitorNote = document.getElementById("monitor-note");
const cameraThumb = document.getElementById("camera-thumb");
const cameraMeta = document.getElementById("camera-meta");
const cameraFocusButton = document.getElementById("camera-focus-btn");
const cameraTaskOverlay = document.getElementById("camera-task-overlay");
const cameraStatusOverlay = document.getElementById("camera-status-overlay");
const cameraControlsOverlay = document.getElementById("camera-controls-overlay");
const cameraRatingOverlay = document.getElementById("camera-rating-overlay");
const d435ColorThumb = document.getElementById("d435-color-thumb");
const d435DepthThumb = document.getElementById("d435-depth-thumb");
const d435Meta = document.getElementById("d435-meta");
const d435ColorState = document.getElementById("d435-color-state");
const d435ColorNote = document.getElementById("d435-color-note");
const d435DepthState = document.getElementById("d435-depth-state");
const d435DepthNote = document.getElementById("d435-depth-note");
const controlsHint = document.getElementById("controls-hint");
const ratingHint = document.getElementById("rating-hint");
const singleD435RgbMode = document.body.classList.contains("single-d435-rgb-mode");

const controls = {
  start: document.getElementById("start-btn"),
  stop: document.getElementById("stop-btn"),
  discard: document.getElementById("discard-btn"),
  estop: document.getElementById("estop-btn"),
  clearFault: document.getElementById("clear-fault-btn"),
  quit: document.getElementById("quit-btn"),
};

const ratingButtons = {
  "1": document.getElementById("rating-1-btn"),
  "2": document.getElementById("rating-2-btn"),
  "3": document.getElementById("rating-3-btn"),
  "4": document.getElementById("rating-4-btn"),
};

const ratingPresets = {
  "1": {
    segment_status: "clean",
    success: "success",
    termination_reason: "goal_reached",
  },
  "2": {
    segment_status: "usable",
    success: "partial",
    termination_reason: "near_goal_stop",
  },
  "3": {
    segment_status: "usable",
    success: "fail",
    termination_reason: "operator_stop",
  },
  "4": {
    segment_status: "discard",
    success: "",
    termination_reason: "",
  },
};

const configFields = {
  sceneId: document.getElementById("cfg-scene-id"),
  operatorId: document.getElementById("cfg-operator-id"),
  instruction: document.getElementById("cfg-instruction"),
  taskFamily: document.getElementById("cfg-task-family"),
  targetDescription: document.getElementById("cfg-target-description"),
  cmdVxMax: document.getElementById("cfg-cmd-vx-max"),
  cmdWzMax: document.getElementById("cfg-cmd-wz-max"),
};

let statusCache = null;
let cameraStreamUrl = "";
let cameraReconnectTimer = null;
let cameraFocusMode = false;

const STATUS_POLL_INTERVAL_MS = 300;
const CAMERA_RECONNECT_DELAY_MS = 1000;
const D435_COLOR_REFRESH_MS = 220;
const D435_DEPTH_REFRESH_MS = 320;
const D435_STALE_REFRESH_MS = 1200;
const MONITOR_MAX_HISTORY = 120;
const MONITOR_THRESHOLDS = {
  turn: 0.18,
  correction: 0.10,
  forward: 0.10,
  stopVx: 0.03,
  stopWz: 0.06,
  recoveryVx: -0.04,
};
const monitorState = {
  history: [],
};
const d435RefreshState = {
  color: {
    streamUrl: "",
    reconnectTimer: null,
  },
  depth: {
    streamUrl: "",
    reconnectTimer: null,
  },
};

const inputHints = {
  wireless_controller: {
    controlsHint: "wireless controller + web buttons",
    ratingHint: "D-pad score / 1-4",
    motionHint: "move: D-pad / yaw: X,B",
    actionHint: "Start begin / A stop / Y discard / R2 estop / F1 clear / F2 stand",
    labelHint: "score: D-pad up/right/down/left = 1/2/3/4",
    buttons: {
      start: ["Start", "Start"],
      stop: ["Stop", "A"],
      discard: ["Discard", "Y"],
      estop: ["E-Stop", "R2"],
      clearFault: ["Clear Fault", "F1"],
      quit: ["Quit", "Web"],
    },
  },
  evdev: {
    controlsHint: "keyboard + web buttons",
    ratingHint: "1-4 shortcuts",
    motionHint: "move: W/S A/D / yaw: Q/E",
    actionHint: "R begin / T stop / ESC discard / Space estop / C clear / V stand",
    labelHint: "score: press 1 / 2 / 3 / 4",
    buttons: {
      start: ["Start", "R"],
      stop: ["Stop", "T"],
      discard: ["Discard", "ESC"],
      estop: ["E-Stop", "Space"],
      clearFault: ["Clear Fault", "C"],
      quit: ["Quit", "Web"],
    },
  },
  tty: {
    controlsHint: "keyboard + web buttons",
    ratingHint: "1-4 shortcuts",
    motionHint: "move: W/S A/D / yaw: Q/E",
    actionHint: "R begin / T stop / ESC discard / Space estop / C clear / V stand",
    labelHint: "score: press 1 / 2 / 3 / 4",
    buttons: {
      start: ["Start", "R"],
      stop: ["Stop", "T"],
      discard: ["Discard", "ESC"],
      estop: ["E-Stop", "Space"],
      clearFault: ["Clear Fault", "C"],
      quit: ["Quit", "Web"],
    },
  },
};

function getInputHints(inputBackend) {
  return inputHints[inputBackend] || inputHints.tty;
}

function setLog(value) {
  if (actionResult) {
    actionResult.textContent = value;
  }
}

function setCameraFocusMode(enabled) {
  cameraFocusMode = enabled;
  document.body.classList.toggle("camera-focus-mode", enabled);
  cameraFocusButton.textContent = enabled ? "Exit Focus" : "Focus View";
  cameraFocusButton.setAttribute("aria-pressed", enabled ? "true" : "false");
}

function clearCameraReconnectTimer() {
  if (cameraReconnectTimer !== null) {
    window.clearTimeout(cameraReconnectTimer);
    cameraReconnectTimer = null;
  }
}

function stopCameraStream() {
  clearCameraReconnectTimer();
  cameraThumb.removeAttribute("src");
  cameraThumb.hidden = true;
  cameraStreamUrl = "";
}

function scheduleCameraReconnect() {
  if (cameraReconnectTimer !== null || !statusCache || !statusCache.robot.image_valid) {
    return;
  }
  cameraReconnectTimer = window.setTimeout(() => {
    cameraReconnectTimer = null;
    ensureCameraStream();
  }, CAMERA_RECONNECT_DELAY_MS);
}

function ensureCameraStream() {
  if (!statusCache || !statusCache.robot.image_valid) {
    stopCameraStream();
    return;
  }

  if (cameraStreamUrl) {
    return;
  }

  clearCameraReconnectTimer();
  const nextUrl = `/api/image/stream.mjpeg?t=${Date.now()}`;
  cameraStreamUrl = nextUrl;
  cameraThumb.src = nextUrl;
  cameraThumb.hidden = false;
}

function setButtonHint(button, label, shortcut) {
  const labelNode = button.querySelector(".btn-label");
  const shortcutNode = button.querySelector(".kbd");
  if (labelNode) {
    labelNode.textContent = label;
  }
  if (shortcutNode) {
    shortcutNode.textContent = shortcut;
  }
}

function renderItems(target, items) {
  target.innerHTML = items.map(([label, value]) => (
    `<div class="item"><span class="label">${label}</span><span class="value">${value ?? "-"}</span></div>`
  )).join("");
}

function renderOverlayLines(target, lines) {
  if (!target) {
    return;
  }
  target.replaceChildren();
  const filteredLines = lines.filter((line) => Boolean(line));
  target.hidden = filteredLines.length === 0;
  filteredLines.forEach((line) => {
    const row = document.createElement("div");
    row.className = "camera-overlay-line";
    row.textContent = line;
    target.appendChild(row);
  });
}

function fmt(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "-";
}

function formatSegmentBrief(status) {
  const durationSeconds = Number(status.collector?.segment_duration_s || 0);
  const bufferedFrames = Number(status.collector?.buffered_frames || 0);
  if (status.collector?.recording) {
    return `${durationSeconds.toFixed(1)}s / ${bufferedFrames}f`;
  }
  if (bufferedFrames > 0) {
    return `${bufferedFrames}f buffered`;
  }
  return "idle";
}

function formatTaskBrief(status) {
  const runContext = status.run_context || {};
  return runContext.target_description || runContext.target_type || runContext.task_family || "manual task";
}

function classifyMonitorPhase(status) {
  const vx = Number(status.command?.vx || 0);
  const wz = Number(status.command?.wz || 0);
  const recording = Boolean(status.collector?.recording);
  const faultReason = String(status.collector?.fault_reason || "");

  if (faultReason || vx <= MONITOR_THRESHOLDS.recoveryVx) {
    return { phase: "RECOVERY", motion: "recover" };
  }
  if (Math.abs(vx) <= MONITOR_THRESHOLDS.stopVx && Math.abs(wz) <= MONITOR_THRESHOLDS.stopWz) {
    return { phase: recording ? "STOP" : "IDLE", motion: recording ? "hold" : "idle" };
  }
  if (Math.abs(wz) >= MONITOR_THRESHOLDS.turn && Math.abs(vx) <= 0.05) {
    return {
      phase: wz > 0 ? "TURN_LEFT" : "TURN_RIGHT",
      motion: wz > 0 ? "left turn" : "right turn",
    };
  }
  if (Math.abs(wz) >= MONITOR_THRESHOLDS.correction && Math.abs(vx) >= 0.05) {
    return { phase: "CORRECTION", motion: "correcting" };
  }
  if (vx >= MONITOR_THRESHOLDS.forward) {
    return { phase: "FORWARD", motion: "forward" };
  }
  return { phase: "CORRECTION", motion: "mixed" };
}

function updateMonitorState(status) {
  const classified = classifyMonitorPhase(status);
  const sample = {
    timestamp: Date.now(),
    vx: Number(status.command?.vx || 0),
    wz: Number(status.command?.wz || 0),
    phase: classified.phase,
    motion: classified.motion,
  };
  monitorState.history.push(sample);
  if (monitorState.history.length > MONITOR_MAX_HISTORY) {
    monitorState.history.shift();
  }
  return sample;
}

async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return response.json();
}

async function postAction(url, body = {}) {
  const result = await postJson(url, body);
  setLog(JSON.stringify(result, null, 2));
  await refreshStatus();
  return result;
}

function syncConfigForm(status) {
  const cfg = status.run_context;
  configFields.sceneId.value = cfg.scene_id || "";
  configFields.operatorId.value = cfg.operator_id || "";
  configFields.instruction.value = cfg.instruction || "";
  configFields.taskFamily.value = cfg.task_family || "";
  configFields.targetDescription.value = cfg.target_description || cfg.target_type || "";
  configFields.cmdVxMax.value = cfg.cmd_vx_max ?? "";
  configFields.cmdWzMax.value = cfg.cmd_wz_max ?? "";
}

function setRatingButtonsDisabled(disabled) {
  Object.values(ratingButtons).forEach((button) => {
    button.disabled = disabled;
  });
}

function describeStopPhase(stopPhase) {
  switch (stopPhase) {
    case "waiting_release":
      return "release motion";
    case "finalizing":
      return "finalizing";
    case "label_ready":
      return "ready / 1-4";
    default:
      return stopPhase || "idle";
  }
}

function applyInputHints(inputBackend) {
  const hints = getInputHints(inputBackend);
  controlsHint.textContent = hints.controlsHint;
  ratingHint.textContent = hints.ratingHint;
  Object.entries(hints.buttons).forEach(([name, [label, shortcut]]) => {
    setButtonHint(controls[name], label, shortcut);
  });
  return hints;
}

async function submitRatingPreset(key) {
  if (!statusCache || !statusCache.actions.can_submit_label) {
    return;
  }
  const preset = ratingPresets[key];
  if (!preset) {
    return;
  }
  await postAction("/api/label/submit", preset);
}

function renderMonitor(status, hints) {
  const current = updateMonitorState(status);
  const segmentBrief = formatSegmentBrief(status);
  const labelBrief = status.actions.can_submit_label ? "ready / submit score" : describeStopPhase(status.collector.stop_phase);
  const commandBrief = `vx ${fmt(status.command.vx)} vy ${fmt(status.command.vy)} wz ${fmt(status.command.wz)}`;

  renderItems(monitorGrid, [
    ["phase", current.phase],
    ["cmd", commandBrief],
    ["segment", segmentBrief],
    ["label", labelBrief],
  ]);

  if (monitorNote) {
    if (status.collector.fault_reason) {
      monitorNote.textContent = `fault ${status.collector.fault_reason}`;
    } else if (status.actions.can_submit_label) {
      monitorNote.textContent = hints.labelHint;
    } else if (status.collector.recording) {
      monitorNote.textContent = `recording ${segmentBrief} / keep target clearly visible`;
    } else {
      monitorNote.textContent = hints.motionHint;
    }
  }

  return {
    phase: current.phase,
    motion: current.motion,
    labelBrief,
    segmentBrief,
    commandBrief,
  };
}

function renderCameraOverlays(status, monitorSnapshot, hints, cameraState) {
  const taskFamily = status.run_context.task_family || "legacy_motion";
  const instruction = status.run_context.instruction || "set instruction before recording";
  const taskBrief = formatTaskBrief(status);
  const captureLine = status.collector.recording
    ? `rec ${monitorSnapshot.segmentBrief}`
    : `capture ${status.collector.capture_state}`;
  const safetyLine = status.collector.fault_reason
    ? `fault ${status.collector.fault_reason}`
    : `safety ${status.collector.safety_state} / camera ${cameraState}`;
  const labelLine = status.actions.can_submit_label
    ? hints.labelHint
    : (status.collector.stop_phase && status.collector.stop_phase !== "idle"
        ? `label ${monitorSnapshot.labelBrief}`
        : (status.collector.recording ? "stop current segment to enter rating" : "start when task setup is ready"));

  renderOverlayLines(cameraTaskOverlay, [
    instruction,
    `${taskFamily} / ${taskBrief}`,
  ]);
  renderOverlayLines(cameraStatusOverlay, [
    `phase ${monitorSnapshot.phase} / ${monitorSnapshot.motion}`,
    `cmd ${monitorSnapshot.commandBrief}`,
    captureLine,
    safetyLine,
  ]);
  renderOverlayLines(cameraControlsOverlay, [
    hints.motionHint,
    hints.actionHint,
  ]);
  renderOverlayLines(cameraRatingOverlay, [
    labelLine,
    status.actions.can_submit_label ? "web buttons 1-4 also work" : hints.ratingHint,
  ]);
}

function formatAgeSeconds(value) {
  const age = Number(value);
  if (!Number.isFinite(age) || age < 0) {
    return "-";
  }
  return age < 1 ? `${age.toFixed(2)}s` : `${age.toFixed(1)}s`;
}

function describeD435Reason(reason) {
  const labels = {
    topic_probe_failed: "话题探测失败",
    external_topics_unavailable: "外部话题未就绪",
  };
  if (!reason) {
    return "未就绪";
  }
  return labels[reason] || reason.split("_").join(" ");
}

function setD435Badge(node, tone, text) {
  if (!node) {
    return;
  }
  node.className = `d435-badge ${tone}`;
  node.textContent = text;
}

function clearD435ReconnectTimer(state) {
  if (state.reconnectTimer !== null) {
    window.clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
  }
}

function resetD435Thumb(target, state) {
  clearD435ReconnectTimer(state);
  target.hidden = true;
  target.removeAttribute("src");
  state.streamUrl = "";
}

function scheduleD435Reconnect(target, state, endpoint) {
  if (state.reconnectTimer !== null) {
    return;
  }
  state.reconnectTimer = window.setTimeout(() => {
    state.reconnectTimer = null;
    ensureD435Stream(target, state, endpoint);
  }, CAMERA_RECONNECT_DELAY_MS);
}

function ensureD435Stream(target, state, endpoint) {
  if (state.streamUrl) {
    return;
  }
  clearD435ReconnectTimer(state);
  const nextUrl = `${endpoint}?t=${Date.now()}`;
  state.streamUrl = nextUrl;
  target.src = nextUrl;
  target.hidden = false;
}

function renderD435(status) {
  const d435 = status.d435i || {};
  if (!d435Meta || !d435ColorThumb || !d435DepthThumb) {
    return;
  }

  if (!d435.enabled) {
    d435Meta.textContent = "D435i 未启用";
    setD435Badge(d435ColorState, "offline", "未启用");
    setD435Badge(d435DepthState, "offline", "未启用");
    if (d435ColorNote) {
      d435ColorNote.textContent = "未打开 D435i 采集。";
    }
    if (d435DepthNote) {
      d435DepthNote.textContent = "未打开 D435i 采集。";
    }
    resetD435Thumb(d435ColorThumb, d435RefreshState.color);
    resetD435Thumb(d435DepthThumb, d435RefreshState.depth);
    return;
  }

  if (d435.degraded) {
    d435Meta.textContent = `D435i 降级：${describeD435Reason(d435.degraded_reason)}`;
  } else if (!d435.active) {
    d435Meta.textContent = "D435i 启动中，等待 RGB-D 话题稳定";
  } else if (singleD435RgbMode) {
    d435Meta.textContent = d435.color_valid
      ? `RGB 在线 ${formatAgeSeconds(d435.color_age_s)}`
      : "RGB 等待首帧";
  } else {
    const colorLine = d435.color_valid
      ? `彩色 ${d435.color_fresh ? "在线" : "变慢"} ${formatAgeSeconds(d435.color_age_s)}`
      : "彩色等待首帧";
    const depthLine = d435.depth_valid
      ? `深度 ${d435.depth_fresh ? "在线" : "变慢"} ${formatAgeSeconds(d435.depth_age_s)}`
      : "深度等待首帧";
    d435Meta.textContent = `${colorLine} / ${depthLine}`;
  }

  if (d435ColorState) {
    if (d435.color_valid && d435.color_fresh) {
      setD435Badge(d435ColorState, "live", "在线");
    } else if (d435.color_valid) {
      setD435Badge(d435ColorState, "stale", "帧变慢");
    } else {
      setD435Badge(d435ColorState, d435.degraded ? "offline" : "waiting", d435.degraded ? "异常" : (d435.active ? "等首帧" : "启动中"));
    }
  }
  if (d435ColorNote) {
    d435ColorNote.textContent = d435.color_valid
      ? `${d435.color_fresh ? "彩色流跟随状态轮询刷新" : "彩色流已连接，当前帧稍旧"} · age ${formatAgeSeconds(d435.color_age_s)}`
      : (d435.degraded ? `彩色流不可用：${describeD435Reason(d435.degraded_reason)}` : "等待彩色图首帧。");
  }

  if (d435DepthState) {
    if (d435.depth_valid && d435.depth_fresh) {
      setD435Badge(d435DepthState, "live", "在线");
    } else if (d435.depth_valid) {
      setD435Badge(d435DepthState, "stale", "低频更新");
    } else {
      setD435Badge(d435DepthState, d435.degraded ? "offline" : "waiting", d435.degraded ? "异常" : (d435.active ? "等首帧" : "启动中"));
    }
  }
  if (d435DepthNote) {
    d435DepthNote.textContent = d435.depth_valid
      ? `${d435.depth_fresh ? "深度图缓冲后再切帧，尽量减少闪烁" : "深度流仍在线，当前帧稍旧"} · age ${formatAgeSeconds(d435.depth_age_s)}`
      : (d435.degraded ? `深度流不可用：${describeD435Reason(d435.degraded_reason)}` : "原始深度预览等待首帧。");
  }

  if (d435.color_valid) {
    ensureD435Stream(d435ColorThumb, d435RefreshState.color, "/api/d435i/color.mjpeg");
  } else {
    resetD435Thumb(d435ColorThumb, d435RefreshState.color);
  }

  if (!singleD435RgbMode && d435.depth_valid) {
    ensureD435Stream(d435DepthThumb, d435RefreshState.depth, "/api/d435i/depth.mjpeg");
  } else {
    resetD435Thumb(d435DepthThumb, d435RefreshState.depth);
  }
}

function renderStatus(status) {
  statusCache = status;
  const hints = applyInputHints(status.process_config.input_backend);
  const monitorSnapshot = renderMonitor(status, hints);
  const cameraState = status.robot.image_valid
    ? `${Number(status.robot.image_age_s).toFixed(2)}s`
    : "waiting";

  renderItems(summaryGrid, [
    ["task", status.run_context.task_family || "legacy_motion"],
    ["capture", status.collector.recording ? `on ${Number(status.collector.segment_duration_s || 0).toFixed(1)}s` : status.collector.capture_state],
    ["phase", monitorSnapshot.phase],
    ["label", monitorSnapshot.labelBrief],
    ["safety", status.collector.safety_state],
    ["target", formatTaskBrief(status)],
  ]);

  const summaryHints = [];
  if (status.run_context.instruction) {
    summaryHints.push(status.run_context.instruction);
  }
  if (status.collector.fault_reason) {
    summaryHints.push(`fault ${status.collector.fault_reason}`);
  } else if (status.collector.stop_phase && status.collector.stop_phase !== "idle") {
    summaryHints.push(`label ${describeStopPhase(status.collector.stop_phase)}`);
  } else {
    summaryHints.push(`camera ${cameraState}`);
  }
  summaryNote.textContent = summaryHints.join("  |  ");

  syncConfigForm(status);

  controls.start.disabled = !status.actions.can_start_recording;
  controls.stop.disabled = !status.actions.can_stop_recording;
  controls.discard.disabled = !status.actions.can_discard_segment;
  controls.estop.disabled = !status.actions.can_estop;
  controls.clearFault.disabled = !status.actions.can_clear_fault;
  setRatingButtonsDisabled(!status.actions.can_submit_label);

  if (!singleD435RgbMode && status.robot.image_valid) {
    if (cameraReconnectTimer !== null) {
      cameraThumb.hidden = true;
      cameraMeta.textContent = "stream reconnecting...";
    } else if (Number(status.robot.image_age_s) > 1.5) {
      stopCameraStream();
      cameraMeta.textContent = "stream stalled, reconnecting...";
      scheduleCameraReconnect();
    } else {
      ensureCameraStream();
      cameraThumb.hidden = false;
      cameraMeta.textContent = status.process_config.preview_mode
        ? `image age ${Number(status.robot.image_age_s).toFixed(2)}s / preview stream`
        : `image age ${Number(status.robot.image_age_s).toFixed(2)}s / live stream`;
    }
  } else {
    stopCameraStream();
    if (cameraMeta) {
      cameraMeta.textContent = "waiting for image...";
    }
  }

  if (!singleD435RgbMode) {
    renderCameraOverlays(status, monitorSnapshot, hints, cameraState);
  }
  renderD435(status);
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    renderStatus(status);
  } catch (error) {
    setLog(String(error));
  }
}

cameraThumb.addEventListener("load", () => {
  clearCameraReconnectTimer();
});

cameraThumb.addEventListener("error", () => {
  cameraMeta.textContent = "stream reconnecting...";
  cameraStreamUrl = "";
  scheduleCameraReconnect();
});

d435ColorThumb.addEventListener("error", () => {
  d435RefreshState.color.streamUrl = "";
  scheduleD435Reconnect(d435ColorThumb, d435RefreshState.color, "/api/d435i/color.mjpeg");
});

d435DepthThumb.addEventListener("error", () => {
  d435RefreshState.depth.streamUrl = "";
  scheduleD435Reconnect(d435DepthThumb, d435RefreshState.depth, "/api/d435i/depth.mjpeg");
});

controls.start.addEventListener("click", async () => {
  await postAction("/api/control/start");
});
controls.stop.addEventListener("click", async () => {
  await postAction("/api/control/stop");
});
controls.discard.addEventListener("click", async () => {
  await postAction("/api/control/discard");
});
controls.estop.addEventListener("click", async () => {
  await postAction("/api/control/estop");
});
controls.clearFault.addEventListener("click", async () => {
  await postAction("/api/control/clear-fault");
});
controls.quit.addEventListener("click", async () => {
  const result = await postJson("/api/control/quit");
  setLog(JSON.stringify(result, null, 2));
});

Object.entries(ratingButtons).forEach(([key, button]) => {
  button.addEventListener("click", async () => {
    await submitRatingPreset(key);
  });
});

document.addEventListener("keydown", async (event) => {
  if (event.repeat) {
    return;
  }

  if (event.key === "Escape" && cameraFocusMode) {
    event.preventDefault();
    setCameraFocusMode(false);
    return;
  }

  const activeTag = document.activeElement ? document.activeElement.tagName : "";
  if (activeTag === "INPUT" || activeTag === "TEXTAREA" || activeTag === "SELECT") {
    return;
  }

  if (statusCache && statusCache.actions.can_submit_label && ["1", "2", "3", "4"].includes(event.key)) {
    event.preventDefault();
    await submitRatingPreset(event.key);
  }
});

cameraFocusButton.addEventListener("click", () => {
  setCameraFocusMode(!cameraFocusMode);
});

cameraThumb.addEventListener("dblclick", () => {
  setCameraFocusMode(!cameraFocusMode);
});

setInterval(refreshStatus, STATUS_POLL_INTERVAL_MS);
refreshStatus();
