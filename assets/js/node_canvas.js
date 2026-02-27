(() => {
  const $ = (id) => document.getElementById(id);

  const dom = {
    celsiusInput: $("nnCelsiusInput"),
    targetF: $("nnTargetF"),
    layerCount: $("nnLayerCount"),
    layerSizes: [$("nnLayerSize1"), $("nnLayerSize2"), $("nnLayerSize3"), $("nnLayerSize4")],
    activation: $("nnActivation"),
    learningRate: $("nnLearningRate"),
    epochsPerRun: $("nnEpochsPerRun"),
    animateSignal: $("nnAnimateSignal"),
    showWeightLabels: $("nnShowWeightLabels"),
    btnForward: $("btnRunForward"),
    btnTrain: $("btnTrainStep"),
    btnReset: $("btnResetWeights"),
    graphStats: $("graphStats"),
    connectionHint: $("connectionHint"),
    canvas: $("nodeCanvas"),
    svg: $("nnSvg"),
    trainingChart: $("trainingChart"),
    chartSvg: $("nnChartSvg"),
    chartSummary: $("nnChartSummary"),
    tooltip: $("nnTooltip"),
  };

  if (!dom.svg || !dom.canvas) {
    return;
  }

  const TRAINING_SET = Array.from({ length: 100 }, (_, index) => {
    const celsius = -40 + index;
    return { celsius, fahrenheit: celsiusToFahrenheit(celsius) };
  });

  const TEST_SET = Array.from({ length: 25 }, (_, index) => {
    const celsius = -37.5 + (index * 5);
    return { celsius, fahrenheit: celsiusToFahrenheit(celsius) };
  });

  const state = {
    network: null,
    hiddenSizes: [],
    activation: "tanh",
    epochs: 0,
    lastRun: null,
    hoverNodeKey: null,
    hoverEdgeKey: null,
    animationTimers: [],
    history: [],
    trainingTimer: null,
    isTrainingBatch: false,
    trainingPreview: null,
  };

  function celsiusToFahrenheit(celsius) {
    return (celsius * 9 / 5) + 32;
  }

  function normalizeInput(celsius) {
    return celsius / 100;
  }

  function normalizeTarget(fahrenheit) {
    return (fahrenheit - 32) / 180;
  }

  function denormalizeOutput(value) {
    return (value * 180) + 32;
  }

  function parseNumber(value, fallback) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function parseEpochsPerRun() {
    const parsed = Number.parseInt(dom.epochsPerRun?.value || "10", 10);
    return Math.max(1, Math.min(500, Number.isFinite(parsed) ? parsed : 10));
  }

  function syncEpochsPerRunInput() {
    const epochs = parseEpochsPerRun();
    if (dom.epochsPerRun) {
      dom.epochsPerRun.value = String(epochs);
    }
    return epochs;
  }

  function setTrainingUiState(isTraining) {
    state.isTrainingBatch = isTraining;

    if (dom.btnTrain) {
      dom.btnTrain.disabled = isTraining;
      dom.btnTrain.textContent = isTraining ? "Entrenando..." : "Entrenar";
    }

    if (dom.btnForward) {
      dom.btnForward.disabled = isTraining;
    }

    if (dom.btnReset) {
      dom.btnReset.disabled = isTraining;
    }
  }

  function stopTrainingBatch() {
    if (state.trainingTimer) {
      clearTimeout(state.trainingTimer);
      state.trainingTimer = null;
    }

    state.trainingPreview = null;

    if (state.isTrainingBatch) {
      setTrainingUiState(false);
    }
  }

  function activationFn(name, value) {
    if (name === "sigmoid") {
      return 1 / (1 + Math.exp(-value));
    }
    if (name === "tanh") {
      return Math.tanh(value);
    }
    if (name === "relu") {
      return Math.max(0, value);
    }
    return value;
  }

  function activationDerivative(name, z, activated) {
    if (name === "sigmoid") {
      return activated * (1 - activated);
    }
    if (name === "tanh") {
      return 1 - (activated * activated);
    }
    if (name === "relu") {
      return z > 0 ? 1 : 0;
    }
    return 1;
  }

  function randomWeight(prevSize) {
    const scale = Math.sqrt(2 / Math.max(prevSize, 1));
    return (Math.random() * 2 - 1) * 0.55 * scale;
  }

  function activeHiddenSizes() {
    const count = Math.max(1, Math.min(4, Number.parseInt(dom.layerCount?.value || "1", 10)));
    return dom.layerSizes
      .slice(0, count)
      .map((select) => Math.max(1, Math.min(5, Number.parseInt(select?.value || "1", 10))));
  }

  function updateTargetField() {
    if (!dom.targetF) {
      return;
    }
    const celsius = parseNumber(dom.celsiusInput?.value, 25);
    dom.targetF.value = celsiusToFahrenheit(celsius).toFixed(1).replace(/\.0$/, "");
  }

  function updateLayerControls() {
    const count = Math.max(1, Math.min(4, Number.parseInt(dom.layerCount?.value || "1", 10)));
    dom.layerSizes.forEach((select, index) => {
      if (!select || !select.parentElement) {
        return;
      }
      const enabled = index < count;
      select.disabled = !enabled;
      select.parentElement.classList.toggle("is-disabled", !enabled);
    });
  }

  function architectureSignature(hiddenSizes) {
    return `1-${hiddenSizes.join("-")}-1`;
  }

  function computeLossMetrics() {
    return {
      epoch: state.epochs,
      trainLoss: evaluateDataset(TRAINING_SET),
      testLoss: evaluateDataset(TEST_SET),
    };
  }

  function recordTrainingSnapshot() {
    const metrics = computeLossMetrics();
    const lastItem = state.history[state.history.length - 1];

    if (lastItem && lastItem.epoch === metrics.epoch) {
      state.history[state.history.length - 1] = metrics;
    } else {
      state.history.push(metrics);
      if (state.history.length > 240) {
        state.history = state.history.slice(-240);
      }
    }

    return metrics;
  }

  function resetNetwork() {
    stopTrainingBatch();
    state.hiddenSizes = activeHiddenSizes();
    state.activation = dom.activation?.value || "relu";
    state.epochs = 0;
    state.hoverNodeKey = null;
    state.hoverEdgeKey = null;
    state.history = [];

    const layerSizes = [1, ...state.hiddenSizes, 1];
    const weights = [];
    const biases = [];

    for (let layerIndex = 0; layerIndex < layerSizes.length - 1; layerIndex += 1) {
      const prevSize = layerSizes[layerIndex];
      const nextSize = layerSizes[layerIndex + 1];
      const layerWeights = Array.from({ length: nextSize }, () =>
        Array.from({ length: prevSize }, () => randomWeight(prevSize))
      );
      const layerBiases = Array.from({ length: nextSize }, () => randomWeight(prevSize) * 0.2);
      weights.push(layerWeights);
      biases.push(layerBiases);
    }

    state.network = {
      signature: architectureSignature(state.hiddenSizes),
      layerSizes,
      weights,
      biases,
    };

    state.lastRun = evaluateCurrentInput();
    recordTrainingSnapshot();
  }

  function ensureNetwork() {
    const nextHiddenSizes = activeHiddenSizes();
    const nextSignature = architectureSignature(nextHiddenSizes);
    const nextActivation = dom.activation?.value || "relu";

    if (!state.network || state.network.signature !== nextSignature) {
      resetNetwork();
      return;
    }

    if (state.activation !== nextActivation) {
      state.activation = nextActivation;
      state.lastRun = evaluateCurrentInput();
    }
  }

  function forwardNormalized(normalizedInput) {
    ensureNetwork();

    const activations = [[normalizedInput]];
    const zs = [];

    for (let layerIndex = 0; layerIndex < state.network.weights.length; layerIndex += 1) {
      const weightMatrix = state.network.weights[layerIndex];
      const biasVector = state.network.biases[layerIndex];
      const previousActivation = activations[layerIndex];
      const isOutputLayer = layerIndex === state.network.weights.length - 1;

      const zVector = weightMatrix.map((row, neuronIndex) => {
        let sum = biasVector[neuronIndex];
        for (let inputIndex = 0; inputIndex < row.length; inputIndex += 1) {
          sum += row[inputIndex] * previousActivation[inputIndex];
        }
        return sum;
      });

      const aVector = zVector.map((z) => (
        isOutputLayer ? z : activationFn(state.activation, z)
      ));

      zs.push(zVector);
      activations.push(aVector);
    }

    return { activations, zs };
  }

  function evaluateSample(celsius) {
    const forward = forwardNormalized(normalizeInput(celsius));
    const normalizedOutput = forward.activations[forward.activations.length - 1][0];
    return {
      ...forward,
      normalizedOutput,
      output: denormalizeOutput(normalizedOutput),
    };
  }

  function evaluateCurrentInput() {
    const celsius = parseNumber(dom.celsiusInput?.value, 25);
    const target = celsiusToFahrenheit(celsius);
    const result = evaluateSample(celsius);
    const loss = 0.5 * ((result.output - target) ** 2);

    return {
      celsius,
      target,
      ...result,
      loss,
    };
  }

  function evaluateDataset(dataset) {
    let totalLoss = 0;
    for (const sample of dataset) {
      const result = evaluateSample(sample.celsius);
      totalLoss += 0.5 * ((result.output - sample.fahrenheit) ** 2);
    }
    return totalLoss / dataset.length;
  }

  function trainOneSample(sample, options = {}) {
    const { skipEnsure = false } = options;
    if (!skipEnsure) {
      ensureNetwork();
    }

    const learningRate = Math.max(0.0001, parseNumber(dom.learningRate?.value, 0.02));
    const hiddenActivation = dom.activation?.value || "relu";

    const input = normalizeInput(sample.celsius);
    const target = normalizeTarget(sample.fahrenheit);
    const { activations, zs } = forwardNormalized(input);
    const deltas = new Array(state.network.weights.length);
    const lastIndex = state.network.weights.length - 1;
    const normalizedOutput = activations[activations.length - 1][0];

    deltas[lastIndex] = [
      normalizedOutput - target,
    ];

    for (let layerIndex = lastIndex - 1; layerIndex >= 0; layerIndex -= 1) {
      const nextWeights = state.network.weights[layerIndex + 1];
      const nextDelta = deltas[layerIndex + 1];
      const currentZ = zs[layerIndex];
      const currentA = activations[layerIndex + 1];

      deltas[layerIndex] = currentZ.map((z, neuronIndex) => {
        let weightedSum = 0;
        for (let nextNeuron = 0; nextNeuron < nextWeights.length; nextNeuron += 1) {
          weightedSum += nextWeights[nextNeuron][neuronIndex] * nextDelta[nextNeuron];
        }
        return weightedSum * activationDerivative(hiddenActivation, z, currentA[neuronIndex]);
      });
    }

    for (let layerIndex = 0; layerIndex < state.network.weights.length; layerIndex += 1) {
      const previousActivation = activations[layerIndex];
      const layerDelta = deltas[layerIndex];

      for (let neuronIndex = 0; neuronIndex < state.network.weights[layerIndex].length; neuronIndex += 1) {
        for (let inputIndex = 0; inputIndex < state.network.weights[layerIndex][neuronIndex].length; inputIndex += 1) {
          state.network.weights[layerIndex][neuronIndex][inputIndex] -= (
            learningRate * layerDelta[neuronIndex] * previousActivation[inputIndex]
          );
        }
        state.network.biases[layerIndex][neuronIndex] -= learningRate * layerDelta[neuronIndex];
      }
    }

    const postUpdate = forwardNormalized(input);
    const postNormalizedOutput = postUpdate.activations[postUpdate.activations.length - 1][0];
    const output = denormalizeOutput(postNormalizedOutput);
    const loss = 0.5 * ((output - sample.fahrenheit) ** 2);

    return {
      celsius: sample.celsius,
      target: sample.fahrenheit,
      activations: postUpdate.activations,
      zs: postUpdate.zs,
      normalizedOutput: postNormalizedOutput,
      output,
      loss,
    };
  }

  function trainOneEpoch(options = {}) {
    const { skipEnsure = false, deferSnapshot = false } = options;
    if (!skipEnsure) {
      ensureNetwork();
    }

    for (const sample of TRAINING_SET) {
      trainOneSample(sample, { skipEnsure: true });
    }

    state.epochs += 1;
    if (!deferSnapshot) {
      state.lastRun = evaluateCurrentInput();
      recordTrainingSnapshot();
    }
  }

  function clearAnimationTimers() {
    while (state.animationTimers.length) {
      clearTimeout(state.animationTimers.pop());
    }
  }

  function scheduleAnimation(className, cycles = 1) {
    clearAnimationTimers();
    if (!dom.animateSignal?.checked) {
      return 0;
    }

    const nodes = Array.from(dom.svg.querySelectorAll(".nn-node"));
    const edges = Array.from(dom.svg.querySelectorAll(".nn-edge, .nn-edge-label"));
    const sequence = [...edges, ...nodes];
    const totalCycles = Math.max(1, Math.min(500, Number.isFinite(cycles) ? cycles : 1));

    if (!sequence.length) {
      return 0;
    }

    const targetDurationMs = totalCycles === 1
      ? 700
      : Math.min(6000, Math.max(1200, totalCycles * 90));
    const totalSteps = sequence.length * totalCycles;
    const stepDelay = Math.max(8, Math.floor(targetDurationMs / Math.max(1, totalSteps)));
    const activeDuration = Math.max(120, Math.min(240, stepDelay * 5));

    for (let cycleIndex = 0; cycleIndex < totalCycles; cycleIndex += 1) {
      sequence.forEach((element, index) => {
        const offset = ((cycleIndex * sequence.length) + index) * stepDelay;
        state.animationTimers.push(setTimeout(() => {
          element.classList.add(className);
        }, offset));
        state.animationTimers.push(setTimeout(() => {
          element.classList.remove(className);
        }, offset + activeDuration));
      });
    }

    return ((totalCycles * sequence.length) * stepDelay) + activeDuration;
  }

  function runTrainingBatch(epochCount) {
    stopTrainingBatch();
    ensureNetwork();

    const totalEpochs = Math.max(1, epochCount);
    let completedEpochs = 0;
    let sampleIndex = 0;
    setTrainingUiState(true);

    const runNextSample = () => {
      if (completedEpochs >= totalEpochs) {
        state.trainingTimer = null;
        state.trainingPreview = null;
        state.lastRun = evaluateCurrentInput();
        render();
        setTrainingUiState(false);
        return;
      }

      const sample = TRAINING_SET[sampleIndex];
      const sampleRun = trainOneSample(sample, { skipEnsure: true });
      const epochDisplayIndex = Math.min(totalEpochs, completedEpochs + 1);

      state.trainingPreview = {
        epochIndex: epochDisplayIndex,
        epochCount: totalEpochs,
        sampleIndex: sampleIndex + 1,
        sampleCount: TRAINING_SET.length,
        run: sampleRun,
      };

      sampleIndex += 1;

      if (sampleIndex >= TRAINING_SET.length) {
        sampleIndex = 0;
        completedEpochs += 1;
        state.epochs = completedEpochs;
        recordTrainingSnapshot();
      }

      render({ skipHistory: true });

      const animationDuration = scheduleAnimation("is-backprop", 1);
      const waitTime = dom.animateSignal?.checked
        ? Math.max(90, animationDuration)
        : 16;

      state.trainingTimer = setTimeout(runNextSample, waitTime);
    };

    runNextSample();
  }

  function colorForWeight(weight) {
    const magnitude = Math.min(1, Math.abs(weight));
    if (weight >= 0) {
      return `rgba(63, 122, 219, ${0.38 + (magnitude * 0.52)})`;
    }
    return `rgba(209, 106, 73, ${0.38 + (magnitude * 0.52)})`;
  }

  function edgeWidth(weight) {
    return (1.3 + (Math.min(1.6, Math.abs(weight)) * 1.2)).toFixed(2);
  }

  function createSvg(tagName, attrs = {}) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tagName);
    Object.entries(attrs).forEach(([key, value]) => {
      element.setAttribute(key, String(value));
    });
    return element;
  }

  function pointOnNodeBoundary(sourceNode, targetNode, gap = 0) {
    const dx = targetNode.x - sourceNode.x;
    const dy = targetNode.y - sourceNode.y;
    const halfWidth = (sourceNode.halfWidth ?? 24) + gap;
    const halfHeight = (sourceNode.halfHeight ?? 24) + gap;
    const ratio = Math.max(
      Math.abs(dx) / Math.max(halfWidth, 1),
      Math.abs(dy) / Math.max(halfHeight, 1),
      0.0001
    );

    return {
      x: sourceNode.x + (dx / ratio),
      y: sourceNode.y + (dy / ratio),
    };
  }

  function cubicBezierPoint(start, control1, control2, end, t) {
    const u = 1 - t;
    const tt = t * t;
    const uu = u * u;
    const uuu = uu * u;
    const ttt = tt * t;

    return {
      x: (uuu * start.x) + (3 * uu * t * control1.x) + (3 * u * tt * control2.x) + (ttt * end.x),
      y: (uuu * start.y) + (3 * uu * t * control1.y) + (3 * u * tt * control2.y) + (ttt * end.y),
    };
  }

  function cubicBezierTForX(start, control1, control2, end, targetX) {
    const increasing = end.x >= start.x;
    let low = 0;
    let high = 1;

    for (let iteration = 0; iteration < 18; iteration += 1) {
      const mid = (low + high) / 2;
      const point = cubicBezierPoint(start, control1, control2, end, mid);

      if ((increasing && point.x < targetX) || (!increasing && point.x > targetX)) {
        low = mid;
      } else {
        high = mid;
      }
    }

    return (low + high) / 2;
  }

  function computeEdgeLabelLaneFactor(sourceIndex, sourceSize) {
    if (sourceSize <= 1) {
      return 0.5;
    }

    const outerGapUnits = 0.6;
    const laneFromLeft = sourceSize - sourceIndex;
    const totalUnits = Math.max(1, (sourceSize - 1) + (outerGapUnits * 2));

    return (outerGapUnits + (laneFromLeft - 1)) / totalUnits;
  }

  function buildEdgeGeometry(fromNode, toNode, labelLaneX = null) {
    const start = pointOnNodeBoundary(fromNode, toNode, 8);
    const end = pointOnNodeBoundary(toNode, fromNode, 8);
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const direction = Math.sign(dx) || 1;
    const handleOffset = Math.max(28, Math.min(96, Math.abs(dx) * 0.38));
    const verticalLift = dy * 0.08;
    const control1 = {
      x: start.x + (handleOffset * direction),
      y: start.y + verticalLift,
    };
    const control2 = {
      x: end.x - (handleOffset * direction),
      y: end.y - verticalLift,
    };
    const minX = Math.min(start.x, end.x) + 1;
    const maxX = Math.max(start.x, end.x) - 1;
    const targetX = labelLaneX === null
      ? start.x + (dx * 0.5)
      : Math.max(minX, Math.min(maxX, labelLaneX));
    const labelT = cubicBezierTForX(start, control1, control2, end, targetX);
    const labelPoint = cubicBezierPoint(start, control1, control2, end, labelT);

    return {
      pathData: `M ${start.x} ${start.y} C ${control1.x} ${control1.y}, ${control2.x} ${control2.y}, ${end.x} ${end.y}`,
      labelX: labelPoint.x,
      labelY: labelPoint.y,
    };
  }

  function setTooltip(html, clientX, clientY) {
    if (!dom.tooltip) {
      return;
    }
    const bounds = dom.canvas.getBoundingClientRect();
    dom.tooltip.innerHTML = html;
    dom.tooltip.hidden = false;
    dom.tooltip.style.left = `${clientX - bounds.left + 12}px`;
    dom.tooltip.style.top = `${clientY - bounds.top + 12}px`;
  }

  function hideTooltip() {
    if (dom.tooltip) {
      dom.tooltip.hidden = true;
    }
  }

  function relatedNodeKeysFromEdge(edgeMeta) {
    return new Set([edgeMeta.fromKey, edgeMeta.toKey]);
  }

  function updateStatusText(metrics, displayRun) {
    const hiddenLabel = state.hiddenSizes.length === 1 ? "capa oculta" : "capas ocultas";
    const output = displayRun?.output ?? 0;
    const target = displayRun?.target ?? 0;
    const trainLoss = metrics?.trainLoss ?? 0;
    const testLoss = metrics?.testLoss ?? 0;
    const progressText = state.trainingPreview
      ? ` | epoch=${state.trainingPreview.epochIndex}/${state.trainingPreview.epochCount} | muestra=${state.trainingPreview.sampleIndex}/${state.trainingPreview.sampleCount} | x=${state.trainingPreview.run.celsius.toFixed(1)}`
      : "";

    if (dom.graphStats) {
      dom.graphStats.textContent = `Arquitectura: 1 entrada | ${state.hiddenSizes.length} ${hiddenLabel} | 1 salida`;
    }

    if (dom.connectionHint) {
      dom.connectionHint.textContent = `y=${output.toFixed(2)} | target=${target.toFixed(2)} | train loss=${trainLoss.toFixed(3)} | test loss=${testLoss.toFixed(3)} | epocas=${state.epochs}${progressText}`;
    }
  }

  function renderTrainingChart(metrics) {
    if (!dom.chartSvg || !dom.trainingChart) {
      return;
    }

    const width = Math.max(320, Math.floor(dom.chartSvg.clientWidth || dom.trainingChart.clientWidth || 320));
    const height = Math.max(110, Math.floor(dom.chartSvg.clientHeight || 110));
    const padding = { top: 10, right: 12, bottom: 24, left: 44 };
    const plotWidth = Math.max(1, width - padding.left - padding.right);
    const plotHeight = Math.max(1, height - padding.top - padding.bottom);
    const maxLoss = Math.max(
      0.001,
      ...state.history.flatMap((item) => [item.trainLoss, item.testLoss])
    );
    const tickCount = 4;
    const xDenominator = Math.max(1, state.history.length - 1);
    const xForIndex = (index) => (
      state.history.length === 1
        ? padding.left + (plotWidth / 2)
        : padding.left + ((plotWidth * index) / xDenominator)
    );
    const yForLoss = (loss) => (
      padding.top + plotHeight - ((loss / maxLoss) * plotHeight)
    );
    const latestEntry = state.history[state.history.length - 1];
    const firstEntry = state.history[0];

    dom.chartSvg.innerHTML = "";
    dom.chartSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);

    dom.chartSvg.appendChild(createSvg("rect", {
      x: padding.left,
      y: padding.top,
      width: plotWidth,
      height: plotHeight,
      class: "nn-chart-frame",
    }));

    for (let tick = 0; tick <= tickCount; tick += 1) {
      const y = padding.top + ((plotHeight * tick) / tickCount);
      const lossValue = maxLoss * (1 - (tick / tickCount));

      dom.chartSvg.appendChild(createSvg("line", {
        x1: padding.left,
        y1: y,
        x2: width - padding.right,
        y2: y,
        class: "nn-chart-grid-line",
      }));

      const label = createSvg("text", {
        x: padding.left - 8,
        y: y + 4,
        "text-anchor": "end",
        class: "nn-chart-axis-label",
      });
      label.textContent = lossValue.toFixed(lossValue >= 10 ? 1 : 2);
      dom.chartSvg.appendChild(label);
    }

    const xTicks = Array.from(new Set([
      0,
      Math.max(0, Math.floor((state.history.length - 1) / 2)),
      Math.max(0, state.history.length - 1),
    ]));

    xTicks.forEach((tickIndex) => {
      const x = xForIndex(tickIndex);

      dom.chartSvg.appendChild(createSvg("line", {
        x1: x,
        y1: padding.top,
        x2: x,
        y2: padding.top + plotHeight,
        class: "nn-chart-grid-line nn-chart-grid-line--vertical",
      }));

      const label = createSvg("text", {
        x,
        y: height - 6,
        "text-anchor": "middle",
        class: "nn-chart-axis-label",
      });
      label.textContent = String(state.history[tickIndex]?.epoch ?? tickIndex);
      dom.chartSvg.appendChild(label);
    });

    const trainPoints = [];
    const testPoints = [];

    state.history.forEach((item, index) => {
      const x = xForIndex(index);
      const trainY = yForLoss(item.trainLoss);
      const testY = yForLoss(item.testLoss);
      trainPoints.push(`${x},${trainY}`);
      testPoints.push(`${x},${testY}`);

      dom.chartSvg.appendChild(createSvg("circle", {
        cx: x,
        cy: trainY,
        r: 1.8,
        class: "nn-chart-point nn-chart-point--train nn-chart-point--history",
      }));

      dom.chartSvg.appendChild(createSvg("circle", {
        cx: x,
        cy: testY,
        r: 1.8,
        class: "nn-chart-point nn-chart-point--test nn-chart-point--history",
      }));
    });

    dom.chartSvg.appendChild(createSvg("polyline", {
      points: trainPoints.join(" "),
      class: "nn-chart-line nn-chart-line--train",
    }));

    dom.chartSvg.appendChild(createSvg("polyline", {
      points: testPoints.join(" "),
      class: "nn-chart-line nn-chart-line--test",
    }));

    if (state.history.length === 1) {
      const firstPoint = state.history[0];
      const x = xForIndex(0);
      const trainY = yForLoss(firstPoint.trainLoss);
      const testY = yForLoss(firstPoint.testLoss);

      dom.chartSvg.appendChild(createSvg("circle", {
        cx: x,
        cy: trainY,
        r: 3.2,
        class: "nn-chart-point nn-chart-point--train",
      }));

      dom.chartSvg.appendChild(createSvg("circle", {
        cx: x,
        cy: testY,
        r: 3.2,
        class: "nn-chart-point nn-chart-point--test",
      }));
    }

    if (latestEntry) {
      const latestX = xForIndex(state.history.length - 1);
      const latestTrainY = yForLoss(latestEntry.trainLoss);
      const latestTestY = yForLoss(latestEntry.testLoss);

      dom.chartSvg.appendChild(createSvg("line", {
        x1: latestX,
        y1: padding.top,
        x2: latestX,
        y2: padding.top + plotHeight,
        class: "nn-chart-cursor",
      }));

      dom.chartSvg.appendChild(createSvg("circle", {
        cx: latestX,
        cy: latestTrainY,
        r: 3.2,
        class: "nn-chart-point nn-chart-point--train nn-chart-point--current",
      }));

      dom.chartSvg.appendChild(createSvg("circle", {
        cx: latestX,
        cy: latestTestY,
        r: 3.2,
        class: "nn-chart-point nn-chart-point--test nn-chart-point--current",
      }));
    }

    if (dom.chartSummary) {
      const initialTrain = firstEntry?.trainLoss ?? metrics.trainLoss;
      const currentEpoch = latestEntry?.epoch ?? metrics.epoch;
      dom.chartSummary.textContent = `Epoch ${currentEpoch} | Train ${initialTrain.toFixed(3)} -> ${metrics.trainLoss.toFixed(3)} | Test ${metrics.testLoss.toFixed(3)}`;
    }
  }

  function render(options = {}) {
    const { skipHistory = false } = options;
    ensureNetwork();
    updateTargetField();
    updateLayerControls();
    if (!state.trainingPreview?.run) {
      state.lastRun = evaluateCurrentInput();
    }

    const displayRun = state.trainingPreview?.run || state.lastRun || evaluateCurrentInput();
    const metrics = skipHistory ? computeLossMetrics() : recordTrainingSnapshot();
    updateStatusText(metrics, displayRun);
    renderTrainingChart(metrics);

    const width = Math.max(720, Math.floor(dom.canvas.clientWidth || 720));
    const height = Math.max(420, Math.floor(dom.canvas.clientHeight || 420));
    dom.svg.innerHTML = "";
    dom.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

    const layerSizes = [1, ...state.hiddenSizes, 1];
    const layerTitles = ["Entrada", ...state.hiddenSizes.map((_, index) => `Capa oculta ${index + 1}`), "Salida"];
    const layerType = (index) => {
      if (index === 0) return "input";
      if (index === layerSizes.length - 1) return "output";
      return "hidden";
    };

    const maxLayerSize = Math.max(...layerSizes);
    const chartReservedHeight = dom.trainingChart ? (dom.trainingChart.offsetHeight + 20) : 0;
    const xPadding = 78;
    const xStart = xPadding;
    const xEnd = width - xPadding;
    const xStep = layerSizes.length > 1 ? (xEnd - xStart) / (layerSizes.length - 1) : 0;
    const topMargin = 90;
    const bottomMargin = 70 + chartReservedHeight;
    const nodeRadius = Math.max(
      18,
      Math.min(
        28,
        Math.floor(Math.min(
          height / Math.max(10, (maxLayerSize * 2.6)),
          (xStep || width) / 4.2
        ))
      )
    );
    const visibleWeightLabels = !!dom.showWeightLabels?.checked;

    const nodeMap = new Map();
    const edgeMeta = [];

    for (let layerIndex = 0; layerIndex < layerSizes.length; layerIndex += 1) {
      const size = layerSizes[layerIndex];
      const x = xStart + (layerIndex * xStep);
      const yStep = size === 1 ? 0 : (height - topMargin - bottomMargin) / (size - 1);
      const regionTop = topMargin - 54;
      const regionHeight = height - topMargin - bottomMargin + 96;
      const regionWidth = Math.max(84, Math.min(124, (xStep || 124) - 28));
      const type = layerType(layerIndex);

      dom.svg.appendChild(createSvg("rect", {
        x: x - (regionWidth / 2),
        y: regionTop,
        width: regionWidth,
        height: regionHeight,
        rx: 18,
        class: `nn-layer-region nn-layer-region--${type}`,
      }));

      const title = createSvg("text", {
        x,
        y: topMargin - 36,
        "text-anchor": "middle",
        class: "nn-layer-title",
      });
      title.textContent = layerTitles[layerIndex];
      dom.svg.appendChild(title);

      const layerActivation = displayRun.activations[layerIndex];
      const layerZ = layerIndex === 0 ? layerActivation : displayRun.zs[layerIndex - 1];

      for (let neuronIndex = 0; neuronIndex < size; neuronIndex += 1) {
        const y = size === 1 ? ((topMargin + height - bottomMargin) / 2) : (topMargin + (neuronIndex * yStep));
        const key = `L${layerIndex}N${neuronIndex}`;
        const group = createSvg("g", { class: "nn-node", "data-key": key });
        const shape = createSvg("rect", {
          x: x - nodeRadius,
          y: y - nodeRadius,
          width: nodeRadius * 2,
          height: nodeRadius * 2,
          rx: Math.max(8, Math.round(nodeRadius * 0.42)),
          ry: Math.max(8, Math.round(nodeRadius * 0.42)),
          class: `nn-node-shape nn-node-shape--${type}`,
        });
        const label = createSvg("text", {
          x,
          y: y - 8,
          "text-anchor": "middle",
          class: "nn-node-title",
        });
        const valueA = createSvg("text", {
          x,
          y: y + 5,
          "text-anchor": "middle",
          class: "nn-node-value",
        });
        const valueZ = createSvg("text", {
          x,
          y: y + 17,
          "text-anchor": "middle",
          class: "nn-node-value",
        });

        if (type === "input") {
          label.textContent = "x1";
          valueA.textContent = `a=${displayRun.celsius.toFixed(1).replace(/\.0$/, "")}`;
          valueZ.textContent = `z=${displayRun.celsius.toFixed(1).replace(/\.0$/, "")}`;
        } else if (type === "output") {
          label.textContent = "y";
          valueA.textContent = `a=${displayRun.output.toFixed(2)}`;
          valueZ.textContent = `t=${displayRun.target.toFixed(2)}`;
        } else {
          label.textContent = `h${neuronIndex + 1}`;
          valueA.textContent = `a=${layerActivation[neuronIndex].toFixed(3)}`;
          valueZ.textContent = `z=${layerZ[neuronIndex].toFixed(3)}`;
        }

        group.append(shape, label, valueA, valueZ);
        dom.svg.appendChild(group);

        nodeMap.set(key, {
          key,
          layerIndex,
          neuronIndex,
          x,
          y,
          halfWidth: nodeRadius,
          halfHeight: nodeRadius,
          type,
          activation: layerActivation[neuronIndex],
          z: layerZ[neuronIndex],
          group,
        });
      }
    }

    for (let layerIndex = 0; layerIndex < state.network.weights.length; layerIndex += 1) {
      const weightMatrix = state.network.weights[layerIndex];
      const prevActivation = displayRun.activations[layerIndex];

      for (let toIndex = 0; toIndex < weightMatrix.length; toIndex += 1) {
        for (let fromIndex = 0; fromIndex < weightMatrix[toIndex].length; fromIndex += 1) {
          const fromKey = `L${layerIndex}N${fromIndex}`;
          const toKey = `L${layerIndex + 1}N${toIndex}`;
          const fromNode = nodeMap.get(fromKey);
          const toNode = nodeMap.get(toKey);
          const weight = weightMatrix[toIndex][fromIndex];
          const key = `${fromKey}->${toKey}`;
          const contribution = prevActivation[fromIndex] * weight;
          const labelLaneFactor = computeEdgeLabelLaneFactor(fromIndex, weightMatrix[toIndex].length);
          const laneStartX = fromNode.x + fromNode.halfWidth + 12;
          const laneEndX = toNode.x - toNode.halfWidth - 12;
          const labelLaneX = laneStartX + ((laneEndX - laneStartX) * labelLaneFactor);
          const edgeGeometry = buildEdgeGeometry(fromNode, toNode, labelLaneX);

          const line = createSvg("path", {
            d: edgeGeometry.pathData,
            class: `nn-edge ${weight >= 0 ? "nn-edge--pos" : "nn-edge--neg"}`,
            "data-key": key,
            style: `--edge-color:${colorForWeight(weight)};--edge-width:${edgeWidth(weight)};`,
          });
          dom.svg.appendChild(line);

          let labelGroup = null;
          if (visibleWeightLabels) {
            const labelText = weight.toFixed(4);
            const labelWidth = Math.max(42, Math.min(62, Math.round((labelText.length * 5.7) + 6)));
            const labelX = edgeGeometry.labelX;
            const labelY = edgeGeometry.labelY;
            labelGroup = createSvg("g", { class: "nn-edge-label", "data-key": key });
            const bg = createSvg("rect", {
              x: labelX - (labelWidth / 2),
              y: labelY - 10,
              width: labelWidth,
              height: 18,
              class: "nn-edge-label-bg",
            });
            const text = createSvg("text", {
              x: labelX,
              y: labelY + 3,
              "text-anchor": "middle",
              class: "nn-edge-label-text",
            });
            text.textContent = labelText;
            labelGroup.append(bg, text);
            dom.svg.appendChild(labelGroup);
          }

          edgeMeta.push({
            key,
            fromKey,
            toKey,
            weight,
            contribution,
            line,
            labelGroup,
          });
        }
      }
    }

    const syncHighlightState = () => {
      const hoveredEdge = edgeMeta.find((item) => item.key === state.hoverEdgeKey);
      const relatedKeys = hoveredEdge ? relatedNodeKeysFromEdge(hoveredEdge) : new Set();

      nodeMap.forEach((meta) => {
        const isHoverNode = meta.key === state.hoverNodeKey;
        const isRelated = relatedKeys.has(meta.key);
        meta.group.classList.toggle("is-highlight", isHoverNode);
        meta.group.classList.toggle("is-related", isRelated);
      });

      edgeMeta.forEach((meta) => {
        const isHover = meta.key === state.hoverEdgeKey;
        const isRelated = relatedKeys.has(meta.fromKey) || relatedKeys.has(meta.toKey);
        meta.line.classList.toggle("is-highlight", isHover);
        meta.line.classList.toggle("is-related", !isHover && isRelated);
        if (meta.labelGroup) {
          meta.labelGroup.classList.toggle("is-highlight", isHover);
          meta.labelGroup.classList.toggle("is-signal", false);
        }
      });
    };

    nodeMap.forEach((meta) => {

      meta.group.addEventListener("mouseenter", (event) => {
        state.hoverNodeKey = meta.key;
        state.hoverEdgeKey = null;
        syncHighlightState();
        const valueLabel = meta.type === "output"
          ? `Salida estimada: ${displayRun.output.toFixed(2)} F`
          : `Activacion: ${meta.activation.toFixed(4)}`;
        const zLabel = meta.type === "input"
          ? `Entrada real: ${displayRun.celsius.toFixed(2)} C`
          : `z: ${meta.z.toFixed(4)}`;
        setTooltip(`<strong>${meta.type === "hidden" ? `Neurona ${meta.neuronIndex + 1}` : meta.type === "input" ? "Entrada" : "Salida"}</strong><br>${valueLabel}<br>${zLabel}`, event.clientX, event.clientY);
      });

      meta.group.addEventListener("mousemove", (event) => {
        if (!dom.tooltip?.hidden) {
          setTooltip(dom.tooltip.innerHTML, event.clientX, event.clientY);
        }
      });

      meta.group.addEventListener("mouseleave", () => {
        state.hoverNodeKey = null;
        hideTooltip();
        syncHighlightState();
      });
    });

    edgeMeta.forEach((meta) => {
      const enterHandler = (event) => {
        state.hoverEdgeKey = meta.key;
        state.hoverNodeKey = null;
        syncHighlightState();
        setTooltip(
          `<strong>Conexion</strong><br>Peso: ${meta.weight.toFixed(4)}<br>Contribucion: ${meta.contribution.toFixed(4)}`,
          event.clientX,
          event.clientY
        );
      };

      const moveHandler = (event) => {
        if (!dom.tooltip?.hidden) {
          setTooltip(dom.tooltip.innerHTML, event.clientX, event.clientY);
        }
      };

      const leaveHandler = () => {
        state.hoverEdgeKey = null;
        hideTooltip();
        syncHighlightState();
      };

      meta.line.addEventListener("mouseenter", enterHandler);
      meta.line.addEventListener("mousemove", moveHandler);
      meta.line.addEventListener("mouseleave", leaveHandler);

      if (meta.labelGroup) {
        meta.labelGroup.addEventListener("mouseenter", enterHandler);
        meta.labelGroup.addEventListener("mousemove", moveHandler);
        meta.labelGroup.addEventListener("mouseleave", leaveHandler);
      }
    });

    syncHighlightState();
  }

  function handleForward() {
    stopTrainingBatch();
    state.lastRun = evaluateCurrentInput();
    render();
    scheduleAnimation("is-signal", 1);
  }

  function handleTrain() {
    const epochsToRun = syncEpochsPerRunInput();
    runTrainingBatch(epochsToRun);
  }

  function bindEvents() {
    dom.celsiusInput?.addEventListener("input", () => {
      updateTargetField();
      render();
    });

    dom.layerCount?.addEventListener("change", () => {
      updateLayerControls();
      resetNetwork();
      render();
    });

    dom.layerSizes.forEach((select) => {
      select?.addEventListener("change", () => {
        resetNetwork();
        render();
      });
    });

    dom.activation?.addEventListener("change", () => {
      ensureNetwork();
      render();
    });

    dom.learningRate?.addEventListener("input", () => {
      render();
    });

    dom.epochsPerRun?.addEventListener("change", () => {
      syncEpochsPerRunInput();
    });

    dom.showWeightLabels?.addEventListener("change", () => {
      render();
    });

    dom.btnForward?.addEventListener("click", handleForward);
    dom.btnTrain?.addEventListener("click", handleTrain);
    dom.btnReset?.addEventListener("click", () => {
      resetNetwork();
      render();
    });

    window.addEventListener("resize", render);
  }

  updateTargetField();
  updateLayerControls();
  resetNetwork();
  bindEvents();
  render();
})();
