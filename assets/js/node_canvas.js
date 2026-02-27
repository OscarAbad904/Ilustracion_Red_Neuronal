(() => {
  const $ = (id) => document.getElementById(id);

  const dom = {
    celsiusInput: $("nnCelsiusInput"),
    targetF: $("nnTargetF"),
    layerCount: $("nnLayerCount"),
    layerSizes: [$("nnLayerSize1"), $("nnLayerSize2"), $("nnLayerSize3"), $("nnLayerSize4")],
    activation: $("nnActivation"),
    learningRate: $("nnLearningRate"),
    animateSignal: $("nnAnimateSignal"),
    showWeightLabels: $("nnShowWeightLabels"),
    btnForward: $("btnRunForward"),
    btnTrain: $("btnTrainStep"),
    btnReset: $("btnResetWeights"),
    graphStats: $("graphStats"),
    connectionHint: $("connectionHint"),
    canvas: $("nodeCanvas"),
    svg: $("nnSvg"),
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
    activation: "relu",
    epochs: 0,
    lastRun: null,
    hoverNodeKey: null,
    hoverEdgeKey: null,
    animationTimers: [],
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

  function resetNetwork() {
    state.hiddenSizes = activeHiddenSizes();
    state.activation = dom.activation?.value || "relu";
    state.epochs = 0;
    state.hoverNodeKey = null;
    state.hoverEdgeKey = null;

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

  function trainOneEpoch() {
    ensureNetwork();

    const learningRate = Math.max(0.0001, parseNumber(dom.learningRate?.value, 0.02));
    const hiddenActivation = dom.activation?.value || "relu";

    for (const sample of TRAINING_SET) {
      const input = normalizeInput(sample.celsius);
      const target = normalizeTarget(sample.fahrenheit);
      const { activations, zs } = forwardNormalized(input);
      const deltas = new Array(state.network.weights.length);
      const lastIndex = state.network.weights.length - 1;

      deltas[lastIndex] = [
        activations[activations.length - 1][0] - target,
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
    }

    state.epochs += 1;
    state.lastRun = evaluateCurrentInput();
  }

  function clearAnimationTimers() {
    while (state.animationTimers.length) {
      clearTimeout(state.animationTimers.pop());
    }
  }

  function scheduleAnimation(className) {
    clearAnimationTimers();
    if (!dom.animateSignal?.checked) {
      return;
    }

    const nodes = Array.from(dom.svg.querySelectorAll(".nn-node"));
    const edges = Array.from(dom.svg.querySelectorAll(".nn-edge, .nn-edge-label"));
    const sequence = [...edges, ...nodes];

    sequence.forEach((element, index) => {
      state.animationTimers.push(setTimeout(() => {
        element.classList.add(className);
      }, index * 28));
      state.animationTimers.push(setTimeout(() => {
        element.classList.remove(className);
      }, (index * 28) + 350));
    });
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

  function updateStatusText() {
    const hiddenLabel = state.hiddenSizes.length === 1 ? "capa oculta" : "capas ocultas";
    const trainLoss = evaluateDataset(TRAINING_SET);
    const testLoss = evaluateDataset(TEST_SET);
    const output = state.lastRun?.output ?? 0;
    const target = state.lastRun?.target ?? 0;

    if (dom.graphStats) {
      dom.graphStats.textContent = `Arquitectura: 1 entrada | ${state.hiddenSizes.length} ${hiddenLabel} | 1 salida`;
    }

    if (dom.connectionHint) {
      dom.connectionHint.textContent = `y=${output.toFixed(2)} | target=${target.toFixed(2)} | train loss=${trainLoss.toFixed(3)} | test loss=${testLoss.toFixed(3)} | epocas=${state.epochs}`;
    }
  }

  function render() {
    ensureNetwork();
    updateTargetField();
    updateLayerControls();
    state.lastRun = evaluateCurrentInput();
    updateStatusText();

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
    const xPadding = 78;
    const xStart = xPadding;
    const xEnd = width - xPadding;
    const xStep = layerSizes.length > 1 ? (xEnd - xStart) / (layerSizes.length - 1) : 0;
    const topMargin = 90;
    const bottomMargin = 70;
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
      const regionTop = topMargin - 42;
      const regionHeight = height - topMargin - bottomMargin + 84;
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
        y: topMargin - 18,
        "text-anchor": "middle",
        class: "nn-layer-title",
      });
      title.textContent = layerTitles[layerIndex];
      dom.svg.appendChild(title);

      const layerActivation = state.lastRun.activations[layerIndex];
      const layerZ = layerIndex === 0 ? layerActivation : state.lastRun.zs[layerIndex - 1];

      for (let neuronIndex = 0; neuronIndex < size; neuronIndex += 1) {
        const y = size === 1 ? ((topMargin + height - bottomMargin) / 2) : (topMargin + (neuronIndex * yStep));
        const key = `L${layerIndex}N${neuronIndex}`;
        const group = createSvg("g", { class: "nn-node", "data-key": key });
        const circle = createSvg("circle", {
          cx: x,
          cy: y,
          r: nodeRadius,
          class: `nn-node-circle nn-node-circle--${type}`,
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
          valueA.textContent = `a=${state.lastRun.celsius.toFixed(1).replace(/\.0$/, "")}`;
          valueZ.textContent = `z=${state.lastRun.celsius.toFixed(1).replace(/\.0$/, "")}`;
        } else if (type === "output") {
          label.textContent = "y";
          valueA.textContent = `a=${state.lastRun.output.toFixed(2)}`;
          valueZ.textContent = `t=${state.lastRun.target.toFixed(2)}`;
        } else {
          label.textContent = `h${neuronIndex + 1}`;
          valueA.textContent = `a=${layerActivation[neuronIndex].toFixed(3)}`;
          valueZ.textContent = `z=${layerZ[neuronIndex].toFixed(3)}`;
        }

        group.append(circle, label, valueA, valueZ);
        dom.svg.appendChild(group);

        nodeMap.set(key, {
          key,
          layerIndex,
          neuronIndex,
          x,
          y,
          type,
          activation: layerActivation[neuronIndex],
          z: layerZ[neuronIndex],
          group,
        });
      }
    }

    for (let layerIndex = 0; layerIndex < state.network.weights.length; layerIndex += 1) {
      const weightMatrix = state.network.weights[layerIndex];
      const prevActivation = state.lastRun.activations[layerIndex];

      for (let toIndex = 0; toIndex < weightMatrix.length; toIndex += 1) {
        for (let fromIndex = 0; fromIndex < weightMatrix[toIndex].length; fromIndex += 1) {
          const fromKey = `L${layerIndex}N${fromIndex}`;
          const toKey = `L${layerIndex + 1}N${toIndex}`;
          const fromNode = nodeMap.get(fromKey);
          const toNode = nodeMap.get(toKey);
          const weight = weightMatrix[toIndex][fromIndex];
          const key = `${fromKey}->${toKey}`;
          const contribution = prevActivation[fromIndex] * weight;

          const line = createSvg("line", {
            x1: fromNode.x,
            y1: fromNode.y,
            x2: toNode.x,
            y2: toNode.y,
            class: `nn-edge ${weight >= 0 ? "nn-edge--pos" : "nn-edge--neg"}`,
            "data-key": key,
            style: `--edge-color:${colorForWeight(weight)};--edge-width:${edgeWidth(weight)};`,
          });
          dom.svg.appendChild(line);

          let labelGroup = null;
          if (visibleWeightLabels) {
            const midX = (fromNode.x + toNode.x) / 2;
            const midY = (fromNode.y + toNode.y) / 2;
            labelGroup = createSvg("g", { class: "nn-edge-label", "data-key": key });
            const bg = createSvg("rect", {
              x: midX - 22,
              y: midY - 10,
              width: 44,
              height: 18,
              class: "nn-edge-label-bg",
            });
            const text = createSvg("text", {
              x: midX,
              y: midY + 3,
              "text-anchor": "middle",
              class: "nn-edge-label-text",
            });
            text.textContent = `w=${weight.toFixed(2)}`;
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
          ? `Salida estimada: ${state.lastRun.output.toFixed(2)} F`
          : `Activacion: ${meta.activation.toFixed(4)}`;
        const zLabel = meta.type === "input"
          ? `Entrada real: ${state.lastRun.celsius.toFixed(2)} C`
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
    state.lastRun = evaluateCurrentInput();
    render();
    scheduleAnimation("is-signal");
  }

  function handleTrain() {
    trainOneEpoch();
    render();
    scheduleAnimation("is-backprop");
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
