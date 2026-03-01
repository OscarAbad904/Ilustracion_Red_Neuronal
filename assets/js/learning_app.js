(function () {
  "use strict";

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function formatNumber(value, digits) {
    if (!Number.isFinite(value)) {
      return "-";
    }
    return Number(value).toFixed(digits || 3);
  }

  function erf(x) {
    var sign = x >= 0 ? 1 : -1;
    var abs = Math.abs(x);
    var a1 = 0.254829592;
    var a2 = -0.284496736;
    var a3 = 1.421413741;
    var a4 = -1.453152027;
    var a5 = 1.061405429;
    var p = 0.3275911;
    var t = 1 / (1 + p * abs);
    var y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-abs * abs);
    return sign * y;
  }

  function sigmoid(x) {
    if (x >= 0) {
      var z = Math.exp(-x);
      return 1 / (1 + z);
    }
    var zz = Math.exp(x);
    return zz / (1 + zz);
  }

  function softmax(values) {
    var max = Math.max.apply(null, values);
    var exps = values.map(function (value) {
      return Math.exp(value - max);
    });
    var sum = exps.reduce(function (acc, value) {
      return acc + value;
    }, 0);
    return exps.map(function (value) {
      return value / sum;
    });
  }

  function applyActivation(name, x) {
    switch (name) {
      case "sigmoid": return sigmoid(x);
      case "tanh": return Math.tanh(x);
      case "relu": return Math.max(0, x);
      case "leaky_relu": return x > 0 ? x : 0.05 * x;
      case "elu": return x >= 0 ? x : Math.expm1(x);
      case "softplus": return Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0);
      case "gelu": return 0.5 * x * (1 + erf(x / Math.sqrt(2)));
      case "swish": return x * sigmoid(x);
      default: return x;
    }
  }

  function activationDerivative(name, x, y) {
    switch (name) {
      case "sigmoid": return y * (1 - y);
      case "tanh": return 1 - y * y;
      case "relu": return x > 0 ? 1 : 0;
      case "leaky_relu": return x > 0 ? 1 : 0.05;
      case "elu": return x >= 0 ? 1 : y + 1;
      case "softplus": return sigmoid(x);
      case "gelu": {
        var cdf = 0.5 * (1 + erf(x / Math.sqrt(2)));
        var pdf = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
        return cdf + x * pdf;
      }
      case "swish": {
        var s = sigmoid(x);
        return s + x * s * (1 - s);
      }
      default: return 1;
    }
  }

  function saturationFlag(name, x, y) {
    switch (name) {
      case "sigmoid": return y < 0.05 || y > 0.95;
      case "tanh": return Math.abs(y) > 0.95;
      case "relu":
      case "leaky_relu": return x <= 0;
      case "elu": return x < -3;
      case "softplus": return x < -4 || x > 6;
      case "gelu":
      case "swish": return Math.abs(x) > 4.5;
      default: return false;
    }
  }

  var contentNode = document.getElementById("appContentJson");
  if (!contentNode) {
    return;
  }

  var content = JSON.parse(contentNode.textContent || "{}");
  var examples = content.examples || [];
  var activationCatalog = content.activation_catalog || [];
  var outputActivationCatalog = content.output_activations || [];
  var constraints = content.constraints || { hidden_layers: { min: 0, max: 5 }, neurons: { min: 1, max: 128 } };
  if (!examples.length) {
    return;
  }

  var state = {
    exampleId: examples[0].id,
    hiddenLayers: [],
    hiddenLayerCount: 0,
    outputActivation: examples[0].default_output_activation,
    inputValues: {},
    trainingEpoch: 0,
    trainingHistory: []
  };
  var elements = {
    exampleSelects: Array.prototype.slice.call(document.querySelectorAll("[data-example-select]")),
    exampleSummary: document.getElementById("homeExampleSummary"),
    practiceList: document.getElementById("practiceList"),
    hiddenLayerCount: document.getElementById("hiddenLayerCount"),
    hiddenLayerControls: document.getElementById("hiddenLayerControls"),
    outputActivation: document.getElementById("outputActivation"),
    inputControls: document.getElementById("inputControls"),
    scenarioButtons: document.getElementById("scenarioButtons"),
    resetToRecommended: document.getElementById("resetToRecommended"),
    learningRateInput: document.getElementById("learningRateInput"),
    epochsPerRunInput: document.getElementById("epochsPerRunInput"),
    trainRunButton: document.getElementById("trainRunButton"),
    trainResetButton: document.getElementById("trainResetButton"),
    trainingHint: document.getElementById("trainingHint"),
    trainingEpochValue: document.getElementById("trainingEpochValue"),
    trainingLossTrainValue: document.getElementById("trainingLossTrainValue"),
    trainingLossValValue: document.getElementById("trainingLossValValue"),
    trainingStatusValue: document.getElementById("trainingStatusValue"),
    trainingChartSvg: document.getElementById("trainingChartSvg"),
    trainingNarrative: document.getElementById("trainingNarrative"),
    trainedExpectedValue: document.getElementById("trainedExpectedValue"),
    trainedPredictedValue: document.getElementById("trainedPredictedValue"),
    trainedResidualValue: document.getElementById("trainedResidualValue"),
    trainedReadingValue: document.getElementById("trainedReadingValue"),
    trainedNarrative: document.getElementById("trainedNarrative"),
    targetOutputValue: document.getElementById("targetOutputValue"),
    predictedOutputValue: document.getElementById("predictedOutputValue"),
    lossValue: document.getElementById("lossValue"),
    outputInterpretation: document.getElementById("outputInterpretation"),
    forwardNarrative: document.getElementById("forwardNarrative"),
    layerSummary: document.getElementById("layerSummary"),
    gradientOverview: document.getElementById("gradientOverview"),
    gradientDetails: document.getElementById("gradientDetails"),
    builderDidacticNotes: document.getElementById("builderDidacticNotes"),
    visualArchitecture: document.getElementById("visualArchitecture"),
    visualActivation: document.getElementById("visualActivation"),
    networkCanvasSvg: document.getElementById("networkCanvasSvg"),
    networkCanvasLegend: document.getElementById("networkCanvasLegend"),
    visualNarrative: document.getElementById("visualNarrative"),
    explanationExample: document.getElementById("explanationExample"),
    minimalConfigPanel: document.getElementById("minimalConfigPanel"),
    activationTable: document.getElementById("activationTable"),
    commonErrorsGrid: document.getElementById("commonErrorsGrid")
  };

  function getExampleById(exampleId) {
    return examples.find(function (item) { return item.id === exampleId; }) || examples[0];
  }

  function exampleIndex(exampleId) {
    var index = examples.findIndex(function (item) { return item.id === exampleId; });
    return index < 0 ? 0 : index;
  }

  function buildEmptyLayer() {
    return { neurons: 8, activation: "relu" };
  }

  function copyRecommendedConfig(example) {
    var recommended = (example.recommended_config && example.recommended_config.hidden_layers) || [];
    resetTrainingState();
    state.hiddenLayers = [];
    for (var i = 0; i < constraints.hidden_layers.max; i += 1) {
      state.hiddenLayers.push(recommended[i] ? { neurons: recommended[i].neurons, activation: recommended[i].activation } : buildEmptyLayer());
    }
    state.hiddenLayerCount = recommended.length;
    state.outputActivation = (example.recommended_config && example.recommended_config.output_activation) || example.default_output_activation;
    state.inputValues = {};
    (example.input_features || []).forEach(function (feature) {
      state.inputValues[feature.key] = feature.default;
    });
  }

  function lossName(example) {
    return example.loss_recommendation || "Perdida orientativa";
  }

  function activationLabel(key) {
    var item = activationCatalog.find(function (entry) { return entry.key === key; }) ||
      outputActivationCatalog.find(function (entry) { return entry.key === key; });
    return item ? item.label : key;
  }

  function resetTrainingState() {
    state.trainingEpoch = 0;
    state.trainingHistory = [];
  }

  function parseLearningRate() {
    var parsed = Number(elements.learningRateInput && elements.learningRateInput.value);
    return Number.isFinite(parsed) ? clamp(parsed, 0.001, 1) : 0.05;
  }

  function parseEpochsPerRun() {
    var parsed = Number(elements.epochsPerRunInput && elements.epochsPerRunInput.value);
    return Number.isFinite(parsed) ? Math.round(clamp(parsed, 1, 200)) : 25;
  }

  function trainingSuitability(example, assessment) {
    var score = 0.56;
    var blocked = assessment.layers.filter(function (item) { return item.status === "blocked"; }).length;
    var weak = assessment.layers.filter(function (item) { return item.status === "weak"; }).length;
    var recommendedLayers = ((example.recommended_config && example.recommended_config.hidden_layers) || []).length;

    if (example.default_output_activation === state.outputActivation) {
      score += 0.18;
    } else {
      score -= 0.22;
    }

    if (example.id === "xor_gate" && state.hiddenLayerCount === 0) {
      score -= 0.45;
    }

    if (example.task_type === "regression" && state.outputActivation === "sigmoid") {
      score -= 0.25;
    }

    if (example.task_type === "multiclass" && state.outputActivation !== "softmax") {
      score -= 0.3;
    }

    if (state.hiddenLayerCount === recommendedLayers) {
      score += 0.1;
    } else if (Math.abs(state.hiddenLayerCount - recommendedLayers) >= 2) {
      score -= 0.08;
    }

    score -= (blocked * 0.22);
    score -= (weak * 0.1);

    if (state.hiddenLayerCount >= 4 && blocked === 0) {
      score -= 0.05;
    }

    return clamp(score, 0.04, 0.94);
  }

  function ensureTrainingBaseline(example, assessment) {
    if (state.trainingHistory.length) {
      return;
    }

    var suitability = trainingSuitability(example, assessment);
    var baseTrain = 1.85 - (suitability * 0.55);
    var baseVal = baseTrain + 0.16 + ((1 - suitability) * 0.12);

    if (example.id === "xor_gate" && state.hiddenLayerCount === 0) {
      baseTrain = 1.7;
      baseVal = 1.86;
    }

    state.trainingHistory.push({
      epoch: 0,
      trainLoss: Number(baseTrain.toFixed(4)),
      valLoss: Number(baseVal.toFixed(4))
    });
  }

  function appendTrainingEpochs(example, assessment) {
    ensureTrainingBaseline(example, assessment);

    var batchEpochs = parseEpochsPerRun();
    var learningRate = parseLearningRate();
    var suitability = trainingSuitability(example, assessment);
    var volatility = (1 - suitability) * 0.03;

    for (var step = 0; step < batchEpochs; step += 1) {
      var previous = state.trainingHistory[state.trainingHistory.length - 1];
      var decay = Math.max(0.012, suitability * learningRate * 0.22);
      var trainLoss = previous.trainLoss * (1 - decay);
      var valDecay = Math.max(0.006, decay * (0.8 - ((1 - suitability) * 0.2)));
      var valLoss = previous.valLoss * (1 - valDecay);

      if (suitability < 0.2) {
        trainLoss = Math.max(trainLoss, previous.trainLoss - 0.004);
        valLoss = Math.max(valLoss, previous.valLoss - 0.002 + volatility);
      } else {
        valLoss += volatility * Math.sin((state.trainingEpoch + step + 1) / 3);
      }

      trainLoss = Math.max(0.02, trainLoss);
      valLoss = Math.max(0.03, valLoss);

      state.trainingEpoch += 1;
      state.trainingHistory.push({
        epoch: state.trainingEpoch,
        trainLoss: Number(trainLoss.toFixed(4)),
        valLoss: Number(valLoss.toFixed(4))
      });
    }
  }

  function fillExampleSelects() {
    elements.exampleSelects.forEach(function (select) {
      select.innerHTML = "";
      examples.forEach(function (example) {
        var option = document.createElement("option");
        option.value = example.id;
        option.textContent = example.menu_label;
        if (example.id === state.exampleId) {
          option.selected = true;
        }
        select.appendChild(option);
      });
      select.addEventListener("change", function (event) {
        state.exampleId = event.target.value;
        copyRecommendedConfig(getExampleById(state.exampleId));
        renderAll();
      });
    });
  }

  function fillHiddenLayerCountSelect() {
    elements.hiddenLayerCount.innerHTML = "";
    for (var count = constraints.hidden_layers.min; count <= constraints.hidden_layers.max; count += 1) {
      var option = document.createElement("option");
      option.value = String(count);
      option.textContent = String(count);
      elements.hiddenLayerCount.appendChild(option);
    }
    elements.hiddenLayerCount.addEventListener("change", function (event) {
      state.hiddenLayerCount = Number(event.target.value);
      resetTrainingState();
      renderAll();
    });
  }

  function renderPracticeList() {
    if (!elements.practiceList) {
      return;
    }
    elements.practiceList.innerHTML = "";
    examples.forEach(function (example) {
      var item = document.createElement("li");
      var layerCount = ((example.recommended_config && example.recommended_config.hidden_layers) || []).length;
      item.textContent = example.menu_label + ": base recomendada con " + layerCount + " capa(s) oculta(s), salida " + example.default_output_activation + " y perdida " + lossName(example) + ".";
      elements.practiceList.appendChild(item);
    });
  }

  function renderExampleSummary(example) {
    if (!elements.exampleSummary) {
      return;
    }
    var recommendedLayers = (example.recommended_config && example.recommended_config.hidden_layers) || [];
    elements.exampleSummary.innerHTML =
      "<div class=\"summary-pill\"><strong>Que aprende:</strong> " + example.expected_pattern + "</div>" +
      "<div class=\"summary-pill\"><strong>Minimo que funciona:</strong> " + example.minimal_config + "</div>" +
      "<div class=\"summary-pill\"><strong>Salida recomendada:</strong> " + example.output_activation_recommendation + "</div>" +
      "<div class=\"summary-pill\"><strong>Base por defecto:</strong> " + recommendedLayers.length + " capa(s) oculta(s), salida " + example.default_output_activation + ".</div>";
  }

  function renderOutputActivationOptions(example) {
    var allowed = example.supported_output_activations || [example.default_output_activation];
    if (allowed.indexOf(state.outputActivation) === -1) {
      state.outputActivation = example.default_output_activation;
    }
    elements.outputActivation.innerHTML = "";
    outputActivationCatalog.forEach(function (activation) {
      if (allowed.indexOf(activation.key) === -1) {
        return;
      }
      var option = document.createElement("option");
      option.value = activation.key;
      option.textContent = activation.label + " - " + activation.summary;
      if (activation.key === state.outputActivation) {
        option.selected = true;
      }
      elements.outputActivation.appendChild(option);
    });
  }

  function renderHiddenLayerControls() {
    elements.hiddenLayerControls.innerHTML = "";
    if (state.hiddenLayerCount === 0) {
      var empty = document.createElement("p");
      empty.className = "plain-note";
      empty.textContent = "Sin capas ocultas: solo hay una transformacion directa de entrada a salida.";
      elements.hiddenLayerControls.appendChild(empty);
      return;
    }

    for (var index = 0; index < state.hiddenLayerCount; index += 1) {
      var layer = state.hiddenLayers[index] || buildEmptyLayer();
      var wrap = document.createElement("div");
      wrap.className = "layer-editor";
      wrap.innerHTML =
        "<div class=\"layer-editor__title\">Capa oculta " + (index + 1) + "</div>" +
        "<div class=\"layer-editor__row\">" +
        "<label class=\"field\"><span>Neuronas</span><input type=\"number\" min=\"" + constraints.neurons.min + "\" max=\"" + constraints.neurons.max + "\" value=\"" + layer.neurons + "\" data-layer-index=\"" + index + "\" data-field=\"neurons\"></label>" +
        "<label class=\"field\"><span>Activacion</span><select data-layer-index=\"" + index + "\" data-field=\"activation\"></select></label>" +
        "</div>";
      var select = wrap.querySelector("select");
      activationCatalog.forEach(function (activation) {
        var option = document.createElement("option");
        option.value = activation.key;
        option.textContent = activation.label;
        if (activation.key === layer.activation) {
          option.selected = true;
        }
        select.appendChild(option);
      });
      elements.hiddenLayerControls.appendChild(wrap);
    }

    Array.prototype.slice.call(elements.hiddenLayerControls.querySelectorAll("input, select")).forEach(function (field) {
      field.addEventListener("change", function (event) {
        var layerIndex = Number(event.target.dataset.layerIndex);
        if (event.target.dataset.field === "neurons") {
          var parsed = Number(event.target.value);
          state.hiddenLayers[layerIndex].neurons = Number.isFinite(parsed) ? clamp(parsed, constraints.neurons.min, constraints.neurons.max) : constraints.neurons.min;
        } else {
          state.hiddenLayers[layerIndex].activation = event.target.value;
        }
        resetTrainingState();
        renderAll();
      });
    });
  }

  function renderInputControls(example) {
    elements.inputControls.innerHTML = "";
    (example.input_features || []).forEach(function (feature) {
      var wrap = document.createElement("div");
      wrap.className = "input-control";
      wrap.innerHTML =
        "<div class=\"input-control__head\"><strong>" + feature.label + "</strong><span>" + feature.description + "</span></div>" +
        "<div class=\"input-control__row\">" +
        "<input type=\"range\" min=\"" + feature.min + "\" max=\"" + feature.max + "\" step=\"" + feature.step + "\" value=\"" + state.inputValues[feature.key] + "\" data-feature-key=\"" + feature.key + "\">" +
        "<input type=\"number\" min=\"" + feature.min + "\" max=\"" + feature.max + "\" step=\"" + feature.step + "\" value=\"" + state.inputValues[feature.key] + "\" data-feature-key=\"" + feature.key + "\">" +
        "<span class=\"input-control__unit\">" + (feature.unit || "") + "</span>" +
        "</div>";
      var slider = wrap.querySelector("input[type='range']");
      var number = wrap.querySelector("input[type='number']");
      function syncValue(nextValue) {
        var parsed = Number(nextValue);
        var safe = Number.isFinite(parsed) ? clamp(parsed, Number(feature.min), Number(feature.max)) : Number(feature.min);
        state.inputValues[feature.key] = safe;
        slider.value = String(safe);
        number.value = String(safe);
        renderSimulation();
      }
      slider.addEventListener("input", function (event) { syncValue(event.target.value); });
      number.addEventListener("change", function (event) { syncValue(event.target.value); });
      elements.inputControls.appendChild(wrap);
    });
  }

  function renderScenarioButtons() {
    elements.scenarioButtons.innerHTML = "";
    (content.gradient_scenarios || []).forEach(function (scenario) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "button button--secondary";
      button.textContent = scenario.label;
      button.title = scenario.summary;
      button.addEventListener("click", function () {
        var apply = scenario.apply || {};
        state.exampleId = apply.example_id || state.exampleId;
        copyRecommendedConfig(getExampleById(state.exampleId));
        var layers = apply.hidden_layers || [];
        state.hiddenLayerCount = Math.min(layers.length, constraints.hidden_layers.max);
        for (var index = 0; index < constraints.hidden_layers.max; index += 1) {
          state.hiddenLayers[index] = layers[index] ? { neurons: layers[index].neurons, activation: layers[index].activation } : buildEmptyLayer();
        }
        if (apply.output_activation) {
          state.outputActivation = apply.output_activation;
        }
        if (apply.input_overrides) {
          Object.keys(apply.input_overrides).forEach(function (key) {
            if (Object.prototype.hasOwnProperty.call(state.inputValues, key)) {
              state.inputValues[key] = apply.input_overrides[key];
            }
          });
        }
        renderAll();
      });
      elements.scenarioButtons.appendChild(button);
    });
  }

  function renderExplanationPanels(example) {
    elements.explanationExample.innerHTML =
      "<p><strong>Problema:</strong> " + example.description + "</p>" +
      "<p><strong>Que debe aprender:</strong> " + example.expected_pattern + "</p>" +
      "<p><strong>Ejemplo real:</strong> " + example.real_example + "</p>" +
      "<p><strong>Perdida adecuada:</strong> " + lossName(example) + "</p>" +
      "<p><strong>Salida recomendada:</strong> " + example.output_activation_recommendation + "</p>";

    var failures = (example.common_failures || []).map(function (failure) {
      return "<li>" + failure + "</li>";
    }).join("");
    elements.minimalConfigPanel.innerHTML =
      "<p><strong>Minimo util:</strong> " + example.minimal_config + "</p>" +
      "<p><strong>Configuracion base:</strong> " + ((example.recommended_config && example.recommended_config.note) || "La arquitectura recomendada prioriza claridad.") + "</p>" +
      "<p><strong>Fallos tipicos:</strong></p><ul class=\"compact-list\">" + failures + "</ul>";
  }

  function renderActivationTable() {
    elements.activationTable.innerHTML = "";
    activationCatalog.forEach(function (activation) {
      var row = document.createElement("tr");
      row.innerHTML =
        "<td><strong>" + activation.label + "</strong></td>" +
        "<td>" + activation.summary + "</td>" +
        "<td>" + activation.best_for + "</td>" +
        "<td>" + activation.watch_out + "</td>" +
        "<td>" + activation.gradient + "</td>";
      elements.activationTable.appendChild(row);
    });
  }

  function renderCommonErrors() {
    elements.commonErrorsGrid.innerHTML = "";
    (content.common_errors || []).forEach(function (item) {
      var card = document.createElement("article");
      card.className = "card";
      card.innerHTML =
        "<h3>" + item.title + "</h3>" +
        "<p><strong>Sintoma:</strong> " + item.symptom + "</p>" +
        "<p><strong>Como detectarlo:</strong> " + item.detect + "</p>" +
        "<p><strong>Que corregir:</strong> " + item.fix + "</p>";
      elements.commonErrorsGrid.appendChild(card);
    });
  }

  function featureVector(example) {
    return (example.input_features || []).map(function (feature) {
      var current = Number(state.inputValues[feature.key]);
      var midpoint = (Number(feature.max) + Number(feature.min)) / 2;
      var span = Math.max(0.0001, (Number(feature.max) - Number(feature.min)) / 2);
      return (current - midpoint) / span;
    });
  }

  function networkWeight(example, layerIndex, neuronIndex, inputIndex) {
    var seed = exampleIndex(example.id) + 1;
    var raw = ((seed * 17) + ((layerIndex + 1) * 13) + ((neuronIndex + 1) * 7) + ((inputIndex + 1) * 5)) % 19;
    return (raw - 9) / 10;
  }

  function networkBias(example, layerIndex, neuronIndex) {
    var seed = exampleIndex(example.id) + 1;
    var raw = ((seed * 11) + ((layerIndex + 2) * 5) + ((neuronIndex + 1) * 3)) % 13;
    return (raw - 6) / 8;
  }

  function computeRegressionTarget(example) {
    if (example.id === "celsius_fahrenheit") {
      return Number(state.inputValues.celsius) * 9 / 5 + 32;
    }
    var temperature = Number(state.inputValues.temperature);
    var humidity = Number(state.inputValues.humidity);
    var wind = Number(state.inputValues.wind);
    var heatBoost = Math.max(0, temperature - 24) * (humidity / 100) * 0.9;
    var coldBoost = temperature < 10 ? (10 - temperature) * 0.12 : 0;
    return temperature + ((humidity - 50) * 0.06) - (wind * 0.18) + heatBoost - coldBoost;
  }

  function computeBinaryTarget(example) {
    if (example.id === "xor_gate") {
      return Number(state.inputValues.x1) !== Number(state.inputValues.x2) ? 1 : 0;
    }
    var caps = Number(state.inputValues.caps_ratio);
    var links = Number(state.inputValues.links);
    var urgency = Number(state.inputValues.urgency);
    var score = -2.6 + (caps * 0.04) + (links * 0.35) + (urgency * 0.28);
    return sigmoid(score) >= 0.5 ? 1 : 0;
  }

  function computeMulticlassTarget(example) {
    var length = Number(state.inputValues.petal_length);
    var width = Number(state.inputValues.petal_width);
    var fragrance = Number(state.inputValues.fragrance);
    var logits = [
      (0.9 * length) + (0.7 * width) - (0.25 * fragrance),
      (0.4 * length) + (1.0 * width) + (0.35 * fragrance),
      (-0.2 * length) + (0.3 * width) + (0.85 * (10 - fragrance))
    ];
    var probabilities = softmax(logits);
    var classIndex = probabilities.indexOf(Math.max.apply(null, probabilities));
    return { probabilities: probabilities, classIndex: classIndex, className: (example.classes || [])[classIndex] || "Clase " + (classIndex + 1) };
  }

  function simulateNetwork(example) {
    var current = featureVector(example);
    var hiddenLayers = state.hiddenLayers.slice(0, state.hiddenLayerCount);
    var layerStats = [];

    hiddenLayers.forEach(function (layerConfig, layerIndex) {
      var outputs = [];
      var preactivations = [];
      var derivatives = [];
      var zeroCount = 0;
      var saturationCount = 0;

      for (var neuronIndex = 0; neuronIndex < layerConfig.neurons; neuronIndex += 1) {
        var weighted = 0;
        for (var inputIndex = 0; inputIndex < current.length; inputIndex += 1) {
          weighted += current[inputIndex] * networkWeight(example, layerIndex, neuronIndex, inputIndex);
        }
        weighted += networkBias(example, layerIndex, neuronIndex);
        var activation = applyActivation(layerConfig.activation, weighted);
        var derivative = Math.abs(activationDerivative(layerConfig.activation, weighted, activation));
        outputs.push(activation);
        preactivations.push(weighted);
        derivatives.push(derivative);
        if (Math.abs(activation) < 1e-6) {
          zeroCount += 1;
        }
        if (saturationFlag(layerConfig.activation, weighted, activation)) {
          saturationCount += 1;
        }
      }

      layerStats.push({
        index: layerIndex + 1,
        neurons: layerConfig.neurons,
        activationName: layerConfig.activation,
        preMin: Math.min.apply(null, preactivations),
        preMax: Math.max.apply(null, preactivations),
        activationMean: outputs.reduce(function (acc, value) { return acc + value; }, 0) / outputs.length,
        derivativeMean: derivatives.reduce(function (acc, value) { return acc + value; }, 0) / derivatives.length,
        zeroRatio: zeroCount / outputs.length,
        saturationRatio: saturationCount / outputs.length,
        values: outputs
      });

      current = outputs;
    });

    if (state.outputActivation === "softmax" || example.output_size > 1) {
      var outputCount = example.output_size || 3;
      var logits = [];
      for (var outIndex = 0; outIndex < outputCount; outIndex += 1) {
        var logit = 0;
        for (var sourceIndex = 0; sourceIndex < current.length; sourceIndex += 1) {
          logit += current[sourceIndex] * networkWeight(example, hiddenLayers.length + 1, outIndex + 20, sourceIndex);
        }
        logit += networkBias(example, hiddenLayers.length + 1, outIndex + 20);
        logits.push(logit);
      }
      var probabilities = softmax(logits);
      var bestIndex = probabilities.indexOf(Math.max.apply(null, probabilities));
      return { layers: layerStats, output: { type: "multiclass", logits: logits, probabilities: probabilities, classIndex: bestIndex, activation: probabilities[bestIndex] } };
    }

    var raw = 0;
    for (var currentIndex = 0; currentIndex < current.length; currentIndex += 1) {
      raw += current[currentIndex] * networkWeight(example, hiddenLayers.length + 1, 0, currentIndex);
    }
    raw += networkBias(example, hiddenLayers.length + 1, 0);
    return { layers: layerStats, output: { type: "scalar", raw: raw, activation: applyActivation(state.outputActivation, raw) } };
  }

  function outputGradientBase(network) {
    if (state.outputActivation === "sigmoid") {
      return clamp(network.output.activation * (1 - network.output.activation), 0.02, 0.25);
    }
    if (state.outputActivation === "softmax") {
      var maxProb = Math.max.apply(null, network.output.probabilities || [0.5]);
      return clamp(maxProb * (1 - maxProb), 0.03, 0.25);
    }
    return 1;
  }

  function evaluateExample(example, network) {
    if (example.task_type === "regression") {
      var targetValue = computeRegressionTarget(example);
      var predictedValue = network.output.activation;
      var loss = Math.pow(predictedValue - targetValue, 2);
      return {
        targetLabel: formatNumber(targetValue, 2),
        predictedLabel: formatNumber(predictedValue, 2),
        lossLabel: formatNumber(loss, 3) + " (" + lossName(example) + ")",
        interpretation: "Salida numerica continua.",
        narrative: "La red intenta aproximar una magnitud continua. Mas capas no siempre mejoran una tarea simple."
        ,
        targetRaw: targetValue,
        predictedRaw: predictedValue,
        predictionType: "scalar"
      };
    }

    if (example.task_type === "binary") {
      var targetBinary = computeBinaryTarget(example);
      var probability = state.outputActivation === "sigmoid" ? network.output.activation : sigmoid(network.output.raw);
      var safeProb = clamp(probability, 1e-6, 1 - 1e-6);
      var loss = -((targetBinary * Math.log(safeProb)) + ((1 - targetBinary) * Math.log(1 - safeProb)));
      return {
        targetLabel: String(targetBinary),
        predictedLabel: state.outputActivation === "sigmoid" ? formatNumber(network.output.activation, 3) : formatNumber(network.output.raw, 3),
        lossLabel: formatNumber(loss, 3) + " (" + lossName(example) + ")",
        interpretation: state.outputActivation === "sigmoid" ? "Probabilidad estimada: " + formatNumber(probability, 3) : "La salida lineal no es probabilidad; se deriva una probabilidad solo para comparar.",
        narrative: "En binaria conviene una salida que se pueda leer como probabilidad."
        ,
        targetRaw: targetBinary,
        predictedRaw: probability,
        predictionType: "binary"
      };
    }

    var targetClass = computeMulticlassTarget(example);
    var predictedIndex = network.output.classIndex;
    var correctProbability = clamp(network.output.probabilities[targetClass.classIndex], 1e-6, 1);
    return {
      targetLabel: targetClass.className,
      predictedLabel: (example.classes || [])[predictedIndex] + " (" + formatNumber(network.output.probabilities[predictedIndex], 3) + ")",
      lossLabel: formatNumber(-Math.log(correctProbability), 3) + " (" + lossName(example) + ")",
      interpretation: "Softmax reparte la probabilidad total entre las clases.",
      narrative: "En multiclase, cada clase compite por parte de la probabilidad total.",
      targetRaw: targetClass.classIndex,
      predictedRaw: predictedIndex,
      targetClassName: targetClass.className,
      predictedClassName: (example.classes || [])[predictedIndex],
      predictedConfidence: network.output.probabilities[predictedIndex],
      predictionType: "categorical"
    };
  }

  function gradientAssessment(example, network) {
    var carry = outputGradientBase(network);
    var reversed = [];

    for (var index = network.layers.length - 1; index >= 0; index -= 1) {
      var layer = network.layers[index];
      carry *= clamp(layer.derivativeMean, 0.01, 1);
      if (layer.activationName === "sigmoid" && network.layers.length >= 4) {
        carry *= 0.75;
      }
      if (layer.activationName === "tanh" && network.layers.length >= 4) {
        carry *= 0.82;
      }
      var status = "strong";
      var why = "El gradiente aun llega con fuerza razonable.";
      if ((layer.activationName === "relu" || layer.activationName === "leaky_relu") && layer.zeroRatio >= 0.65) {
        status = "blocked";
        why = "Muchas neuronas quedan en cero y dejan de corregirse.";
      } else if (carry < 0.05 || layer.saturationRatio > 0.8) {
        status = "blocked";
        why = "La derivada acumulada es demasiado pequena o la capa esta saturada.";
      } else if (carry < 0.2 || layer.saturationRatio > 0.45) {
        status = "weak";
        why = "La correccion aun pasa, pero llega ya debilitada.";
      }
      reversed.push({ index: layer.index, status: status, carry: carry, why: why, saturationRatio: layer.saturationRatio, zeroRatio: layer.zeroRatio });
    }

    reversed.reverse();
    var blocked = reversed.filter(function (item) { return item.status === "blocked"; }).length;
    var weak = reversed.filter(function (item) { return item.status === "weak"; }).length;
    var overview = "Flujo sano.";
    if (blocked > 0) {
      overview = "Hay " + blocked + " capa(s) bloqueada(s): la red corregiria muy poco.";
    } else if (weak > 0) {
      overview = "Hay " + weak + " capa(s) debiles: la red podria aprender, pero mas lento.";
    } else if (!reversed.length) {
      overview = "Sin capas ocultas: no hay gradiente profundo que comparar.";
    }
    if (example.id === "xor_gate" && state.hiddenLayerCount === 0) {
      overview = "XOR sigue fallando con 0 capas ocultas aunque el gradiente no este bloqueado: el limite aqui es geometrico.";
    }
    return { overview: overview, layers: reversed };
  }

  function renderLayerSummary(network) {
    elements.layerSummary.innerHTML = "";
    if (!network.layers.length) {
      var empty = document.createElement("div");
      empty.className = "layer-card";
      empty.innerHTML = "<strong>Sin capas ocultas</strong><span>Solo hay transformacion directa entre entrada y salida.</span>";
      elements.layerSummary.appendChild(empty);
      return;
    }
    network.layers.forEach(function (layer) {
      var card = document.createElement("div");
      card.className = "layer-card";
      card.innerHTML =
        "<strong>Capa " + layer.index + " - " + layer.neurons + " neuronas</strong>" +
        "<span>Activacion: " + layer.activationName + "</span>" +
        "<span>Rango preactivacion: " + formatNumber(layer.preMin, 2) + " a " + formatNumber(layer.preMax, 2) + "</span>" +
        "<span>Media activada: " + formatNumber(layer.activationMean, 3) + "</span>" +
        "<span>Derivada media: " + formatNumber(layer.derivativeMean, 3) + "</span>";
      elements.layerSummary.appendChild(card);
    });
  }

  function renderGradient(assessment) {
    elements.gradientOverview.textContent = assessment.overview;
    elements.gradientDetails.innerHTML = "";
    if (!assessment.layers.length) {
      var empty = document.createElement("div");
      empty.className = "gradient-card gradient-card--strong";
      empty.innerHTML = "<strong>Salida directa</strong><span>La correccion actua sin atravesar capas ocultas.</span>";
      elements.gradientDetails.appendChild(empty);
      return;
    }
    assessment.layers.forEach(function (layer) {
      var label = layer.status === "strong" ? "Fuerte" : layer.status === "weak" ? "Debil" : "Bloqueado";
      var card = document.createElement("div");
      card.className = "gradient-card gradient-card--" + layer.status;
      card.innerHTML =
        "<strong>Capa " + layer.index + ": " + label + "</strong>" +
        "<span>" + layer.why + "</span>" +
        "<span>Gradiente acumulado: " + formatNumber(layer.carry, 3) + "</span>" +
        "<span>Saturacion: " + Math.round(layer.saturationRatio * 100) + "% | Ceros: " + Math.round(layer.zeroRatio * 100) + "%</span>";
      elements.gradientDetails.appendChild(card);
    });
  }

  function renderNetworkCanvas(example, assessment) {
    if (!elements.networkCanvasSvg) {
      return;
    }

    var actualHidden = state.hiddenLayers.slice(0, state.hiddenLayerCount);
    var actualSizes = [example.input_features.length]
      .concat(actualHidden.map(function (layer) { return layer.neurons; }))
      .concat([example.output_size || 1]);
    var displaySizes = actualSizes.map(function (size, index) {
      if (index === 0) {
        return Math.min(size, 4);
      }
      if (index === actualSizes.length - 1) {
        return Math.min(size, 3);
      }
      return Math.min(size, 6);
    });
    var layerTitles = ["Entrada"]
      .concat(actualHidden.map(function (_, index) { return "Oculta " + (index + 1); }))
      .concat(["Salida"]);
    var layerActivations = ["Datos"]
      .concat(actualHidden.map(function (layer) { return activationLabel(layer.activation); }))
      .concat([activationLabel(state.outputActivation)]);
    var width = 900;
    var height = 320;
    var top = 72;
    var bottom = height - 54;
    var left = 68;
    var right = width - 68;
    var xStep = displaySizes.length > 1 ? (right - left) / (displaySizes.length - 1) : 0;
    var assessmentMap = {};
    assessment.layers.forEach(function (layer) {
      assessmentMap[layer.index] = layer;
    });

    function nodeStatusColor(layerIndex, isOutput) {
      if (layerIndex === 0) {
        return { fill: "#eef4ff", stroke: "#5c79b6" };
      }
      if (isOutput) {
        return { fill: "#fff3e8", stroke: "#d17b2a" };
      }
      var layerInfo = assessmentMap[layerIndex];
      if (!layerInfo || layerInfo.status === "strong") {
        return { fill: "#e9f7f0", stroke: "#2e9c6c" };
      }
      if (layerInfo.status === "weak") {
        return { fill: "#fff5e6", stroke: "#e19b22" };
      }
      return { fill: "#ffeaea", stroke: "#cb5656" };
    }

    var nodePositions = [];
    for (var layerIndex = 0; layerIndex < displaySizes.length; layerIndex += 1) {
      var size = displaySizes[layerIndex];
      var x = left + (layerIndex * xStep);
      var yStep = size === 1 ? 0 : (bottom - top) / (size - 1);
      var nodes = [];
      for (var nodeIndex = 0; nodeIndex < size; nodeIndex += 1) {
        nodes.push({
          x: x,
          y: size === 1 ? ((top + bottom) / 2) : (top + (nodeIndex * yStep)),
          actualIndex: actualSizes[layerIndex] === size ? nodeIndex : Math.round((nodeIndex * (actualSizes[layerIndex] - 1)) / Math.max(1, size - 1))
        });
      }
      nodePositions.push(nodes);
    }

    var parts = [];
    parts.push("<rect x=\"8\" y=\"8\" width=\"884\" height=\"304\" rx=\"18\" fill=\"rgba(245,248,255,0.88)\" stroke=\"rgba(90,122,185,0.18)\"></rect>");

    for (var fromLayer = 0; fromLayer < nodePositions.length - 1; fromLayer += 1) {
      var toLayer = fromLayer + 1;
      nodePositions[fromLayer].forEach(function (fromNode) {
        nodePositions[toLayer].forEach(function (toNode) {
          var weight = networkWeight(example, fromLayer, toNode.actualIndex, fromNode.actualIndex);
          var color = weight >= 0 ? "rgba(63,122,219,0.58)" : "rgba(209,106,73,0.58)";
          var strokeWidth = (1.1 + (Math.abs(weight) * 1.8)).toFixed(2);
          parts.push(
            "<line x1=\"" + (fromNode.x + 18) + "\" y1=\"" + fromNode.y + "\" x2=\"" + (toNode.x - 18) + "\" y2=\"" + toNode.y + "\" stroke=\"" + color + "\" stroke-width=\"" + strokeWidth + "\" stroke-linecap=\"round\"></line>"
          );
        });
      });
    }

    for (var titleLayer = 0; titleLayer < nodePositions.length; titleLayer += 1) {
      var titleX = left + (titleLayer * xStep);
      parts.push("<text x=\"" + titleX + "\" y=\"30\" text-anchor=\"middle\" fill=\"#294f87\" font-size=\"13\" font-weight=\"800\">" + layerTitles[titleLayer] + "</text>");
      parts.push("<text x=\"" + titleX + "\" y=\"47\" text-anchor=\"middle\" fill=\"rgba(35,63,113,0.75)\" font-size=\"10\" font-weight=\"700\">" + actualSizes[titleLayer] + " nodo(s) | " + layerActivations[titleLayer] + "</text>");
    }

    for (var drawLayer = 0; drawLayer < nodePositions.length; drawLayer += 1) {
      var isOutput = drawLayer === nodePositions.length - 1;
      nodePositions[drawLayer].forEach(function (node, nodeIndex) {
        var colors = nodeStatusColor(drawLayer, isOutput);
        var label;
        if (drawLayer === 0) {
          label = ((example.input_features[node.actualIndex] || {}).label || ("x" + (node.actualIndex + 1))).slice(0, 10);
        } else if (isOutput) {
          label = example.output_size > 1 ? ("y" + (node.actualIndex + 1)) : "y";
        } else {
          label = "h" + (node.actualIndex + 1);
        }
        parts.push("<circle cx=\"" + node.x + "\" cy=\"" + node.y + "\" r=\"16\" fill=\"" + colors.fill + "\" stroke=\"" + colors.stroke + "\" stroke-width=\"2.4\"></circle>");
        parts.push("<text x=\"" + node.x + "\" y=\"" + (node.y + 4) + "\" text-anchor=\"middle\" fill=\"#173a6d\" font-size=\"10\" font-weight=\"800\">" + label + "</text>");
        if (actualSizes[drawLayer] > displaySizes[drawLayer] && nodeIndex === displaySizes[drawLayer] - 1) {
          parts.push("<text x=\"" + (node.x + 28) + "\" y=\"" + (node.y + 4) + "\" text-anchor=\"start\" fill=\"rgba(35,63,113,0.6)\" font-size=\"12\" font-weight=\"800\">+" + (actualSizes[drawLayer] - displaySizes[drawLayer]) + "</text>");
        }
      });
    }

    elements.networkCanvasSvg.innerHTML = parts.join("");

    if (elements.visualArchitecture) {
      elements.visualArchitecture.textContent = "Arquitectura: " + actualSizes.join(" -> ");
    }
    if (elements.visualActivation) {
      elements.visualActivation.textContent = "Salida: " + activationLabel(state.outputActivation);
    }
    if (elements.networkCanvasLegend) {
      elements.networkCanvasLegend.innerHTML =
        "<span class=\"legend-chip legend-chip--pos\">Peso positivo</span>" +
        "<span class=\"legend-chip legend-chip--neg\">Peso negativo</span>" +
        "<span class=\"legend-chip legend-chip--ok\">Gradiente fuerte</span>" +
        "<span class=\"legend-chip legend-chip--warn\">Gradiente debil</span>" +
        "<span class=\"legend-chip legend-chip--bad\">Gradiente bloqueado</span>";
    }
    if (elements.visualNarrative) {
      elements.visualNarrative.textContent = assessment.overview;
    }
  }

  function renderTrainingPanel(example, assessment) {
    if (!elements.trainingChartSvg) {
      return;
    }

    ensureTrainingBaseline(example, assessment);
    var last = state.trainingHistory[state.trainingHistory.length - 1];
    var first = state.trainingHistory[0];
    var suitability = trainingSuitability(example, assessment);
    var status = suitability >= 0.65 ? "Aprende bien" : suitability >= 0.35 ? "Aprende lento" : "Casi atascada";

    elements.trainingEpochValue.textContent = String(last.epoch);
    elements.trainingLossTrainValue.textContent = formatNumber(last.trainLoss, 3);
    elements.trainingLossValValue.textContent = formatNumber(last.valLoss, 3);
    elements.trainingStatusValue.textContent = status;

    if (elements.trainingHint) {
      elements.trainingHint.textContent = "Learning rate " + formatNumber(parseLearningRate(), 3) + ". Una arquitectura mas sana reduce la loss con mas consistencia.";
    }

    if (elements.trainingNarrative) {
      elements.trainingNarrative.textContent =
        "La curva parte en " + formatNumber(first.trainLoss, 3) +
        " y llega a " + formatNumber(last.trainLoss, 3) +
        ". La validacion cierra en " + formatNumber(last.valLoss, 3) +
        ". Esta simulacion resume si la arquitectura tiene margen real para mejorar.";
    }

    var width = 860;
    var height = 230;
    var padding = { left: 50, right: 18, top: 16, bottom: 34 };
    var plotWidth = width - padding.left - padding.right;
    var plotHeight = height - padding.top - padding.bottom;
    var maxLoss = Math.max.apply(null, state.trainingHistory.map(function (point) { return Math.max(point.trainLoss, point.valLoss); }));
    var minLoss = Math.min.apply(null, state.trainingHistory.map(function (point) { return Math.min(point.trainLoss, point.valLoss); }));
    var safeMin = Math.max(0, minLoss * 0.9);
    var safeMax = Math.max(safeMin + 0.05, maxLoss * 1.05);

    function xForIndex(index) {
      if (state.trainingHistory.length === 1) {
        return padding.left + (plotWidth / 2);
      }
      return padding.left + ((plotWidth * index) / (state.trainingHistory.length - 1));
    }

    function yForLoss(loss) {
      return padding.top + ((safeMax - loss) / (safeMax - safeMin)) * plotHeight;
    }

    var parts = [];
    parts.push("<rect x=\"" + padding.left + "\" y=\"" + padding.top + "\" width=\"" + plotWidth + "\" height=\"" + plotHeight + "\" rx=\"12\" fill=\"rgba(244,248,255,0.42)\" stroke=\"rgba(90,122,185,0.18)\"></rect>");

    for (var grid = 0; grid <= 4; grid += 1) {
      var y = padding.top + ((plotHeight * grid) / 4);
      var lossValue = safeMax - (((safeMax - safeMin) * grid) / 4);
      parts.push("<line x1=\"" + padding.left + "\" y1=\"" + y + "\" x2=\"" + (padding.left + plotWidth) + "\" y2=\"" + y + "\" stroke=\"rgba(92,121,181,0.14)\" stroke-width=\"1\"></line>");
      parts.push("<text x=\"" + (padding.left - 8) + "\" y=\"" + (y + 4) + "\" text-anchor=\"end\" fill=\"rgba(35,63,113,0.72)\" font-size=\"10\" font-weight=\"700\">" + formatNumber(lossValue, 2) + "</text>");
    }

    var trainPoints = [];
    var valPoints = [];
    state.trainingHistory.forEach(function (point, index) {
      trainPoints.push(xForIndex(index) + "," + yForLoss(point.trainLoss));
      valPoints.push(xForIndex(index) + "," + yForLoss(point.valLoss));
    });

    parts.push("<polyline points=\"" + trainPoints.join(" ") + "\" fill=\"none\" stroke=\"#2d8a5f\" stroke-width=\"3\" stroke-linecap=\"round\" stroke-linejoin=\"round\"></polyline>");
    parts.push("<polyline points=\"" + valPoints.join(" ") + "\" fill=\"none\" stroke=\"#2f77d8\" stroke-width=\"3\" stroke-linecap=\"round\" stroke-linejoin=\"round\"></polyline>");

    var lastIndex = state.trainingHistory.length - 1;
    parts.push("<circle cx=\"" + xForIndex(lastIndex) + "\" cy=\"" + yForLoss(last.trainLoss) + "\" r=\"4\" fill=\"#2d8a5f\"></circle>");
    parts.push("<circle cx=\"" + xForIndex(lastIndex) + "\" cy=\"" + yForLoss(last.valLoss) + "\" r=\"4\" fill=\"#2f77d8\"></circle>");

    parts.push("<text x=\"" + padding.left + "\" y=\"" + (height - 10) + "\" fill=\"rgba(35,63,113,0.72)\" font-size=\"10\" font-weight=\"700\">Epoca 0</text>");
    parts.push("<text x=\"" + (width - padding.right) + "\" y=\"" + (height - 10) + "\" text-anchor=\"end\" fill=\"rgba(35,63,113,0.72)\" font-size=\"10\" font-weight=\"700\">Epoca " + last.epoch + "</text>");
    parts.push("<text x=\"" + (padding.left + 8) + "\" y=\"" + (padding.top + 14) + "\" fill=\"#2d8a5f\" font-size=\"10\" font-weight=\"800\">Train</text>");
    parts.push("<text x=\"" + (padding.left + 54) + "\" y=\"" + (padding.top + 14) + "\" fill=\"#2f77d8\" font-size=\"10\" font-weight=\"800\">Validacion</text>");

    elements.trainingChartSvg.innerHTML = parts.join("");
  }

  function trainingProgressRatio() {
    if (state.trainingHistory.length < 2) {
      return 0;
    }
    var first = state.trainingHistory[0];
    var last = state.trainingHistory[state.trainingHistory.length - 1];
    if (!first || !Number.isFinite(first.trainLoss) || first.trainLoss <= 0) {
      return 0;
    }
    return clamp(1 - (last.trainLoss / first.trainLoss), 0, 0.95);
  }

  function renderTrainedInference(example, evaluation, assessment) {
    if (!elements.trainedExpectedValue) {
      return;
    }

    ensureTrainingBaseline(example, assessment);
    var progress = trainingProgressRatio();
    var suitability = trainingSuitability(example, assessment);
    var effectiveProgress = clamp(progress * (0.45 + (suitability * 0.55)), 0, 0.95);

    if (evaluation.predictionType === "scalar") {
      var trainedValue = evaluation.predictedRaw + ((evaluation.targetRaw - evaluation.predictedRaw) * effectiveProgress);
      var residual = Math.abs(evaluation.targetRaw - trainedValue);
      elements.trainedExpectedValue.textContent = formatNumber(evaluation.targetRaw, 2);
      elements.trainedPredictedValue.textContent = formatNumber(trainedValue, 2);
      elements.trainedResidualValue.textContent = formatNumber(residual, 3);
      elements.trainedReadingValue.textContent = effectiveProgress > 0.55 ? "Se acerca" : effectiveProgress > 0.2 ? "Mejora parcial" : "Casi igual";
      elements.trainedNarrative.textContent =
        "Tras la tanda, la inferencia estimada se desplaza desde " +
        formatNumber(evaluation.predictedRaw, 2) +
        " hacia el objetivo " +
        formatNumber(evaluation.targetRaw, 2) +
        ".";
      return;
    }

    if (evaluation.predictionType === "binary") {
      var trainedProbability = clamp(evaluation.predictedRaw + ((evaluation.targetRaw - evaluation.predictedRaw) * effectiveProgress), 0, 1);
      var residualBinary = Math.abs(evaluation.targetRaw - trainedProbability);
      elements.trainedExpectedValue.textContent = String(evaluation.targetRaw);
      elements.trainedPredictedValue.textContent = formatNumber(trainedProbability, 3);
      elements.trainedResidualValue.textContent = formatNumber(residualBinary, 3);
      elements.trainedReadingValue.textContent = trainedProbability >= 0.5 ? "Clase 1" : "Clase 0";
      elements.trainedNarrative.textContent =
        "La inferencia estimada tras entrenar se lee como probabilidad. Cuanto mas se acerca a " +
        String(evaluation.targetRaw) +
        ", mejor encaja la salida con la etiqueta esperada.";
      return;
    }

    var trainedClassName = effectiveProgress > 0.45 ? evaluation.targetClassName : evaluation.predictedClassName;
    var confidence = clamp(evaluation.predictedConfidence + ((effectiveProgress > 0.45 ? 0.88 : 0.62) - evaluation.predictedConfidence) * Math.max(effectiveProgress, 0.15), 0, 0.99);
    elements.trainedExpectedValue.textContent = evaluation.targetClassName;
    elements.trainedPredictedValue.textContent = trainedClassName + " (" + formatNumber(confidence, 3) + ")";
    elements.trainedResidualValue.textContent = trainedClassName === evaluation.targetClassName ? "Coincide" : "Aun falla";
    elements.trainedReadingValue.textContent = trainedClassName;
    elements.trainedNarrative.textContent =
      "En multiclase, el resultado tras entrenamiento se expresa como clase dominante mas su confianza estimada.";
  }

  function renderDidacticNotes(example, evaluation, assessment) {
    var notes = [];
    notes.push("<p><strong>Que intenta aprender:</strong> " + example.expected_pattern + "</p>");
    notes.push("<p><strong>Por que esta salida:</strong> " + example.output_activation_recommendation + "</p>");
    notes.push("<p><strong>Configuracion minima:</strong> " + example.minimal_config + "</p>");
    if (example.id === "xor_gate" && state.hiddenLayerCount === 0) {
      notes.push("<p><strong>Advertencia:</strong> XOR no puede resolverse con una sola capa lineal aunque el gradiente no parezca malo.</p>");
    } else if (assessment.layers.some(function (layer) { return layer.status === "blocked"; })) {
      notes.push("<p><strong>Lectura actual:</strong> Hay al menos una capa donde la correccion practicamente no pasa.</p>");
    } else if (assessment.layers.some(function (layer) { return layer.status === "weak"; })) {
      notes.push("<p><strong>Lectura actual:</strong> La red aun aprende, pero ya consume demasiado gradiente.</p>");
    } else {
      notes.push("<p><strong>Lectura actual:</strong> El flujo es razonable para una herramienta didactica.</p>");
    }
    notes.push("<p><strong>Fallo tipico:</strong> " + (example.common_failures || [])[0] + "</p>");
    notes.push("<p><strong>Forward actual:</strong> " + evaluation.narrative + "</p>");
    elements.builderDidacticNotes.innerHTML = notes.join("");
  }

  function renderSimulation() {
    var example = getExampleById(state.exampleId);
    var network = simulateNetwork(example);
    var evaluation = evaluateExample(example, network);
    var assessment = gradientAssessment(example, network);
    elements.targetOutputValue.textContent = evaluation.targetLabel;
    elements.predictedOutputValue.textContent = evaluation.predictedLabel;
    elements.lossValue.textContent = evaluation.lossLabel;
    elements.outputInterpretation.textContent = evaluation.interpretation;
    elements.forwardNarrative.textContent = example.title + ". " + evaluation.narrative;
    renderNetworkCanvas(example, assessment);
    renderTrainingPanel(example, assessment);
    renderTrainedInference(example, evaluation, assessment);
    renderLayerSummary(network);
    renderGradient(assessment);
    renderDidacticNotes(example, evaluation, assessment);
  }

  function renderAll() {
    var example = getExampleById(state.exampleId);
    elements.exampleSelects.forEach(function (select) {
      select.value = state.exampleId;
    });
    elements.hiddenLayerCount.value = String(state.hiddenLayerCount);
    renderExampleSummary(example);
    renderOutputActivationOptions(example);
    elements.outputActivation.onchange = function (event) {
      state.outputActivation = event.target.value;
      resetTrainingState();
      renderSimulation();
    };
    elements.outputActivation.value = state.outputActivation;
    renderHiddenLayerControls();
    renderInputControls(example);
    renderExplanationPanels(example);
    renderSimulation();
  }

  function wireStaticActions() {
    elements.resetToRecommended.addEventListener("click", function () {
      copyRecommendedConfig(getExampleById(state.exampleId));
      renderAll();
    });

    if (elements.trainRunButton) {
      elements.trainRunButton.addEventListener("click", function () {
        var example = getExampleById(state.exampleId);
        var network = simulateNetwork(example);
        var assessment = gradientAssessment(example, network);
        appendTrainingEpochs(example, assessment);
        renderSimulation();
      });
    }

    if (elements.trainResetButton) {
      elements.trainResetButton.addEventListener("click", function () {
        resetTrainingState();
        renderSimulation();
      });
    }

    if (elements.learningRateInput) {
      elements.learningRateInput.addEventListener("change", function () {
        elements.learningRateInput.value = formatNumber(parseLearningRate(), 3);
        renderSimulation();
      });
    }

    if (elements.epochsPerRunInput) {
      elements.epochsPerRunInput.addEventListener("change", function () {
        elements.epochsPerRunInput.value = String(parseEpochsPerRun());
        renderSimulation();
      });
    }
  }

  function initialRender() {
    copyRecommendedConfig(getExampleById(state.exampleId));
    fillExampleSelects();
    fillHiddenLayerCountSelect();
    renderPracticeList();
    renderActivationTable();
    renderCommonErrors();
    renderScenarioButtons();
    wireStaticActions();
    renderAll();
  }

  function focusActiveView() {
    var activeView = document.body.getAttribute("data-active-view");
    if (!activeView || activeView === "inicio") {
      return;
    }
    var section = document.querySelector("[data-view-section=\"" + activeView + "\"]");
    if (!section) {
      return;
    }
    window.requestAnimationFrame(function () {
      var top = section.getBoundingClientRect().top + window.scrollY - 110;
      window.scrollTo({ top: Math.max(0, top), behavior: "auto" });
    });
  }

  initialRender();
  focusActiveView();
})();
