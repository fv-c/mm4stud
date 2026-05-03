import { Midi } from "https://cdn.jsdelivr.net/npm/@tonejs/midi@2.0.28/+esm";
import JSZip from "https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm";

const RHYTHM_GRID = 0.25;
const BEAT_STRENGTH_GRID = 0.125;
const METRIC_FORM_TOLERANCE = 1e-5;
const DEFAULT_VELOCITY = 64;
const DEFAULT_TEMPO = 120;
const DEFAULT_TIME_SIGNATURE = "4/4";
const DEFAULT_BAR_DURATION_QL = 4;
const DEFAULT_GATE = 0.9;
const DEFAULT_SEED = 42;
const DEFAULT_TRANSITION_WINDOW_SIZE = 5;
const DEFAULT_TRANSITION_WINDOW_STEP = 1;

const DYNAMIC_BINS = [
  [1, 35, "pp", 28],
  [36, 50, "p", 44],
  [51, 65, "mp", 58],
  [66, 80, "mf", 72],
  [81, 100, "f", 92],
  [101, 127, "ff", 112],
];

const form = document.querySelector("#generatorForm");
const statusText = document.querySelector("#statusText");
const fileSummary = document.querySelector("#fileSummary");
const downloadZip = document.querySelector("#downloadZip");
const submitButton = form.querySelector("button[type='submit']");
const pianoRollPanel = document.querySelector("#pianoRollPanel");
const pianoRollCanvas = document.querySelector("#pianoRollCanvas");
const originalAlphaInput = document.querySelector("#originalAlpha");
const generatedAlphaInput = document.querySelector("#generatedAlpha");
let lastPianoRollData = null;

for (const id of ["metricFormVariation", "metricSimilarityThreshold"]) {
  const input = document.querySelector(`#${id}`);
  const output = document.querySelector(`#${id}Value`);
  input.addEventListener("input", () => {
    output.value = Number(input.value).toFixed(2);
  });
}

for (const alphaInput of [originalAlphaInput, generatedAlphaInput]) {
  alphaInput.addEventListener("input", () => {
    if (lastPianoRollData) drawPianoRoll(lastPianoRollData);
  });
}

function sanitizeName(name) {
  const cleaned = String(name || "")
    .trim()
    .replace(/\s+/g, "_")
    .replace(/[^A-Za-z0-9_-]+/g, "");
  return cleaned || "unnamed";
}

function quantizeValue(value, grid) {
  if (grid <= 0) return Number(value);
  return Number((Math.round(Number(value) / grid) * grid).toFixed(6));
}

function valueToCleanString(value) {
  const rounded = Number(Number(value).toFixed(6));
  return Number.isInteger(rounded) ? String(rounded) : String(Number(rounded.toPrecision(12)));
}

function parseDurationString(value) {
  return Number(value);
}

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function velocityToDynamic(velocity) {
  const midiVelocity = clamp(Math.round(velocity ?? DEFAULT_VELOCITY), 1, 127);
  const match = DYNAMIC_BINS.find(([low, high]) => low <= midiVelocity && midiVelocity <= high);
  return match ? match[2] : "mf";
}

function dynamicToVelocity(label) {
  const match = DYNAMIC_BINS.find((bin) => bin[2] === label);
  return match ? match[3] : DEFAULT_VELOCITY;
}

function pitchClassAndRegisterToMidiPitch(pitchClass, register) {
  return clamp(12 * (Number(register) + 1) + Number(pitchClass), 0, 127);
}

function sortStates(states) {
  return [...states].sort((a, b) => {
    const na = Number(a);
    const nb = Number(b);
    const aNumeric = Number.isFinite(na);
    const bNumeric = Number.isFinite(nb);
    if (aNumeric && bNumeric) return na - nb;
    if (aNumeric) return -1;
    if (bNumeric) return 1;
    return String(a).localeCompare(String(b));
  });
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function random() {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function weightedChoice(items, weights, rng) {
  if (!items.length) throw new Error("weighted_choice richiede almeno un elemento.");
  let safeWeights = weights.map((weight) => Number(weight));
  const valid = safeWeights.every(Number.isFinite) && safeWeights.reduce((a, b) => a + b, 0) > 0;
  if (!valid) safeWeights = items.map(() => 1);
  const total = safeWeights.reduce((a, b) => a + b, 0);
  let cursor = rng() * total;
  for (let i = 0; i < items.length; i += 1) {
    cursor -= safeWeights[i];
    if (cursor <= 0) return items[i];
  }
  return items[items.length - 1];
}

function durationNameIt(durationQl, count = 1) {
  const names = new Map([
    [4, ["semibreve", "semibrevi"]],
    [3, ["minima puntata", "minime puntate"]],
    [2, ["minima", "minime"]],
    [1.5, ["semiminima puntata", "semiminime puntate"]],
    [1, ["semiminima", "semiminime"]],
    [0.75, ["croma puntata", "crome puntate"]],
    [0.5, ["croma", "crome"]],
    [0.375, ["semicroma puntata", "semicrome puntate"]],
    [0.25, ["semicroma", "semicrome"]],
    [0.125, ["biscroma", "biscrome"]],
    [0.0625, ["semibiscroma", "semibiscrome"]],
  ]);
  const d = Number(Number(durationQl).toFixed(6));
  if (!names.has(d)) return `durate da ${valueToCleanString(d)} quarterLength`;
  const [singular, plural] = names.get(d);
  return count === 1 ? singular : plural;
}

function metricFormLabelFromDurations(durations) {
  if (!durations.length) return "forma vuota";
  const groups = [];
  let current = durations[0];
  let count = 1;
  for (const duration of durations.slice(1)) {
    if (duration === current) {
      count += 1;
    } else {
      groups.push([current, count]);
      current = duration;
      count = 1;
    }
  }
  groups.push([current, count]);
  return groups
    .map(([duration, groupCount]) => `${groupCount} ${durationNameIt(parseDurationString(duration), groupCount)}`)
    .join(" + ");
}

function getTimeSignatureSegments(midi) {
  const ppq = midi.header.ppq || 480;
  const raw = midi.header.timeSignatures?.length
    ? midi.header.timeSignatures
    : [{ ticks: 0, timeSignature: [4, 4] }];
  const sorted = [...raw].sort((a, b) => (a.ticks || 0) - (b.ticks || 0));
  let startMeasure = 1;
  return sorted.map((entry, index) => {
    const [numerator, denominator] = entry.timeSignature || [4, 4];
    const startTick = entry.ticks || 0;
    const barDurationQl = numerator * (4 / denominator);
    const barTicks = barDurationQl * ppq;
    if (index > 0) {
      const previous = sorted[index - 1];
      const [prevNum, prevDen] = previous.timeSignature || [4, 4];
      const prevBarTicks = prevNum * (4 / prevDen) * ppq;
      startMeasure += Math.floor((startTick - (previous.ticks || 0)) / prevBarTicks);
    }
    return {
      startTick,
      startMeasure,
      numerator,
      denominator,
      timeSignature: `${numerator}/${denominator}`,
      barDurationQl,
      barTicks,
    };
  });
}

function metricContextForTick(tick, segments, ppq) {
  let segment = segments[0];
  for (const candidate of segments) {
    if (candidate.startTick <= tick) segment = candidate;
  }
  const relativeTicks = Math.max(0, tick - segment.startTick);
  const measureOffset = Math.floor(relativeTicks / segment.barTicks);
  const positionTicks = relativeTicks - measureOffset * segment.barTicks;
  return {
    measure_number: segment.startMeasure + measureOffset,
    time_signature: segment.timeSignature,
    bar_duration_ql: Number(segment.barDurationQl.toFixed(6)),
    position_in_measure_ql: valueToCleanString(quantizeValue(positionTicks / ppq, RHYTHM_GRID)),
    numerator: segment.numerator,
    denominator: segment.denominator,
  };
}

function approximateBeatStrength(positionQl, barDurationQl, numerator, denominator) {
  const beatQl = 4 / denominator;
  const pos = Number(positionQl);
  if (Math.abs(pos) < 1e-6) return 1;
  if (Math.abs(pos - barDurationQl / 2) < 1e-6 && numerator % 2 === 0) return 0.75;
  if (Math.abs(pos / beatQl - Math.round(pos / beatQl)) < 1e-6) return 0.5;
  return 0.25;
}

function getPartMetadata(track, partIndex) {
  const fallbackName = `part_${partIndex + 1}`;
  const trackName = track.name || track.instrument?.name || fallbackName;
  const midiProgram = clamp(Number(track.instrument?.number ?? 0), 0, 127);
  return {
    part_index: partIndex,
    part_name: String(trackName),
    part_slug: `${String(partIndex + 1).padStart(2, "0")}_${sanitizeName(trackName)}`,
    midi_program: midiProgram,
  };
}

function extractEventsByPart(midi) {
  const ppq = midi.header.ppq || 480;
  const segments = getTimeSignatureSegments(midi);
  const allEvents = [];
  const partRows = [];
  let partIndex = 0;

  for (const track of midi.tracks) {
    if (!track.notes.length) continue;
    const metadata = getPartMetadata(track, partIndex);
    const events = [];
    for (const note of track.notes) {
      const offsetQl = Number(((note.ticks || 0) / ppq).toFixed(6));
      const durationTicks = note.durationTicks ?? Math.round((note.duration || 0) * ppq);
      const rhythm = quantizeValue(durationTicks / ppq, RHYTHM_GRID);
      const metric = metricContextForTick(note.ticks || 0, segments, ppq);
      const beatStrength = quantizeValue(
        approximateBeatStrength(
          parseDurationString(metric.position_in_measure_ql),
          metric.bar_duration_ql,
          metric.numerator,
          metric.denominator,
        ),
        BEAT_STRENGTH_GRID,
      );
      const midiPitch = Number(note.midi);
      events.push({
        part_index: metadata.part_index,
        part_name: metadata.part_name,
        part_slug: metadata.part_slug,
        midi_program: metadata.midi_program,
        offset_ql: offsetQl,
        rhythm: valueToCleanString(rhythm),
        beat_strength: valueToCleanString(beatStrength),
        dynamic: velocityToDynamic(Math.round((note.velocity ?? 0.5) * 127)),
        velocity: clamp(Math.round((note.velocity ?? 0.5) * 127), 1, 127),
        measure_number: metric.measure_number,
        time_signature: metric.time_signature,
        bar_duration_ql: metric.bar_duration_ql,
        position_in_measure_ql: metric.position_in_measure_ql,
        pitch_class: ((midiPitch % 12) + 12) % 12,
        register: Math.floor(midiPitch / 12) - 1,
        original_midi_pitch: midiPitch,
      });
    }
    events.sort((a, b) => a.offset_ql - b.offset_ql || a.original_midi_pitch - b.original_midi_pitch);
    allEvents.push(...events);
    partRows.push({
      ...metadata,
      n_events: events.length,
    });
    partIndex += 1;
  }

  if (!allEvents.length) throw new Error("Nessuna nota trovata nel file MIDI.");
  allEvents.sort((a, b) => a.part_index - b.part_index || a.offset_ql - b.offset_ql || a.original_midi_pitch - b.original_midi_pitch);
  return { events: allEvents, parts: partRows };
}

function buildMetricFormsForPart(partEvents, includeIncompleteForms = false) {
  const forms = new Map();
  const byMeasure = new Map();
  for (const event of partEvents) {
    if (!byMeasure.has(event.measure_number)) byMeasure.set(event.measure_number, []);
    byMeasure.get(event.measure_number).push(event);
  }

  for (const [measureNumber, measureEvents] of byMeasure) {
    const seen = new Set();
    const rhythmEvents = [];
    for (const event of measureEvents.sort((a, b) => Number(a.position_in_measure_ql) - Number(b.position_in_measure_ql) || Number(a.rhythm) - Number(b.rhythm))) {
      const key = `${event.position_in_measure_ql}::${event.rhythm}`;
      if (!seen.has(key)) {
        seen.add(key);
        rhythmEvents.push(event);
      }
    }
    if (!rhythmEvents.length) continue;
    const durations = rhythmEvents.map((event) => String(event.rhythm));
    const totalDuration = durations.reduce((sum, d) => sum + parseDurationString(d), 0);
    const timeSignature = String(rhythmEvents[0].time_signature || DEFAULT_TIME_SIGNATURE);
    const barDurationQl = Number(rhythmEvents[0].bar_duration_ql || DEFAULT_BAR_DURATION_QL);
    const fillsMeasure = Math.abs(totalDuration - barDurationQl) <= METRIC_FORM_TOLERANCE;
    if (!fillsMeasure && !includeIncompleteForms) continue;
    const durationKey = durations.join("|");
    const key = `${timeSignature}::${durationKey}`;
    if (!forms.has(key)) {
      const readable = metricFormLabelFromDurations(durations);
      forms.set(key, {
        key,
        label: `${timeSignature} | ${readable}`,
        time_signature: timeSignature,
        bar_duration_ql: barDurationQl,
        durations_ql: durations,
        measure_numbers: [],
        count: 0,
        fills_measure: fillsMeasure,
      });
    }
    const form = forms.get(key);
    form.measure_numbers.push(Number(measureNumber));
    form.count += 1;
  }

  for (const metricForm of forms.values()) {
    metricForm.measure_numbers = [...new Set(metricForm.measure_numbers)].sort((a, b) => a - b);
  }
  return Object.fromEntries(forms);
}

function metricFormsToRecords(metricForms) {
  return Object.values(metricForms).sort((a, b) => b.count - a.count || a.key.localeCompare(b.key));
}

function buildTransitionMatrices(sequence) {
  if (sequence.length < 2) throw new Error("La sequenza deve contenere almeno due stati.");
  const states = sortStates([...new Set(sequence)]);
  const counts = Object.fromEntries(states.map((state) => [state, Object.fromEntries(states.map((target) => [target, 0]))]));
  for (let i = 0; i < sequence.length - 1; i += 1) {
    counts[sequence[i]][sequence[i + 1]] += 1;
  }
  const probabilities = Object.fromEntries(states.map((state) => {
    const row = counts[state];
    const total = Object.values(row).reduce((sum, value) => sum + value, 0);
    return [state, Object.fromEntries(states.map((target) => [target, total > 0 ? row[target] / total : 0]))];
  }));
  return { states, counts, probabilities };
}

function buildTimeVaryingTransitionMatrices(sequence, windowSize, windowStep) {
  if (sequence.length < 2) return [];

  const safeWindowSize = Math.max(2, Math.floor(windowSize));
  const safeWindowStep = Math.max(1, Math.floor(windowStep));
  const windows = [];

  if (sequence.length < safeWindowSize) {
    const matrices = buildTransitionMatrices(sequence);
    return [{
      window_index: 1,
      start_event_index: 0,
      end_event_index: sequence.length - 1,
      n_events_in_window: sequence.length,
      n_transitions: sequence.length - 1,
      ...matrices,
      warning: "Sequenza più corta della finestra: usata una finestra ridotta.",
    }];
  }

  for (let start = 0; start + safeWindowSize <= sequence.length; start += safeWindowStep) {
    const windowSequence = sequence.slice(start, start + safeWindowSize);
    const matrices = buildTransitionMatrices(windowSequence);
    windows.push({
      window_index: windows.length + 1,
      start_event_index: start,
      end_event_index: start + safeWindowSize - 1,
      n_events_in_window: safeWindowSize,
      n_transitions: safeWindowSize - 1,
      ...matrices,
      warning: "",
    });
  }

  return windows;
}

function transitionWindowForEvent(windows, eventIndex, windowStep) {
  if (!windows.length) return null;
  const safeWindowStep = Math.max(1, Math.floor(windowStep));
  const index = clamp(Math.floor(Math.max(0, eventIndex) / safeWindowStep), 0, windows.length - 1);
  return windows[index];
}

function empiricalDistribution(sequence) {
  const counter = new Map();
  for (const item of sequence) counter.set(item, (counter.get(item) || 0) + 1);
  const states = sortStates([...counter.keys()]);
  const total = sequence.length;
  return {
    states,
    probabilities: states.map((state) => counter.get(state) / total),
  };
}

function sampleNextState(transitionProbabilities, currentState, originalSequence, rng) {
  const fallback = empiricalDistribution(originalSequence);
  const row = transitionProbabilities?.[currentState];
  if (row) {
    const states = Object.keys(row);
    const weights = states.map((state) => Number(row[state]));
    if (weights.reduce((a, b) => a + b, 0) > 0) return weightedChoice(states, weights, rng);
  }
  return weightedChoice(fallback.states, fallback.probabilities, rng);
}

function sampleNextStateTimeVarying(transitionWindows, currentState, originalSequence, eventIndex, options, rng) {
  const window = transitionWindowForEvent(transitionWindows, eventIndex, options.transitionWindowStep);
  return sampleNextState(window?.probabilities, currentState, originalSequence, rng);
}

function transitionProbability(transitionProbabilities, currentState, nextState) {
  return Number(transitionProbabilities?.[currentState]?.[nextState] || 0);
}

function transitionProbabilityTimeVarying(transitionWindows, currentState, nextState, eventIndex, options) {
  const window = transitionWindowForEvent(transitionWindows, eventIndex, options.transitionWindowStep);
  return transitionProbability(window?.probabilities, currentState, nextState);
}

function sampleMarkovSequence(transitionProbabilities, originalSequence, nEvents, rng) {
  if (nEvents <= 0) throw new Error("n_events deve essere positivo.");
  const fallback = empiricalDistribution(originalSequence);
  let current = weightedChoice(fallback.states, fallback.probabilities, rng);
  const generated = [current];
  for (let i = 1; i < nEvents; i += 1) {
    current = sampleNextState(transitionProbabilities, current, originalSequence, rng);
    generated.push(current);
  }
  return generated;
}

function sampleMarkovSequenceTimeVarying(transitionWindows, originalSequence, nEvents, options, rng, startState = null) {
  if (nEvents <= 0) throw new Error("n_events deve essere positivo.");
  const fallback = empiricalDistribution(originalSequence);
  let current = startState === null || startState === undefined
    ? weightedChoice(fallback.states, fallback.probabilities, rng)
    : startState;
  const generated = [current];
  for (let i = 1; i < nEvents; i += 1) {
    current = sampleNextStateTimeVarying(transitionWindows, current, originalSequence, i - 1, options, rng);
    generated.push(current);
  }
  return generated;
}

function buildIntervalSizeSequence(midiPitches) {
  const intervals = [];
  for (let i = 1; i < midiPitches.length; i += 1) {
    intervals.push(String(Math.abs(Number(midiPitches[i]) - Number(midiPitches[i - 1]))));
  }
  return intervals;
}

function nearestObservedPitch(basePitch, previousPitch, desiredInterval, observedPitches) {
  if (!observedPitches.length) return clamp(Math.round(basePitch), 0, 127);
  if (previousPitch === null || previousPitch === undefined) {
    return observedPitches.reduce((best, pitch) => (
      Math.abs(Number(pitch) - Number(basePitch)) < Math.abs(Number(best) - Number(basePitch)) ? pitch : best
    ), observedPitches[0]);
  }

  const desired = Number(desiredInterval);
  const basePitchClass = ((Math.round(basePitch) % 12) + 12) % 12;
  let bestPitch = observedPitches[0];
  let bestScore = Infinity;

  for (const pitch of observedPitches) {
    const intervalDiff = Math.abs(Math.abs(Number(pitch) - Number(previousPitch)) - desired);
    const baseDiff = Math.abs(Number(pitch) - Number(basePitch));
    const pitchClassPenalty = (((Number(pitch) % 12) + 12) % 12) === basePitchClass ? 0 : 0.5;
    const score = intervalDiff * 6 + baseDiff * 0.25 + pitchClassPenalty;
    if (score < bestScore) {
      bestScore = score;
      bestPitch = Number(pitch);
    }
  }

  return clamp(Math.round(bestPitch), 0, 127);
}

function applyIntervalProfileConstraint(generated, originalMidiPitches, desiredIntervals, deterministicMode) {
  const observedPitches = sortStates([...new Set(originalMidiPitches.map((pitch) => String(pitch)))]).map(Number);
  const nEvents = generated.pitch_class.length;
  const midiPitches = [];
  const rows = [];

  for (let i = 0; i < nEvents; i += 1) {
    const basePitch = pitchClassAndRegisterToMidiPitch(
      Number(generated.pitch_class[i]),
      Number(generated.register[i]),
    );
    const previousPitch = i > 0 ? midiPitches[i - 1] : null;
    const desiredInterval = i > 0 ? (desiredIntervals[i - 1] ?? 0) : 0;
    const constrainedPitch = deterministicMode && originalMidiPitches[i] !== undefined
      ? Number(originalMidiPitches[i])
      : nearestObservedPitch(basePitch, previousPitch, desiredInterval, observedPitches);

    midiPitches.push(constrainedPitch);
    generated.pitch_class[i] = String(((constrainedPitch % 12) + 12) % 12);
    generated.register[i] = String(Math.floor(constrainedPitch / 12) - 1);

    rows.push({
      event_index: i,
      base_midi_pitch: basePitch,
      constrained_midi_pitch: constrainedPitch,
      desired_interval_size: desiredInterval,
      actual_interval_size: i > 0 ? Math.abs(constrainedPitch - midiPitches[i - 1]) : 0,
    });
  }

  generated.midi_pitch = midiPitches;
  return rows;
}

function metricFormSimilarity(anchor, candidate, allowDifferentMeterVariants) {
  if (!allowDifferentMeterVariants && anchor.time_signature !== candidate.time_signature) return 0;
  const a = anchor.durations_ql.map(Number);
  const b = candidate.durations_ql.map(Number);
  const rows = a.length + 1;
  const cols = b.length + 1;
  const dp = Array.from({ length: rows }, () => Array(cols).fill(0));
  for (let i = 1; i < rows; i += 1) dp[i][0] = i;
  for (let j = 1; j < cols; j += 1) dp[0][j] = j;
  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      const substitutionCost = Math.min(Math.abs(a[i - 1] - b[j - 1]) / Math.max(a[i - 1], b[j - 1], RHYTHM_GRID), 1);
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + substitutionCost,
      );
    }
  }
  const sequenceSimilarity = 1 - dp[a.length][b.length] / Math.max(a.length, b.length, 1);
  const totalA = a.reduce((sum, value) => sum + value, 0);
  const totalB = b.reduce((sum, value) => sum + value, 0);
  const durationSimilarity = 1 - Math.min(Math.abs(totalA - totalB) / Math.max(totalA, totalB, RHYTHM_GRID), 1);
  const meterBonus = anchor.time_signature === candidate.time_signature ? 1 : 0.86;
  return clamp(sequenceSimilarity * durationSimilarity * meterBonus, 0, 1);
}

function maybeVaryMetricForm(anchor, records, options, rng) {
  if (options.metricFormVariation <= 0 || rng() >= options.metricFormVariation) return anchor;
  const candidates = [];
  const weights = [];
  for (const candidate of records) {
    const similarity = metricFormSimilarity(anchor, candidate, options.allowDifferentMeterVariants);
    if (candidate.key !== anchor.key && similarity >= options.metricSimilarityThreshold) {
      candidates.push(candidate);
      weights.push(candidate.count * similarity);
    }
  }
  return candidates.length ? weightedChoice(candidates, weights, rng) : anchor;
}

function chooseObservedMetricForm(records, desiredFirstRhythm, previousRhythm, rhythmTransitionWindows, eventIndex, options, rng) {
  if (!records.length) throw new Error("Nessuna forma metrica disponibile.");
  const desiredFirst = String(desiredFirstRhythm);
  const exactCandidates = records.filter((metricForm) => metricForm.durations_ql.length && String(metricForm.durations_ql[0]) === desiredFirst);
  let selected;
  if (exactCandidates.length) {
    selected = weightedChoice(exactCandidates, exactCandidates.map((metricForm) => metricForm.count), rng);
  } else if (previousRhythm !== null && previousRhythm !== undefined) {
    const scoredForms = [];
    const scores = [];
    for (const metricForm of records) {
      if (!metricForm.durations_ql.length) continue;
      const firstDuration = String(metricForm.durations_ql[0]);
      const score = metricForm.count * transitionProbabilityTimeVarying(
        rhythmTransitionWindows,
        previousRhythm,
        firstDuration,
        eventIndex,
        options,
      );
      scoredForms.push(metricForm);
      scores.push(score);
    }
    selected = scores.reduce((a, b) => a + b, 0) > 0
      ? weightedChoice(scoredForms, scores, rng)
      : weightedChoice(records, records.map((metricForm) => metricForm.count), rng);
  } else {
    selected = weightedChoice(records, records.map((metricForm) => metricForm.count), rng);
  }
  return maybeVaryMetricForm(selected, records, options, rng);
}

function generateMetricConstrainedRhythmSequence(rhythmTransitionWindows, originalRhythmSequence, metricForms, nEvents, options, rng) {
  if (nEvents <= 0) throw new Error("n_events deve essere positivo.");
  const records = metricFormsToRecords(metricForms);
  if (!records.length) {
    const generated = sampleMarkovSequenceTimeVarying(rhythmTransitionWindows, originalRhythmSequence, nEvents, options, rng);
    return {
      generated,
      formLog: [{
        generated_measure_index: "",
        metric_form_key: "",
        metric_form_label: "",
        warning: "Nessuna forma metrica completa disponibile: fallback a Markov durata-per-durata.",
      }],
    };
  }

  const empirical = empiricalDistribution(originalRhythmSequence);
  const generated = [];
  const formLog = [];
  let desiredFirstRhythm = weightedChoice(empirical.states, empirical.probabilities, rng);
  let previousRhythm = null;
  let generatedMeasureIndex = 1;

  while (generated.length < nEvents) {
    const transitionEventIndex = Math.max(0, generated.length - 1);
    const selectedForm = chooseObservedMetricForm(
      records,
      desiredFirstRhythm,
      previousRhythm,
      rhythmTransitionWindows,
      transitionEventIndex,
      options,
      rng,
    );
    const durations = selectedForm.durations_ql.map(String);
    generated.push(...durations);
    formLog.push({
      generated_measure_index: generatedMeasureIndex,
      metric_form_key: selectedForm.key,
      metric_form_label: selectedForm.label,
      source_measure_numbers: selectedForm.measure_numbers.join(","),
      form_count_in_original: selectedForm.count,
      desired_first_rhythm: String(desiredFirstRhythm),
      actual_first_rhythm: durations[0] || "",
      warning: String(durations[0]) === String(desiredFirstRhythm)
        ? ""
        : "Prima durata non compatibile: scelta forma osservata più plausibile.",
    });
    if (durations.length) {
      previousRhythm = durations[durations.length - 1];
      desiredFirstRhythm = sampleNextStateTimeVarying(
        rhythmTransitionWindows,
        previousRhythm,
        originalRhythmSequence,
        Math.max(0, generated.length - 1),
        options,
        rng,
      );
    }
    generatedMeasureIndex += 1;
  }
  return { generated, formLog };
}

function generatedPartSequencesToNotes(generated, partName, midiProgram, tempo, gate) {
  const nEvents = generated.pitch_class.length;
  let currentOnsetQl = 0;
  const rows = [];
  for (let i = 0; i < nEvents; i += 1) {
    const pitchClass = Number(generated.pitch_class[i]);
    const register = Number(generated.register[i]);
    const rhythmQl = Math.max(Number(generated.rhythm[i]), RHYTHM_GRID);
    const beatStrength = Number(generated.beat_strength[i]);
    const dynamic = String(generated.dynamic[i]);
    const midiPitch = generated.midi_pitch?.[i] !== undefined
      ? clamp(Number(generated.midi_pitch[i]), 0, 127)
      : pitchClassAndRegisterToMidiPitch(pitchClass, register);
    const finalPitchClass = ((midiPitch % 12) + 12) % 12;
    const finalRegister = Math.floor(midiPitch / 12) - 1;
    const baseVelocity = dynamicToVelocity(dynamic);
    const accentBonus = Math.round(16 * (beatStrength - 0.25));
    const velocity = clamp(baseVelocity + accentBonus, 1, 127);
    rows.push({
      event_index: i,
      part_name: partName,
      offset_ql: Number(currentOnsetQl.toFixed(6)),
      rhythm: rhythmQl,
      pitch_class: finalPitchClass,
      register: finalRegister,
      beat_strength: beatStrength,
      dynamic,
      velocity,
      midi_pitch: midiPitch,
    });
    currentOnsetQl += rhythmQl;
  }
  return { rows, midiProgram };
}

function bytesFromString(value) {
  return [...new TextEncoder().encode(String(value))];
}

function uint16(value) {
  return [(value >> 8) & 0xff, value & 0xff];
}

function uint32(value) {
  return [
    (value >> 24) & 0xff,
    (value >> 16) & 0xff,
    (value >> 8) & 0xff,
    value & 0xff,
  ];
}

function variableLengthQuantity(value) {
  let buffer = value & 0x7f;
  const bytes = [];
  while ((value >>= 7)) {
    buffer <<= 8;
    buffer |= (value & 0x7f) | 0x80;
  }
  while (true) {
    bytes.push(buffer & 0xff);
    if (buffer & 0x80) buffer >>= 8;
    else break;
  }
  return bytes;
}

function midiChunk(type, data) {
  return [...bytesFromString(type), ...uint32(data.length), ...data];
}

function encodeTrack(events) {
  const sorted = [...events].sort((a, b) => a.tick - b.tick || a.priority - b.priority);
  let previousTick = 0;
  const data = [];
  for (const event of sorted) {
    const delta = Math.max(0, event.tick - previousTick);
    data.push(...variableLengthQuantity(delta), ...event.bytes);
    previousTick = event.tick;
  }
  data.push(0, 0xff, 0x2f, 0);
  return midiChunk("MTrk", data);
}

function textMetaEvent(tick, type, text) {
  const payload = bytesFromString(text);
  return {
    tick,
    priority: 0,
    bytes: [0xff, type, ...variableLengthQuantity(payload.length), ...payload],
  };
}

function midiChannelForPart(partPosition) {
  const channel = partPosition % 15;
  return channel >= 9 ? channel + 1 : channel;
}

function buildGeneratedMidiBytes(parts, generatedRows, tempo, gate, ppq = 480) {
  const microsecondsPerBeat = Math.round(60000000 / Math.max(1, tempo));
  const outputParts = parts
    .map((part, partPosition) => ({
      part,
      partPosition,
      channel: midiChannelForPart(partPosition),
      rows: generatedRows.filter((row) => row.part_slug === part.part_slug),
    }))
    .filter((entry) => entry.rows.length > 0);
  const header = midiChunk("MThd", [
    ...uint16(1),
    ...uint16(outputParts.length + 1),
    ...uint16(ppq),
  ]);
  const tempoTrack = encodeTrack([
    {
      tick: 0,
      priority: 0,
      bytes: [
        0xff,
        0x51,
        3,
        (microsecondsPerBeat >> 16) & 0xff,
        (microsecondsPerBeat >> 8) & 0xff,
        microsecondsPerBeat & 0xff,
      ],
    },
    { tick: 0, priority: 1, bytes: [0xff, 0x58, 4, 4, 2, 24, 8] },
  ]);
  const tracks = [tempoTrack];

  for (const [outputIndex, entry] of outputParts.entries()) {
    const { part, channel, rows: trackRows } = entry;
    const trackName = `${String(outputIndex + 1).padStart(2, "0")} ${part.part_name}`;
    const events = [
      { tick: 0, priority: 0, bytes: [0xff, 0x00, 0x02, ...uint16(outputIndex + 1)] },
      { tick: 0, priority: 1, bytes: [0xff, 0x20, 0x01, channel] },
      textMetaEvent(0, 0x03, trackName),
      textMetaEvent(0, 0x04, trackName),
      {
        tick: 0,
        priority: 4,
        bytes: [0xc0 + channel, clamp(Number(part.midi_program), 0, 127)],
      },
    ];

    for (const row of trackRows) {
      const startTick = Math.round(Number(row.offset_ql) * ppq);
      const durationTick = Math.max(1, Math.round(Number(row.rhythm) * gate * ppq));
      const endTick = startTick + durationTick;
      const pitch = clamp(Number(row.midi_pitch), 0, 127);
      const velocity = clamp(Number(row.velocity), 1, 127);
      events.push(
        { tick: startTick, priority: 3, bytes: [0x90 + channel, pitch, velocity] },
        { tick: endTick, priority: 2, bytes: [0x80 + channel, pitch, 64] },
      );
    }

    tracks.push(encodeTrack(events));
  }

  return new Uint8Array([...header, ...tracks.flat()]);
}

function generateMultitrackMidi(generatedByPart, parts, tempo, gate) {
  const generatedRows = [];
  const trackManifestRows = [];
  for (const part of parts) {
    const generated = generatedByPart[part.part_slug];
    if (!generated) continue;
    const converted = generatedPartSequencesToNotes(generated, part.part_name, part.midi_program, tempo, gate);
    for (const row of converted.rows) {
      generatedRows.push({
        part_index: part.part_index,
        part_slug: part.part_slug,
        ...row,
      });
    }
  }
  for (const [partPosition, part] of parts.entries()) {
    const nEvents = generatedRows.filter((row) => row.part_slug === part.part_slug).length;
    if (nEvents <= 0) continue;
    trackManifestRows.push({
      generated_track_index: trackManifestRows.length + 1,
      part_index: part.part_index,
      part_slug: part.part_slug,
      part_name: part.part_name,
      midi_program: part.midi_program,
      midi_channel: midiChannelForPart(partPosition) + 1,
      n_generated_events: nEvents,
    });
  }
  return {
    midiBytes: buildGeneratedMidiBytes(parts, generatedRows, tempo, gate),
    generatedRows,
    trackManifestRows,
  };
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function toCsv(rows) {
  if (!rows.length) return "";
  const columns = [...rows.reduce((set, row) => {
    Object.keys(row).forEach((key) => set.add(key));
    return set;
  }, new Set())];
  return `${columns.join(",")}\n${rows.map((row) => columns.map((column) => csvEscape(row[column])).join(",")).join("\n")}\n`;
}

function matrixToCsv(matrix, states) {
  const header = ["state", ...states].map(csvEscape).join(",");
  const rows = states.map((state) => [state, ...states.map((target) => matrix[state]?.[target] ?? 0)].map(csvEscape).join(","));
  return `${header}\n${rows.join("\n")}\n`;
}

function sparseMatrixRows(part, param, window) {
  const rows = [];
  for (const source of window.states) {
    for (const target of window.states) {
      const count = Number(window.counts[source]?.[target] || 0);
      const probability = Number(window.probabilities[source]?.[target] || 0);
      if (count <= 0 && probability <= 0) continue;
      rows.push({
        part_index: part.part_index,
        part_name: part.part_name,
        part_slug: part.part_slug,
        parameter: param,
        window_index: window.window_index,
        start_event_index: window.start_event_index,
        end_event_index: window.end_event_index,
        source_state: source,
        target_state: target,
        transition_count: count,
        transition_probability: Number(probability.toFixed(8)),
      });
    }
  }
  return rows;
}

function yieldToBrowser() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function buildSummaryText(inputName, outdir, tempo, parts, options) {
  const lines = [
    "Markov MIDI multitrack generation summary",
    "=========================================",
    "",
    `Input MIDI: ${inputName}`,
    `Output MIDI: ${outdir}/generated_markov_multitrack.mid`,
    `Tempo MIDI generato: ${tempo} BPM`,
    `Seed: ${DEFAULT_SEED}`,
    `Gate: ${DEFAULT_GATE}`,
    `Numero parti con note: ${parts.length}`,
    `Metric form variation: ${options.metricFormVariation}`,
    `Metric similarity threshold: ${options.metricSimilarityThreshold}`,
    `Allow different meter variants: ${options.allowDifferentMeterVariants}`,
    `Transition window size: ${options.transitionWindowSize}`,
    `Transition window step: ${options.transitionWindowStep}`,
    `Modalità: ${isDeterministicMarkovMode(options) ? "ricostruzione con catene di Markov" : "generazione"}`,
    "",
    "Parti rilevate:",
    ...parts.map((part) => `- ${part.part_slug} | name=${part.part_name} | program=${part.midi_program} | events=${part.n_events}`),
    "",
    "Parametri modellati per ogni parte:",
    "- pitch_class",
    "- rhythm",
    "- beat_strength",
    "- register",
    "- dynamic",
    "- interval_size",
    "",
    "Output rilevanti:",
    "- original_events_by_part.csv",
    "- parts_metadata.csv",
    "- metric_forms_summary.csv",
    "- metric_forms_by_part/*.json",
    "- time_varying_transition_matrices_sparse.csv",
    "- time_varying_windows_by_part.csv",
    "- interval_profile_by_part.csv",
    "- generation_logs_by_part/*.csv",
    "- metric_form_generation_log.csv",
    "- generated_tracks_manifest.csv",
    "- generated_events_by_part.csv",
    "- generated_markov_multitrack.mid",
  ];
  return `${lines.join("\n")}\n`;
}

function addFile(files, path, content) {
  files.push({ path, content, size: content.byteLength ?? new Blob([content]).size });
}

function isDeterministicMarkovMode(options) {
  return Number(options.metricFormVariation) === 0
    && Number(options.metricSimilarityThreshold) === 1
    && Number(options.transitionWindowSize) === 2
    && Number(options.transitionWindowStep) === 1;
}

async function runPipeline(file, options) {
  const inputBuffer = await file.arrayBuffer();
  const midi = new Midi(inputBuffer);
  const tempo = Math.round(midi.header.tempos?.[0]?.bpm || DEFAULT_TEMPO);
  const rng = mulberry32(DEFAULT_SEED);
  const outdir = sanitizeName(options.outdir || "risultato");
  options.transitionWindowSize = Math.max(2, Math.floor(Number(options.transitionWindowSize) || DEFAULT_TRANSITION_WINDOW_SIZE));
  options.transitionWindowStep = Math.max(1, Math.floor(Number(options.transitionWindowStep) || DEFAULT_TRANSITION_WINDOW_STEP));
  const { events, parts } = extractEventsByPart(midi);
  const parameters = ["pitch_class", "rhythm", "beat_strength", "register", "dynamic"];
  const files = [];
  const generatedByPart = {};
  const summaryRows = [];
  const transitionWindowRows = [];
  const transitionMatrixRows = [];
  const intervalProfileRows = [];
  const metricFormRows = [];
  const allGenerationFormLogRows = [];
  const deterministicMarkovMode = isDeterministicMarkovMode(options);

  addFile(files, `${outdir}/original_events_by_part.csv`, toCsv(events));
  addFile(files, `${outdir}/parts_metadata.csv`, toCsv(parts));

  for (const part of parts) {
    const partEvents = events.filter((event) => event.part_slug === part.part_slug);
    const requestedEvents = partEvents.length;
    const metricForms = buildMetricFormsForPart(partEvents, false);
    addFile(
      files,
      `${outdir}/metric_forms_by_part/${part.part_slug}_metric_forms.json`,
      `${JSON.stringify(metricForms, null, 2)}\n`,
    );
    for (const metricForm of metricFormsToRecords(metricForms)) {
      metricFormRows.push({
        part_index: part.part_index,
        part_name: part.part_name,
        part_slug: part.part_slug,
        metric_form_key: metricForm.key,
        metric_form_label: metricForm.label,
        time_signature: metricForm.time_signature,
        bar_duration_ql: metricForm.bar_duration_ql,
        durations_ql: metricForm.durations_ql.join("|"),
        measure_numbers: metricForm.measure_numbers.join(","),
        count: metricForm.count,
        fills_measure: metricForm.fills_measure,
      });
    }

    const transitionByParam = {};
    const originalByParam = {};
    generatedByPart[part.part_slug] = {};

    for (const param of parameters) {
      const sequence = partEvents.map((event) => String(event[param]));
      originalByParam[param] = sequence;
      if (sequence.length < 2) {
        summaryRows.push({
          part_index: part.part_index,
          part_name: part.part_name,
          part_slug: part.part_slug,
          parameter: param,
          n_states: 1,
          n_original_events: sequence.length,
          n_requested_events: requestedEvents,
          n_generated_events: requestedEvents,
          warning: "Sequenza troppo breve: stato ripetuto.",
        });
        continue;
      }
      const transitionWindows = buildTimeVaryingTransitionMatrices(
        sequence,
        options.transitionWindowSize,
        options.transitionWindowStep,
      );
      transitionByParam[param] = transitionWindows;

      for (const window of transitionWindows) {
        transitionWindowRows.push({
          part_index: part.part_index,
          part_name: part.part_name,
          part_slug: part.part_slug,
          parameter: param,
          window_index: window.window_index,
          start_event_index: window.start_event_index,
          end_event_index: window.end_event_index,
          n_events_in_window: window.n_events_in_window,
          n_transitions: window.n_transitions,
          n_states: window.states.length,
          window_size_requested: options.transitionWindowSize,
          window_step_requested: options.transitionWindowStep,
          warning: window.warning,
        });
        transitionMatrixRows.push(...sparseMatrixRows(part, param, window));
      }
      await yieldToBrowser();
    }

    let generatedRhythm;
    let formLog;
    if (deterministicMarkovMode && transitionByParam.rhythm?.length && originalByParam.rhythm.length >= 2) {
      generatedRhythm = sampleMarkovSequenceTimeVarying(
        transitionByParam.rhythm,
        originalByParam.rhythm,
        requestedEvents,
        options,
        rng,
        originalByParam.rhythm[0],
      );
      formLog = [{
        generated_measure_index: "",
        metric_form_key: "",
        metric_form_label: "",
        source_measure_numbers: "",
        form_count_in_original: "",
        desired_first_rhythm: originalByParam.rhythm[0],
        actual_first_rhythm: generatedRhythm[0],
        warning: "Ricostruzione deterministica con finestra 2 e passo 1.",
      }];
    } else if (transitionByParam.rhythm?.length && originalByParam.rhythm.length >= 2) {
      const result = generateMetricConstrainedRhythmSequence(
        transitionByParam.rhythm,
        originalByParam.rhythm,
        metricForms,
        requestedEvents,
        options,
        rng,
      );
      generatedRhythm = result.generated;
      formLog = result.formLog;
    } else {
      const rhythmState = originalByParam.rhythm[0];
      generatedRhythm = Array(requestedEvents).fill(rhythmState);
      formLog = [{
        generated_measure_index: "",
        metric_form_key: "",
        metric_form_label: "",
        source_measure_numbers: "",
        form_count_in_original: "",
        desired_first_rhythm: rhythmState,
        actual_first_rhythm: rhythmState,
        warning: "Sequenza ritmica troppo breve: stato ripetuto.",
      }];
    }
    generatedByPart[part.part_slug].rhythm = generatedRhythm;
    const actualEvents = generatedRhythm.length;
    const partFormRows = formLog.map((row) => ({
      part_index: part.part_index,
      part_name: part.part_name,
      part_slug: part.part_slug,
      ...row,
    }));
    allGenerationFormLogRows.push(...partFormRows);
    addFile(files, `${outdir}/generation_logs_by_part/${part.part_slug}_metric_form_generation_log.csv`, toCsv(partFormRows));

    for (const param of parameters) {
      const sequence = originalByParam[param];
      let generatedSequence;
      if (param === "rhythm") {
        generatedSequence = generatedRhythm;
      } else if (sequence.length < 2 || !transitionByParam[param]?.length) {
        generatedSequence = Array(actualEvents).fill(sequence[0]);
      } else {
        generatedSequence = sampleMarkovSequenceTimeVarying(
          transitionByParam[param],
          sequence,
          actualEvents,
          options,
          rng,
          deterministicMarkovMode ? sequence[0] : null,
        );
      }
      generatedByPart[part.part_slug][param] = generatedSequence;
      summaryRows.push({
        part_index: part.part_index,
        part_name: part.part_name,
        part_slug: part.part_slug,
        parameter: param,
        n_states: new Set(sequence).size,
        n_original_events: sequence.length,
        n_requested_events: requestedEvents,
        n_generated_events: generatedSequence.length,
        transition_window_size: options.transitionWindowSize,
        transition_window_step: options.transitionWindowStep,
        n_transition_windows: transitionByParam[param]?.length || 0,
        mode: deterministicMarkovMode ? "ricostruzione con catene di Markov" : "generazione",
        warning: sequence.length >= 2 ? "" : "Sequenza troppo breve: stato ripetuto.",
      });
    }

    const originalMidiPitches = partEvents.map((event) => Number(event.original_midi_pitch));
    const intervalSequence = buildIntervalSizeSequence(originalMidiPitches);
    let desiredIntervals = [];
    let intervalWindows = [];

    if (intervalSequence.length >= 2) {
      intervalWindows = buildTimeVaryingTransitionMatrices(
        intervalSequence,
        options.transitionWindowSize,
        options.transitionWindowStep,
      );
      for (const window of intervalWindows) {
        transitionWindowRows.push({
          part_index: part.part_index,
          part_name: part.part_name,
          part_slug: part.part_slug,
          parameter: "interval_size",
          window_index: window.window_index,
          start_event_index: window.start_event_index,
          end_event_index: window.end_event_index,
          n_events_in_window: window.n_events_in_window,
          n_transitions: window.n_transitions,
          n_states: window.states.length,
          window_size_requested: options.transitionWindowSize,
          window_step_requested: options.transitionWindowStep,
          warning: window.warning,
        });
        transitionMatrixRows.push(...sparseMatrixRows(part, "interval_size", window));
      }
      desiredIntervals = sampleMarkovSequenceTimeVarying(
        intervalWindows,
        intervalSequence,
        Math.max(1, actualEvents - 1),
        options,
        rng,
        deterministicMarkovMode ? intervalSequence[0] : null,
      );
    } else {
      desiredIntervals = Array(Math.max(0, actualEvents - 1)).fill(intervalSequence[0] ?? "0");
    }

    const intervalRows = applyIntervalProfileConstraint(
      generatedByPart[part.part_slug],
      originalMidiPitches,
      desiredIntervals,
      deterministicMarkovMode,
    ).map((row) => ({
      part_index: part.part_index,
      part_name: part.part_name,
      part_slug: part.part_slug,
      ...row,
    }));
    intervalProfileRows.push(...intervalRows);

    summaryRows.push({
      part_index: part.part_index,
      part_name: part.part_name,
      part_slug: part.part_slug,
      parameter: "interval_size",
      n_states: new Set(intervalSequence).size,
      n_original_events: intervalSequence.length,
      n_requested_events: Math.max(0, requestedEvents - 1),
      n_generated_events: desiredIntervals.length,
      transition_window_size: options.transitionWindowSize,
      transition_window_step: options.transitionWindowStep,
      n_transition_windows: intervalWindows.length,
      mode: "vincolo profilo intervallare",
      warning: intervalSequence.length >= 1 ? "" : "Profilo intervallare non disponibile.",
    });
    await yieldToBrowser();
  }

  addFile(files, `${outdir}/metric_forms_summary.csv`, toCsv(metricFormRows));
  addFile(files, `${outdir}/time_varying_windows_by_part.csv`, toCsv(transitionWindowRows));
  addFile(files, `${outdir}/time_varying_transition_matrices_sparse.csv`, toCsv(transitionMatrixRows));
  addFile(files, `${outdir}/interval_profile_by_part.csv`, toCsv(intervalProfileRows));
  addFile(files, `${outdir}/metric_form_generation_log.csv`, toCsv(allGenerationFormLogRows));
  const { midiBytes, generatedRows, trackManifestRows } = generateMultitrackMidi(generatedByPart, parts, tempo, DEFAULT_GATE);
  addFile(files, `${outdir}/generated_markov_multitrack.mid`, midiBytes);
  addFile(files, `${outdir}/generated_tracks_manifest.csv`, toCsv(trackManifestRows));
  addFile(files, `${outdir}/generated_events_by_part.csv`, toCsv(generatedRows));
  addFile(files, `${outdir}/summary_by_part.csv`, toCsv(summaryRows));
  addFile(files, `${outdir}/summary.txt`, buildSummaryText(file.name, outdir, tempo, parts, options));

  const zip = new JSZip();
  for (const generatedFile of files) zip.file(generatedFile.path, generatedFile.content);
  const zipBlob = await zip.generateAsync({ type: "blob", compression: "DEFLATE" });
  return {
    files,
    zipBlob,
    outdir,
    pianoRoll: {
      original: events.map((event) => ({
        offset_ql: Number(event.offset_ql),
        rhythm: Number(event.rhythm),
        midi_pitch: Number(event.original_midi_pitch),
      })),
      generated: generatedRows.map((event) => ({
        offset_ql: Number(event.offset_ql),
        rhythm: Number(event.rhythm),
        midi_pitch: Number(event.midi_pitch),
      })),
    },
  };
}

function renderFileSummary(files, outdir) {
  const totalBytes = files.reduce((sum, generatedFile) => sum + generatedFile.size, 0);
  fileSummary.textContent = `${files.length} elementi preparati in ${outdir}. Dimensione archivio: ${formatBytes(totalBytes)}.`;
  fileSummary.hidden = false;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function drawPianoRoll(data) {
  const canvas = pianoRollCanvas;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#fffdf8";
  ctx.fillRect(0, 0, rect.width, rect.height);

  const notes = [...data.original, ...data.generated].filter((note) => Number.isFinite(note.midi_pitch));
  if (!notes.length) return;

  const minPitch = Math.max(0, Math.min(...notes.map((note) => note.midi_pitch)) - 2);
  const maxPitch = Math.min(127, Math.max(...notes.map((note) => note.midi_pitch)) + 2);
  const maxTime = Math.max(
    RHYTHM_GRID,
    ...notes.map((note) => Number(note.offset_ql) + Math.max(Number(note.rhythm), RHYTHM_GRID)),
  );
  const padLeft = 34;
  const padRight = 12;
  const padTop = 12;
  const padBottom = 24;
  const plotWidth = Math.max(1, rect.width - padLeft - padRight);
  const plotHeight = Math.max(1, rect.height - padTop - padBottom);
  const pitchSpan = Math.max(1, maxPitch - minPitch + 1);
  const x = (ql) => padLeft + (Number(ql) / maxTime) * plotWidth;
  const y = (pitch) => padTop + ((maxPitch - Number(pitch)) / pitchSpan) * plotHeight;
  const noteHeight = Math.max(2, plotHeight / pitchSpan);

  ctx.strokeStyle = "rgba(22,22,22,0.08)";
  ctx.lineWidth = 1;
  for (let pitch = Math.ceil(minPitch / 12) * 12; pitch <= maxPitch; pitch += 12) {
    const yy = y(pitch);
    ctx.beginPath();
    ctx.moveTo(padLeft, yy);
    ctx.lineTo(rect.width - padRight, yy);
    ctx.stroke();
    ctx.fillStyle = "rgba(22,22,22,0.5)";
    ctx.font = "11px sans-serif";
    ctx.fillText(String(pitch), 6, yy + 4);
  }

  const beatStep = maxTime > 96 ? 16 : maxTime > 48 ? 8 : 4;
  for (let beat = 0; beat <= maxTime; beat += beatStep) {
    const xx = x(beat);
    ctx.strokeStyle = beat === 0 ? "rgba(22,22,22,0.24)" : "rgba(22,22,22,0.08)";
    ctx.beginPath();
    ctx.moveTo(xx, padTop);
    ctx.lineTo(xx, rect.height - padBottom);
    ctx.stroke();
    ctx.fillStyle = "rgba(22,22,22,0.5)";
    ctx.font = "11px sans-serif";
    ctx.fillText(String(beat), xx - 4, rect.height - 7);
  }

  function drawNotes(notesToDraw, color, alpha) {
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    for (const note of notesToDraw) {
      const xx = x(note.offset_ql);
      const yy = y(note.midi_pitch);
      const ww = Math.max(1.5, (Math.max(Number(note.rhythm), RHYTHM_GRID) / maxTime) * plotWidth);
      ctx.fillRect(xx, yy, ww, Math.max(2, noteHeight * 0.82));
    }
    ctx.globalAlpha = 1;
  }

  drawNotes(data.original, "#253f7a", Number(originalAlphaInput.value));
  drawNotes(data.generated, "#b34a28", Number(generatedAlphaInput.value));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(form);
  const file = data.get("midiFile");
  if (!file || !file.size) return;
  submitButton.disabled = true;
  statusText.textContent = "Ascolto del brano e creazione in corso...";
  downloadZip.hidden = true;
  fileSummary.hidden = true;
  pianoRollPanel.hidden = true;
  lastPianoRollData = null;
  try {
    await yieldToBrowser();
    const result = await runPipeline(file, {
      outdir: data.get("outdir"),
      metricFormVariation: Number(data.get("metricFormVariation")),
      metricSimilarityThreshold: Number(data.get("metricSimilarityThreshold")),
      allowDifferentMeterVariants: data.has("allowDifferentMeterVariants"),
      transitionWindowSize: Number(data.get("transitionWindowSize")),
      transitionWindowStep: Number(data.get("transitionWindowStep")),
    });
    renderFileSummary(result.files, result.outdir);
    lastPianoRollData = result.pianoRoll;
    pianoRollPanel.hidden = false;
    drawPianoRoll(lastPianoRollData);
    const url = URL.createObjectURL(result.zipBlob);
    downloadZip.href = url;
    downloadZip.download = `${result.outdir}.zip`;
    downloadZip.hidden = false;
    statusText.textContent = "Pronto. Disegno delle note e archivio disponibili.";
  } catch (error) {
    statusText.textContent = `Errore: ${error.message}`;
  } finally {
    submitButton.disabled = false;
  }
});

function drawHero() {
  const canvas = document.querySelector("#heroCanvas");
  const ctx = canvas.getContext("2d");
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#171716";
  ctx.fillRect(0, 0, rect.width, rect.height);
  const rows = 16;
  const cellHeight = rect.height / rows;
  const colors = ["#f0b15b", "#0c6b5f", "#b34a28", "#fffdf8", "#253f7a"];
  for (let row = 0; row < rows; row += 1) {
    ctx.strokeStyle = "rgba(255,253,248,0.07)";
    ctx.beginPath();
    ctx.moveTo(0, row * cellHeight);
    ctx.lineTo(rect.width, row * cellHeight);
    ctx.stroke();
  }
  let seed = 8;
  for (let i = 0; i < 120; i += 1) {
    seed = (seed * 9301 + 49297) % 233280;
    const random = seed / 233280;
    const x = (i / 120) * rect.width;
    const y = Math.floor(random * rows) * cellHeight + cellHeight * 0.25;
    const width = 20 + ((i * 17) % 76);
    ctx.fillStyle = colors[i % colors.length];
    ctx.globalAlpha = i % 5 === 3 ? 0.55 : 0.82;
    ctx.fillRect(x, y, width, Math.max(3, cellHeight * 0.16));
  }
  ctx.globalAlpha = 1;
  const gradient = ctx.createLinearGradient(0, 0, 0, rect.height);
  gradient.addColorStop(0, "rgba(23,23,22,0.05)");
  gradient.addColorStop(0.72, "rgba(23,23,22,0.44)");
  gradient.addColorStop(1, "rgba(23,23,22,0.92)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, rect.width, rect.height);
}

window.addEventListener("resize", drawHero);
window.addEventListener("resize", () => {
  if (lastPianoRollData) drawPianoRoll(lastPianoRollData);
});
drawHero();
