#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
markov_midi_multitrack.py

Modello di Markov discreto del primo ordine separato per traccia/parte,
con vincolo metrico basato sulle forme ritmiche effettivamente rilevate
in ciascuna parte.

Dato un file MIDI, lo script:

1. legge il MIDI;
2. separa gli eventi per parte/traccia;
3. per ogni parte estrae:
   - pitch_class: classe di altezza 0-11;
   - rhythm: durata in quarterLength quantizzata;
   - beat_strength: peso metrico music21 quantizzato;
   - register: ottava MIDI/music21;
   - dynamic: classe dinamica derivata dalla velocity;
   - measure_number: numero di battuta;
   - time_signature: metro rilevato;
   - position_in_measure_ql: posizione dell'evento nella battuta;
4. per ogni parte costruisce un dizionario delle forme metriche effettivamente rilevate:

       forma_metrica -> battute in cui la forma compare

   Esempi di forma:

       4/4 | 4 semiminime
       4/4 | 2 semiminime + 1 minima
       3/4 | 1 minima + 1 semiminima

5. per ogni parametro e per ogni parte costruisce:
   - matrice dei conteggi;
   - matrice delle probabilità di transizione;
   - heatmap della matrice;
   - grafo di transizione;
6. salva il piano roll del MIDI originale;
7. genera un nuovo MIDI multitraccia;
8. durante la generazione ritmica, non inventa liberamente sequenze di durate:
   - genera o propone una prima durata;
   - cerca forme metriche osservate che cominciano con quella durata;
   - se esistono, sceglie fra quelle forme osservate;
   - se non esistono, sceglie comunque una forma osservata, pesata dalla probabilità
     markoviana della sua prima durata rispetto alla durata precedente;
   - la battuta generata è quindi sempre una forma metrica effettivamente rilevata,
     salvo fallback esplicito quando nessuna forma completa è disponibile.

Installazione dipendenze:

    pip install music21 pretty_midi numpy pandas matplotlib networkx

Esecuzione:

    python markov_midi_multitrack.py input.mid --outdir markov_multitrack_output

Con piano roll separati per parte:

    python markov_midi_multitrack.py input.mid --outdir markov_multitrack_output --per-part-pianorolls

Con numero indicativo di eventi per parte:

    python markov_midi_multitrack.py input.mid --events 200 --outdir markov_multitrack_output

Nota:
Per default lo script preserva le forme metriche complete. Questo significa che, se --events 200
interromperebbe una forma metrica, la parte può generare qualche evento in più per completare
l'ultima forma. Per forzare il taglio esatto, usare:

    --allow-partial-final-form

Questo però può produrre un'ultima battuta metricamente incompleta.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pretty_midi
from music21 import chord, converter, meter, note, stream


# -------------------------------------------------------------------
# Parametri globali di discretizzazione
# -------------------------------------------------------------------

# In music21 la durata 1.0 corrisponde a una semiminima.
RHYTHM_GRID = 0.25

# Griglia per beatStrength.
BEAT_STRENGTH_GRID = 0.125

# Tolleranza per verificare se una forma riempie la battuta.
METRIC_FORM_TOLERANCE = 1e-5

# Classi dinamiche derivate da velocity MIDI.
# Formato: min_velocity, max_velocity, label, velocity_rappresentativa.
DYNAMIC_BINS = [
    (1, 35, "pp", 28),
    (36, 50, "p", 44),
    (51, 65, "mp", 58),
    (66, 80, "mf", 72),
    (81, 100, "f", 92),
    (101, 127, "ff", 112),
]

DEFAULT_VELOCITY = 64
DEFAULT_TEMPO = 120.0
DEFAULT_TIME_SIGNATURE = "4/4"
DEFAULT_BAR_DURATION_QL = 4.0


# -------------------------------------------------------------------
# Utility generali
# -------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    """
    Rende una stringa utilizzabile come nome file/cartella.
    """
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]+", "", name)
    return name or "unnamed"


def quantize_value(value: float, grid: float) -> float:
    """
    Quantizza un valore sulla griglia specificata.
    """
    if grid <= 0:
        return float(value)

    q = round(value / grid) * grid
    return round(q, 6)


def value_to_clean_string(value: float) -> str:
    """
    Converte un float quantizzato in stringa leggibile.
    """
    value = round(float(value), 6)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def parse_duration_string(value: Any) -> float:
    """
    Converte una durata salvata come stringa in float.
    """
    return float(value)


def velocity_to_dynamic(velocity: int | None) -> str:
    """
    Converte una velocity MIDI in una classe dinamica simbolica.
    """
    if velocity is None:
        velocity = DEFAULT_VELOCITY

    velocity = int(np.clip(velocity, 1, 127))

    for low, high, label, _representative in DYNAMIC_BINS:
        if low <= velocity <= high:
            return label

    return "mf"


def dynamic_to_velocity(label: str) -> int:
    """
    Converte una classe dinamica simbolica in una velocity rappresentativa.
    """
    for _low, _high, dyn_label, representative in DYNAMIC_BINS:
        if dyn_label == label:
            return representative

    return DEFAULT_VELOCITY


def pitch_class_and_register_to_midi_pitch(
    pitch_class: int,
    register: int,
) -> int:
    """
    Ricombina classe di altezza e registro in un pitch MIDI.

    Convenzione:
        C4 = 60

    Formula:
        midi = 12 * (ottava + 1) + pitch_class
    """
    midi_pitch = 12 * (int(register) + 1) + int(pitch_class)
    return int(np.clip(midi_pitch, 0, 127))


def sort_states(states: List[Any]) -> List[Any]:
    """
    Ordina stati numerici come numeri e stati simbolici come stringhe.
    """
    def key(x: Any):
        try:
            return (0, float(x))
        except Exception:
            return (1, str(x))

    return sorted(states, key=key)


def weighted_choice(
    items: List[Any],
    weights: List[float],
    rng: np.random.Generator,
) -> Any:
    """
    Scelta pesata robusta.
    Se i pesi sono tutti nulli o non validi, usa distribuzione uniforme.
    """
    if not items:
        raise ValueError("weighted_choice richiede almeno un elemento.")

    weights_arr = np.array(weights, dtype=float)

    if not np.all(np.isfinite(weights_arr)) or weights_arr.sum() <= 0:
        weights_arr = np.ones(len(items), dtype=float)

    weights_arr = weights_arr / weights_arr.sum()
    index = int(rng.choice(np.arange(len(items)), p=weights_arr))
    return items[index]


# -------------------------------------------------------------------
# Nomi italiani delle durate
# -------------------------------------------------------------------

def duration_name_it(duration_ql: float, count: int = 1) -> str:
    """
    Nome italiano approssimato della durata, assumendo:
        1.0 = semiminima.

    Per valori non standard, restituisce una forma numerica.
    """
    d = round(float(duration_ql), 6)

    names = {
        4.0: ("semibreve", "semibrevi"),
        3.0: ("minima puntata", "minime puntate"),
        2.0: ("minima", "minime"),
        1.5: ("semiminima puntata", "semiminime puntate"),
        1.0: ("semiminima", "semiminime"),
        0.75: ("croma puntata", "crome puntate"),
        0.5: ("croma", "crome"),
        0.375: ("semicroma puntata", "semicrome puntate"),
        0.25: ("semicroma", "semicrome"),
        0.125: ("biscroma", "biscrome"),
        0.0625: ("semibiscroma", "semibiscrome"),
    }

    if d in names:
        singular, plural = names[d]
        return singular if count == 1 else plural

    return f"durate da {value_to_clean_string(d)} quarterLength"


def metric_form_label_from_durations(durations: List[str]) -> str:
    """
    Produce una label leggibile preservando l'ordine della forma.

    Esempi:
        ["1", "1", "1", "1"] -> "4 semiminime"
        ["1", "1", "2"]      -> "2 semiminime + 1 minima"
        ["1", "2", "1"]      -> "1 semiminima + 1 minima + 1 semiminima"
    """
    if not durations:
        return "forma vuota"

    groups: List[Tuple[str, int]] = []
    current = durations[0]
    count = 1

    for d in durations[1:]:
        if d == current:
            count += 1
        else:
            groups.append((current, count))
            current = d
            count = 1

    groups.append((current, count))

    chunks: List[str] = []
    for duration_str, group_count in groups:
        duration_float = parse_duration_string(duration_str)
        chunks.append(
            f"{group_count} {duration_name_it(duration_float, group_count)}"
        )

    return " + ".join(chunks)


# -------------------------------------------------------------------
# Lettura parti/tracce
# -------------------------------------------------------------------

def ensure_measures(score: stream.Stream) -> stream.Stream:
    """
    Prova ad assicurare la presenza di battute nello score.

    Molti MIDI hanno già informazioni metriche. In altri casi music21 può creare
    battute tramite makeMeasures(). Se l'operazione fallisce, lo score viene usato
    così com'è e i numeri di battuta vengono stimati con fallback 4/4.
    """
    try:
        measures = list(score.recurse().getElementsByClass(stream.Measure))
        if measures:
            return score
    except Exception:
        pass

    try:
        return score.makeMeasures(inPlace=False)
    except Exception:
        return score


def get_score_parts(score: stream.Score) -> List[stream.Stream]:
    """
    Restituisce le parti del MIDI.

    In alcuni MIDI music21 produce score.parts.
    In altri casi può restituire uno stream senza parti esplicite.
    In quel caso usiamo l'intero score come unica parte.
    """
    try:
        parts = list(score.parts)
    except Exception:
        parts = []

    if parts:
        return parts

    return [score]


def get_part_metadata(part: stream.Stream, part_index: int) -> Dict[str, Any]:
    """
    Estrae nome parte e programma MIDI, quando disponibili.

    Il programma viene usato per creare lo strumento nel MIDI generato.
    Se non viene trovato, si usa Acoustic Grand Piano, cioè program 0.
    """
    part_name = getattr(part, "partName", None)

    found_instrument = None
    try:
        found_instrument = part.getInstrument(returnDefault=False)
    except Exception:
        found_instrument = None

    if found_instrument is not None:
        if not part_name:
            part_name = found_instrument.partName or found_instrument.instrumentName

        midi_program = getattr(found_instrument, "midiProgram", None)
        if midi_program is None:
            midi_program = 0
    else:
        midi_program = 0

    if not part_name:
        part_name = f"part_{part_index + 1}"

    part_slug = f"{part_index + 1:02d}_{sanitize_name(part_name)}"

    return {
        "part_index": part_index,
        "part_name": str(part_name),
        "part_slug": part_slug,
        "midi_program": int(np.clip(midi_program, 0, 127)),
    }


def get_event_metric_context(
    el: stream.Music21Object,
    score: stream.Stream,
    offset_ql: float,
) -> Dict[str, Any]:
    """
    Restituisce informazioni metriche per un evento:
    - numero di battuta;
    - metro;
    - durata della battuta in quarterLength;
    - posizione dell'evento dentro la battuta.
    """
    ts = None
    try:
        ts = el.getContextByClass(meter.TimeSignature)
    except Exception:
        ts = None

    if ts is not None:
        time_signature = ts.ratioString
        bar_duration_ql = float(ts.barDuration.quarterLength)
    else:
        time_signature = DEFAULT_TIME_SIGNATURE
        bar_duration_ql = DEFAULT_BAR_DURATION_QL

    containing_measure = None
    try:
        containing_measure = el.getContextByClass(stream.Measure)
    except Exception:
        containing_measure = None

    if containing_measure is not None:
        measure_number_raw = getattr(containing_measure, "measureNumber", None)
        try:
            measure_number = int(measure_number_raw)
        except Exception:
            measure_number = int(np.floor(offset_ql / bar_duration_ql)) + 1

        try:
            measure_start_ql = float(containing_measure.getOffsetInHierarchy(score))
        except Exception:
            measure_start_ql = (measure_number - 1) * bar_duration_ql

        position_in_measure_ql = offset_ql - measure_start_ql
    else:
        measure_number = int(np.floor(offset_ql / bar_duration_ql)) + 1
        measure_start_ql = (measure_number - 1) * bar_duration_ql
        position_in_measure_ql = offset_ql - measure_start_ql

    position_in_measure_ql = quantize_value(position_in_measure_ql, RHYTHM_GRID)

    return {
        "measure_number": measure_number,
        "time_signature": time_signature,
        "bar_duration_ql": round(bar_duration_ql, 6),
        "position_in_measure_ql": value_to_clean_string(position_in_measure_ql),
    }


def extract_note_events_from_part(
    part: stream.Stream,
    score: stream.Score,
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Estrae eventi nota da una singola parte.

    Se la parte contiene accordi, ogni nota dell'accordo diventa un evento distinto.
    Questo preserva il contenuto armonico, ma non modella ancora accordi come oggetti unitari.
    Per l'estrazione delle forme metriche, gli eventi simultanei verranno poi compressi.
    """
    events: List[Dict[str, Any]] = []

    for el in part.recurse().notes:
        try:
            offset_ql = float(el.getOffsetInHierarchy(score))
        except Exception:
            offset_ql = float(el.offset)

        duration_ql = float(el.duration.quarterLength)
        rhythm = quantize_value(duration_ql, RHYTHM_GRID)

        try:
            beat_strength = float(el.beatStrength)
        except Exception:
            beat_strength = 0.25

        beat_strength = quantize_value(beat_strength, BEAT_STRENGTH_GRID)

        velocity = getattr(el.volume, "velocity", None)
        dynamic = velocity_to_dynamic(velocity)
        metric_context = get_event_metric_context(el, score, offset_ql)

        common = {
            "part_index": metadata["part_index"],
            "part_name": metadata["part_name"],
            "part_slug": metadata["part_slug"],
            "midi_program": metadata["midi_program"],
            "offset_ql": round(offset_ql, 6),
            "rhythm": value_to_clean_string(rhythm),
            "beat_strength": value_to_clean_string(beat_strength),
            "dynamic": dynamic,
            **metric_context,
        }

        if isinstance(el, note.Note):
            p = el.pitch
            events.append(
                {
                    **common,
                    "pitch_class": int(p.pitchClass),
                    "register": int(p.octave if p.octave is not None else 4),
                    "original_midi_pitch": int(p.midi),
                }
            )

        elif isinstance(el, chord.Chord):
            for p in el.pitches:
                events.append(
                    {
                        **common,
                        "pitch_class": int(p.pitchClass),
                        "register": int(p.octave if p.octave is not None else 4),
                        "original_midi_pitch": int(p.midi),
                    }
                )

    return events


def extract_events_by_part(midi_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Legge un MIDI e restituisce:

    - events_df: tutti gli eventi nota con identificativo della parte;
    - parts_df: metadati delle parti.
    """
    score_raw = converter.parse(str(midi_path))
    score = ensure_measures(score_raw)
    parts = get_score_parts(score)

    all_events: List[Dict[str, Any]] = []
    part_rows: List[Dict[str, Any]] = []

    for part_index, part in enumerate(parts):
        metadata = get_part_metadata(part, part_index)
        events = extract_note_events_from_part(part, score, metadata)

        # Salviamo solo parti che contengono almeno una nota.
        if not events:
            continue

        all_events.extend(events)

        part_rows.append(
            {
                "part_index": metadata["part_index"],
                "part_name": metadata["part_name"],
                "part_slug": metadata["part_slug"],
                "midi_program": metadata["midi_program"],
                "n_events": len(events),
            }
        )

    if not all_events:
        raise ValueError("Nessuna nota trovata nel file MIDI.")

    events_df = pd.DataFrame(all_events)
    events_df = events_df.sort_values(
        by=["part_index", "offset_ql", "original_midi_pitch"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    parts_df = pd.DataFrame(part_rows)
    parts_df = parts_df.sort_values(by="part_index").reset_index(drop=True)

    return events_df, parts_df


# -------------------------------------------------------------------
# Forme metriche osservate
# -------------------------------------------------------------------

def build_metric_forms_for_part(
    part_df: pd.DataFrame,
    include_incomplete_forms: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Costruisce il dizionario delle forme metriche effettivamente rilevate
    in una singola parte.

    La chiave è costruita così:

        <time_signature>::<durata1>|<durata2>|...

    Esempio:

        4/4::1|1|1|1
        4/4::1|1|2

    Il valore contiene:
    - label leggibile;
    - metro;
    - durata della battuta;
    - sequenza di durate;
    - battute in cui la forma compare;
    - conteggio delle occorrenze.

    Gli accordi vengono compressi ritmicamente: più note allo stesso onset
    e con la stessa durata valgono come un solo evento ritmico.
    """
    forms: Dict[str, Dict[str, Any]] = {}

    if part_df.empty:
        return forms

    required_cols = {
        "measure_number",
        "position_in_measure_ql",
        "rhythm",
        "time_signature",
        "bar_duration_ql",
    }

    if not required_cols.issubset(set(part_df.columns)):
        return forms

    for measure_number, measure_df in part_df.groupby("measure_number"):
        # Compressione degli eventi simultanei per evitare che un accordo venga letto
        # come ripetizione ritmica.
        rhythm_events = measure_df.drop_duplicates(
            subset=["position_in_measure_ql", "rhythm"]
        ).sort_values(
            by=["position_in_measure_ql", "rhythm"],
            ascending=[True, True],
        )

        if rhythm_events.empty:
            continue

        durations = [str(x) for x in rhythm_events["rhythm"].tolist()]
        total_duration = sum(parse_duration_string(d) for d in durations)

        time_signature = str(rhythm_events["time_signature"].iloc[0])
        bar_duration_ql = float(rhythm_events["bar_duration_ql"].iloc[0])

        fills_measure = abs(total_duration - bar_duration_ql) <= METRIC_FORM_TOLERANCE

        # Per default consideriamo solo forme che riempiono il metro rilevato.
        # Questo evita di usare frammenti con pause implicite come forme metriche complete.
        if not fills_measure and not include_incomplete_forms:
            continue

        duration_key = "|".join(durations)
        key = f"{time_signature}::{duration_key}"
        readable = metric_form_label_from_durations(durations)
        label = f"{time_signature} | {readable}"

        if key not in forms:
            forms[key] = {
                "key": key,
                "label": label,
                "time_signature": time_signature,
                "bar_duration_ql": bar_duration_ql,
                "durations_ql": durations,
                "measure_numbers": [],
                "count": 0,
                "fills_measure": fills_measure,
            }

        forms[key]["measure_numbers"].append(int(measure_number))
        forms[key]["count"] += 1

    # Ordinamento interno delle battute.
    for form in forms.values():
        form["measure_numbers"] = sorted(set(form["measure_numbers"]))

    return forms


def metric_forms_to_records(metric_forms: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converte il dizionario delle forme metriche in una lista ordinata.
    """
    records = list(metric_forms.values())
    return sorted(
        records,
        key=lambda x: (-int(x["count"]), str(x["key"])),
    )


def save_metric_forms(
    metric_forms: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Salva le forme metriche in JSON leggibile.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metric_forms, f, indent=2, ensure_ascii=False)


# -------------------------------------------------------------------
# Markov del primo ordine
# -------------------------------------------------------------------

def build_transition_matrices(
    sequence: List[Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Costruisce matrice dei conteggi e matrice di probabilità.

    La probabilità modellata è:

        P(x_{t+1} | x_t)
    """
    if len(sequence) < 2:
        raise ValueError("La sequenza deve contenere almeno due stati.")

    states = sort_states(list(set(sequence)))

    counts = pd.DataFrame(
        data=0,
        index=states,
        columns=states,
        dtype=float,
    )

    for current_state, next_state in zip(sequence[:-1], sequence[1:]):
        counts.loc[current_state, next_state] += 1

    row_sums = counts.sum(axis=1)
    probs = counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)

    return counts.astype(int), probs


def empirical_distribution(sequence: List[Any]) -> Tuple[List[Any], np.ndarray]:
    """
    Distribuzione empirica degli stati.

    Viene usata come fallback se uno stato non ha transizioni uscenti.
    """
    counter = Counter(sequence)
    states = sort_states(list(counter.keys()))
    counts = np.array([counter[s] for s in states], dtype=float)
    probs = counts / counts.sum()

    return states, probs


def sample_next_state(
    transition_probs: pd.DataFrame,
    current_state: Any,
    original_sequence: List[Any],
    rng: np.random.Generator,
) -> Any:
    """
    Campiona lo stato successivo da una matrice di transizione.
    Se lo stato corrente non ha uscite valide, usa la distribuzione empirica.
    """
    fallback_states, fallback_probs = empirical_distribution(original_sequence)

    if current_state in transition_probs.index:
        row = transition_probs.loc[current_state].to_numpy(dtype=float)
        states = list(transition_probs.columns)

        if row.sum() > 0:
            row = row / row.sum()
            return rng.choice(states, p=row)

    return rng.choice(fallback_states, p=fallback_probs)


def transition_probability(
    transition_probs: pd.DataFrame,
    current_state: Any,
    next_state: Any,
) -> float:
    """
    Restituisce P(next_state | current_state) se disponibile.
    """
    if current_state not in transition_probs.index:
        return 0.0
    if next_state not in transition_probs.columns:
        return 0.0
    return float(transition_probs.loc[current_state, next_state])


def sample_markov_sequence(
    transition_probs: pd.DataFrame,
    original_sequence: List[Any],
    n_events: int,
    rng: np.random.Generator,
) -> List[Any]:
    """
    Genera una nuova sequenza da una matrice di transizione.
    """
    if n_events <= 0:
        raise ValueError("n_events deve essere positivo.")

    fallback_states, fallback_probs = empirical_distribution(original_sequence)

    current_state = rng.choice(fallback_states, p=fallback_probs)
    generated = [current_state]

    for _ in range(n_events - 1):
        next_state = sample_next_state(
            transition_probs=transition_probs,
            current_state=current_state,
            original_sequence=original_sequence,
            rng=rng,
        )
        generated.append(next_state)
        current_state = next_state

    return generated


# -------------------------------------------------------------------
# Generazione ritmica vincolata dalle forme metriche osservate
# -------------------------------------------------------------------

def choose_observed_metric_form(
    metric_form_records: List[Dict[str, Any]],
    desired_first_rhythm: Any,
    previous_rhythm: Any | None,
    rhythm_transition_probs: pd.DataFrame,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Sceglie una forma metrica osservata.

    Strategia:

    1. Cerca forme osservate la cui prima durata coincide con desired_first_rhythm.
       Se le trova, sceglie fra queste pesando per frequenza osservata.

    2. Se non le trova, non inventa una forma nuova. Sceglie comunque fra le forme
       osservate, pesando ogni forma con:

           frequenza_forma * P(prima_durata_forma | previous_rhythm)

       Se anche questo produce pesi nulli, usa solo la frequenza osservata.
    """
    if not metric_form_records:
        raise ValueError("Nessuna forma metrica disponibile.")

    desired_first = str(desired_first_rhythm)

    exact_candidates = [
        form for form in metric_form_records
        if form["durations_ql"] and str(form["durations_ql"][0]) == desired_first
    ]

    if exact_candidates:
        weights = [float(form["count"]) for form in exact_candidates]
        return weighted_choice(exact_candidates, weights, rng)

    if previous_rhythm is not None:
        scored_forms: List[Dict[str, Any]] = []
        scores: List[float] = []

        for form in metric_form_records:
            if not form["durations_ql"]:
                continue

            first_duration = str(form["durations_ql"][0])
            prob = transition_probability(
                rhythm_transition_probs,
                previous_rhythm,
                first_duration,
            )
            score = float(form["count"]) * prob
            scored_forms.append(form)
            scores.append(score)

        if sum(scores) > 0:
            return weighted_choice(scored_forms, scores, rng)

    # Fallback non cieco: sempre su forme osservate, pesate per frequenza.
    weights = [float(form["count"]) for form in metric_form_records]
    return weighted_choice(metric_form_records, weights, rng)


def generate_metric_constrained_rhythm_sequence(
    rhythm_transition_probs: pd.DataFrame,
    original_rhythm_sequence: List[Any],
    metric_forms: Dict[str, Dict[str, Any]],
    n_events: int,
    rng: np.random.Generator,
    allow_partial_final_form: bool = False,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Genera una sequenza ritmica concatenando forme metriche osservate.

    A differenza di sample_markov_sequence(), qui la generazione non procede
    durata per durata in modo libero. Procede per battute/forme metriche:

    - si propone una prima durata con il modello di Markov;
    - si seleziona una forma metrica realmente osservata compatibile con quella durata;
    - si aggiunge l'intera forma;
    - si ripete fino a raggiungere il numero desiderato di eventi.

    Ritorna:
    - lista delle durate generate;
    - log delle forme metriche scelte.
    """
    if n_events <= 0:
        raise ValueError("n_events deve essere positivo.")

    records = metric_forms_to_records(metric_forms)

    # Se non sono state trovate forme metriche complete, fallback alla catena Markov classica.
    # Il log segnala chiaramente il fallback.
    if not records:
        generated = sample_markov_sequence(
            transition_probs=rhythm_transition_probs,
            original_sequence=original_rhythm_sequence,
            n_events=n_events,
            rng=rng,
        )
        return generated, [
            {
                "generated_measure_index": None,
                "metric_form_key": None,
                "metric_form_label": None,
                "warning": "Nessuna forma metrica completa disponibile: fallback a Markov durata-per-durata.",
            }
        ]

    empirical_states, empirical_probs = empirical_distribution(original_rhythm_sequence)

    generated: List[Any] = []
    form_log: List[Dict[str, Any]] = []

    # Primo ritmo generato empiricamente: da qui si cerca una forma osservata compatibile.
    desired_first_rhythm = rng.choice(empirical_states, p=empirical_probs)
    previous_rhythm: Any | None = None
    generated_measure_index = 1

    while len(generated) < n_events:
        selected_form = choose_observed_metric_form(
            metric_form_records=records,
            desired_first_rhythm=desired_first_rhythm,
            previous_rhythm=previous_rhythm,
            rhythm_transition_probs=rhythm_transition_probs,
            rng=rng,
        )

        durations = [str(d) for d in selected_form["durations_ql"]]
        generated.extend(durations)

        form_log.append(
            {
                "generated_measure_index": generated_measure_index,
                "metric_form_key": selected_form["key"],
                "metric_form_label": selected_form["label"],
                "source_measure_numbers": selected_form["measure_numbers"],
                "form_count_in_original": selected_form["count"],
                "desired_first_rhythm": str(desired_first_rhythm),
                "actual_first_rhythm": str(durations[0]) if durations else None,
                "warning": "" if str(durations[0]) == str(desired_first_rhythm) else "Prima durata non compatibile: scelta forma osservata più plausibile.",
            }
        )

        if durations:
            previous_rhythm = durations[-1]
            desired_first_rhythm = sample_next_state(
                transition_probs=rhythm_transition_probs,
                current_state=previous_rhythm,
                original_sequence=original_rhythm_sequence,
                rng=rng,
            )

        generated_measure_index += 1

    if allow_partial_final_form:
        generated = generated[:n_events]

    return generated, form_log


# -------------------------------------------------------------------
# Visualizzazione matrici, grafi e piano roll
# -------------------------------------------------------------------

def save_matrix_heatmap(
    matrix: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    """
    Salva una heatmap della matrice.
    """
    fig_width = max(6, 0.45 * len(matrix.columns))
    fig_height = max(5, 0.45 * len(matrix.index))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Stato successivo")
    ax.set_ylabel("Stato corrente")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))

    ax.set_xticklabels([str(x) for x in matrix.columns], rotation=90)
    ax.set_yticklabels([str(x) for x in matrix.index])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_transition_graph(
    transition_probs: pd.DataFrame,
    title: str,
    out_path: Path,
    edge_threshold: float = 0.0,
) -> None:
    """
    Salva un grafo orientato delle transizioni.
    """
    graph = nx.DiGraph()

    for source in transition_probs.index:
        graph.add_node(str(source))

    for source in transition_probs.index:
        for target in transition_probs.columns:
            prob = float(transition_probs.loc[source, target])
            if prob > edge_threshold:
                graph.add_edge(str(source), str(target), weight=prob)

    fig, ax = plt.subplots(figsize=(10, 8))

    if len(graph.nodes) == 0:
        ax.set_title(f"{title} - grafo vuoto")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    pos = nx.spring_layout(graph, seed=42, weight="weight")

    weights = np.array(
        [graph[u][v]["weight"] for u, v in graph.edges],
        dtype=float,
    )

    widths = 1.0 + 4.0 * weights if len(weights) > 0 else 1.0

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=1000,
        ax=ax,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=9,
        ax=ax,
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        width=widths,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )

    edge_labels = {
        (u, v): f"{graph[u][v]['weight']:.2f}"
        for u, v in graph.edges
    }

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        ax=ax,
    )

    ax.set_title(title)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_piano_roll(
    midi_path: Path,
    out_path: Path,
    title: str,
    fs: int = 25,
) -> None:
    """
    Salva il piano roll complessivo di un MIDI.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    piano_roll = pm.get_piano_roll(fs=fs)

    if piano_roll.size == 0:
        raise ValueError(f"Piano roll vuoto per {midi_path}")

    times = np.arange(piano_roll.shape[1]) / fs

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.imshow(
        piano_roll,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[times[0], times[-1] if len(times) > 1 else 0, 0, 127],
    )

    ax.set_title(title)
    ax.set_xlabel("Tempo, secondi")
    ax.set_ylabel("Pitch MIDI")
    ax.set_ylim(0, 127)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_part_piano_rolls(
    midi_path: Path,
    outdir: Path,
    title_prefix: str,
    fs: int = 25,
) -> None:
    """
    Salva un piano roll separato per ogni strumento/traccia pretty_midi.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    for i, inst in enumerate(pm.instruments):
        if inst.is_drum or not inst.notes:
            continue

        piano_roll = inst.get_piano_roll(fs=fs)
        if piano_roll.size == 0:
            continue

        times = np.arange(piano_roll.shape[1]) / fs
        name = inst.name or pretty_midi.program_to_instrument_name(inst.program)
        slug = f"{i + 1:02d}_{sanitize_name(name)}"

        fig, ax = plt.subplots(figsize=(14, 4))

        ax.imshow(
            piano_roll,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[times[0], times[-1] if len(times) > 1 else 0, 0, 127],
        )

        ax.set_title(f"{title_prefix} - {name}")
        ax.set_xlabel("Tempo, secondi")
        ax.set_ylabel("Pitch MIDI")
        ax.set_ylim(0, 127)

        fig.tight_layout()
        fig.savefig(outdir / f"{slug}_piano_roll.png", dpi=200)
        plt.close(fig)


# -------------------------------------------------------------------
# Generazione MIDI multitraccia
# -------------------------------------------------------------------

def generated_part_sequences_to_instrument(
    generated: Dict[str, List[Any]],
    part_name: str,
    midi_program: int,
    tempo: float,
    gate: float,
) -> Tuple[pretty_midi.Instrument, pd.DataFrame]:
    """
    Converte le sequenze generate di una parte in uno strumento pretty_midi.
    """
    n_events = len(generated["pitch_class"])

    instrument_pm = pretty_midi.Instrument(
        program=int(np.clip(midi_program, 0, 127)),
        name=part_name,
    )

    seconds_per_quarter = 60.0 / tempo
    current_onset_ql = 0.0

    generated_rows: List[Dict[str, Any]] = []

    for i in range(n_events):
        pitch_class = int(generated["pitch_class"][i])
        register = int(generated["register"][i])
        rhythm_ql = max(float(generated["rhythm"][i]), RHYTHM_GRID)
        beat_strength = float(generated["beat_strength"][i])
        dynamic = str(generated["dynamic"][i])

        midi_pitch = pitch_class_and_register_to_midi_pitch(
            pitch_class=pitch_class,
            register=register,
        )

        base_velocity = dynamic_to_velocity(dynamic)

        # Uso beatStrength come accento dinamico.
        # Il beatStrength non esiste come evento MIDI autonomo.
        accent_bonus = int(round(16 * (beat_strength - 0.25)))
        velocity = int(np.clip(base_velocity + accent_bonus, 1, 127))

        start_sec = current_onset_ql * seconds_per_quarter
        duration_ql = max(rhythm_ql * gate, RHYTHM_GRID * 0.5)
        end_sec = (current_onset_ql + duration_ql) * seconds_per_quarter

        midi_note = pretty_midi.Note(
            velocity=velocity,
            pitch=midi_pitch,
            start=start_sec,
            end=end_sec,
        )

        instrument_pm.notes.append(midi_note)

        generated_rows.append(
            {
                "event_index": i,
                "part_name": part_name,
                "offset_ql": round(current_onset_ql, 6),
                "rhythm": rhythm_ql,
                "pitch_class": pitch_class,
                "register": register,
                "beat_strength": beat_strength,
                "dynamic": dynamic,
                "velocity": velocity,
                "midi_pitch": midi_pitch,
            }
        )

        current_onset_ql += rhythm_ql

    generated_df = pd.DataFrame(generated_rows)
    return instrument_pm, generated_df


def generate_multitrack_midi(
    generated_by_part: Dict[str, Dict[str, List[Any]]],
    parts_df: pd.DataFrame,
    output_midi_path: Path,
    tempo: float,
    gate: float,
) -> pd.DataFrame:
    """
    Costruisce un MIDI multitraccia dalle sequenze generate per ogni parte.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    generated_tables: List[pd.DataFrame] = []

    for _, part_row in parts_df.iterrows():
        part_slug = part_row["part_slug"]
        part_name = part_row["part_name"]
        midi_program = int(part_row["midi_program"])

        if part_slug not in generated_by_part:
            continue

        instrument_pm, part_generated_df = generated_part_sequences_to_instrument(
            generated=generated_by_part[part_slug],
            part_name=part_name,
            midi_program=midi_program,
            tempo=tempo,
            gate=gate,
        )

        pm.instruments.append(instrument_pm)

        part_generated_df.insert(0, "part_slug", part_slug)
        part_generated_df.insert(0, "part_index", int(part_row["part_index"]))
        generated_tables.append(part_generated_df)

    pm.write(str(output_midi_path))

    if generated_tables:
        return pd.concat(generated_tables, ignore_index=True)

    return pd.DataFrame()


# -------------------------------------------------------------------
# Pipeline principale
# -------------------------------------------------------------------

def run_pipeline(
    input_midi: Path,
    outdir: Path,
    n_events: int | None,
    tempo: float,
    seed: int,
    edge_threshold: float,
    gate: float,
    per_part_pianorolls: bool,
    include_incomplete_metric_forms: bool,
    allow_partial_final_form: bool,
) -> None:
    """
    Esegue l'intera pipeline multitraccia.
    """
    rng = np.random.default_rng(seed)

    outdir.mkdir(parents=True, exist_ok=True)

    matrices_dir = outdir / "matrices_by_part"
    heatmaps_dir = outdir / "matrix_heatmaps_by_part"
    graphs_dir = outdir / "transition_graphs_by_part"
    pianorolls_dir = outdir / "piano_rolls"
    metric_forms_dir = outdir / "metric_forms_by_part"
    generation_logs_dir = outdir / "generation_logs_by_part"

    matrices_dir.mkdir(exist_ok=True)
    heatmaps_dir.mkdir(exist_ok=True)
    graphs_dir.mkdir(exist_ok=True)
    pianorolls_dir.mkdir(exist_ok=True)
    metric_forms_dir.mkdir(exist_ok=True)
    generation_logs_dir.mkdir(exist_ok=True)

    # 1. Estrazione eventi per parte.
    events_df, parts_df = extract_events_by_part(input_midi)

    events_df.to_csv(outdir / "original_events_by_part.csv", index=False)
    parts_df.to_csv(outdir / "parts_metadata.csv", index=False)

    # 2. Piano roll originale complessivo.
    save_piano_roll(
        midi_path=input_midi,
        out_path=pianorolls_dir / "original_piano_roll_global.png",
        title="Piano roll globale - MIDI originale",
    )

    if per_part_pianorolls:
        original_parts_pr_dir = pianorolls_dir / "original_parts"
        original_parts_pr_dir.mkdir(exist_ok=True)
        save_part_piano_rolls(
            midi_path=input_midi,
            outdir=original_parts_pr_dir,
            title_prefix="Piano roll originale",
        )

    parameters = [
        "pitch_class",
        "rhythm",
        "beat_strength",
        "register",
        "dynamic",
    ]

    generated_by_part: Dict[str, Dict[str, List[Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []
    all_metric_form_rows: List[Dict[str, Any]] = []
    all_generation_form_log_rows: List[Dict[str, Any]] = []

    # 3. Matrici, grafi, forme metriche e generazione separata per ciascuna parte.
    for _, part_row in parts_df.iterrows():
        part_slug = part_row["part_slug"]
        part_name = part_row["part_name"]
        part_index = int(part_row["part_index"])

        part_df = events_df[events_df["part_slug"] == part_slug].copy()
        part_df = part_df.sort_values(
            by=["offset_ql", "original_midi_pitch"],
            ascending=[True, True],
        ).reset_index(drop=True)

        # Se l'utente non imposta --events, manteniamo per ogni parte
        # lo stesso numero di eventi dell'originale come obiettivo indicativo.
        part_requested_events = int(n_events if n_events is not None else len(part_df))

        part_matrices_dir = matrices_dir / part_slug
        part_heatmaps_dir = heatmaps_dir / part_slug
        part_graphs_dir = graphs_dir / part_slug

        part_matrices_dir.mkdir(exist_ok=True)
        part_heatmaps_dir.mkdir(exist_ok=True)
        part_graphs_dir.mkdir(exist_ok=True)

        generated_by_part[part_slug] = {}

        # 3a. Forme metriche osservate nella parte.
        metric_forms = build_metric_forms_for_part(
            part_df=part_df,
            include_incomplete_forms=include_incomplete_metric_forms,
        )

        save_metric_forms(
            metric_forms=metric_forms,
            out_path=metric_forms_dir / f"{part_slug}_metric_forms.json",
        )

        for form in metric_forms_to_records(metric_forms):
            all_metric_form_rows.append(
                {
                    "part_index": part_index,
                    "part_name": part_name,
                    "part_slug": part_slug,
                    "metric_form_key": form["key"],
                    "metric_form_label": form["label"],
                    "time_signature": form["time_signature"],
                    "bar_duration_ql": form["bar_duration_ql"],
                    "durations_ql": "|".join(form["durations_ql"]),
                    "measure_numbers": ",".join(str(x) for x in form["measure_numbers"]),
                    "count": form["count"],
                    "fills_measure": form["fills_measure"],
                }
            )

        # 3b. Costruzione matrici per tutti i parametri.
        transition_probs_by_param: Dict[str, pd.DataFrame] = {}
        original_sequences_by_param: Dict[str, List[Any]] = {}

        for param in parameters:
            sequence = part_df[param].tolist()
            original_sequences_by_param[param] = sequence

            # Se una parte ha una sola nota, non può produrre una matrice di transizione.
            if len(sequence) < 2:
                summary_rows.append(
                    {
                        "part_index": part_index,
                        "part_name": part_name,
                        "part_slug": part_slug,
                        "parameter": param,
                        "n_states": 1,
                        "n_original_events": len(sequence),
                        "n_generated_events": part_requested_events,
                        "warning": "Sequenza troppo breve: stato ripetuto.",
                    }
                )
                continue

            counts, probs = build_transition_matrices(sequence)
            transition_probs_by_param[param] = probs

            counts.to_csv(part_matrices_dir / f"{param}_transition_counts.csv")
            probs.to_csv(part_matrices_dir / f"{param}_transition_probabilities.csv")

            save_matrix_heatmap(
                matrix=probs,
                title=f"{part_name} - matrice di transizione - {param}",
                out_path=part_heatmaps_dir / f"{param}_transition_matrix.png",
            )

            save_transition_graph(
                transition_probs=probs,
                title=f"{part_name} - grafo di transizione - {param}",
                out_path=part_graphs_dir / f"{param}_transition_graph.png",
                edge_threshold=edge_threshold,
            )

        # 3c. Generazione del ritmo con vincolo sulle forme metriche osservate.
        if "rhythm" in transition_probs_by_param and len(original_sequences_by_param["rhythm"]) >= 2:
            generated_rhythm, form_log = generate_metric_constrained_rhythm_sequence(
                rhythm_transition_probs=transition_probs_by_param["rhythm"],
                original_rhythm_sequence=original_sequences_by_param["rhythm"],
                metric_forms=metric_forms,
                n_events=part_requested_events,
                rng=rng,
                allow_partial_final_form=allow_partial_final_form,
            )
        else:
            # Fallback per parti con un solo evento.
            rhythm_state = original_sequences_by_param["rhythm"][0]
            generated_rhythm = [rhythm_state] * part_requested_events
            form_log = [
                {
                    "generated_measure_index": None,
                    "metric_form_key": None,
                    "metric_form_label": None,
                    "source_measure_numbers": [],
                    "form_count_in_original": None,
                    "desired_first_rhythm": rhythm_state,
                    "actual_first_rhythm": rhythm_state,
                    "warning": "Sequenza ritmica troppo breve: stato ripetuto.",
                }
            ]

        generated_by_part[part_slug]["rhythm"] = generated_rhythm
        actual_part_n_events = len(generated_rhythm)

        part_form_log_rows: List[Dict[str, Any]] = []
        for row in form_log:
            log_row = {
                "part_index": part_index,
                "part_name": part_name,
                "part_slug": part_slug,
                **row,
            }
            part_form_log_rows.append(log_row)
            all_generation_form_log_rows.append(log_row)

        pd.DataFrame(part_form_log_rows).to_csv(
            generation_logs_dir / f"{part_slug}_metric_form_generation_log.csv",
            index=False,
        )

        # 3d. Generazione degli altri parametri sulla lunghezza ritmica effettiva.
        # Se l'ultima forma viene preservata completa, actual_part_n_events può essere
        # maggiore di part_requested_events.
        for param in parameters:
            sequence = original_sequences_by_param[param]

            if param == "rhythm":
                generated_sequence = generated_rhythm
            elif len(sequence) < 2 or param not in transition_probs_by_param:
                generated_sequence = [sequence[0]] * actual_part_n_events
            else:
                generated_sequence = sample_markov_sequence(
                    transition_probs=transition_probs_by_param[param],
                    original_sequence=sequence,
                    n_events=actual_part_n_events,
                    rng=rng,
                )

            generated_by_part[part_slug][param] = generated_sequence

            summary_rows.append(
                {
                    "part_index": part_index,
                    "part_name": part_name,
                    "part_slug": part_slug,
                    "parameter": param,
                    "n_states": len(set(sequence)),
                    "n_original_events": len(sequence),
                    "n_requested_events": part_requested_events,
                    "n_generated_events": len(generated_sequence),
                    "warning": "" if len(sequence) >= 2 else "Sequenza troppo breve: stato ripetuto.",
                }
            )

    # 4. Salvataggio riepilogo forme metriche.
    metric_forms_summary_df = pd.DataFrame(all_metric_form_rows)
    metric_forms_summary_df.to_csv(outdir / "metric_forms_summary.csv", index=False)

    generation_form_log_df = pd.DataFrame(all_generation_form_log_rows)
    generation_form_log_df.to_csv(outdir / "metric_form_generation_log.csv", index=False)

    # 5. MIDI generato multitraccia.
    generated_midi_path = outdir / "generated_markov_multitrack.mid"

    generated_events_df = generate_multitrack_midi(
        generated_by_part=generated_by_part,
        parts_df=parts_df,
        output_midi_path=generated_midi_path,
        tempo=tempo,
        gate=gate,
    )

    generated_events_df.to_csv(outdir / "generated_events_by_part.csv", index=False)

    # 6. Piano roll generato complessivo.
    save_piano_roll(
        midi_path=generated_midi_path,
        out_path=pianorolls_dir / "generated_piano_roll_global.png",
        title="Piano roll globale - MIDI generato multitraccia",
    )

    if per_part_pianorolls:
        generated_parts_pr_dir = pianorolls_dir / "generated_parts"
        generated_parts_pr_dir.mkdir(exist_ok=True)
        save_part_piano_rolls(
            midi_path=generated_midi_path,
            outdir=generated_parts_pr_dir,
            title_prefix="Piano roll generato",
        )

    # 7. Riepiloghi.
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "summary_by_part.csv", index=False)

    summary_path = outdir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Markov MIDI multitrack generation summary\n")
        f.write("=========================================\n\n")
        f.write(f"Input MIDI: {input_midi}\n")
        f.write(f"Output MIDI: {generated_midi_path}\n")
        f.write(f"Tempo MIDI generato: {tempo} BPM\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Gate: {gate}\n")
        f.write(f"Numero parti con note: {len(parts_df)}\n")
        f.write(f"Include forme metriche incomplete: {include_incomplete_metric_forms}\n")
        f.write(f"Permetti ultima forma parziale: {allow_partial_final_form}\n\n")

        f.write("Parti rilevate:\n")
        for _, row in parts_df.iterrows():
            f.write(
                f"- {row['part_slug']} | "
                f"name={row['part_name']} | "
                f"program={row['midi_program']} | "
                f"events={row['n_events']}\n"
            )

        f.write("\nParametri modellati per ogni parte:\n")
        for param in parameters:
            f.write(f"- {param}\n")

        f.write("\nOutput rilevanti:\n")
        f.write("- original_events_by_part.csv\n")
        f.write("- metric_forms_summary.csv\n")
        f.write("- metric_forms_by_part/*.json\n")
        f.write("- metric_form_generation_log.csv\n")
        f.write("- generated_events_by_part.csv\n")
        f.write("- generated_markov_multitrack.mid\n")

    print("\nPipeline multitraccia completata.")
    print(f"Cartella output: {outdir.resolve()}")
    print(f"MIDI generato: {generated_midi_path.resolve()}")
    print(f"Eventi originali: {(outdir / 'original_events_by_part.csv').resolve()}")
    print(f"Eventi generati: {(outdir / 'generated_events_by_part.csv').resolve()}")
    print(f"Forme metriche: {(outdir / 'metric_forms_summary.csv').resolve()}")
    print(f"Log forme generate: {(outdir / 'metric_form_generation_log.csv').resolve()}")
    print(f"Metadati parti: {(outdir / 'parts_metadata.csv').resolve()}")
    print(f"Riepilogo: {summary_path.resolve()}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generazione MIDI multitraccia con modelli di Markov discreti del primo ordine "
            "separati per parte e vincolo sulle forme metriche osservate."
        )
    )

    parser.add_argument(
        "input_midi",
        type=Path,
        help="Percorso del file MIDI di input.",
    )

    parser.add_argument(
        "--events",
        type=int,
        default=None,
        help=(
            "Numero indicativo di eventi da generare per ciascuna parte. "
            "Se omesso, ogni parte usa come obiettivo lo stesso numero di eventi dell'originale. "
            "Per default l'ultima forma metrica viene completata, quindi il numero finale può essere maggiore."
        ),
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("markov_multitrack_output"),
        help="Cartella di output.",
    )

    parser.add_argument(
        "--tempo",
        type=float,
        default=DEFAULT_TEMPO,
        help="Tempo del MIDI generato in BPM.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed casuale per rendere la generazione riproducibile.",
    )

    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help=(
            "Soglia minima per disegnare un arco nei grafi. "
            "Esempio: 0.05 mostra solo transizioni con probabilità >= 5%."
        ),
    )

    parser.add_argument(
        "--gate",
        type=float,
        default=0.9,
        help="Fattore di durata delle note generate rispetto al ritmo. Default: 0.9.",
    )

    parser.add_argument(
        "--per-part-pianorolls",
        action="store_true",
        help="Salva anche i piano roll separati per parte/traccia.",
    )

    parser.add_argument(
        "--include-incomplete-metric-forms",
        action="store_true",
        help=(
            "Include anche forme che non riempiono interamente la battuta rilevata. "
            "Utile se il MIDI contiene pause implicite, ma meno rigoroso metricamente."
        ),
    )

    parser.add_argument(
        "--allow-partial-final-form",
        action="store_true",
        help=(
            "Taglia la sequenza ritmica esattamente al numero di eventi richiesto, "
            "anche se questo interrompe l'ultima forma metrica. Di default lo script completa l'ultima forma."
        ),
    )

    parser.add_argument(
        "--metric-similarity-threshold",
        type=float,
        default=0.6,
        help=(
            "Soglia di similarità, fra 0 e 1, per consentire la sostituzione di una forma metrica "
            "con un'altra forma osservata. 1.0 = quasi solo forme identiche; 0.6 = varianti vicine, "
            "per esempio fusioni/suddivisioni parziali; 0.4 = varianti più divergenti ma comunque osservate. "
            "Default: 0.6."
        ),
    )

    parser.add_argument(
        "--metric-form-variation",
        type=float,
        default=0.0,
        help=(
            "Probabilità, fra 0 e 1, di sostituire la forma metrica ancora con una forma osservata "
            "sufficientemente simile. 0.0 = nessuna variazione; 0.3 = circa 30%% di battute variate. "
            "Default: 0.0."
        ),
    )

    parser.add_argument(
        "--allow-different-meter-variants",
        action="store_true",
        help=(
            "Permette che una forma metrica venga sostituita da una forma osservata con metro diverso. "
            "Di default le varianti sono cercate solo nello stesso metro della forma-ancora."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        input_midi=args.input_midi,
        outdir=args.outdir,
        n_events=args.events,
        tempo=args.tempo,
        seed=args.seed,
        edge_threshold=args.edge_threshold,
        gate=args.gate,
        per_part_pianorolls=args.per_part_pianorolls,
        include_incomplete_metric_forms=args.include_incomplete_metric_forms,
        allow_partial_final_form=args.allow_partial_final_form,
        metric_similarity_threshold=args.metric_similarity_threshold,
        metric_form_variation_probability=args.metric_form_variation,
        allow_different_meter_variants=args.allow_different_meter_variants,
    )
