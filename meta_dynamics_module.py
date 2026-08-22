"""meta_dynamics_module.py — TIMDR-META-DYNAMICS jako modul wpiety w
Analizator Gieldowy v3 (a NIE osobne okno Tkinter).

KONTEKST: w repo TIMDR-META-DYNAMICS main.py oryginalnie zakladal import
`from analizator3_core import load_market_series` - modulu, ktory nigdy
nie istnial (komentarz w oryginale: "zakladam, ze masz taka funkcje").
Ten plik to REALNA integracja zamiast tamtego zalozenia: podlacza
timdr_meta_dynamics (meta-warstwe nad polem Λ-τ-ρ-J) bezposrednio do
PRAWDZIWEGO TimdrPacket z pipeline.py tego repo (ten sam pakiet, ktory
napedza dashboard pod /api/analyze), zamiast do nieistniejacego modulu.

Wymaga, zeby folder TIMDR-META-DYNAMICS lezal jako SASIAD tego repo
(ten sam ukladu, w jakim oba repo faktycznie siedza w Downloads\\a) -
patrz _ensure_timdr_meta_dynamics_on_path() nizej. Jesli go tam nie ma,
rzuca czytelny ImportError zamiast cichego niepowodzenia (ta sama
konwencja co DataLoaderError w data_loader.py - zaden `except: pass`).

MAPOWANIE Λ-τ-ρ-J -> realne sygnaly TIMDR z tego repo (decyzja, nie
jedyna mozliwa - patrz uzasadnienie przy kazdym polu):

    Λ (struktura)      = packet.trm.values      (mediana krocząca ceny -
                          to jest doslownie "linia struktury/trendu" w
                          tym repo, blizej "struktury" niz surowa cena)
    τ (transformacja)  = packet.flow.values      (tempo zmiany trm - to
                          jest doslownie "transformacja w toku", flow()
                          w timdr_core_finance.py to pochodna trm)
    ρ (anomalia)       = packet.resonance.values  (juz gotowy, ciagly
                          wskaznik [0,1] per bar - 0/3..3/3 zgodnosci
                          trzech niezaleznych sprawdzen anomalii/defektu/
                          skretu - blizej "anomalii" niz pojedynczy
                          surowy sygnal anomalies()/defect(), bo jest
                          juz zsyntetyzowany)
    J (operator punktowy) = wolumen (surowy, per bar - jak w oryginalnym
                          main.py z TIMDR-META-DYNAMICS, ktory tez uzywal
                          wolumenu jako J)

UWAGA: to INNE mapowanie niz w oryginalnym main.py z TIMDR-META-DYNAMICS
(tam bylo Λ=cena, τ=trm, ρ=flow, J=wolumen) - tamto bylo prowizorka na
danych demo bez dostepu do prawdziwych trm/flow/resonance. To tutaj jest
bardziej spojne z glosariuszem README TIMDR-META-DYNAMICS (τ=transformacja
pasuje do flow=pochodnej lepiej niz do trm=linii wygladzonej), bo mamy
teraz prawdziwe, juz policzone sygnaly do wyboru zamiast zgadywania z
samej ceny/wolumenu.
"""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pandas as pd


def _ensure_timdr_meta_dynamics_on_path() -> None:
    """Dodaje folder-siostre TIMDR-META-DYNAMICS do sys.path, jesli
    jeszcze go tam nie ma. Zaklada uklad Downloads\\a\\analizator-gieldowy-v3
    i Downloads\\a\\TIMDR-META-DYNAMICS jako katalogi na tym samym poziomie."""
    here = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.join(here, "..", "TIMDR-META-DYNAMICS")
    sibling = os.path.abspath(sibling)

    if not os.path.isdir(sibling):
        raise ImportError(
            "meta_dynamics_module wymaga folderu 'TIMDR-META-DYNAMICS' jako "
            f"sasiada tego repo (szukano w: {sibling}). Jesli lezy gdzie "
            "indziej, popraw sciezke w _ensure_timdr_meta_dynamics_on_path()."
        )
    if sibling not in sys.path:
        sys.path.insert(0, sibling)


_ensure_timdr_meta_dynamics_on_path()

from timdr_meta_dynamics import (  # noqa: E402  (import po sys.path.insert - celowo)
    MetaState,
    MetaOperatorM,
    FieldEvolution,
    MetaMap,
    MetaPredict,
)

from pipeline import TimdrEngine, TimdrPacket  # noqa: E402


def build_meta_states_from_packet(packet: TimdrPacket, volume: np.ndarray) -> List[MetaState]:
    """Konwertuje prawdziwy TimdrPacket (+ wolumen z OHLCV) na serie MetaState.

    Wymaga len(volume) == len(packet.trm.values) == ... (wszystkie tablice
    per-bar tej samej dlugosci, tak jak buduje je TimdrEngine.compute_packet() -
    trm/flow/resonance sa liczone na calej serii, nie tylko w indeksach
    zdarzen jak twist/anomaly/defect).
    """
    trm_vals = np.asarray(packet.trm.values, dtype=float)
    flow_vals = np.asarray(packet.flow.values, dtype=float)
    resonance_vals = np.asarray(packet.resonance.values, dtype=float)
    volume = np.asarray(volume, dtype=float)

    n = len(trm_vals)
    if not (len(flow_vals) == len(resonance_vals) == len(volume) == n):
        raise ValueError(
            "Niespojne dlugosci tablic w TimdrPacket/wolumenie: "
            f"trm={len(trm_vals)} flow={len(flow_vals)} "
            f"resonance={len(resonance_vals)} volume={len(volume)}. "
            "To by nie powinno sie zdarzyc przy normalnym TimdrEngine.compute_packet() "
            "- sprawdz, czy packet pochodzi z tego repo."
        )

    return [
        MetaState(Lambda=trm_vals[i], tau=flow_vals[i], rho=resonance_vals[i], J=volume[i])
        for i in range(n)
    ]


def analyze_ticker_meta(ohlcv: pd.DataFrame, dt: float = 1.0, predict_last_n: int = 10) -> dict:
    """Pelny pipeline meta-warstwy na REALNYCH danych jednego tickera.

    Przyjmuje juz pobrane `ohlcv` (patrz data_loader.fetch_ohlcv) - ta
    funkcja NIE pobiera danych sama, zeby dalo sie ja testowac bez sieci
    (ten sam wzorzec co run_pipeline(ohlcv) w pipeline.py) i zeby wywolujacy
    (np. api.py) mogl reuzyc juz pobrane ohlcv zamiast pobierac drugi raz.

    Zwraca dict gotowy do jsonify (listy/floaty, nie numpy) - konwersje
    analogiczne do _clean() w api.py, zrobione tutaj lokalnie zeby modul
    nie zalezal od api.py.
    """
    if len(ohlcv) < 2:
        raise ValueError(f"Za malo barow w ohlcv ({len(ohlcv)}) - potrzeba co najmniej 2.")

    engine = TimdrEngine(ohlcv)
    packet = engine.compute_packet()
    volume = ohlcv["volume"].values

    states = build_meta_states_from_packet(packet, volume)

    meta_operator = MetaOperatorM()
    field_evolution = FieldEvolution(meta_operator)
    M_series = field_evolution.simulate(states, dt)

    meta_map = MetaMap(meta_operator)
    phases = meta_map.detect_transitions(M_series)

    predictor = MetaPredict()
    future_states = predictor.simulate_future(states[-1], M_series[-predict_last_n:], dt)

    return {
        "n_bars": len(states),
        "phases": phases,
        "phases_last_20": phases[-20:],
        "current_phase": phases[-1] if phases else None,
        "future_lambda": [float(s.Lambda) for s in future_states],
        "future_tau": [float(s.tau) for s in future_states],
        "future_rho": [float(s.rho) for s in future_states],
        "future_J": [float(s.J) for s in future_states],
        "mapping": {
            "Lambda": "trm (mediana krocząca ceny, k=5)",
            "tau": "flow (tempo zmiany trm)",
            "rho": "resonance (zgodnosc 3 niezaleznych sprawdzen, 0..1)",
            "J": "volume (surowy wolumen)",
        },
    }
