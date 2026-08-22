"""
api.py — TIMDR → Analizator Giełdowy v3, lokalne REST API + dashboard
================================================================================
Pojedynczy proces Flask (ten sam wzorzec co inne dashboardy TIMDR w tym
zestawie repo) - serwuje `/` (dashboard) oraz endpointy JSON.

`full_analysis(ticker)` łączy WSZYSTKIE warstwy zbudowane w v3:
  1. pipeline.run_pipeline()      - silnik TIMDR (trm/flow/twist/rhythm/
                                     anomalie/defekt/rezonans) + werdykt
                                     bazowy (Emergencja/Ufność/RSI/backtest)
  2. cascade.py                   - kaskada przepływu kapitału (surowce→
                                     waluty→obligacje→indeksy→sektory)
                                     PODNOSI Ufność, jeśli wcześniejsze
                                     ogniwa łańcucha pokazują spójny,
                                     potwierdzony wolumenem sygnał
  3. state.py (StateStore)        - wykrywa zmianę werdyktu względem
                                     poprzedniego uruchomienia (trwałe
                                     na dysku)
  4. state.py (PredictionLog)     - samo-uczenie: loguje predykcję,
                                     potwierdza poprzednie gdy nadejdzie
                                     czas, liczy bias/trafność z plakietką

Endpointy:
  GET  /                          -> dashboard
  GET  /api/health
  GET  /api/analyze?ticker=...&period=1y  -> pełna analiza
  POST /api/state/clear           -> {ticker} - czyści stan + log predykcji dla tickera
"""

from __future__ import annotations

import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from data_loader import DataLoaderError, fetch_ohlcv, CachedReferenceLoader, fetch_cascade_stage_data
from pipeline import run_pipeline
from analizator_gieldowy import EMERGENCE_CONFIDENCE_THRESHOLD
from cascade import analyze_cascade, adjust_confidence, sector_for_ticker, sector_etf_for_ticker, currency_unit_for_ticker
from state import StateStore, PredictionLog

# DODANE: integracja z TIMDR-META-DYNAMICS (meta-warstwa nad polem
# Λ-τ-ρ-J, patrz meta_dynamics_module.py). Import odizolowany w
# try/except - jesli folder-siostra TIMDR-META-DYNAMICS nie istnieje na
# tej maszynie (np. sklonowano tylko to jedno repo), reszta API (dashboard,
# /api/analyze, kaskada, samouczenie) dziala normalnie, tylko /api/meta
# zwraca czytelny 501 zamiast wywalac caly proces na starcie.
try:
    from meta_dynamics_module import analyze_ticker_meta
    _META_DYNAMICS_AVAILABLE = True
    _META_DYNAMICS_IMPORT_ERROR = None
except ImportError as _e:
    _META_DYNAMICS_AVAILABLE = False
    _META_DYNAMICS_IMPORT_ERROR = str(_e)

app = Flask(__name__, static_folder="static", static_url_path="")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
state_store = StateStore(os.path.join(DATA_DIR, "state"))
prediction_log = PredictionLog(os.path.join(DATA_DIR, "predictions"))
reference_cache = CachedReferenceLoader(os.path.join(DATA_DIR, "cache"))

DISCLAIMER = (
    "Narzędzie badawczo-edukacyjne. NIE jest doradztwem inwestycyjnym ani "
    "rekomendacją. Wszystkie metryki (Emergencja, Ufność, sharpe_n, "
    "winrate_n, dd_n, RSI) to statystyczne odchylenia względem lokalnej "
    "historii sygnału, nie porady finansowe. Decyzje inwestycyjne "
    "podejmuj na własną odpowiedzialność, najlepiej po konsultacji z "
    "licencjonowanym doradcą."
)

LEAD_TIME_BARS = 5  # ile barów naprzód "przewiduje" logowana predykcja (samo-uczenie)


def _clean(obj):
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _clean(float(obj))
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def full_analysis(ticker: str, period: str = "1y", include_cascade: bool = True) -> dict:
    ohlcv = fetch_ohlcv(ticker, period=period)
    close = ohlcv["close"].values

    base = run_pipeline(ohlcv)

    # --- samo-uczenie: najpierw potwierdź WCZEŚNIEJ zalogowane predykcje
    # (mogły dojrzeć od ostatniego uruchomienia), DOPIERO POTEM zaloguj
    # nową - w tej kolejności, żeby nowa predykcja nie próbowała
    # "potwierdzić się sama sobą" na tych samych danych.
    n_confirmed = prediction_log.confirm_due_predictions(ticker, close)
    bias_info = prediction_log.bias_by_lead_time(ticker, min_samples=5)

    flow_last = base.get("flow_last")
    direction = "up" if (flow_last or 0) > 0 else ("down" if (flow_last or 0) < 0 else "flat")
    prediction_log.log_prediction(
        ticker, bar_index=len(close) - 1, bar_timestamp=str(ohlcv.index[-1]),
        direction=direction, confidence=base.get("ufnosc_procent", 0.0),
        lead_time_bars=LEAD_TIME_BARS,
    )

    result = dict(base)
    result["ticker"] = ticker
    result["sector"] = sector_for_ticker(ticker)
    result["sector_etf"] = sector_etf_for_ticker(ticker)

    # Wolumen (ostatnia bara + średnia z całego okresu) - był w
    # OHLCV od początku, ale nigdy nie trafiał do odpowiedzi API/
    # dashboardu. Waluta/jednostka: patrz currency_unit_for_ticker -
    # heurystyka z sufiksu tickera, jawnie oznaczona jako przybliżenie.
    volume_col = ohlcv["volume"].values
    result["last_volume"] = float(volume_col[-1]) if len(volume_col) else None
    result["avg_volume"] = float(np.mean(volume_col)) if len(volume_col) else None
    currency_info = currency_unit_for_ticker(ticker)
    result["currency"] = currency_info["currency"]
    result["price_unit_label"] = currency_info["unit_label"]
    result["learning"] = {
        "n_newly_confirmed": n_confirmed,
        "bias_by_lead_time": bias_info,
    }

    # --- kaskada przepływu kapitału (opcjonalna warstwa - błąd pobrania
    # NIE przerywa analizy podstawowej, tylko pomija wzbogacenie)
    cascade_info = None
    if include_cascade:
        try:
            fetch_result = fetch_cascade_stage_data(ticker, reference_cache)
            cascade_info = analyze_cascade("akcje", fetch_result["stage_data"])
            cascade_info["fetch_errors"] = fetch_result["errors"]
            new_ufnosc, bonus = adjust_confidence(result["ufnosc_procent"], cascade_info)
            result["ufnosc_procent_bazowa"] = result["ufnosc_procent"]
            result["ufnosc_procent"] = new_ufnosc
            result["cascade_bonus"] = bonus
            if bonus > 0:
                result["emergencja_label"] = (
                    "EMERGENCJA" if new_ufnosc >= EMERGENCE_CONFIDENCE_THRESHOLD else "szum (brak emergencji)"
                )
        except Exception as e:
            cascade_info = {"error": str(e)}
    result["cascade"] = cascade_info

    # --- wykrywanie zmiany stanu względem poprzedniego uruchomienia
    # (trwałe na dysku - przetrwa restart procesu, patrz state.py)
    state_change = state_store.compare_and_update(ticker, {
        "emergencja_label": result["emergencja_label"],
        "ufnosc_procent": result["ufnosc_procent"],
    })
    result["state_change"] = state_change

    result["disclaimer"] = DISCLAIMER
    return result


@app.route("/")
def dashboard():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "disclaimer": DISCLAIMER})


@app.route("/api/analyze")
def analyze():
    ticker = request.args.get("ticker", "EURPLN=X")
    period = request.args.get("period", "1y")
    include_cascade = request.args.get("cascade", "1") not in ("0", "false", "False")

    try:
        result = full_analysis(ticker, period=period, include_cascade=include_cascade)
    except DataLoaderError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(_clean(result))


@app.route("/api/meta")
def meta():
    """Meta-warstwa TIMDR-META-DYNAMICS nad polem Λ-τ-ρ-J tego tickera.

    Mapowanie sygnalow -> pole (patrz meta_dynamics_module.py):
    Λ=trm, τ=flow, ρ=resonance, J=volume. To DODATKOWA, eksperymentalna
    warstwa nad tym, co juz robi /api/analyze - nie zastepuje jej i nie
    wplywa na Emergencja/Ufnosc/RSI/backtest z tamtego endpointu.
    """
    if not _META_DYNAMICS_AVAILABLE:
        return jsonify({
            "error": (
                "meta_dynamics_module niedostepny - folder-siostra "
                "TIMDR-META-DYNAMICS nie zostal znaleziony obok tego repo."
            ),
            "detail": _META_DYNAMICS_IMPORT_ERROR,
        }), 501

    ticker = request.args.get("ticker", "EURPLN=X")
    period = request.args.get("period", "1y")

    try:
        ohlcv = fetch_ohlcv(ticker, period=period)
        result = analyze_ticker_meta(ohlcv)
    except DataLoaderError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result["ticker"] = ticker
    result["disclaimer"] = DISCLAIMER
    return jsonify(_clean(result))


@app.route("/api/state/clear", methods=["POST"])
def clear_state():
    body = request.get_json(force=True, silent=True) or {}
    ticker = body.get("ticker")
    if not ticker:
        return jsonify({"error": "wymagane pole 'ticker'"}), 400
    cleared_state = state_store.clear(ticker)
    cleared_log = prediction_log.clear(ticker)
    return jsonify({"ticker": ticker, "state_cleared": cleared_state, "prediction_log_cleared": cleared_log})


if __name__ == "__main__":
    # UWAGA: port 5060 (SIP) jest na liście "zakazanych portów" przeglądarek
    # i Node/undici fetch() (ERR_UNSAFE_PORT) - dashboard.html odpalony w
    # realnej przeglądarce NIE mógłby się połączyć z API na tym porcie.
    # Znaleziono podczas weryfikacji end-to-end tego dashboardu - patrz README.
    app.run(host="127.0.0.1", port=8060, debug=False, threaded=True)
