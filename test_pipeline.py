"""
test_pipeline.py — testy end-to-end dla pipeline.py (TimdrEngine +
AnalizatorGieldowy przez run_pipeline), na syntetycznym OHLCV.

Dokumentuje Bug 1 (brak surowej ceny w pakiecie użytkownika):
oryginalny moduł-łącznik dostarczony przez użytkownika przekazywał do
analizatora tylko `trm` (cenę WYGŁADZONĄ medianą kroczącą k=5) - RSI i
backtest liczone na tej linii byłyby systematycznie inne niż na
standardowych platformach (gdzie RSI zawsze liczy się z surowej ceny
zamknięcia) i zaniżałyby realną zmienność zwrotów w backteście.
Naprawiono dodaniem `PriceSignal`/`packet.price` (surowa cena) - patrz
docstring w pipeline.py i analizator_gieldowy.py.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline import run_pipeline, TimdrEngine


def _make_ohlcv(n=300, seed=42, trend=0.0, vol_period=5):
    rng = np.random.default_rng(seed)
    close = 100 + trend * np.arange(n) + np.cumsum(rng.normal(0, 0.5, n))
    volume = 100000 + 20000 * np.sin(2 * np.pi * np.arange(n) / vol_period) + rng.normal(0, 3000, n)
    return pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998,
        "close": close, "volume": np.abs(volume),
    })


def test_run_pipeline_zwraca_oczekiwany_ksztalt_wyniku():
    ohlcv = _make_ohlcv()
    wynik = run_pipeline(ohlcv)
    for key in ("emergencja_label", "ufnosc_procent", "sharpe_n", "winrate_n",
                "dd_n", "rsi", "rsi_interpretation", "n_bars"):
        assert key in wynik
    assert wynik["n_bars"] == 300
    assert wynik["emergencja_label"] in ("EMERGENCJA", "szum (brak emergencji)")
    assert 0 <= wynik["rsi"] <= 100


def test_packet_ma_surowa_cene_nie_tylko_wygladzona():
    """Regresja Bug 1: packet.price MUSI istnieć i być RÓŻNE od packet.trm
    (poza trywialnym przypadkiem stałej ceny) - inaczej wracamy do
    błędu, gdzie RSI/backtest liczą się na wygładzonej linii."""
    ohlcv = _make_ohlcv()
    engine = TimdrEngine(ohlcv)
    packet = engine.compute_packet()
    assert hasattr(packet, "price")
    assert not np.allclose(packet.price.values, packet.trm.values)
    assert np.allclose(packet.price.values, ohlcv["close"].values)


def test_silny_trend_wzrostowy_daje_wyzszy_rsi_niz_trend_spadkowy():
    up = _make_ohlcv(trend=0.3, seed=1)
    down = _make_ohlcv(trend=-0.3, seed=1)
    r_up = run_pipeline(up)
    r_down = run_pipeline(down)
    assert r_up["rsi"] > r_down["rsi"]


def test_pipeline_dziala_na_minimalnej_liczbie_barow():
    ohlcv = _make_ohlcv(n=15)
    wynik = run_pipeline(ohlcv)
    assert wynik["n_bars"] == 15
