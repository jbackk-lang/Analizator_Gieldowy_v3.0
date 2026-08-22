"""test_meta_dynamics_module.py - testy dla meta_dynamics_module.py
(integracja TIMDR-META-DYNAMICS <-> Analizator Gieldowy v3), na
syntetycznym OHLCV (ten sam wzorzec _make_ohlcv co w test_pipeline.py,
zeby nie zalezec od sieci/yfinance)."""
import numpy as np
import pandas as pd
import pytest

from pipeline import TimdrEngine
from meta_dynamics_module import build_meta_states_from_packet, analyze_ticker_meta


def _make_ohlcv(n=120, seed=7, trend=0.05):
    rng = np.random.default_rng(seed)
    close = 100 + trend * np.arange(n) + np.cumsum(rng.normal(0, 0.5, n))
    volume = 100000 + np.abs(rng.normal(0, 5000, n))
    return pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998,
        "close": close, "volume": volume,
    })


def test_build_meta_states_from_packet_dlugosc_i_mapowanie():
    ohlcv = _make_ohlcv()
    packet = TimdrEngine(ohlcv).compute_packet()
    volume = ohlcv["volume"].values

    states = build_meta_states_from_packet(packet, volume)

    assert len(states) == len(ohlcv)
    # Lambda = trm, tau = flow, rho = resonance (w [0,1]), J = volume
    assert states[0].Lambda == pytest.approx(packet.trm.values[0])
    assert states[0].tau == pytest.approx(packet.flow.values[0])
    assert states[0].rho == pytest.approx(packet.resonance.values[0])
    assert states[0].J == pytest.approx(volume[0])
    assert all(0.0 <= s.rho <= 1.0 for s in states)


def test_build_meta_states_niespojne_dlugosci_rzuca_czytelny_blad():
    ohlcv = _make_ohlcv()
    packet = TimdrEngine(ohlcv).compute_packet()
    zla_dlugosc_wolumenu = ohlcv["volume"].values[:-5]  # celowo za krotki
    with pytest.raises(ValueError, match="Niespojne dlugosci"):
        build_meta_states_from_packet(packet, zla_dlugosc_wolumenu)


def test_analyze_ticker_meta_pelny_pipeline():
    ohlcv = _make_ohlcv(n=150)
    result = analyze_ticker_meta(ohlcv)

    assert result["n_bars"] == 150
    assert len(result["phases"]) == 149  # M-seria ma o 1 mniej niz stany
    assert result["current_phase"] in ("stabilna", "przejsciowa", "krytyczna")
    assert len(result["phases_last_20"]) == 20
    assert len(result["future_lambda"]) >= 1
    assert "mapping" in result
    # wynik musi byc JSON-serializable (floaty/listy, nie numpy)
    import json
    json.dumps(result)


def test_analyze_ticker_meta_za_malo_barow_rzuca_blad():
    ohlcv = _make_ohlcv(n=1)
    with pytest.raises(ValueError, match="Za malo barow"):
        analyze_ticker_meta(ohlcv)
