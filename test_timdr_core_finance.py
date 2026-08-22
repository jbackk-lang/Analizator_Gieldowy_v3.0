"""
test_timdr_core_finance.py — testy dla timdr_core_finance.py

Dokumentuje kluczowy błąd znaleziony przy budowie v3:

  Bug 1 (defect): dosłowna implementacja przepisu ze skilla TIMDR
  ("próg = jump_factor * rozstęp POZIOMÓW parametru w oknie") sprawdza
  się dla wolno zmiennych odczytów pogodowych, ale dla cen giełdowych
  (naturalny random walk) dała próg drastycznie za czuły - zwykły szum
  dnia-do-dnia był flagowany jako "defekt" w ~20% próbek. Naprawiono
  zmianą odniesienia na rozstęp SAMYCH RÓŻNIC (nie poziomów) + podniesienie
  jump_factor z 0.3 do 3.0 - patrz test_defect_* niżej i docstring
  defect() w timdr_core_finance.py.
"""

import numpy as np
import pytest

from timdr_core_finance import trm, flow, twist, rhythm, anomalies, defect, resonance


@pytest.fixture()
def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------
# trm / flow (funkcje pomocnicze)
# ---------------------------------------------------------------------

def test_trm_odporniejszy_na_pojedynczy_spike_niz_srednia_krocząca(rng):
    n = 100
    price = 100 + np.cumsum(rng.normal(0, 0.3, n))
    price_spiked = price.copy()
    price_spiked[50] += 20
    trm_line = trm(price_spiked, k=5)
    ma_line = np.convolve(price_spiked, np.ones(5) / 5, mode="same")
    assert abs(trm_line[50] - price[50]) < abs(ma_line[50] - price[50])


def test_trm_pusty_sygnal():
    assert len(trm([], k=5)) == 0


def test_flow_odzyskuje_nachylenie_liniowego_trendu():
    lin = np.arange(100, dtype=float) * 2.0
    flow_lin = flow(trm(lin, k=5), window=5)
    assert np.allclose(flow_lin[10:], 2.0, atol=0.5)


# ---------------------------------------------------------------------
# twist (skręt)
# ---------------------------------------------------------------------

def test_twist_wykrywa_odwrocenie_trendu():
    up = np.arange(60, dtype=float)
    down = 59 - np.arange(60, dtype=float)
    reversal_price = np.concatenate([up, down])
    flow_r = flow(trm(reversal_price, k=3), window=5)
    idx = twist(flow_r, window=5)
    assert len(idx) > 0
    assert any(abs(i - 60) <= 10 for i in idx)


def test_twist_niski_falszywy_alarm_na_monotonicznym_trendzie(rng):
    mono = np.arange(150, dtype=float) + rng.normal(0, 0.1, 150)
    idx = twist(flow(trm(mono, k=3), window=5), window=5)
    assert len(idx) <= 2


def test_twist_za_krotki_sygnal():
    assert len(twist([1, 2, 3], window=5)) == 0


# ---------------------------------------------------------------------
# rhythm (okresowość wolumenu)
# ---------------------------------------------------------------------

def test_rhythm_wykrywa_prawdziwa_okresowosc(rng):
    t = np.arange(300)
    vol = 1000 + 300 * np.sin(2 * np.pi * t / 10) + rng.normal(0, 20, 300)
    periods, power = rhythm(vol, max_lag=30)
    assert power > 0.5
    assert 10 in periods or 9 in periods or 11 in periods


def test_rhythm_brak_periodycznosci_na_losowym_wolumenie(rng):
    periods, power = rhythm(rng.normal(1000, 50, 300), max_lag=30)
    assert power < 0.35


# ---------------------------------------------------------------------
# anomalies (anomalia)
# ---------------------------------------------------------------------

def test_anomalies_wykrywa_wstrzykniety_spike(rng):
    price = 100 + rng.normal(0, 1, 200)
    price[100] = 200
    assert 100 in anomalies(price)


def test_anomalies_pusty_sygnal():
    assert len(anomalies([])) == 0


# ---------------------------------------------------------------------
# defect (defekt) - patrz Bug 1 w docstringu modułu
# ---------------------------------------------------------------------

def test_defect_wykrywa_prawdziwy_skok(rng):
    price = 100 + np.cumsum(rng.normal(0, 0.1, 200))
    price[150:] += 15
    idx = defect(price)
    assert any(145 <= i <= 155 for i in idx)


def test_defect_rzadki_na_zwyklym_szumie_random_walk(rng):
    """Regresja Bug 1: na CZYSTYM random walk (bez żadnego prawdziwego
    skoku) defect() nie powinien flagować systematycznie dużej części
    próbek - to był dokładnie objaw błędu (dosłowna implementacja ze
    skilla dawała ~20% flagowanych próbek tutaj)."""
    price = 100 + np.cumsum(rng.normal(0, 0.2, 500))
    idx = defect(price)
    assert len(idx) < 25  # < 5%


def test_defect_podloga_zapobiega_falszywym_alarmom_na_plaskich_danych(rng):
    flat = np.full(100, 50.0) + rng.normal(0, 1e-6, 100)
    assert len(defect(flat)) < 10


def test_defect_za_krotki_sygnal():
    assert len(defect([1, 2])) == 0


# ---------------------------------------------------------------------
# resonance (rezonans)
# ---------------------------------------------------------------------

def test_resonance_wykrywa_wspolny_event(rng):
    price = 100 + np.cumsum(rng.normal(0, 0.2, 200))
    price[120] -= 10
    score, strong_idx = resonance(price)
    assert len(strong_idx) > 0
    assert score.min() >= 0.0 and score.max() <= 1.0


def test_resonance_za_krotki_sygnal():
    score, strong_idx = resonance([1, 2, 3])
    assert len(score) == 3
    assert len(strong_idx) == 0
