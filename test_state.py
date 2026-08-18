"""
test_state.py — testy dla state.py (StateStore + PredictionLog)

Wszystkie testy używają IZOLOWANEGO katalogu tymczasowego (pytest
`tmp_path`), żeby nie dotykać ewentualnego realnego, zgromadzonego
stanu użytkownika (lekcja #8 ze Synoptyka: testy nie mogą zanieczyszczać
prawdziwych danych - tu zapewnione strukturalnie, przez pełną izolację
katalogu, a nie przez ręczny backup/restore).
"""

import os

import pytest

from state import StateStore, PredictionLog


# ---------------------------------------------------------------------
# StateStore
# ---------------------------------------------------------------------

def test_pierwszy_zapis_nie_ma_poprzedniego(tmp_path):
    ss = StateStore(str(tmp_path / "state"))
    r = ss.compare_and_update("TEST", {"emergencja_label": "szum", "ufnosc_procent": 6.0})
    assert r["previous"] is None
    assert r["changed"] is False


def test_wykrywa_zmiane_etykiety_i_skok_ufnosci(tmp_path):
    ss = StateStore(str(tmp_path / "state"))
    ss.compare_and_update("TEST", {"emergencja_label": "szum", "ufnosc_procent": 6.0})
    r = ss.compare_and_update("TEST", {"emergencja_label": "EMERGENCJA", "ufnosc_procent": 40.0})
    assert r["changed"] is True
    assert r["confidence_jump"] is True


def test_brak_zmiany_gdy_etykieta_i_ufnosc_stabilne(tmp_path):
    ss = StateStore(str(tmp_path / "state"))
    ss.compare_and_update("TEST", {"emergencja_label": "EMERGENCJA", "ufnosc_procent": 40.0})
    r = ss.compare_and_update("TEST", {"emergencja_label": "EMERGENCJA", "ufnosc_procent": 42.0})
    assert r["changed"] is False
    assert r["confidence_jump"] is False


def test_stan_przetrwa_restart_procesu(tmp_path):
    """Regresja kluczowa - lekcja #5 ze Synoptyka: stan MUSI być na
    dysku. Symulujemy restart procesu tworząc NOWĄ instancję StateStore
    wskazującą na ten sam katalog."""
    path = str(tmp_path / "state")
    StateStore(path).compare_and_update("XYZ", {"emergencja_label": "szum", "ufnosc_procent": 5.0})
    r = StateStore(path).compare_and_update("XYZ", {"emergencja_label": "EMERGENCJA", "ufnosc_procent": 50.0})
    assert r["changed"] is True


def test_uszkodzony_plik_stanu_nie_wywala_wyjatku_tylko_ostrzega(tmp_path, capsys):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "BAD.json").write_text("{niepoprawny json", encoding="utf-8")
    ss = StateStore(str(state_dir))
    result = ss.load_last("BAD")
    assert result is None
    captured = capsys.readouterr()
    assert "UWAGA" in captured.out  # błąd jest WIDOCZNY, nie cichy (lekcja #4)


def test_clear_usuwa_stan(tmp_path):
    ss = StateStore(str(tmp_path / "state"))
    ss.compare_and_update("TEST", {"emergencja_label": "szum", "ufnosc_procent": 6.0})
    assert ss.clear("TEST") is True
    assert ss.load_last("TEST") is None
    assert ss.clear("TEST") is False  # drugie czyszczenie - nic do usunięcia


def test_sanityzacja_nazwy_tickera_ze_znakiem_specjalnym(tmp_path):
    """EURPLN=X (typowy ticker walutowy z yfinance) musi dać się
    zapisać/odczytać bez błędu systemu plików."""
    ss = StateStore(str(tmp_path / "state"))
    ss.compare_and_update("EURPLN=X", {"emergencja_label": "szum", "ufnosc_procent": 1.0})
    assert ss.load_last("EURPLN=X") is not None


# ---------------------------------------------------------------------
# PredictionLog
# ---------------------------------------------------------------------

def test_confirm_due_predictions_dopasowuje_do_rzeczywistej_ceny(tmp_path):
    pl = PredictionLog(str(tmp_path / "predlog"))
    prices = [100.0, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    pl.log_prediction("TEST", bar_index=0, bar_timestamp="t0", direction="up", confidence=0.8, lead_time_bars=5)
    pl.log_prediction("TEST", bar_index=1, bar_timestamp="t1", direction="down", confidence=0.6, lead_time_bars=3)

    n = pl.confirm_due_predictions("TEST", prices)
    assert n == 2

    bias = pl.bias_by_lead_time("TEST", min_samples=5)
    assert bias[5]["accuracy"] == 1.0   # przewidziano "up", cena faktycznie rosła
    assert bias[3]["accuracy"] == 0.0   # przewidziano "down", cena faktycznie rosła


def test_korekta_wylaczona_ponizej_min_samples(tmp_path):
    pl = PredictionLog(str(tmp_path / "predlog"))
    prices = [100.0, 105.0]
    pl.log_prediction("TEST", bar_index=0, bar_timestamp="t0", direction="up", confidence=0.8, lead_time_bars=1)
    pl.confirm_due_predictions("TEST", prices)
    bias = pl.bias_by_lead_time("TEST", min_samples=5)
    assert bias[1]["apply_correction"] is False
    assert bias[1]["badge"] == "🔴"


def test_korekta_wlaczona_powyzej_min_samples(tmp_path):
    pl = PredictionLog(str(tmp_path / "predlog"))
    prices = list(range(100, 130))  # rosnący ciąg
    for i in range(10):
        pl.log_prediction("TEST", bar_index=i, bar_timestamp=f"t{i}", direction="up", confidence=0.7, lead_time_bars=2)
    pl.confirm_due_predictions("TEST", prices)
    bias = pl.bias_by_lead_time("TEST", min_samples=5)
    assert bias[2]["n"] == 10
    assert bias[2]["apply_correction"] is True
    assert bias[2]["badge"] in ("🟠", "🟢")


def test_pending_predykcja_bez_wystarczajacych_danych_nie_jest_potwierdzana(tmp_path):
    pl = PredictionLog(str(tmp_path / "predlog"))
    pl.log_prediction("TEST", bar_index=0, bar_timestamp="t0", direction="up", confidence=0.8, lead_time_bars=10)
    n = pl.confirm_due_predictions("TEST", [100.0, 101.0, 102.0])  # tylko 3 bary, target=10
    assert n == 0


def test_clear_usuwa_log_predykcji(tmp_path):
    pl = PredictionLog(str(tmp_path / "predlog"))
    pl.log_prediction("TEST", bar_index=0, bar_timestamp="t0", direction="up", confidence=0.8, lead_time_bars=1)
    assert pl.clear("TEST") is True
    assert pl.clear("TEST") is False
