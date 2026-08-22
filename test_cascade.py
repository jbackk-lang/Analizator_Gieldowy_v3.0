"""
test_cascade.py — testy dla cascade.py (kaskada przepływu kapitału
surowce → waluty → obligacje → indeksy → sektory → akcje)

Historia zmian tego modułu (dwie iteracje odrzucone/poprawione w trakcie
budowy, na żywo z użytkownikiem):

  v1 (odrzucona): łańcuch sektorów gospodarki (surowce→energetyka→
  przemysł→...→technologia), wagi ze STAŁEJ tabeli "siła wpływu" (★1-5).
  Użytkownik doprecyzował: chodzi o PRZEPŁYW KAPITAŁU między klasami
  aktywów, nie propagację kosztów przez sektory.

  v2 (bieżąca): łańcuch surowce→waluty→obligacje→indeksy→sektory→akcje.
  Pierwsza wersja WAG nadal używała stałej tabeli "★ ważności" - kolejna
  uwaga użytkownika: "mówimy o przepływie finansów nie szkolnej
  definicji ważności". Naprawiono: waga liczona jest TERAZ z DANYCH
  (`_flow_intensity` - iloczyn wielkości ruchu ceny i wolumenu
  względnego), nie z góry przypisanej opinii o ważności ogniwa.

  Bug znaleziony przy weryfikacji tej poprawki: dla bardzo dużych
  szoków testowych (shock_size=8 na serii o std szumu 0.1) znormalizowana,
  UCIĘTA waga (flow_weight, capped na 1.0) saturuje niezależnie od
  wolumenu - to zamierzone (nie trzeba "więcej niż pełnej wagi" dla
  ewidentnego zdarzenia), ale ZAMASKOWAŁO to realną wrażliwość na wolumen
  w pierwszym naiwnym teście. Poprawne testy sprawdzają SUROWĄ,
  nieuciętą `flow_intensity` (zawsze rośnie z wolumenem), nie tylko
  ucięty `flow_weight`/bonus.
"""

import numpy as np
import pytest

from cascade import (
    analyze_cascade, adjust_confidence, sector_for_ticker, sector_etf_for_ticker,
    ASSET_CHAIN, STAGE_TIMING, stage_index, _flow_intensity,
    currency_unit_for_ticker,
)


def _calm_series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(0, 0.1, n))


def _shocked_series(n=200, seed=0, shock_at=190, shock_size=8):
    x = _calm_series(n, seed)
    x[shock_at:] += shock_size
    return x


def _calm_volume(n=200, seed=0, base=100000.0):
    rng = np.random.default_rng(seed)
    return np.abs(base + rng.normal(0, 5000, n))


def _volume_with_spike(n=200, seed=0, base=100000.0, spike_at=190, spike_mult=4.0):
    v = _calm_volume(n, seed, base)
    v[spike_at - 1:spike_at + 3] *= spike_mult
    return np.abs(v)


# ---------------------------------------------------------------------
# _flow_intensity - miara przepływu kapitału (cena x wolumen względny)
# ---------------------------------------------------------------------

def test_flow_intensity_rosnie_z_wolumenem_wzglednym():
    """Kluczowa regresja - dokładnie ta właściwość, o którą poprosił
    użytkownik: ten sam ruch ceny na WYŻSZYM wolumenie względnym musi
    dawać WYŻSZĄ intensywność przepływu."""
    prices = _shocked_series(seed=1)
    fi_normal = _flow_intensity(prices, _calm_volume(seed=1), peak_local_idx=190)
    fi_spiked = _flow_intensity(prices, _volume_with_spike(seed=1, spike_mult=5.0), peak_local_idx=190)
    assert fi_spiked > fi_normal * 3


def test_flow_intensity_bez_wolumenu_spada_do_samej_zmiany_ceny():
    prices = _shocked_series(seed=1)
    fi_no_volume = _flow_intensity(prices, None, peak_local_idx=190)  # domyślne window=3 -> [187,193]
    expected = abs(prices[193] - prices[187]) / abs(prices[187])
    assert fi_no_volume == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------
# analyze_cascade / adjust_confidence
# ---------------------------------------------------------------------

def test_wykrywa_spojny_przeplyw_z_wolumenem_w_kilku_ogniwach():
    stage_data = {
        "surowce": {"close": _shocked_series(seed=1), "volume": _volume_with_spike(seed=1, spike_mult=4.0)},
        "waluty": {"close": _shocked_series(seed=2), "volume": _volume_with_spike(seed=2, spike_mult=3.0)},
        "obligacje": {"close": _calm_series(seed=3), "volume": _calm_volume(seed=3)},
        "indeksy": {"close": _calm_series(seed=4), "volume": _calm_volume(seed=4)},
        "sektory": {"close": _calm_series(seed=5), "volume": _calm_volume(seed=5)},
    }
    res = analyze_cascade("akcje", stage_data, lookback=15)
    assert res["upstream_pressure_score"] > 0
    assert res["consistent_direction"] == "wzrostowa"
    assert len(res["alerts"]) >= 2

    new_conf, bonus = adjust_confidence(6.0, res)
    assert bonus > 0
    assert new_conf > 6.0


def test_surowa_flow_intensity_wyzsza_z_potwierdzeniem_wolumenowym():
    """Regresja: nawet gdy UCIĘTA waga (flow_weight) saturuje przy
    bardzo dużych szokach testowych, SUROWA flow_intensity musi zawsze
    pokazywać wrażliwość na wolumen (to ona faktycznie mierzy przepływ,
    ucięcie na 1.0 to tylko limit górny dla samego bonusu)."""
    confirmed = {"surowce": {"close": _shocked_series(seed=1), "volume": _volume_with_spike(seed=1, spike_mult=4.0)}}
    unconfirmed = {"surowce": {"close": _shocked_series(seed=1), "volume": _calm_volume(seed=1)}}

    res_c = analyze_cascade("waluty", confirmed, lookback=15)
    res_u = analyze_cascade("waluty", unconfirmed, lookback=15)

    fi_c = res_c["stage_details"]["surowce"]["flow_intensity"]
    fi_u = res_u["stage_details"]["surowce"]["flow_intensity"]
    assert fi_c > fi_u * 2


def test_brak_bonusu_gdy_wszystko_spokojne():
    stage_data = {s: {"close": _calm_series(seed=10 + i), "volume": _calm_volume(seed=10 + i)}
                  for i, s in enumerate(ASSET_CHAIN[:-1])}
    res = analyze_cascade("akcje", stage_data, lookback=15)
    assert res["upstream_pressure_score"] == 0.0
    assert res["consistent_direction"] is None
    new_conf, bonus = adjust_confidence(6.0, res)
    assert bonus == 0.0
    assert new_conf == 6.0


def test_brak_bonusu_przy_sprzecznych_kierunkach():
    stage_data = {s: {"close": _calm_series(seed=30 + i), "volume": _calm_volume(seed=30 + i)}
                  for i, s in enumerate(ASSET_CHAIN[:-1])}
    stage_data["surowce"] = {"close": _shocked_series(seed=1, shock_size=8), "volume": _volume_with_spike(seed=1)}
    stage_data["obligacje"] = {"close": _shocked_series(seed=20, shock_size=-8), "volume": _volume_with_spike(seed=20)}
    res = analyze_cascade("akcje", stage_data, lookback=15)
    assert res["consistent_direction"] is None
    _, bonus = adjust_confidence(6.0, res)
    assert bonus == 0.0


def test_ufnosc_nigdy_nie_spada_ponizej_bazowej():
    stage_data = {s: {"close": _calm_series(seed=40 + i), "volume": _calm_volume(seed=40 + i)}
                  for i, s in enumerate(ASSET_CHAIN[:-1])}
    res = analyze_cascade("akcje", stage_data, lookback=15)
    for base in (0.0, 5.0, 50.0, 99.0):
        new_conf, _ = adjust_confidence(base, res)
        assert new_conf >= base


def test_ufnosc_nie_przekracza_100():
    stage_data = {s: {"close": _shocked_series(seed=1 + i), "volume": _volume_with_spike(seed=1 + i, spike_mult=5.0)}
                  for i, s in enumerate(ASSET_CHAIN[:-1])}
    res = analyze_cascade("akcje", stage_data, lookback=15)
    new_conf, _ = adjust_confidence(95.0, res)
    assert new_conf <= 100.0


def test_target_surowce_nie_ma_wczesniejszych_ogniw():
    stage_data = {"surowce": {"close": _shocked_series(seed=1), "volume": _volume_with_spike(seed=1)}}
    res = analyze_cascade("surowce", stage_data, lookback=15)
    assert res["upstream_pressure_score"] == 0.0
    assert res["stage_details"] == {}
    assert res["alerts"] == []


def test_brakujace_ogniwo_pomijane_bez_bledu():
    stage_data = {"surowce": {"close": _shocked_series(seed=1), "volume": None}}
    res = analyze_cascade("akcje", stage_data, lookback=15)
    assert "surowce" in res["stage_details"]
    assert "waluty" not in res["stage_details"]


def test_zbyt_krotkie_dane_pomijane():
    stage_data = {"surowce": {"close": np.array([100.0, 101.0, 102.0]), "volume": None}}
    res = analyze_cascade("akcje", stage_data, lookback=15)
    assert res["stage_details"] == {}


def test_nieznane_ogniwo_traktowane_jako_akcje():
    assert stage_index("cos_nieznanego") == stage_index("akcje")


def test_stage_timing_kompletny_dla_calego_lancucha():
    for stage in ASSET_CHAIN:
        assert stage in STAGE_TIMING, f"brak {stage} w STAGE_TIMING"


def test_sector_for_ticker_i_etf():
    assert sector_for_ticker("AAPL") == "technologia"
    assert sector_for_ticker("PKN.WA") == "energetyka"
    assert sector_etf_for_ticker("PKN.WA") == "XLE"
    assert sector_etf_for_ticker("AAPL") == "XLK"


def test_sector_for_ticker_domyslny_dla_nieznanego():
    assert sector_for_ticker("COS_ZUPELNIE_NIEZNANEGO") == "technologia"


def test_prosty_format_bez_wolumenu_dziala_tez_jako_gola_tablica():
    """stage_data może też być {ogniwo: np.ndarray} (bez volume) dla
    wygody - nie tylko {ogniwo: {"close":..., "volume":...}}."""
    stage_data = {"surowce": _shocked_series(seed=1)}
    res = analyze_cascade("waluty", stage_data, lookback=15)
    assert "surowce" in res["stage_details"]


# ---------------------------------------------------------------------
# currency_unit_for_ticker - regresja dla braku waluty/jednostki przy
# cenie w dashboardzie (zgłoszenie: "cena nie ma woluminu w jakiej
# walucie za jaki pakiet")
# ---------------------------------------------------------------------

def test_currency_forex_pary_kwotowana_w_drugiej_walucie():
    assert currency_unit_for_ticker("EURPLN=X") == {"currency": "PLN", "unit_label": "PLN"}
    assert currency_unit_for_ticker("USDJPY=X") == {"currency": "JPY", "unit_label": "JPY"}


def test_currency_gpw_akcje_pln():
    res = currency_unit_for_ticker("PKN.WA")
    assert res["currency"] == "PLN"


def test_currency_krypto_usd():
    res = currency_unit_for_ticker("BTC-USD")
    assert res["currency"] == "USD"


def test_currency_futures_ma_jednostke_fizyczna():
    res = currency_unit_for_ticker("CL=F")
    assert res["currency"] == "USD"
    assert "baryłka" in res["unit_label"]

    res_gold = currency_unit_for_ticker("GC=F")
    assert "uncja" in res_gold["unit_label"]


def test_currency_futures_nieznany_symbol_ma_generyczna_etykiete():
    res = currency_unit_for_ticker("ZZZ=F")
    assert res["currency"] == "USD"
    assert res["unit_label"]  # niepusty fallback, nie wywala się


def test_currency_indeks_punkty_bez_waluty():
    res = currency_unit_for_ticker("^GSPC")
    assert res["currency"] is None
    assert "pkt" in res["unit_label"]


def test_currency_domyslnie_usd_dla_zwyklych_akcji_us():
    res = currency_unit_for_ticker("AAPL")
    assert res["currency"] == "USD"


def test_currency_case_insensitive():
    assert currency_unit_for_ticker("aapl") == currency_unit_for_ticker("AAPL")
