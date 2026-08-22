"""
test_api.py — testy dla api.py (full_analysis + endpointy Flask).

Ten plik NIE istniał wcześniej - api.py był dotąd testowany tylko
ręcznie (Flask test_client + live-server + weryfikacja dashboardu przez
Node.js podczas budowy), nigdy jako trwała regresja w pytest. Dodany
przy okazji naprawy zgłoszenia "cena nie ma woluminu w jakiej walucie
za jaki pakiet", żeby last_volume/avg_volume/currency/price_unit_label
miały trwałe pokrycie testowe, nie tylko jednorazową ręczną weryfikację.

Sieć jest zamockowana (yf.download) - ten sam wzorzec co
test_data_loader.py (w tym sandboxie dostęp do Yahoo Finance jest
zablokowany przez proxy).
"""

import numpy as np
import pandas as pd
import pytest


def _fake_ohlcv_df(n=60, seed=0, base=100.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = base + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": np.abs(rng.normal(500000, 20000, n)),
    }, index=idx)


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    """Importuje api.py ze stanem (data/) przekierowanym do tmp_path,
    żeby testy nigdy nie dotykały realnych, zakumulowanych plików stanu
    w data/ tego repo (patrz lekcja ze Synoptyka: nie zanieczyszczaj
    prawdziwych danych użytkownika testami)."""
    import importlib
    import data_loader as dl

    def fake_download(ticker, period=None, interval=None, progress=None, auto_adjust=None):
        seed = abs(hash(ticker)) % (2**31)
        return _fake_ohlcv_df(seed=seed)

    monkeypatch.setattr(dl.yf, "download", fake_download)

    import api
    importlib.reload(api)  # świeży moduł, ale yf.download już zamockowany na poziomie data_loader
    api.state_store.base_dir = str(tmp_path / "state")
    api.prediction_log.base_dir = str(tmp_path / "predictions")
    api.reference_cache.cache_dir = str(tmp_path / "cache")
    import os
    os.makedirs(api.state_store.base_dir, exist_ok=True)
    os.makedirs(api.prediction_log.base_dir, exist_ok=True)
    os.makedirs(api.reference_cache.cache_dir, exist_ok=True)

    api.app.config["TESTING"] = True
    with api.app.test_client() as client:
        yield client


def test_health(app_client):
    res = app_client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_analyze_zwraca_cene_wolumen_i_walute(app_client):
    """Regresja zgłoszenia: 'cena nie ma woluminu w jakiej walucie za
    jaki pakiet' - last_price/last_volume/currency/price_unit_label
    muszą być obecne i sensowne dla zwykłej akcji US."""
    res = app_client.get("/api/analyze?ticker=AAPL&period=1y&cascade=0")
    assert res.status_code == 200
    data = res.get_json()

    assert data["last_price"] is not None
    assert data["last_volume"] is not None and data["last_volume"] > 0
    assert data["avg_volume"] is not None and data["avg_volume"] > 0
    assert data["currency"] == "USD"
    assert data["price_unit_label"] == "USD"
    assert data["n_bars"] > 0
    assert isinstance(data["trend"], list)
    assert len(data["trend"]) == len(data["x"])


def test_analyze_futures_ma_jednostke_fizyczna(app_client):
    res = app_client.get("/api/analyze?ticker=CL=F&period=1y&cascade=0")
    assert res.status_code == 200
    data = res.get_json()
    assert data["currency"] == "USD"
    assert "baryłka" in data["price_unit_label"]


def test_analyze_forex_waluta_kwotowana(app_client):
    res = app_client.get("/api/analyze?ticker=EURPLN=X&period=1y&cascade=0")
    assert res.status_code == 200
    data = res.get_json()
    assert data["currency"] == "PLN"


def test_meta_endpoint_zwraca_faze_i_mapowanie(app_client):
    """DODANE: /api/meta - integracja z TIMDR-META-DYNAMICS (patrz
    meta_dynamics_module.py). Testuje, ze endpoint dziala end-to-end na
    zamockowanym OHLCV i zwraca oczekiwany ksztalt."""
    res = app_client.get("/api/meta?ticker=AAPL&period=1y")
    assert res.status_code == 200
    data = res.get_json()

    assert data["ticker"] == "AAPL"
    assert data["current_phase"] in ("stabilna", "przejsciowa", "krytyczna")
    assert isinstance(data["phases"], list)
    assert len(data["future_lambda"]) >= 1
    assert data["mapping"]["J"] == "volume (surowy wolumen)"
    assert "disclaimer" in data


def test_meta_endpoint_zwraca_501_gdy_modul_niedostepny(app_client, monkeypatch):
    """Gdy folder-siostra TIMDR-META-DYNAMICS nie jest dostepny (inna
    maszyna, sklonowano tylko to repo), /api/meta ma zwracac czytelny
    501, NIE wywalac calego procesu przy imporcie api.py."""
    import api
    monkeypatch.setattr(api, "_META_DYNAMICS_AVAILABLE", False)
    monkeypatch.setattr(api, "_META_DYNAMICS_IMPORT_ERROR", "symulowany brak folderu")

    res = app_client.get("/api/meta?ticker=AAPL")
    assert res.status_code == 501
    data = res.get_json()
    assert "error" in data


def test_analyze_nieznany_ticker_zwraca_czytelny_blad(app_client, monkeypatch):
    import data_loader as dl

    def raise_error(ticker, **kw):
        raise ConnectionError("brak takiego tickera")
    monkeypatch.setattr(dl.yf, "download", raise_error)

    res = app_client.get("/api/analyze?ticker=NIEISTNIEJACY&cascade=0")
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_state_clear_wymaga_tickera(app_client):
    res = app_client.post("/api/state/clear", json={})
    assert res.status_code == 400
