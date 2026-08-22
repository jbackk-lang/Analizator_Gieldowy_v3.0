"""
test_data_loader.py — testy dla data_loader.py, z zamockowanym
yf.download() (bez sieci - w tym sandboxie dostęp do Yahoo Finance jest
zablokowany przez proxy, tak jak potwierdzono ręcznie: 'curl: (7)
CONNECT tunnel failed, response 403'. Realny test na żywo musi być
wykonany na maszynie użytkownika - patrz README.md).

Dokumentuje lekcje ze Synoptyka zaadresowane w tym module:
  Lekcja #4 (schema mismatch = ciche wywalenie): każdy test poniżej
  sprawdza, że błędne dane/schemat/wyjątek sieciowy kończą się
  WIDOCZNYM DataLoaderError, nigdy cichym pustym wynikiem.
  Lekcja #3 (cache, nie pobieraj w kółko): cache dzienny weryfikowany
  bezpośrednio przez liczenie wywołań yf.download().
"""

import numpy as np
import pandas as pd
import pytest

import data_loader


def _fake_ohlcv_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": np.abs(rng.normal(100000, 5000, n)),
    }, index=idx)


@pytest.fixture()
def mock_yf(monkeypatch):
    """Podmienia yf.download na kontrolowaną funkcję - domyślnie zwraca
    poprawne dane; testy nadpisują `mock_yf.impl` żeby symulować błędy."""
    state = {"impl": lambda ticker, **kw: _fake_ohlcv_df()}

    def fake_download(ticker, period=None, interval=None, progress=None, auto_adjust=None):
        return state["impl"](ticker, period=period, interval=interval)

    monkeypatch.setattr(data_loader.yf, "download", fake_download)
    return state


# ---------------------------------------------------------------------
# fetch_ohlcv
# ---------------------------------------------------------------------

def test_fetch_ohlcv_poprawne_dane(mock_yf):
    df = data_loader.fetch_ohlcv("TEST")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 50


def test_fetch_ohlcv_puste_dane_rzuca_czytelny_blad(mock_yf):
    mock_yf["impl"] = lambda ticker, **kw: pd.DataFrame()
    with pytest.raises(data_loader.DataLoaderError, match="puste dane"):
        data_loader.fetch_ohlcv("EMPTY")


def test_fetch_ohlcv_zly_schemat_rzuca_czytelny_blad(mock_yf):
    """Regresja lekcji #4 - brak wymaganych kolumn NIE MOŻE kończyć się
    cichym pustym wynikiem, tylko jawnym błędem z listą brakujących kolumn."""
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    mock_yf["impl"] = lambda ticker, **kw: pd.DataFrame({"price": np.arange(20.0)}, index=idx)
    with pytest.raises(data_loader.DataLoaderError, match="nieoczekiwany schemat"):
        data_loader.fetch_ohlcv("BADSCHEMA")


def test_fetch_ohlcv_wyjatek_sieciowy_opakowany_widocznie(mock_yf):
    def raise_conn_error(ticker, **kw):
        raise ConnectionError("network down")
    mock_yf["impl"] = raise_conn_error
    with pytest.raises(data_loader.DataLoaderError, match="Błąd pobierania"):
        data_loader.fetch_ohlcv("NETERR")


def test_fetch_ohlcv_za_malo_barow(mock_yf):
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    mock_yf["impl"] = lambda ticker, **kw: pd.DataFrame({
        "Open": [1, 2, 3], "High": [1, 2, 3], "Low": [1, 2, 3],
        "Close": [1, 2, 3], "Volume": [100, 100, 100],
    }, index=idx)
    with pytest.raises(data_loader.DataLoaderError, match="Za mało"):
        data_loader.fetch_ohlcv("SHORT")


def test_fetch_ohlcv_multiindex_kolumny_splaszczone(mock_yf):
    """yfinance czasem zwraca MultiIndex kolumn nawet dla pojedynczego
    tickera - musi zostać poprawnie spłaszczony, nie potraktowany jako
    zły schemat."""
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["TEST"]])
    df = pd.DataFrame(np.random.rand(20, 5) * 100, index=idx, columns=cols)
    mock_yf["impl"] = lambda ticker, **kw: df
    result = data_loader.fetch_ohlcv("TEST")
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------
# CachedReferenceLoader
# ---------------------------------------------------------------------

def test_cache_unika_powtornego_pobrania_tego_samego_dnia(mock_yf, tmp_path):
    call_count = {"n": 0}

    def counting_impl(ticker, **kw):
        call_count["n"] += 1
        return _fake_ohlcv_df()
    mock_yf["impl"] = counting_impl

    cache = data_loader.CachedReferenceLoader(str(tmp_path / "cache"))
    cache.get_close_array("CL=F")
    cache.get_close_array("CL=F")
    assert call_count["n"] == 1


def test_cache_przetrwa_restart_procesu(mock_yf, tmp_path):
    call_count = {"n": 0}

    def counting_impl(ticker, **kw):
        call_count["n"] += 1
        return _fake_ohlcv_df()
    mock_yf["impl"] = counting_impl

    path = str(tmp_path / "cache")
    data_loader.CachedReferenceLoader(path).get_close_array("CL=F")
    data_loader.CachedReferenceLoader(path).get_close_array("CL=F")  # nowa instancja = symulacja restartu
    assert call_count["n"] == 1


def test_cache_uszkodzony_plik_nie_wywala_wyjatku(mock_yf, tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "CL_F.json").write_text("{zly json", encoding="utf-8")
    cache = data_loader.CachedReferenceLoader(str(cache_dir))
    close, volume = cache.get_close_volume("CL=F")  # powinien odzyskać się pobierając na nowo
    assert len(close) > 0
    captured = capsys.readouterr()
    assert "UWAGA" in captured.out


# ---------------------------------------------------------------------
# fetch_cascade_stage_data
# ---------------------------------------------------------------------

def test_fetch_cascade_stage_data_zbiera_wszystkie_ogniwa(mock_yf, tmp_path):
    cache = data_loader.CachedReferenceLoader(str(tmp_path / "cache"))
    result = data_loader.fetch_cascade_stage_data("AAPL", cache)
    assert set(result["stage_data"].keys()) == {"surowce", "waluty", "obligacje", "indeksy", "sektory"}
    for stage, entry in result["stage_data"].items():
        assert "close" in entry and "volume" in entry
        assert len(entry["close"]) > 0


def test_fetch_cascade_stage_data_sektor_zalezy_od_tickera(mock_yf, tmp_path):
    """Ogniwo 'sektory' musi pobrać RÓŻNY ETF w zależności od tickera -
    inaczej byłoby bez znaczenia w analizie kaskadowej."""
    from cascade import sector_etf_for_ticker

    requested_tickers = []

    def tracking_impl(ticker, **kw):
        requested_tickers.append(ticker)
        return _fake_ohlcv_df()
    mock_yf["impl"] = tracking_impl

    cache = data_loader.CachedReferenceLoader(str(tmp_path / "cache"))
    data_loader.fetch_cascade_stage_data("AAPL", cache)
    assert sector_etf_for_ticker("AAPL") in requested_tickers

    requested_tickers.clear()
    data_loader.fetch_cascade_stage_data("PKN.WA", cache)
    assert sector_etf_for_ticker("PKN.WA") in requested_tickers
    assert sector_etf_for_ticker("PKN.WA") != sector_etf_for_ticker("AAPL")


def test_fetch_cascade_stage_data_blad_pojedynczego_instrumentu_nie_przerywa_calosci(mock_yf, tmp_path):
    def selective_fail(ticker, **kw):
        if ticker == "CL=F":
            raise ConnectionError("symulowany błąd tylko dla CL=F")
        return _fake_ohlcv_df()
    mock_yf["impl"] = selective_fail

    cache = data_loader.CachedReferenceLoader(str(tmp_path / "cache"))
    result = data_loader.fetch_cascade_stage_data("AAPL", cache)
    assert "surowce" in result["errors"]
    assert any("CL=F" in msg for msg in result["errors"]["surowce"])
    # pozostałe ogniwa (surowce ma jeszcze 3 inne tickery + inne ogniwa) nadal obecne
    assert "waluty" in result["stage_data"]
