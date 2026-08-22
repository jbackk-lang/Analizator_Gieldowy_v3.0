"""
data_loader.py — pobieranie OHLCV (yfinance) dla tickera głównego i dla
instrumentów referencyjnych kaskady sektorowej, z jawną walidacją
schematu i cache'em dziennym.
================================================================================
Lekcje ze Synoptyka zastosowane tutaj (patrz README.md):

  Lekcja #4 (schema mismatch = ciche wywalenie): każda funkcja tutaj
  WALIDUJE jawnie obecność kolumn open/high/low/close/volume i rzuca
  czytelny wyjątek, jeśli ich brakuje - ŻADNEGO `except Exception: pass`.
  Błędy pobierania są zawsze widoczne (zwracane jako komunikat, nie
  cichy pusty wynik).

  Lekcja #3 (cache, nie przeliczaj/nie pobieraj w kółko): kaskada
  sektorowa potrzebuje do 8 dodatkowych pobrań (referencyjne instrumenty
  per ogniwo) OPRÓCZ tickera głównego - bez cache'u każda pojedyncza
  analiza biłaby w yfinance 9 razy. Cache dzienny (jeden plik na ticker
  w `data/cache/`, ważny do końca dnia) - realny handel giełdowy i tak
  aktualizuje się raz dziennie po zamknięciu sesji dla danych EOD, więc
  częstsze odświeżanie referencyjnych instrumentów kaskady nie daje
  dodatkowej wartości.

Zna też Bug 5/Bug 6 z analizator-gieldowy (v1) tego zestawu repo:
  Bug 5: `yf.download(..., show_errors=False)` - parametr USUNIĘTY w
  nowszych yfinance, powodował TypeError. Tutaj w ogóle nie używamy
  tego parametru.
  Bug 6: yfinance bez `curl_cffi` dostaje puste dane od Yahoo (ochrona
  antybotowa) - stąd run.bat wymusza `pip install --upgrade yfinance`
  przy każdym uruchomieniu (nie tylko install-if-missing).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


class DataLoaderError(Exception):
    """Jawny, czytelny błąd pobierania/walidacji danych - NIGDY nie
    połykany bare exceptem gdziekolwiek w tym repo."""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    return df


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Pobiera OHLCV z yfinance i WALIDUJE schemat przed zwróceniem.
    Rzuca DataLoaderError z czytelnym komunikatem (nie zwraca cicho
    pustego/błędnego DataFrame) w każdym przypadku niepowodzenia.
    """
    try:
        raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    except Exception as e:
        raise DataLoaderError(
            f"Błąd pobierania danych dla '{ticker}': {e}. "
            f"Sprawdź połączenie z internetem i czy masz aktualną wersję yfinance "
            f"(pip install --upgrade yfinance)."
        ) from e

    if raw is None or len(raw) == 0:
        raise DataLoaderError(
            f"Yahoo Finance zwróciło puste dane dla '{ticker}'. Najczęstsze przyczyny: "
            f"(1) nieprawidłowy ticker, (2) ochrona antybotowa Yahoo przy starej wersji "
            f"yfinance - spróbuj `pip install --upgrade yfinance`."
        )

    # yfinance czasem zwraca MultiIndex kolumn (Price, Ticker) przy
    # pojedynczym tickerze w niektórych wersjach - spłaszcz do prostych nazw
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = _normalize_columns(raw)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        # Lekcja #4: WIDOCZNY błąd, nie ciche przejście do "brak danych"
        raise DataLoaderError(
            f"Dane dla '{ticker}' mają nieoczekiwany schemat - brakuje kolumn: "
            f"{sorted(missing)}. Dostępne kolumny: {sorted(df.columns)}. "
            f"To może oznaczać zmianę formatu odpowiedzi Yahoo Finance / yfinance."
        )

    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    if len(df) < 10:
        raise DataLoaderError(
            f"Za mało poprawnych barów danych dla '{ticker}' ({len(df)}) - potrzeba min. 10."
        )

    return df[["open", "high", "low", "close", "volume"]]


class CachedReferenceLoader:
    """
    Cache dzienny dla instrumentów referencyjnych kaskady (patrz lekcja
    #3 w docstringu modułu) - jeden plik JSON na ticker w `cache_dir`,
    ważny do końca dnia UTC, w którym został zapisany.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, ticker: str) -> str:
        safe = "".join(c if c.isalnum() else "_" for c in ticker)
        return os.path.join(self.cache_dir, f"{safe}.json")

    def _is_fresh(self, saved_at_iso: str) -> bool:
        try:
            saved_at = datetime.fromisoformat(saved_at_iso)
        except ValueError:
            return False
        now = datetime.now(timezone.utc)
        return saved_at.date() == now.date()

    def get(self, ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        path = self._path(ticker)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if self._is_fresh(payload.get("saved_at", "")):
                    df = pd.DataFrame(payload["data"])
                    df.index = pd.to_datetime(payload["index"])
                    return df
            except (json.JSONDecodeError, KeyError, OSError, ValueError) as e:
                # Widoczne ostrzeżenie, nie cichy fallback (lekcja #4) -
                # i tak próbujemy świeżo pobrać poniżej.
                print(f"[data_loader.py] UWAGA: uszkodzony cache dla '{ticker}': {e} - pobieram od nowa.")

        df = fetch_ohlcv(ticker, period=period, interval=interval)
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "index": [str(i) for i in df.index],
            "data": df.reset_index(drop=True).to_dict(orient="list"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return df

    def get_close_array(self, ticker: str, period: str = "6mo", interval: str = "1d"):
        df = self.get(ticker, period=period, interval=interval)
        return df["close"].values

    def get_close_volume(self, ticker: str, period: str = "6mo", interval: str = "1d"):
        df = self.get(ticker, period=period, interval=interval)
        return df["close"].values, df["volume"].values


def fetch_cascade_stage_data(target_ticker: str, cache: CachedReferenceLoader, period: str = "6mo") -> dict:
    """
    Pobiera (z cache'u) ceny + wolumen REPREZENTATYWNEGO instrumentu dla
    każdego ogniwa łańcucha PRZEPŁYWU KAPITAŁU (surowce/waluty/obligacje/
    indeksy/sektory) wcześniejszego niż "akcje" - patrz cascade.py.

    Dla ogniw z wieloma instrumentami referencyjnymi (np. "surowce" ma 4
    towary) uśrednia znormalizowane (do 100 na starcie) serie ceny -
    jeden "syntetyczny indeks koszyka" na ogniwo. Wolumen koszyka to
    SUMA wolumenów składowych (sensowniejsze niż średnia dla wolumenu -
    reprezentuje łączną aktywność handlową koszyka).

    Ogniwo "sektory" używa ETF-u WŁAŚCIWEGO dla `target_ticker`
    (cascade.sector_etf_for_ticker) - inaczej byłoby bez znaczenia
    (ten sam koszyk niezależnie od analizowanej spółki).

    Błędy pobrania POJEDYNCZEGO instrumentu referencyjnego NIE
    przerywają całej analizy (kaskada jest funkcją DODATKOWĄ) - są
    zbierane i zwracane osobno jako `errors` (lekcja #4: widoczne, nie
    ciche pominięcie).
    """
    from cascade import REFERENCE_TICKERS, sector_etf_for_ticker

    stage_data = {}
    errors = {}

    for stage, tickers in REFERENCE_TICKERS.items():
        price_list = []
        volume_list = []
        for tk in tickers:
            try:
                close, volume = cache.get_close_volume(tk, period=period)
                if len(close) >= 10:
                    price_list.append(close / close[0] * 100.0)
                    volume_list.append(volume)
            except DataLoaderError as e:
                errors.setdefault(stage, []).append(f"{tk}: {e}")

        if price_list:
            min_len = min(len(s) for s in price_list)
            price_trimmed = [s[-min_len:] for s in price_list]
            basket_price = sum(price_trimmed) / len(price_trimmed)
            volume_trimmed = [v[-min_len:] for v in volume_list]
            basket_volume = sum(volume_trimmed)
            stage_data[stage] = {"close": basket_price, "volume": basket_volume}

    sector_etf = sector_etf_for_ticker(target_ticker)
    try:
        close, volume = cache.get_close_volume(sector_etf, period=period)
        if len(close) >= 10:
            stage_data["sektory"] = {"close": close, "volume": volume}
    except DataLoaderError as e:
        errors.setdefault("sektory", []).append(f"{sector_etf}: {e}")

    return {"stage_data": stage_data, "errors": errors}
