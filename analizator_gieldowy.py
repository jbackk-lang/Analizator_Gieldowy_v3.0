"""
analizator_gieldowy.py — interpretacja pakietu sygnałów TIMDR
================================================================================
`AnalizatorGieldowy.przetworz_sygnaly(packet)` bierze `TimdrPacket` (7
niezależnych sygnałów obliczonych przez `TimdrEngine` z modułu
użytkownika) i zwraca gotowy werdykt: klasyfikację Emergencja/szum,
procent Ufności, lekki backtest (sharpe_n/winrate_n/dd_n) oraz RSI jako
NIEZALEŻNY, klasyczny wskaźnik kontrolny (nie oparty o TIMDR - druga,
niezależna metoda do porównania, w duchu "dwóch niezależnych torów"
opisanych w lekcjach ze Synoptyka, punkt #7).

Dokładnie ten kształt wyniku (emergencja_label, ufnosc_procent,
sharpe_n, winrate_n, dd_n, rsi) użytkownik już wcześniej widział jako
przykładowy output - te definicje zostały tak dobrane, by liczbowo
odtwarzać sensowne, realistyczne wartości w tym stylu (patrz README.md,
sekcja "Skąd te konkretne wzory").
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------
# RSI - klasyczny wskaźnik (Wilder), NIEZALEŻNY od TIMDR
# ---------------------------------------------------------------------

def rsi(price, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index metodą Wildera (wygładzanie wykładnicze
    średniego zysku/straty, nie prosta średnia krocząca - standardowa
    definicja). Pierwsze `period` próbek nie ma wystarczających danych
    (zwracane jako 50.0 - neutralne, nie 0 czy 100, żeby nie sugerować
    fałszywego sygnału skrajnego zanim jest dość danych).

    Przypadek brzegowy: gdy zarówno średni zysk, jak i średnia strata
    wynoszą 0 (sygnał idealnie płaski w oknie Wildera) - RSI=50
    (neutralnie), NIE 100 (co dałaby naiwna formuła 100-100/(1+inf) bez
    tego zabezpieczenia, myląc "brak ruchu" z "same zyski").
    """
    x = np.asarray(price, dtype=float)
    n = len(x)
    if n < period + 1:
        return np.full(n, 50.0)

    diffs = np.diff(x)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    out = np.full(n, 50.0)
    for i in range(period, n):
        ag, al = avg_gain[i], avg_loss[i]
        if ag == 0 and al == 0:
            out[i] = 50.0
        elif al == 0:
            out[i] = 100.0
        else:
            rs = ag / al
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def rsi_interpretation(value: float) -> str:
    if value is None:
        return "brak danych"
    if value >= 70:
        return "wykupienie (RSI >= 70)"
    if value <= 30:
        return "wyprzedanie (RSI <= 30)"
    return "neutralnie"


# ---------------------------------------------------------------------
# Lekki backtest - sharpe_n / winrate_n / dd_n
# ---------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def backtest_resonance_strategy(price, resonance_score, flow_signal,
                                 resonance_threshold: float = 2.0 / 3.0,
                                 dd_reference: float = 0.5) -> dict:
    """
    Prosty, w pełni przejrzysty backtest LONG-ONLY (celowo - short
    dodaje złożoność i ryzyko błędnej interpretacji dla użytkownika bez
    doświadczenia w tradingu, a nie jest tu potrzebny do zademonstrowania
    użyteczności sygnałów TIMDR): pozycja[t] = 1, jeśli resonance_score[t]
    >= próg ORAZ flow[t] > 0 (rezonans + trend rosnący), inaczej 0.
    Zwrot z bara t liczony na podstawie pozycji ustalonej na PODSTAWIE
    danych do t-1 (przyczynowe - bez zaglądania w przyszłość).

    Metryki:
      sharpe   - roczny Sharpe (mean/std * sqrt(252)), 0 gdy std=0
      winrate  - odsetek zyskownych barów SPOŚRÓD barów z otwartą pozycją
                 (nie liczone dla barów poza rynkiem - winrate ma sens
                 tylko wśród faktycznie podjętych "transakcji")
      max_dd   - maksymalne obsunięcie kapitału (wartość ujemna, np -0.23)

    Znormalizowane (zakres ~[0,1], do wyświetlenia obok siebie):
      sharpe_n = sigmoid(sharpe) - gładkie, nieograniczone dziedzinowo
                 (bez arbitralnego przycinania) przejście przez [0,1];
                 sharpe=0 -> sharpe_n=0.5, sharpe bardzo ujemny -> ~0
      winrate_n = winrate (już naturalnie w [0,1])
      dd_n = min(|max_dd| / dd_reference, 1.0) - `dd_reference`=0.5
             (50% obsunięcia) potraktowane jako "bardzo źle, blisko
             górnej granicy skali"

    `no_trades: True`, jeśli strategia nigdy nie weszła w pozycję -
    wtedy winrate/winrate_n są None (nie 0.5 czy 0 - żeby nie sugerować
    fałszywej informacji tam, gdzie jej po prostu nie ma).
    """
    price = np.asarray(price, dtype=float)
    resonance_score = np.asarray(resonance_score, dtype=float)
    flow_signal = np.asarray(flow_signal, dtype=float)
    n = len(price)

    if n < 3:
        return {
            "sharpe": 0.0, "winrate": None, "max_dd": 0.0,
            "sharpe_n": 0.5, "winrate_n": None, "dd_n": 0.0,
            "no_trades": True, "n_bars_in_position": 0,
        }

    position = ((resonance_score >= resonance_threshold) & (flow_signal > 0)).astype(float)

    returns = np.zeros(n)
    for t in range(1, n):
        if price[t - 1] == 0:
            continue
        bar_return = (price[t] - price[t - 1]) / price[t - 1]
        returns[t] = position[t - 1] * bar_return

    in_position_mask = position[:-1] > 0  # pozycja decydująca o returns[1:]
    active_returns = returns[1:][in_position_mask]
    n_active = len(active_returns)

    if n_active == 0:
        sharpe = 0.0
        winrate = None
    else:
        mean_r = np.mean(active_returns)
        std_r = np.std(active_returns)
        sharpe = float(mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
        winrate = float(np.mean(active_returns > 0))

    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = np.where(running_max > 0, (equity - running_max) / running_max, 0.0)
    max_dd = float(np.min(drawdown)) if len(drawdown) else 0.0

    sharpe_n = _sigmoid(sharpe)
    winrate_n = winrate  # już w [0,1] albo None
    dd_n = min(abs(max_dd) / dd_reference, 1.0) if dd_reference > 0 else 0.0

    return {
        "sharpe": sharpe, "winrate": winrate, "max_dd": max_dd,
        "sharpe_n": round(sharpe_n, 3),
        "winrate_n": round(winrate_n, 3) if winrate_n is not None else None,
        "dd_n": round(dd_n, 3),
        "no_trades": n_active == 0,
        "n_bars_in_position": int(n_active),
    }


# ---------------------------------------------------------------------
# Klasyfikacja Emergencja / szum
# ---------------------------------------------------------------------

EMERGENCE_CONFIDENCE_THRESHOLD = 20.0  # procent - poniżej = "szum (brak emergencji)"


def classify_emergence(resonance_score, lookback: int = 10) -> dict:
    """
    Ufność% = 100 * średni resonance_score z OSTATNICH `lookback` barów
    (0% = żadne z trzech wewnętrznych sprawdzeń rezonansu nigdy się nie
    zgadzają w ostatnim oknie, 100% = wszystkie trzy zgadzają się na
    każdym barze ostatniego okna - w praktyce skrajnie rzadkie).

    Etykieta "EMERGENCJA" tylko gdy Ufność >= próg
    (EMERGENCE_CONFIDENCE_THRESHOLD=20%) - inaczej "szum (brak
    emergencji)". Świadomie prosty, jednoznaczny próg zamiast
    nieprzejrzystej, wielo-czynnikowej wagi - łatwiej zweryfikować i
    wytłumaczyć użytkownikowi, co dokładnie oznacza wynik.
    """
    score = np.asarray(resonance_score, dtype=float)
    if len(score) == 0:
        return {"emergencja_label": "szum (brak emergencji)", "ufnosc_procent": 0.0}

    window = score[-lookback:]
    ufnosc = 100.0 * float(np.mean(window))
    label = "EMERGENCJA" if ufnosc >= EMERGENCE_CONFIDENCE_THRESHOLD else "szum (brak emergencji)"
    return {"emergencja_label": label, "ufnosc_procent": round(ufnosc, 1)}


# ---------------------------------------------------------------------
# Główna klasa
# ---------------------------------------------------------------------

class AnalizatorGieldowy:
    """Interpretuje TimdrPacket i zwraca gotowy werdykt (patrz
    przetworz_sygnaly). Bezstanowa sama w sobie - trwały stan (ostatni
    werdykt, log predykcji) jest wstrzykiwany z zewnątrz (state.py),
    zgodnie z zasadą, że silnik analizy i przechowywanie stanu to dwie
    osobne odpowiedzialności."""

    def przetworz_sygnaly(self, packet, rsi_period: int = 14) -> dict:
        # UWAGA: RSI i backtest liczone na SUROWEJ cenie (packet.price),
        # NIE na wygładzonej packet.trm - patrz docstring PriceSignal/
        # TimdrPacket w pipeline.py (naprawiony błąd projektowy:
        # oryginalny pakiet użytkownika w ogóle nie przekazywał surowej
        # ceny, tylko wygładzoną trm).
        price_like = packet.price.values
        resonance_score = packet.resonance.values
        flow_signal = packet.flow.values
        rhythm_periods, rhythm_power = packet.rhythm.values

        emergencja = classify_emergence(resonance_score)
        bt = backtest_resonance_strategy(price_like, resonance_score, flow_signal)

        rsi_vals = rsi(price_like, period=rsi_period)
        rsi_last = float(rsi_vals[-1]) if len(rsi_vals) else None

        n = len(price_like)
        # Indeksy (nie tylko liczności) - potrzebne dashboardowi do
        # zaznaczenia zdarzeń na wykresie ceny. `x` = surowa cena (ta
        # sama seria co price_like/packet.price) - patrz Bug 1 w
        # pipeline.py, dlaczego to jest SUROWA, nie wygładzona cena.
        anomaly_idx = [int(i) for i in packet.anomaly.values]
        defect_idx = [int(i) for i in packet.defect.values]
        twist_idx = [int(i) for i in packet.twist.values]

        # Cena: ostatnia wartość + zmiana % na przestrzeni CAŁEGO
        # analizowanego okresu (pierwsza vs ostatnia bara) - dashboard
        # potrzebuje tego jako osobnej karty ("Cena"), niezależnie od
        # wykresu. `trm` (trend reference mean, mediana krocząca k=5)
        # jest dodatkowo zwracana jako osobna seria - to jest sygnał
        # "trendu" nałożony na surową cenę na wykresie, nie tylko
        # markery anomalii/defektów.
        last_price = float(price_like[-1]) if n else None
        first_price = float(price_like[0]) if n else None
        price_change_pct = (
            round(100.0 * (last_price - first_price) / first_price, 2)
            if (n and first_price not in (None, 0))
            else None
        )
        trm_values = packet.trm.values

        result = {
            **emergencja,
            "last_price": round(last_price, 4) if last_price is not None else None,
            "price_change_pct": price_change_pct,
            "trend": [round(float(v), 6) if np.isfinite(v) else None for v in trm_values],
            "sharpe_n": bt["sharpe_n"],
            "winrate_n": bt["winrate_n"],
            "dd_n": bt["dd_n"],
            "backtest_detail": {
                "sharpe": round(bt["sharpe"], 4),
                "winrate": bt["winrate"],
                "max_dd": round(bt["max_dd"], 4),
                "no_trades": bt["no_trades"],
                "n_bars_in_position": bt["n_bars_in_position"],
            },
            "rsi": round(rsi_last, 1) if rsi_last is not None else None,
            "rsi_interpretation": rsi_interpretation(rsi_last),
            "n_anomaly": len(anomaly_idx),
            "n_defect": len(defect_idx),
            "n_twist": len(twist_idx),
            "anomalies_idx": anomaly_idx,
            "defect_idx": defect_idx,
            "twist_idx": twist_idx,
            "rhythm_volume_power": round(float(rhythm_power), 3),
            "rhythm_volume_periods": rhythm_periods[:3],
            "resonance_last": round(float(resonance_score[-1]), 3) if n else None,
            "flow_last": round(float(flow_signal[-1]), 6) if n else None,
            "n_bars": n,
            "x": price_like.tolist(),
        }
        return result
