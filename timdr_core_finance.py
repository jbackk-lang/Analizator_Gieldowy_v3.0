"""
timdr_core_finance.py — budulec sygnałów TIMDR dla danych giełdowych (OHLCV)
================================================================================
v3 tego repo. Nazwy funkcji (trm, flow, twist, rhythm, anomalies, defect,
resonance) pochodzą z modułu-łącznika dostarczonego przez użytkownika
("MODUŁ POD ANALIZATOR") - same funkcje NIE były w nim zaimplementowane
(moduł użytkownika to czysty "plumbing": pakuje wyniki w obiekty i przekazuje
dalej). Implementacje poniżej są MOJEGO autorstwa, zaprojektowane tak, by:

  1) trzymać się ogólnej ramy sygnałów TIMDR (anomalia / defekt / rezonans /
     skręt) wypracowanej i opisanej przy budowie Synoptyka (system pogodowy) -
     patrz README.md, sekcja "Lekcje z Synoptyka zastosowane w v3",
  2) być wektoryzowane (numpy), bez powtórnego przeliczania progów w pętli
     Python per-wiersz - to była realna, zmierzona przyczyna problemu
     wydajności w Synoptyku (O(n²), patrz lekcja #3),
  3) kalibrować progi ŻYWO z okna analizowanych danych, nie z jednej,
     sztywnej stałej uniwersalnej (lekcja #2).

Mapowanie na 4 typy sygnałów TIMDR (definicje ze skilla):
  anomalies() → anomalia  (pojedynczy odczyt poza normalnym zakresem)
  defect()    → defekt    (nagły skok między kolejnymi odczytami)
  resonance() → rezonans  (kilka niezależnych sprawdzeń zgadza się naraz)
  twist()     → skręt     (odwrócenie kierunku trendu)

`trm`, `flow`, `rhythm` to funkcje POMOCNICZE specyficzne dla finansów
(przygotowują dane wejściowe pod powyższe cztery), nie są bezpośrednio
częścią czwórki anomalia/defekt/rezonans/skręt.
"""

from __future__ import annotations

import numpy as np


def _mad_z(x: np.ndarray) -> np.ndarray:
    """Standardowy robust z-score (mediana + MAD, fallback do rozstępu/4
    przy płaskim sygnale) - ten sam wzorzec co we wszystkich innych
    modułach TIMDR w tym zestawie repo."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        span = np.max(x) - np.min(x)
        if span == 0:
            return np.zeros_like(x)
        return (x - med) / (span / 4)
    return 0.6745 * (x - med) / mad


def _rolling_median(x: np.ndarray, k: int) -> np.ndarray:
    """Mediana krocząca, okno k, min_periods=1 (brzegi liczone z
    dostępnej, krótszej próbki - bez tego pierwsze k-1 punktów byłoby
    NaN, co komplikowałoby wszystkie funkcje pochodne)."""
    n = len(x)
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - k + 1)
        out[i] = np.median(x[lo:i + 1])
    return out


def _rolling_percentile_spread(x: np.ndarray, window: int, p_lo=10, p_hi=90) -> np.ndarray:
    """Rozstęp (p_hi - p_lo) percentyla w kroczącym oknie `window`
    KOŃCZĄCYM się w każdym punkcie (tylko przeszłość + bieżący punkt -
    przyczynowe, nie zagląda w przyszłość). Brzegi: rosnące okno."""
    n = len(x)
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = x[lo:i + 1]
        if len(seg) < 2:
            out[i] = 0.0
        else:
            out[i] = np.percentile(seg, p_hi) - np.percentile(seg, p_lo)
    return out


# ---------------------------------------------------------------------
# Funkcje pomocnicze (przygotowanie danych)
# ---------------------------------------------------------------------

def trm(price, k: int = 5) -> np.ndarray:
    """
    TRM ("Trend Reference Mean/Median") - odporna linia referencyjna
    trendu: mediana krocząca w oknie `k`. Mediana (nie średnia krocząca)
    świadomie - jak wszędzie w tym zestawie repo, mediana jest odporna
    na pojedyncze świece odstające (np. spike na otwarciu, dane błędnie
    zaraportowane przez giełdę), których średnia krocząca nie odfiltruje.

    To NIE jest sygnał sam w sobie - to WEJŚCIE dla flow()/twist().
    """
    price = np.asarray(price, dtype=float)
    if len(price) == 0:
        return price.copy()
    return _rolling_median(price, k)


def flow(trm_price, window: int = 5) -> np.ndarray:
    """
    "Flow" - tempo/kierunek zmiany linii trendu (trm) w oknie `window`:
    flow[t] = (trm[t] - trm[t-window]) / window (dyskretna pochodna,
    przybliżenie prędkości ruchu trendu; brzegi liczone z dostępnej,
    krótszej różnicy - okno rosnące od 1 do `window`).

    Dodatnie flow = trend rośnie, ujemne = trend spada, ~0 = trend płaski.
    """
    x = np.asarray(trm_price, dtype=float)
    n = len(x)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        span = min(window, i)
        if span == 0:
            out[i] = 0.0
        else:
            out[i] = (x[i] - x[i - span]) / span
    return out


def twist(flow_price, window: int = 5, factor: float = 1.5) -> np.ndarray:
    """
    "Skręt" (trend reversal) - DOKŁADNIE wg definicji ze skilla TIMDR:
    znak lokalnego nachylenia odwraca się między DWOMA KOLEJNYMI oknami,
    a wielkość tego odwrócenia przekracza próg (`factor` * std(flow)).

    Implementacja: dla każdego punktu t >= 2*window porównujemy średnie
    flow w oknie POPRZEDZAJĄCYM [t-2w, t-w) ze średnim flow w oknie
    BIEŻĄCYM [t-w, t) - obie leżą w przeszłości względem t (przyczynowe,
    bez zaglądania w przyszłość, spójne z tym, jak repo analizuje dane
    historyczne "do teraz").

    Zwraca: indeksy próbek, w których wykryto skręt.
    """
    x = np.asarray(flow_price, dtype=float)
    n = len(x)
    if n < 2 * window + 1:
        return np.array([], dtype=int)

    std = np.std(x)
    if std == 0:
        return np.array([], dtype=int)
    threshold = factor * std

    idx = []
    for t in range(2 * window, n):
        prev = np.mean(x[t - 2 * window:t - window])
        curr = np.mean(x[t - window:t])
        sign_flip = (prev > 0 and curr < 0) or (prev < 0 and curr > 0)
        magnitude = abs(curr - prev)
        if sign_flip and magnitude > threshold:
            idx.append(t)
    return np.array(idx, dtype=int)


def rhythm(volume, max_lag: int = 30, power_thresh: float = 0.35):
    """
    Okresowość wolumenu (np. tygodniowy/miesięczny wzorzec aktywności) -
    TEN SAM, zweryfikowany wzorzec autokorelacji + TYLKO lokalne maksima
    co w innych modułach TIMDR tego repo (bio_core.rhythm,
    catalog_core.py) - pełny detrend (np.polyfit) na sygnale ZE ZNAKIEM,
    filtr tylko lokalnych maksimów autokorelacji (unika artefaktu
    rektyfikacji, wielokrotnie udokumentowanego w innych modułach tego
    zestawu repo).

    Zwraca: (periods, power) - periods to lista okresów (w barach) z
    lokalnym maksimum autokorelacji, power to moc najsilniejszego z nich.
    """
    x = np.asarray(volume, dtype=float)
    n = len(x)
    if n < 8:
        return [], 0.0

    idx = np.arange(n, dtype=float)
    coeffs = np.polyfit(idx, x, 1)
    detrended = x - np.polyval(coeffs, idx)

    std = np.std(detrended)
    if std == 0:
        return [], 0.0

    max_lag = min(max_lag, n - 2)
    if max_lag < 2:
        return [], 0.0

    acf = np.zeros(max_lag + 1)
    for lag in range(1, max_lag + 1):
        a, b = detrended[:-lag], detrended[lag:]
        denom = np.std(a) * np.std(b) * len(a)
        acf[lag] = 0.0 if denom == 0 else np.sum(a * b) / denom

    peaks = []
    for lag in range(2, max_lag):
        if acf[lag] > acf[lag - 1] and acf[lag] > acf[lag + 1] and acf[lag] > power_thresh:
            peaks.append((lag, float(acf[lag])))

    if not peaks:
        return [], 0.0
    peaks.sort(key=lambda p: -p[1])
    return [p[0] for p in peaks], peaks[0][1]


# ---------------------------------------------------------------------
# Cztery główne sygnały TIMDR (anomalia / defekt / rezonans / skręt)
# ---------------------------------------------------------------------

def anomalies(price, factor: float = 3.0) -> np.ndarray:
    """
    "Anomalia" wg definicji ze skilla: pojedynczy odczyt poza
    statystycznie normalnym zakresem - MAD-z (mediana+MAD), próg
    KALIBROWANY ŻYWO z samego analizowanego okna (lekcja #2 ze
    Synoptyka: NIGDY sztywna, uniwersalna stała - realne ceny mają
    kompletnie różne skale między spółkami/walutami).
    """
    x = np.asarray(price, dtype=float)
    if len(x) == 0:
        return np.array([], dtype=int)
    z = _mad_z(x)
    return np.where(np.abs(z) > factor)[0]


def defect(price, window: int = 20, jump_factor: float = 3.0, min_floor_frac: float = 1e-4) -> np.ndarray:
    """
    "Defekt" wg definicji ze skilla: nagły skok między kolejnymi
    odczytami, większy niż próg wyprowadzony z NIEDAWNEGO rozstępu TEJ
    ZMIENNEJ. Skala odniesienia to rozstęp (p90-p10) SAMYCH RÓŻNIC
    dzień-do-dnia w kroczącym oknie `window` (licząc tylko wstecz -
    przyczynowe), NIE rozstęp poziomów ceny.

    POPRAWKA WZGLĘDEM DOSŁOWNEGO PRZEPISU ZE SKILLA (znaleziona
    empirycznie przy testowaniu): oryginalna definicja ze skilla
    ("próg = 0.3 * rozstęp POZIOMÓW parametru") sprawdza się dla
    wolno zmiennych odczytów pogodowych (temperatura), ale dla CEN
    AKCJI/WALUT dała próg dramatycznie za czuły - losowy spacer cenowy
    (zwykły szum dnia-do-dnia, BEZ żadnego prawdziwego skoku) był
    flagowany jako "defekt" w ~20% próbek (zweryfikowano na syntetycznym
    random walk, patrz test_timdr_core_finance.py). Przyczyna: dla
    trendującego random walk rozstęp POZIOMÓW w krótkim oknie jest tego
    samego rzędu wielkości co pojedyncza różnica dzień-do-dnia - więc
    próg 0.3x tego rozstępu jest mniejszy niż typowa, zupełnie normalna
    różnica. Naprawiono: odniesienie to rozstęp SAMYCH RÓŻNIC (o ile
    typowo różnią się między sobą kolejne przyrosty ceny), z
    podniesionym `jump_factor` (0.3 → 3.0, ten sam rząd wielkości co
    próg MAD-z=3 używany w anomalies() - analogiczna logika: "typowa
    zmienność x kilka" = prawdziwy wyjątek, nie każda różnica).

    ZABEZPIECZENIE (lekcja #2, dopisek o parametrach "zero-inflated"):
    jeśli różnice w danym oknie są praktycznie zerowe (rozstęp p90-p10
    ~ 0, np. bardzo niska płynność), próg mógłby skolapsować do ~0 i
    oznaczyć każdy najmniejszy ruch jako "defekt". Dlatego próg ma
    DOLNĄ PODŁOGĘ: `min_floor_frac * mediana(|price|)` (domyślnie
    0.01% poziomu ceny) - nigdy nie spada poniżej tego minimum.

    Zwraca: indeksy próbek, w których wykryto defekt (skok).
    """
    x = np.asarray(price, dtype=float)
    n = len(x)
    if n < 3:
        return np.array([], dtype=int)

    diffs = np.diff(x)  # diffs[i] = x[i+1] - x[i], dlugosc n-1
    spread = _rolling_percentile_spread(diffs, window)  # rozstep SAMYCH ROZNIC, przyczynowe
    floor = min_floor_frac * np.median(np.abs(x)) if np.median(np.abs(x)) > 0 else 1e-9
    threshold = np.maximum(jump_factor * spread, floor)

    flagged = np.where(np.abs(diffs) > threshold)[0]
    return flagged + 1  # +1 bo diffs[i] odpowiada przejsciu do probki i+1


def resonance(price, factor: float = 3.0, defect_window: int = 20, twist_window: int = 5):
    """
    "Rezonans" wg definicji ze skilla: kilka NIEZALEŻNYCH sprawdzeń
    zgadza się naraz - w oryginalnym sformułowaniu (Synoptyk) to kilka
    RÓŻNYCH PARAMETRÓW (np. temperatura + ciśnienie + wiatr) flagujących
    anomalię w tej samej chwili. `resonance()` tutaj przyjmuje tylko
    JEDEN szereg (`price`) - zgodnie z sygnaturą w module-łączniku
    użytkownika, gdzie jest wołane niezależnie od anomalies()/defect()/
    twist() (nie jako ich kompozycja na poziomie pakietu).

    Adaptacja: zamiast wielu RÓŻNYCH parametrów, `resonance()` liczy
    WŁASNE, wewnętrzne wersje trzech sprawdzeń na tym samym szeregu
    (anomalia poziomu, defekt skoku, skręt kierunku - każde liczone TU,
    niezależnie od zewnętrznych anomalies()/defect()/twist(), żeby
    funkcja była samodzielna) i zwraca, w ilu z trzech dana próbka
    "świeci się" naraz - to samo w sobie jest silniejszym, bardziej
    wiarygodnym sygnałem niż pojedyncze sprawdzenie (duch definicji ze
    skilla zachowany: SIŁA sygnału rośnie ze ZGODNOŚCIĄ niezależnych
    sprawdzeń, nie z jednym czułym progiem).

    Zwraca: (score, strong_idx)
      score      - np.ndarray we/wy [0,1] dla KAŻDEJ próbki: 0/3, 1/3,
                   2/3 albo 3/3 sprawdzeń zgodnych w tym punkcie
      strong_idx - indeksy, gdzie score >= 2/3 (co najmniej 2 z 3
                   niezależnych sprawdzeń zgodne naraz)
    """
    x = np.asarray(price, dtype=float)
    n = len(x)
    if n < max(8, 2 * twist_window + 1):
        return np.zeros(n, dtype=float), np.array([], dtype=int)

    anomaly_idx = set(anomalies(x, factor=factor).tolist())
    defect_idx = set(defect(x, window=defect_window).tolist())

    trm_x = trm(x, k=twist_window)
    flow_x = flow(trm_x, window=twist_window)
    twist_idx = set(twist(flow_x, window=twist_window).tolist())

    score = np.zeros(n, dtype=float)
    for i in range(n):
        hits = (i in anomaly_idx) + (i in defect_idx) + (i in twist_idx)
        score[i] = hits / 3.0

    strong_idx = np.where(score >= 2.0 / 3.0)[0]
    return score, strong_idx
