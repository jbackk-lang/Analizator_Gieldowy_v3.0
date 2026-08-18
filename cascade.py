"""
cascade.py — kaskada PRZEPŁYWU KAPITAŁU (surowce → waluty → obligacje →
indeksy → sektory → akcje)
================================================================================
Łańcuch: surowce → waluty → obligacje → indeksy → sektory → akcje.
Szok cenowy surowców jest wyceniany w USD, więc najpierw widać go na
rynku walutowym; to zmienia oczekiwania inflacyjne/stóp procentowych,
więc reaguje rynek obligacji; dalej przecenia się szeroki rynek akcji
(indeksy), potem konkretne sektory, na końcu pojedyncze spółki.

POPRAWKA (uwaga użytkownika: "mówimy o przepływie finansów nie szkolnej
definicji ważności"): pierwsza wersja tego modułu ważyła ogniwa STAŁĄ,
z góry przypisaną "siłą wpływu" (★1-5, moja szacunkowa, podręcznikowa
ocena "jak ważne jest to ogniwo"). To było niezgodne z intencją -
kaskada ma mierzyć RZECZYWISTY PRZEPŁYW KAPITAŁU, nie opinię o
ważności. Naprawiono: waga każdego zdarzenia liczona jest TERAZ z
DANYCH - `_flow_intensity()` łączy wielkość ruchu ceny (%) z
WOLUMENEM WZGLĘDNYM (ile razy wyższy niż typowy dla tego instrumentu) w
momencie zdarzenia. Duży ruch ceny na wolumenie w normie = mało
przekonujący ("mogła się przelać płynność, ale niewielu graczy to
zrobiło"); ten sam ruch na 3x wolumenie = silny sygnał realnego
przepływu kapitału. `delay_days`/`delay_label` pozostają WYŁĄCZNIE
informacyjne (orientacyjny czas dotarcia efektu) - nie wchodzą już do
wzoru na wagę/Ufność, tylko do treści alertu.

ZASTRZEŻENIE UCZCIWOŚCI: `delay_days` to nadal heurystyczne,
niezmierzone przybliżenie (patrz alert text) - narzędzie daje
orientacyjny, jakościowy sygnał "gdzie realnie płynie kapitał wcześniej
w łańcuchu", nie ilościową prognozę.
"""

from __future__ import annotations

import numpy as np

from timdr_core_finance import resonance


ASSET_CHAIN = ["surowce", "waluty", "obligacje", "indeksy", "sektory", "akcje"]

ASSET_LABELS = {
    "surowce": "Surowce", "waluty": "Waluty", "obligacje": "Obligacje",
    "indeksy": "Indeksy", "sektory": "Sektor", "akcje": "Akcje",
}

REFERENCE_TICKERS = {
    "surowce": ["CL=F", "NG=F", "HG=F", "GC=F"],
    "waluty": ["DX-Y.NYB", "EURUSD=X", "USDPLN=X"],
    "obligacje": ["^TNX"],
    "indeksy": ["^GSPC", "^DJI"],
}

# WYŁĄCZNIE informacyjne (orientacyjny czas dotarcia efektu w treści
# alertu) - NIE wchodzi do wzoru na wagę/Ufność (patrz docstring modułu:
# to była pierwsza, odrzucona wersja - waga liczy się teraz z
# rzeczywistego przepływu kapitału, nie z góry założonej ważności).
STAGE_TIMING = {
    # "reaction" = uzasadnienie użytkownika, dlaczego to ogniwo w ogóle
    # trzeba obserwować (patrz jego wiadomość: "A. surowce - tam
    # zaczyna się fala. B. waluty - tam widać kierunek kapitału.
    # C. obligacje - tam widać koszt pieniądza. D. indeksy - tam widać
    # reakcję systemu. E. sektory - tam widać dystrybucję energii.
    # F. akcje - tam widać efekt końcowy.")
    "surowce":   {"delay_days": 0,  "delay_label": "źródło",          "reaction": "źródło fali"},
    "waluty":    {"delay_days": 1,  "delay_label": "minimalne",       "reaction": "kierunek kapitału"},
    "obligacje": {"delay_days": 3,  "delay_label": "niskie",          "reaction": "koszt pieniądza"},
    "indeksy":   {"delay_days": 6,  "delay_label": "średnie",         "reaction": "reakcja systemu"},
    "sektory":   {"delay_days": 9,  "delay_label": "średnie-wysokie", "reaction": "dystrybucja energii/kapitału"},
    "akcje":     {"delay_days": 14, "delay_label": "wysokie",         "reaction": "efekt końcowy"},
}

SECTOR_ETF_MAP = {
    "technologia": "XLK", "energetyka": "XLE", "finanse": "XLF",
    "konsumpcja": "XLY", "przemysl": "XLI", "rolnictwo": "DBA",
    "transport": "IYT", "surowce": "XLB",
}

TICKER_SECTOR_MAP = {
    "AAPL": "technologia", "MSFT": "technologia", "GOOGL": "technologia",
    "GOOG": "technologia", "NVDA": "technologia", "META": "technologia",
    "AMZN": "konsumpcja", "TSLA": "konsumpcja",
    "PKN.WA": "energetyka", "PKO.WA": "finanse", "PZU.WA": "finanse",
    "KGH.WA": "surowce", "CDR.WA": "technologia", "ALE.WA": "konsumpcja",
    "CL=F": "surowce", "NG=F": "surowce", "HG=F": "surowce", "GC=F": "surowce",
    "SI=F": "surowce", "ZW=F": "surowce", "ZC=F": "surowce",
    "EURPLN=X": "finanse", "USDPLN=X": "finanse", "EURUSD=X": "finanse",
    "^GSPC": "finanse", "^DJI": "finanse",
}

DEFAULT_SECTOR = "technologia"

# Kalibracja normalizacji flow_intensity (patrz _flow_intensity) - "pełna
# waga" (1.0) osiągana orientacyjnie przy: 3% ruchu ceny na typowym
# wolumenie, ALBO 1.5% ruchu na ~2x typowym wolumenie, itd. (iloczyn).
# Heurystyczne, jawnie oznaczone - nie skalibrowane na realnych danych
# (brak dostępu do wystarczająco długiej, zróżnicowanej historii do
# rzetelnej kalibracji w ramach budowy tego narzędzia).
FLOW_INTENSITY_REFERENCE = 0.03


def sector_for_ticker(ticker: str) -> str:
    return TICKER_SECTOR_MAP.get(ticker.upper(), DEFAULT_SECTOR)


def sector_etf_for_ticker(ticker: str) -> str:
    return SECTOR_ETF_MAP.get(sector_for_ticker(ticker), "XLK")


# ---------------------------------------------------------------------
# Waluta / jednostka ceny - HEURYSTYKA z konwencji sufiksów tickerów
# Yahoo Finance, NIE pobrana z autorytatywnego pola (yfinance.download()
# w tym pipeline nie zwraca metadanych waluty - tylko OHLCV; osobne
# zapytanie o Ticker.info dla KAŻDEJ analizy byłoby kolejnym wolnym
# wywołaniem sieciowym per żądanie, którego świadomie unikamy - patrz
# lekcja #3 ze Synoptyka, cache/nie-przeliczaj-w-kółko). Jawnie
# oznaczone jako przybliżenie w UI i README - NIE traktować jako
# potwierdzone dane rynkowe.
# ---------------------------------------------------------------------

FUTURES_UNIT_MAP = {
    "GC=F": "USD/uncja (złoto)", "SI=F": "USD/uncja (srebro)",
    "CL=F": "USD/baryłka (ropa WTI)", "BZ=F": "USD/baryłka (ropa Brent)",
    "NG=F": "USD/MMBtu (gaz ziemny)", "HG=F": "USD/funt (miedź)",
    "ZW=F": "USD/buszel (pszenica)", "ZC=F": "USD/buszel (kukurydza)",
}


def currency_unit_for_ticker(ticker: str) -> dict:
    """
    Zwraca {"currency": "USD"/"PLN"/..., "unit_label": tekst do pokazania
    obok ceny}. Heurystyka na podstawie sufiksu tickera (konwencja
    Yahoo Finance), NIE autorytatywne dane - patrz komentarz modułu.
    """
    t = ticker.upper().strip()

    if t.endswith("=X") and len(t) >= 8:
        # para walutowa BAZAKWOTA=X, np. EURPLN=X -> kwotowana w PLN
        quote = t[3:6]
        return {"currency": quote, "unit_label": quote}

    if t.endswith(".WA"):
        return {"currency": "PLN", "unit_label": "PLN"}

    if t.endswith("-USD"):
        return {"currency": "USD", "unit_label": "USD"}

    if t in FUTURES_UNIT_MAP:
        return {"currency": "USD", "unit_label": FUTURES_UNIT_MAP[t]}
    if t.endswith("=F"):
        return {"currency": "USD", "unit_label": "USD (kontrakt terminowy)"}

    if t.startswith("^"):
        return {"currency": None, "unit_label": "pkt (punkty indeksowe)"}

    # domyślnie: akcje bez sufiksu giełdy krajowej - notowane w USD
    # (NASDAQ/NYSE, zdecydowana większość tickerów bez kropki w tym
    # narzędziu) - PRZYBLIŻENIE, nie sprawdzone dla każdej giełdy.
    return {"currency": "USD", "unit_label": "USD"}


def stage_index(stage: str) -> int:
    return ASSET_CHAIN.index(stage) if stage in ASSET_CHAIN else ASSET_CHAIN.index("akcje")


def _flow_intensity(prices: np.ndarray, volumes, peak_local_idx: int, window: int = 3) -> float:
    """
    Miara RZECZYWISTEGO przepływu kapitału wokół zdarzenia: wielkość
    ruchu ceny (% zmiany) razy wolumen WZGLĘDNY (ile razy wyższy niż
    mediana całej serii). Bez danych o wolumenie (volumes=None) wolumen
    względny = 1.0 (neutralny mnożnik - miara spada wtedy do samej
    wielkości ruchu ceny).
    """
    n = len(prices)
    lo = max(0, peak_local_idx - window)
    hi = min(n - 1, peak_local_idx + window)
    if prices[lo] == 0:
        price_change_pct = 0.0
    else:
        price_change_pct = abs(prices[hi] - prices[lo]) / abs(prices[lo])

    if volumes is not None and len(volumes) == n:
        volumes = np.asarray(volumes, dtype=float)
        typical_vol = np.median(volumes)
        event_vol = np.mean(volumes[lo:hi + 1])
        rel_volume = (event_vol / typical_vol) if typical_vol > 0 else 1.0
    else:
        rel_volume = 1.0

    return price_change_pct * rel_volume


def analyze_cascade(target_stage: str, stage_data: dict, lookback: int = 10,
                     active_threshold: float = 2.0 / 3.0) -> dict:
    """
    target_stage: zwykle "akcje".
    stage_data:   {ogniwo: {"close": np.ndarray, "volume": np.ndarray | None}}
                  dla ogniw WCZEŚNIEJSZYCH niż target_stage. "volume"
                  może być None/pominięte - wtedy waga liczy się z
                  samej wielkości ruchu ceny (patrz _flow_intensity).

    Zwraca dict: target_stage, upstream_pressure_score (0..1, ważone
    ZMIERZONYM przepływem kapitału - patrz docstring modułu),
    consistent_direction, alerts, stage_details.
    """
    t_idx = stage_index(target_stage)
    stage_details = {}
    directions = []
    alerts = []
    weighted_active = 0.0
    weight_total = 0.0

    for stage in ASSET_CHAIN[:t_idx]:
        entry = stage_data.get(stage)
        if entry is None:
            continue
        prices = entry.get("close") if isinstance(entry, dict) else entry
        volumes = entry.get("volume") if isinstance(entry, dict) else None
        if prices is None or len(prices) < 10:
            continue
        prices = np.asarray(prices, dtype=float)
        timing = STAGE_TIMING.get(stage, {"delay_days": 7, "delay_label": "średnie", "reaction": "?"})

        score, _ = resonance(prices)
        recent_start = max(0, len(score) - lookback)
        recent = score[recent_start:]
        mean_recent = float(np.mean(recent)) if len(recent) else 0.0
        peak_recent = float(np.max(recent)) if len(recent) else 0.0
        peak_local_idx = recent_start + int(np.argmax(recent)) if len(recent) else None

        is_active = peak_recent >= active_threshold

        if peak_local_idx is not None:
            flow_intensity = _flow_intensity(prices, volumes, peak_local_idx)
            lo = max(0, peak_local_idx - 2)
            hi = min(len(prices) - 1, peak_local_idx + 2)
            event_delta = float(prices[hi] - prices[lo])
        else:
            flow_intensity = 0.0
            event_delta = 0.0
        direction = "wzrostowa" if event_delta > 0 else ("spadkowa" if event_delta < 0 else "neutralna")

        # Waga = zmierzony przepływ kapitału (0..1+, ucięte do 1.0),
        # NIE z góry przypisana "ważność" ogniwa (patrz docstring modułu).
        weight = min(flow_intensity / FLOW_INTENSITY_REFERENCE, 1.0) if is_active else 0.0

        stage_details[stage] = {
            "mean_resonance_recent": round(mean_recent, 3),
            "active": is_active,
            "direction": direction,
            "flow_intensity": round(flow_intensity, 5),
            "flow_weight": round(weight, 3),
            "delay_days": timing["delay_days"],
            "delay_label": timing["delay_label"],
            "reaction": timing["reaction"],
        }

        weight_total += 1.0  # każde rozpatrzone ogniwo liczy się równo do mianownika
        if is_active:
            weighted_active += weight
            directions.append(direction)
            alerts.append(
                f"{ASSET_LABELS[stage]}: aktywny przepływ kapitału w ostatnich {lookback} barach "
                f"(intensywność={flow_intensity:.4f}, waga={weight:.2f}), kierunek {direction} "
                f"({timing['reaction']}) - orientacyjne dotarcie do "
                f"{ASSET_LABELS.get(target_stage, target_stage)}: {timing['delay_label']} "
                f"(~{timing['delay_days']} dni)."
            )

    upstream_pressure = (weighted_active / weight_total) if weight_total > 0 else 0.0

    consistent_direction = None
    if directions and all(d == directions[0] for d in directions) and directions[0] != "neutralna":
        consistent_direction = directions[0]

    return {
        "target_stage": target_stage,
        "upstream_pressure_score": round(upstream_pressure, 3),
        "consistent_direction": consistent_direction,
        "alerts": alerts,
        "stage_details": stage_details,
    }


def adjust_confidence(base_ufnosc_procent: float, cascade_result: dict,
                       bonus_per_active_stage: float = 5.0, max_bonus: float = 20.0):
    """
    Podnosi Ufność, jeśli wcześniejsze ogniwa łańcucha pokazują SPÓJNY
    kierunek aktywności. Bonus per ogniwo skalowany jego ZMIERZONĄ WAGĄ
    PRZEPŁYWU (flow_weight, 0..1 - iloczyn wielkości ruchu i wolumenu
    względnego), NIE z góry przypisaną ważnością. NIGDY nie obniża
    Ufności.

    Zwraca: (nowa_ufnosc, zastosowany_bonus)
    """
    if cascade_result["consistent_direction"] is None:
        return base_ufnosc_procent, 0.0
    bonus = 0.0
    for details in cascade_result["stage_details"].values():
        if details["active"]:
            bonus += bonus_per_active_stage * details["flow_weight"]
    bonus = min(bonus, max_bonus)
    return min(base_ufnosc_procent + bonus, 100.0), round(bonus, 2)
