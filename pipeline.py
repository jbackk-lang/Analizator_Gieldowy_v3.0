"""
pipeline.py — TIMDR → ANALIZATOR GIEŁDOWY, moduł łączący
================================================================================
To jest moduł-łącznik dostarczony przez użytkownika ("MODUŁ POD
ANALIZATOR"), zachowany niemal 1:1 (tylko poprawione importy pod
faktyczną strukturę tego repo - oryginał odwoływał się do
`timdr_core_finance` i `analizator_gieldowy` jako gotowych modułów;
obie te implementacje zostały napisane od zera w ramach budowy v3,
patrz timdr_core_finance.py i analizator_gieldowy.py oraz README.md).

TimdrEngine liczy 7 niezależnych sygnałów z OHLCV i pakuje je w
TimdrPacket; AnalizatorGieldowy.przetworz_sygnaly() interpretuje cały
pakiet naraz (analizator "widzi pełny zestaw, nie pojedyncze strzały" -
komentarz z oryginalnego modułu użytkownika, zachowany jako trafny opis
architektury).
"""

from __future__ import annotations

import numpy as np

from timdr_core_finance import (
    trm, flow, twist, rhythm,
    anomalies, defect, resonance
)
from ringdown import ringdown_resonance
from analizator_gieldowy import AnalizatorGieldowy


class TrmSignal:
    def __init__(self, values):
        self.values = values


class FlowSignal:
    def __init__(self, values):
        self.values = values


class TwistSignal:
    def __init__(self, values):
        self.values = values


class RhythmSignal:
    def __init__(self, values):
        self.values = values


class AnomalySignal:
    def __init__(self, values):
        self.values = values


class DefectSignal:
    def __init__(self, values):
        self.values = values


class ResonanceSignal:
    def __init__(self, values):
        self.values = values


class RingdownSignal:
    """Wynik ringdown_resonance() (patrz ringdown.py) dla każdego bloku
    zdarzeń z defect() - NIE to samo co ResonanceSignal (licznik
    koincydencji) - to fizyczny rezonans: czy powrót ceny po skoku jest
    oscylacyjny (overreaction + korekta) czy monotoniczny (trwała
    przecena). `values` to lista dictów, patrz ringdown_resonance()."""
    def __init__(self, values):
        self.values = values


class PriceSignal:
    def __init__(self, values):
        self.values = values


class KhipuRegimeSignal:
    """Wynik KHIPURegimeSignal.score_series_indexed() (khipu_bottleneck.py) -
    seria zgodności reżimu między sąsiadującymi oknami świec, [-1, 1], oraz
    odpowiadające im bar-indeksy (do nanoszenia alertów na wykres ceny).
    OPCJONALNE - patrz TimdrPacket.khipu_regime niżej."""
    def __init__(self, values, bar_indices=None):
        self.values = values
        self.bar_indices = bar_indices if bar_indices is not None else np.arange(len(values))


class TimdrPacket:
    """
    Pakiet sygnałów TIMDR przekazywany do analizatora.
    Analizator widzi pełny zestaw, nie pojedyncze strzały.

    POPRAWKA względem oryginalnego modułu użytkownika: dodano pole
    `price` (SUROWA cena zamknięcia, nieprzetworzona). Oryginalny pakiet
    zawierał tylko `trm` (cena wygładzona medianą kroczącą, k=5) jako
    najbliższy substytut ceny - ale RSI i backtest liczone na
    WYGŁADZONEJ linii dają: (a) RSI systematycznie inny niż na dowolnej
    zwykłej platformie tradingowej (RSI zawsze liczone na surowej
    cenie zamknięcia - użytkownik porównujący z wykresem giełdowym
    zobaczyłby rozbieżność i słusznie by jej nie ufał), (b) backtest
    zaniżający realną zmienność zwrotów (bo trm już wygładził
    dzień-do-dnia szum, którego prawdziwy trader i tak by doświadczył).
    `trm` w pakiecie pozostaje - jest właściwym wejściem dla flow/twist/
    resonance (do tego był projektowany), ale RSI/backtest w
    analizator_gieldowy.py używają teraz `packet.price`.
    """

    def __init__(
        self,
        trm_signal,
        flow_signal,
        twist_signal,
        rhythm_signal,
        anomaly_signal,
        defect_signal,
        resonance_signal,
        price_signal,
        khipu_regime_signal=None,
        ringdown_signal=None,
    ):
        self.trm = trm_signal
        self.flow = flow_signal
        self.twist = twist_signal
        self.rhythm = rhythm_signal
        self.anomaly = anomaly_signal
        self.defect = defect_signal
        self.resonance = resonance_signal
        self.price = price_signal
        # OPCJONALNE - None gdy brak zdarzeń defect() (nic do analizy
        # ringdown) - patrz ringdown.py i TimdrEngine.compute_packet().
        self.ringdown = ringdown_signal
        # OPCJONALNE (patrz khipu_bottleneck.py) - None dopóki
        # KHIPU_BOTTLENECK_ENABLED=False (domyślnie), więc istniejący
        # kod czytający TimdrPacket nie widzi żadnej zmiany.
        self.khipu_regime = khipu_regime_signal


class TimdrEngine:
    """
    Silnik TIMDR: liczy sygnały z OHLCV i buduje pakiet dla analizatora.
    """

    def __init__(self, ohlcv):
        self.ohlcv = ohlcv

    def compute_packet(self) -> TimdrPacket:
        price = self.ohlcv["close"].values
        volume = self.ohlcv["volume"].values

        trm_price = trm(price, k=5)
        flow_price = flow(trm_price, window=5)
        twist_idx = twist(flow_price)
        rhythm_result = rhythm(volume)
        anomaly_idx = anomalies(price)
        defect_idx = defect(price)
        resonance_score, resonance_strong_idx = resonance(price)

        # Rezonans w sensie fizycznym (ringdown.py): dla każdego bloku
        # zdarzeń z defect() (skok ceny), czy powrót w stronę poziomu
        # sprzed skoku jest oscylacyjny (overreaction + korekta) czy
        # monotoniczny (trwała przecena). Bloki = ciągłe grupy indeksów w
        # defect_idx (te same, sąsiadujące skoki nie powinny dawać wielu
        # nakładających się analiz tego samego zdarzenia - ten sam wzorzec
        # co _detect_frequency_ringdown w TIMDR-Grid-Monitor).
        # pre_event_window=20 dopasowane do window= domyślnego w defect();
        # max_lookahead=40 (2x to okno) - świadomie ograniczone, żeby nie
        # złapać zupełnie innego, późniejszego skoku jako część tego samego
        # "powrotu"; wartość nie skalibrowana na realnych danych (patrz
        # README, sekcja Ograniczenia).
        ringdown_results = []
        if len(defect_idx):
            sorted_defects = sorted(set(int(i) for i in defect_idx))
            blocks = [sorted_defects[0]]
            prev = sorted_defects[0]
            for i in sorted_defects[1:]:
                if i != prev + 1:
                    blocks.append(i)
                prev = i
            bar_idx = np.arange(len(price), dtype=float)
            for event_idx in blocks:
                if event_idx == 0:
                    continue  # brak historii przed zdarzeniem - nie da się oszacować szumu
                res = ringdown_resonance(
                    bar_idx, price, event_idx,
                    pre_event_window=min(event_idx, 20),
                    max_lookahead=40,
                )
                ringdown_results.append({"event_idx": event_idx, **res})

        # OPCJONALNY sygnał KHIPU (krok integracji "features -> KHIPU
        # bottleneck -> dalsza logika TRM/FLOW/TWIST") - liczony TYLKO
        # gdy KHIPU_BOTTLENECK_ENABLED=True (domyślnie False, patrz
        # khipu_bottleneck.py). Failuje cicho (except Exception: None) -
        # to dodatkowy, opcjonalny sygnał, nie krytyczna ścieżka; awaria
        # tutaj NIE ma prawa wywrócić reszty pipeline'u.
        khipu_regime_signal = None
        try:
            from khipu_bottleneck import KHIPU_BOTTLENECK_ENABLED, KHIPURegimeSignal
            if KHIPU_BOTTLENECK_ENABLED:
                khipu_scores, khipu_bar_idx = KHIPURegimeSignal().score_series_indexed(self.ohlcv)
                khipu_regime_signal = KhipuRegimeSignal(khipu_scores, khipu_bar_idx)
        except Exception:
            khipu_regime_signal = None

        return TimdrPacket(
            trm_signal=TrmSignal(trm_price),
            flow_signal=FlowSignal(flow_price),
            twist_signal=TwistSignal(twist_idx),
            rhythm_signal=RhythmSignal(rhythm_result),
            anomaly_signal=AnomalySignal(anomaly_idx),
            defect_signal=DefectSignal(defect_idx),
            resonance_signal=ResonanceSignal(resonance_score),
            price_signal=PriceSignal(price),
            khipu_regime_signal=khipu_regime_signal,
            ringdown_signal=RingdownSignal(ringdown_results),
        )


def run_pipeline(ohlcv) -> dict:
    """
    Główny pipeline:
    - TIMDR liczy sygnały
    - analizator giełdowy je interpretuje
    """
    timdr = TimdrEngine(ohlcv)
    packet = timdr.compute_packet()
    analizator = AnalizatorGieldowy()
    wynik = analizator.przetworz_sygnaly(packet)
    return wynik
