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

from timdr_core_finance import (
    trm, flow, twist, rhythm,
    anomalies, defect, resonance
)
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


class PriceSignal:
    def __init__(self, values):
        self.values = values


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
    ):
        self.trm = trm_signal
        self.flow = flow_signal
        self.twist = twist_signal
        self.rhythm = rhythm_signal
        self.anomaly = anomaly_signal
        self.defect = defect_signal
        self.resonance = resonance_signal
        self.price = price_signal


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

        return TimdrPacket(
            trm_signal=TrmSignal(trm_price),
            flow_signal=FlowSignal(flow_price),
            twist_signal=TwistSignal(twist_idx),
            rhythm_signal=RhythmSignal(rhythm_result),
            anomaly_signal=AnomalySignal(anomaly_idx),
            defect_signal=DefectSignal(defect_idx),
            resonance_signal=ResonanceSignal(resonance_score),
            price_signal=PriceSignal(price),
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
