"""_vendor_timdr_meta_dynamics_core.py -- ZWENDOROWANA (skopiowana 1:1, NIE
sibling-importowana) kopia rdzenia formalizmu TIMDR-META-DYNAMICS: MetaState,
MetaOperatorM, MetaMap, MetaTrigger, MetaTriggerResult.

POCHODZENIE: TIMDR-META-DYNAMICS (github.com/jbackk-lang/TIMDR-META-DYNAMICS),
core_meta/meta_state.py + core_meta/meta_operator_M.py + analysis/meta_map.py +
analysis/meta_trigger.py -- skopiowane bez zmian (ta sama matematyka, te same
progi classify_phase 0.1/1.0) dnia 2026-09-10.

DLACZEGO ZWENDOROWANE, NIE SIBLING-IMPORT: to repo ma byc SAMOWYSTARCZALNE --
dzialac po sklonowaniu WYLACZNIE tego jednego repo, bez wymogu, zeby
TIMDR-META-DYNAMICS lezalo obok jako folder-siostra na dysku (poprzedni
wzorzec, sys.path.insert() + import z sibling-folderu, dzialal tylko w
konkretnym ukladzie katalogow jednej maszyny -- nie po sklonowaniu samego
tego repo gdziekolwiek indziej, np. na innym komputerze, w CI, czy przez
kogos innego). Decyzja podjeta swiadomie na wyrazna prosbe: "repozytoria
kodu maja byc niezalezne od siebie".

UWAGA O UTRZYMANIU: to jest KOPIA, nie link. Jesli TIMDR-META-DYNAMICS
kiedys zmieni te klasy (np. inne progi classify_phase, nowe pola
MetaState), ta kopia NIE zaktualizuje sie automatycznie -- trzeba by
recznie zsynchronizowac. Ten kompromis (mozliwy przyszly rozjazd wersji)
jest swiadomie zaakceptowany w zamian za niezaleznosc tego repo od
struktury/obecnosci innego repo na dysku.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class MetaState:
    """Globalny stan pola TIMDR:
    Λ (Lambda) – struktura
    τ (tau)    – transformacja
    ρ (rho)    – anomalia
    J          – operator punktowy
    """

    Lambda: float
    tau: float
    rho: float
    J: float

    def delta(self, other: "MetaState") -> "MetaState":
        """Różnica (other - self), zgodnie z konwencją compute_meta_time(S_prev, S_now)."""
        return MetaState(
            Lambda=other.Lambda - self.Lambda,
            tau=other.tau - self.tau,
            rho=other.rho - self.rho,
            J=other.J - self.J,
        )


class MetaOperatorM:
    """Operator ewolucji pola: M = d/dt (Λ, τ, ρ, J)."""

    def compute(self, prev_state: MetaState, next_state: MetaState, dt: float) -> MetaState:
        if dt == 0:
            raise ValueError("dt musi być różne od zera")

        delta = prev_state.delta(next_state)

        return MetaState(
            Lambda=delta.Lambda / dt,
            tau=delta.tau / dt,
            rho=delta.rho / dt,
            J=delta.J / dt,
        )

    def magnitude(self, M_state: MetaState) -> float:
        """Suma wartości bezwzględnych składowych M."""
        return (
            abs(M_state.Lambda)
            + abs(M_state.tau)
            + abs(M_state.rho)
            + abs(M_state.J)
        )

    def classify_phase(self, M_state: MetaState) -> str:
        """Klasyfikacja fazy pola na podstawie wielkości M:
        - "stabilna"    magnitude < 0.1
        - "przejsciowa" 0.1 <= magnitude < 1.0
        - "krytyczna"   magnitude >= 1.0

        UWAGA: progi (0.1 / 1.0) są dobrane arbitralnie/heurystycznie w
        oryginalnym formalizmie TIMDR-META-DYNAMICS, nie skalibrowane na
        żadnych realnych danych tej konkretnej domeny -- patrz zastrzeżenia
        w module, który używa tej klasy.
        """
        magnitude = self.magnitude(M_state)

        if magnitude < 0.1:
            return "stabilna"
        elif magnitude < 1.0:
            return "przejsciowa"
        else:
            return "krytyczna"


class MetaMap:
    """Mapa zmian pola w czasie."""

    def __init__(self, meta_operator: MetaOperatorM):
        self.meta_operator = meta_operator

    def build(self, states: List[MetaState], M_series: List[MetaState]) -> dict:
        return {
            "states": states,
            "meta_operator": M_series,
        }

    def detect_transitions(self, M_series: List[MetaState]) -> List[str]:
        """Zwraca listę faz (stabilna/przejsciowa/krytyczna) per krok M-serii.

        UWAGA: to NIE jest "wykrywanie przejść" w sensie punktowym (nie
        zwraca tylko momentów ZMIANY fazy) -- zwraca fazę dla KAŻDEGO kroku.
        """
        transitions: List[str] = []

        for M in M_series:
            phase = self.meta_operator.classify_phase(M)
            transitions.append(phase)

        return transitions


class MetaTriggerResult:
    def __init__(self, triggered: bool = False, trigger_type: str = "none",
                 location: Optional[int] = None, message: str = ""):
        self.triggered = triggered
        self.trigger_type = trigger_type
        self.location = location
        self.message = message

    def as_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "type": self.trigger_type,
            "location": self.location,
            "message": self.message,
        }


class MetaTrigger:
    """Dispatcher nad sekwencją etykiet fazy (lista stringów, jedna faza na
    krok) -- zwraca pierwszy krok, w którym pole osiągnęło NAJPOWAZNIEJSZY z
    monitorowanych poziomów (kolejność wg `severity_order`), gdziekolwiek
    wystąpił w serii. Patrz TIMDR-META-DYNAMICS/analysis/meta_trigger.py po
    pełny docstring/uzasadnienie."""

    DEFAULT_SEVERITY_ORDER = ("krytyczna", "przejsciowa")

    def __init__(self, severity_order: Optional[Sequence[str]] = None):
        self.severity_order = tuple(severity_order) if severity_order else self.DEFAULT_SEVERITY_ORDER
        self.last_result = MetaTriggerResult()

    def analyze(self, phases: List[str]) -> MetaTriggerResult:
        for label in self.severity_order:
            for i, phase in enumerate(phases):
                if phase == label:
                    return self._set_result(
                        True, label, i,
                        f"Faza '{label}' osiągnięta pierwszy raz w kroku {i}."
                    )
        return self._set_result(
            False, "none", None,
            "Żaden z monitorowanych poziomów nie został osiągnięty w całej serii."
        )

    def _set_result(self, triggered, trigger_type, location, message):
        self.last_result = MetaTriggerResult(triggered, trigger_type, location, message)
        return self.last_result

    def get_last(self) -> MetaTriggerResult:
        return self.last_result


class FieldEvolution:
    """Ewolucja globalnego pola TIMDR: seria M-stanów liczona z serii S-stanów.
    Zwendorowane z TIMDR-META-DYNAMICS/models/evolution_field.py (patrz
    nagłówek modułu -- ten sam powód wendorowania)."""

    def __init__(self, meta_operator: MetaOperatorM):
        self.meta_operator = meta_operator

    def simulate(self, states: List[MetaState], dt: float) -> List[MetaState]:
        M_series: List[MetaState] = []

        for i in range(len(states) - 1):
            M = self.meta_operator.compute(states[i], states[i + 1], dt)
            M_series.append(M)

        return M_series


class MetaPredict:
    """Predykcja przyszłego stanu pola TIMDR: S_future = S_now + M_now * dt
    (prosta ekstrapolacja Eulera pierwszego rzędu, NIE model NWP/ML).
    Zwendorowane z TIMDR-META-DYNAMICS/analysis/meta_predict.py."""

    def predict_next(self, current_state: MetaState, M_state: MetaState, dt: float) -> MetaState:
        """Jeden krok ekstrapolacji: S(t+dt) = S(t) + M(t) * dt."""
        return MetaState(
            Lambda=current_state.Lambda + M_state.Lambda * dt,
            tau=current_state.tau + M_state.tau * dt,
            rho=current_state.rho + M_state.rho * dt,
            J=current_state.J + M_state.J * dt,
        )

    def simulate_future(
        self,
        current_state: MetaState,
        M_series: List[MetaState],
        dt: float,
        damping: float = 1.0,
    ) -> List[MetaState]:
        """Wielokrokowa ekstrapolacja z tłumieniem do kotwicy `current_state`:

            E_i = S_{i-1} + M_i * dt                 (czysta ekstrapolacja)
            w   = damping ** i
            S_i = w * E_i + (1 - w) * current_state    (tłumienie do kotwicy)

        `damping=1.0` (DOMYŚLNE) odtwarza czystą, nietłumioną ekstrapolację
        Eulera (zachowanie oryginału) -- patrz TIMDR-META-DYNAMICS/analysis/
        meta_predict.py po pełne uzasadnienie i zalecaną wartość ~0.85 dla
        realnego użycia na dłuższym horyzoncie."""
        states: List[MetaState] = [current_state]
        extrapolated = current_state

        for i, M in enumerate(M_series, start=1):
            extrapolated = self.predict_next(extrapolated, M, dt)
            w = damping ** i
            blended = MetaState(
                Lambda=w * extrapolated.Lambda + (1 - w) * current_state.Lambda,
                tau=w * extrapolated.tau + (1 - w) * current_state.tau,
                rho=w * extrapolated.rho + (1 - w) * current_state.rho,
                J=w * extrapolated.J + (1 - w) * current_state.J,
            )
            states.append(blended)

        return states
