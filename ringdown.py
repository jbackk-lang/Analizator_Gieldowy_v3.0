"""
ringdown.py — czy powrót ceny do poziomu sprzed skoku jest REZONANSOWY (oscylacyjny)
================================================================================
`timdr_core_finance.py::resonance()` to licznik koincydencji (ile z trzech
niezależnych sprawdzeń - anomalia/defekt/skręt - zgadza się naraz w tej
samej próbce) - nazwa pożyczona z fizyki, ale mechanizm inny (patrz jego
docstring). Ten moduł liczy coś, co faktycznie odpowiada fizycznemu
rezonansowi: po `defekcie` (nagłym skoku ceny, patrz `defect()`) cena
wraca (albo nie) w stronę poziomu sprzed zaburzenia - pytanie brzmi, CZY
ten powrót jest OSCYLACYJNY (cena "przewahnęła" przez poziom odniesienia w
obie strony, typowy wzorzec overreaction + korekty/mean-reversion) czy
MONOTONICZNY (permanentne przecenienie/przewartościowanie - nowy poziom
się utrzymuje, bez odbicia w drugą stronę).

Port 1:1 (bez zmian w matematyce) z jbackk-lang/universal-state-analyzer
(`timdr_core/ringdown.py`) - tam metoda jest zweryfikowana numerycznie na
tłumionym oscylatorze o znanej częstotliwości/stałej czasowej (odzyskana
częstotliwość i tłumienie zgodne z teorią w granicach szumu, historia
znalezionych i naprawionych błędów w tamtym README) - te gwarancje
przenoszą się tutaj bez zmian, bo matematyka jest identyczna.

RÓŻNICA względem sieci energetycznej (TIMDR-Grid-Monitor): tam istnieje
REALNY, bezwzględny punkt odniesienia (f_nominal=50Hz, norma EN 50160).
Cena akcji/waluty/surowca NIE MA takiego stałego punktu odniesienia - więc
tutaj `baseline` jest domyślnie liczone jako średnia z okna PRZED
zdarzeniem (tryb domyślny funkcji), tak jak w domenach EKG/bio-sygnałów.

Metoda: histereza Schmitta na wykrywaniu stanu (HIGH/LOW potwierdzane
dopiero powyżej progu szumu `noise_floor_factor * std(cena przed
zdarzeniem)`) + interpolowane przejścia przez poziom odniesienia między
potwierdzonymi zmianami stanu + logarithmic decrement między szczytami
tego samego znaku, częstotliwość liczona z MEDIANY (nie średniej) odstępów
między przejściami - standardowa technika inżynierska, NIE dopasowanie
nieliniowe (za kruche numerycznie na krótkich, zaszumionych oknach).
"""
from __future__ import annotations

import numpy as np


def ringdown_resonance(
    t,
    s,
    event_idx: int,
    baseline: float | None = None,
    pre_event_window: int = 10,
    max_lookahead: int | None = None,
    noise_floor_factor: float = 3.0,
) -> dict:
    """Analizuje powrót `s` do poziomu odniesienia PO indeksie `event_idx`.
    Patrz docstring modułu i universal-state-analyzer/timdr_core/ringdown.py
    po pełne uzasadnienie parametrów i metody.

    Zwraca dict: baseline, noise_floor, is_oscillatory, n_crossings,
    n_peaks_used, period_s, frequency_hz, log_decrement, damping_ratio,
    peak_times, peak_amplitudes.

    UWAGA nazewnictwo parametrów: `t`/`period_s`/`frequency_hz` to
    dziedzictwo wspólnego API z pozostałymi portami tej funkcji (czas w
    sekundach). Tutaj `t` to po prostu indeks bara (1 jednostka = 1 bar,
    nie sekunda) - `period_s` wychodzi więc w BARACH, nie w sekundach;
    `frequency_hz` analogicznie w cyklach/bar. Interpretuj jako "okres w
    barach" / "cykle na bar", nie dosłowną częstotliwość fizyczną.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    n = len(s)
    if n == 0 or not (0 <= event_idx < n):
        raise ValueError(f"event_idx={event_idx} poza zakresem serii o długości {n}")

    pre_start = max(0, event_idx - pre_event_window)
    pre_samples = s[pre_start:event_idx]

    if baseline is None:
        baseline = float(np.mean(pre_samples)) if len(pre_samples) else float(s[event_idx])

    noise_std = float(np.std(pre_samples)) if len(pre_samples) >= 2 else 0.0
    noise_floor = noise_floor_factor * noise_std

    end = n if max_lookahead is None else min(n, event_idx + max_lookahead)
    t_post = t[event_idx:end]
    d = s[event_idx:end] - baseline

    result: dict = {
        "baseline": float(baseline),
        "noise_floor": float(noise_floor),
        "is_oscillatory": False,
        "n_crossings": 0,
        "n_peaks_used": 0,
        "period_s": None,
        "frequency_hz": None,
        "log_decrement": None,
        "damping_ratio": None,
        "peak_times": [],
        "peak_amplitudes": [],
    }

    if len(d) < 3:
        return result

    # --- histereza Schmitta NA STANIE, nie doklejona po fakcie do już
    # policzonych szczytów: stan HIGH/LOW jest "potwierdzany" dopiero gdy
    # |d| > noise_floor; przejście zapisujemy dopiero gdy stan faktycznie
    # PRZEŁĄCZY się na przeciwny, potwierdzony stan. Próbki w paśmie
    # [-noise_floor, noise_floor] nigdy nie potwierdzają nowego stanu, więc
    # nie generują fałszywych przejść — to standardowy komparator z
    # histerezą (Schmitt trigger) zastosowany wprost do detekcji stanu.
    band = noise_floor
    confirmed_idx: list[int] = []
    state = 0
    for i in range(len(d)):
        if d[i] > band:
            new_state = 1
        elif d[i] < -band:
            new_state = -1
        else:
            continue
        if new_state != state:
            confirmed_idx.append(i)
            state = new_state

    # przejścia = surowy moment zmiany znaku d (interpolowany), leżący
    # MIĘDZY dwoma kolejnymi potwierdzonymi punktami przeciwnego stanu.
    crossing_times: list[float] = []
    for prev_i, cur_i in zip(confirmed_idx[:-1], confirmed_idx[1:]):
        found = None
        for k in range(prev_i, cur_i):
            if d[k] == 0 or (d[k] > 0) != (d[k + 1] > 0):
                frac = 0.0 if d[k] == 0 else -d[k] / (d[k + 1] - d[k])
                found = float(t_post[k] + frac * (t_post[k + 1] - t_post[k]))
                break
        if found is None:
            found = float((t_post[prev_i] + t_post[cur_i]) / 2.0)
        crossing_times.append(found)

    # szczyty: lokalne ekstremum |d| w segmentach ograniczonych
    # potwierdzonymi punktami stanu (nie surowymi przejściami) — każdy
    # segment z definicji zawiera punkt przekraczający próg szumu, więc
    # każdy zwrócony szczyt jest już "zaufany" (>= noise_floor).
    # (deduplikacja: gdy sygnał PRZEKRACZA próg już w pierwszej/ostatniej
    # próbce okna, confirmed_idx[0]/[-1] pokrywa się z granicą 0/len(d)-1 -
    # bez `set()` dawałoby to zdegenerowany, jednopunktowy segment i
    # podwójnie liczony ten sam fizyczny szczyt.)
    bounds_idx = sorted(set([0] + confirmed_idx + [len(d) - 1]))
    peak_times: list[float] = []
    peak_amps: list[float] = []
    for a, b in zip(bounds_idx[:-1], bounds_idx[1:]):
        if b < a:
            continue
        seg = d[a:b + 1]
        local_idx = int(np.argmax(np.abs(seg)))
        peak_times.append(float(t_post[a + local_idx]))
        peak_amps.append(float(seg[local_idx]))

    used_crossings = crossing_times

    result["n_crossings"] = len(used_crossings)
    result["n_peaks_used"] = len(peak_amps)
    result["peak_times"] = peak_times
    result["peak_amplitudes"] = peak_amps

    if len(used_crossings) >= 2 and len(peak_amps) >= 2:
        result["is_oscillatory"] = True

        # mediana, nie średnia: ostatni potwierdzony półokres bywa tuż nad
        # progiem szumu (niepewny dokładny czas) - mediana jest odporna na
        # taki pojedynczy zanieczyszczony półokres.
        crossing_diffs = np.diff(used_crossings)
        if len(crossing_diffs) and np.median(crossing_diffs) > 0:
            period = 2.0 * float(np.median(crossing_diffs))
            result["period_s"] = period
            result["frequency_hz"] = 1.0 / period

        log_ratios = []
        for i in range(len(peak_amps) - 2):
            a, b = peak_amps[i], peak_amps[i + 2]
            if np.sign(a) == np.sign(b) and a != 0 and b != 0:
                ratio = abs(a) / abs(b)
                if ratio > 0:
                    log_ratios.append(np.log(ratio))
        if log_ratios:
            delta = float(np.mean(log_ratios))
            result["log_decrement"] = delta
            result["damping_ratio"] = float(delta / np.sqrt(4 * np.pi ** 2 + delta ** 2))

    return result
