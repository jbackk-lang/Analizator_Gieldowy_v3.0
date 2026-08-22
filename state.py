"""
state.py — trwały stan na dysku: ostatni werdykt (wykrywanie zmiany
stanu między uruchomieniami) + log predykcja/realizacja (samo-uczenie).
================================================================================
Dwie funkcje, obie bezpośrednio zaadresowane wg lekcji ze Synoptyka
(patrz README.md, "Lekcje z Synoptyka zastosowane w v3"):

  Lekcja #5 (EV / jump detection między uruchomieniami): stan MUSI być
  na DYSKU, nie tylko w pamięci procesu - inaczej restart procesu
  cichnie resetuje historię i wykrywacz zmiany stanu prawie nigdy nie
  zadziała (tylko gdy dwa uruchomienia nastąpią jedno po drugim bez
  restartu). Tutaj: JEDEN plik JSON na ticker w `data/state/`.

  Lekcja #6 (samo-uczenie z par predykcja/rzeczywistość): log predykcji
  z lead_time, dopasowanie do późniejszej rzeczywistej ceny, bias/MAE
  per lead_time, korekta tylko przy n>=5 próbek, zawsze widoczna plakietka
  (nawet czerwona - "za mało danych" to też informacja, nie cisza).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def _safe_ticker(ticker: str) -> str:
    """Sanityzacja nazwy tickera do bezpiecznej nazwy pliku (np. EURPLN=X
    zawiera znak '=', niedozwolony/kłopotliwy w niektórych systemach
    plików) - zamieniamy wszystko poza alfanumerykami na '_'."""
    return "".join(c if c.isalnum() else "_" for c in ticker)


class StateStore:
    """Ostatni werdykt per ticker, trwały na dysku (JSON). Pozwala
    wykryć, czy klasyfikacja (Emergencja/Ufność) ZMIENIŁA SIĘ od
    poprzedniego uruchomienia - użyteczne jako "co nowego" alert,
    bez potrzeby ręcznego porównywania historii przez użytkownika."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, ticker: str) -> str:
        return os.path.join(self.base_dir, f"{_safe_ticker(ticker)}.json")

    def load_last(self, ticker: str) -> dict | None:
        path = self._path(ticker)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # NIGDY bare except:pass (lekcja #4 ze Synoptyka) - błąd
            # odczytu stanu jest widoczny, nie cichnie do "brak historii"
            print(f"[state.py] UWAGA: nie udało się odczytać stanu dla '{ticker}': {e}")
            return None

    def save(self, ticker: str, verdict: dict) -> None:
        path = self._path(ticker)
        payload = dict(verdict)
        payload["_saved_at"] = datetime.now(timezone.utc).isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def compare_and_update(self, ticker: str, new_verdict: dict, confidence_jump_threshold: float = 15.0) -> dict:
        """
        Zwraca dict opisujący zmianę względem poprzedniego zapisanego
        werdyktu, PO CZYM zapisuje nowy werdykt jako najnowszy (kolejność
        celowa: najpierw porównanie, dopiero potem nadpisanie).

        Zwraca: {
          "changed": bool,          # etykieta emergencja się zmieniła
          "confidence_jump": bool,  # Ufność skoczyła o >= threshold p.p.
          "previous": dict | None,  # poprzedni werdykt (albo None za pierwszym razem)
        }
        """
        previous = self.load_last(ticker)
        changed = False
        confidence_jump = False
        if previous is not None:
            changed = previous.get("emergencja_label") != new_verdict.get("emergencja_label")
            prev_conf = previous.get("ufnosc_procent")
            new_conf = new_verdict.get("ufnosc_procent")
            if prev_conf is not None and new_conf is not None:
                confidence_jump = abs(new_conf - prev_conf) >= confidence_jump_threshold

        self.save(ticker, new_verdict)
        return {"changed": changed, "confidence_jump": confidence_jump, "previous": previous}

    def clear(self, ticker: str) -> bool:
        """Ręczne czyszczenie stanu dla tickera (lekcja #5: użytkownik
        MUSI mieć kontrolę nad czyszczeniem, inaczej stare wpisy z
        testów/innych sesji zalegają w nieskończoność).

        Zwraca False (bez wyjątku) zarówno gdy nie było czego czyścić,
        jak i gdy system plików odmówił usunięcia (np. zablokowany
        plik, prawa dostępu) - błąd jest WIDOCZNY (wypisany), ale nie
        wywala całego żądania API (lekcja #4: widoczne, nie ciche;
        ale też nie fatalne tam, gdzie nie musi być)."""
        path = self._path(ticker)
        if not os.path.exists(path):
            return False
        try:
            os.remove(path)
            return True
        except OSError as e:
            print(f"[state.py] UWAGA: nie udało się usunąć stanu dla '{ticker}': {e}")
            return False


class PredictionLog:
    """
    Log predykcja/realizacja per ticker (JSONL - jeden wpis na linię,
    łatwe dopisywanie bez wczytywania całości). Dwa stany wpisu:
    "pending" (predykcja jeszcze niepotwierdzona - nie minął lead_time)
    i "confirmed" (dopasowano do rzeczywistej, późniejszej ceny).
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, ticker: str) -> str:
        return os.path.join(self.base_dir, f"{_safe_ticker(ticker)}_predictions.jsonl")

    def _read_all(self, ticker: str) -> list[dict]:
        path = self._path(ticker)
        if not os.path.exists(path):
            return []
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    # widoczny błąd, nie cichy skip (lekcja #4)
                    print(f"[state.py] UWAGA: uszkodzona linia {line_no} w {path}: {e}")
        return entries

    def _write_all(self, ticker: str, entries: list[dict]) -> None:
        path = self._path(ticker)
        with open(path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def log_prediction(self, ticker: str, bar_index: int, bar_timestamp: str,
                        direction: str, confidence: float, lead_time_bars: int) -> None:
        """Zapisuje nową, jeszcze niepotwierdzoną predykcję.
        direction: "up" | "down" | "flat" - oczekiwany kierunek ceny za
        `lead_time_bars` barów od `bar_index`."""
        entries = self._read_all(ticker)
        entries.append({
            "status": "pending",
            "bar_index": bar_index,
            "bar_timestamp": bar_timestamp,
            "direction": direction,
            "confidence": confidence,
            "lead_time_bars": lead_time_bars,
            "target_bar_index": bar_index + lead_time_bars,
        })
        self._write_all(ticker, entries)

    def confirm_due_predictions(self, ticker: str, price_series) -> int:
        """
        Sprawdza wszystkie "pending" predykcje, których target_bar_index
        mieści się już w dostępnym `price_series` (nowe dane od czasu
        zalogowania predykcji) - oblicza rzeczywisty kierunek/błąd i
        przenosi je do stanu "confirmed". Zwraca liczbę nowo potwierdzonych.
        """
        entries = self._read_all(ticker)
        n_prices = len(price_series)
        confirmed_count = 0

        for e in entries:
            if e["status"] != "pending":
                continue
            target_idx = e["target_bar_index"]
            if target_idx >= n_prices:
                continue  # jeszcze nie ma tylu nowych danych

            start_idx = e["bar_index"]
            if start_idx >= n_prices:
                continue

            start_price = price_series[start_idx]
            target_price = price_series[target_idx]
            realized_return = (target_price - start_price) / start_price if start_price else 0.0
            realized_direction = "up" if realized_return > 0 else ("down" if realized_return < 0 else "flat")

            predicted_sign = {"up": 1, "down": -1, "flat": 0}[e["direction"]]
            realized_sign = {"up": 1, "down": -1, "flat": 0}[realized_direction]
            error = predicted_sign - realized_sign  # >0: przewidziano zbyt optymistycznie, <0: zbyt pesymistycznie

            e["status"] = "confirmed"
            e["realized_direction"] = realized_direction
            e["realized_return"] = realized_return
            e["error"] = error
            e["correct"] = (predicted_sign == realized_sign)
            confirmed_count += 1

        if confirmed_count:
            self._write_all(ticker, entries)
        return confirmed_count

    def bias_by_lead_time(self, ticker: str, min_samples: int = 5) -> dict:
        """
        Grupuje potwierdzone predykcje po lead_time_bars, liczy bias
        (średni błąd) i MAE. Korekta ma sens (n >= min_samples) tylko
        dla lead_time'ów z wystarczającą liczbą potwierdzonych próbek -
        w przeciwnym razie zwraca surowe liczby, ale `apply_correction:
        False`, żeby wołający NIE stosował korekty na podstawie szumu.

        Zwraca: {lead_time_bars: {"n": int, "bias": float, "mae": float,
                                    "accuracy": float, "apply_correction": bool,
                                    "badge": "🔴"|"🟠"|"🟢"}}
        """
        entries = [e for e in self._read_all(ticker) if e["status"] == "confirmed"]
        by_lead: dict[int, list[dict]] = {}
        for e in entries:
            by_lead.setdefault(e["lead_time_bars"], []).append(e)

        result = {}
        for lead, group in by_lead.items():
            n = len(group)
            errors = [g["error"] for g in group]
            bias = sum(errors) / n
            mae = sum(abs(e) for e in errors) / n
            accuracy = sum(1 for g in group if g["correct"]) / n
            if n < min_samples:
                badge = "🔴"
            elif n < 15:
                badge = "🟠"
            else:
                badge = "🟢"
            result[lead] = {
                "n": n, "bias": bias, "mae": mae, "accuracy": accuracy,
                "apply_correction": n >= min_samples, "badge": badge,
            }
        return result

    def clear(self, ticker: str) -> bool:
        path = self._path(ticker)
        if not os.path.exists(path):
            return False
        try:
            os.remove(path)
            return True
        except OSError as e:
            print(f"[state.py] UWAGA: nie udało się usunąć logu predykcji dla '{ticker}': {e}")
            return False
