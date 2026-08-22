"""
khipu_bottleneck.py — opcjonalny moduł "KHIPU regime bottleneck"
================================================================================
Wpięcie State9Bottleneck (z jbackk-lang/KHIPU-NEURAL) w miejsce
features/embedding -> dalsza logika TRM/FLOW/TWIST, zgodnie z planem
integracji dostarczonym przez użytkownika ("Wyznacz miejsce bottlenecku
w pipeline Analizatora 3.0" - 6 kroków). Realizuje kroki 1-4 i 6 w pełni;
krok 5 (kalibracja/trening) jest zaimplementowany, ale z jawnym
zastrzeżeniem, że etykiety "ta sama faza" są tu HEURYSTYCZNYM bootstrapem
z istniejących sygnałów TRM/TWIST, NIE prawdziwą, niezależną etykietą
rynkową (nikt takiej nie dostarczył) - patrz MarketPhaseDataset niżej.

UWAGA NA GRANICE (krok 6 planu, i główny wniosek z całego KHIPU-NEURAL,
patrz jego README "Wniosek końcowy: gdzie to się nadaje, a gdzie nie"):
State9Bottleneck W ZWERYFIKOWANYCH TESTACH pomaga na zadaniach
KATEGORIALNYCH ("czy te dwa stany należą do tej samej klasy/fazy"), i
WYRAŹNIE SZKODZI na zadaniach wymagających precyzyjnej wartości ciągłej
(regresja ceny, dokładnej odległości, wielkości wolumenu). Dlatego:

  - Ten moduł NIGDY nie zwraca ceny, zmiany ceny, ani żadnej innej
    ciągłej wielkości - tylko DYSKRETNY sygnał zgodności reżimu
    (regime_agreement_score, [-1, 1], oraz twardy kod 9-osiowy).
  - KHIPU_BOTTLENECK_ENABLED (domyślnie False) - "wyłącznik" na module.
    Domyślnie WYŁĄCZONY: dopóki ktoś świadomie go nie włączy,
    pipeline.py/analizator_gieldowy.py działają DOKŁADNIE tak jak przed
    tym modułem (zero zmiany istniejącego zachowania).
  - Jeśli używasz tego gdziekolwiek do przewidywania ciągłej wartości
    (cena, wolumen, dokładna odległość) - to jest DOKŁADNIE to
    zastosowanie, przed którym ostrzega KHIPU-NEURAL. Nie rób tego.

Pochodzenie kodu: State9Bottleneck/balance_correct poniżej to WIERNY
PORT (bez zmian w matematyce) z jbackk-lang/KHIPU-NEURAL
(khipu_neural/quantize.py) - tam gradienty są zweryfikowane numerycznie
do ~1e-11 (tests/test_gradients_quantize.py). Port nie zmienia tej
matematyki, więc te gwarancje się przenoszą; nie zostały tu ponownie
wyprowadzone od zera.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# "Wyłącznik" na cały moduł (patrz ostrzeżenie w docstringu wyżej) —
# krok 6 planu integracji: łatwe wyłączenie, jeśli okaże się, że to
# szkodzi na jakimś fragmencie pipeline'u.
# ---------------------------------------------------------------------
KHIPU_BOTTLENECK_ENABLED = False

N_AXES = 9  # jak w State9/F4-RED (KHIPU) i KHIPU-NEURAL


# ---------------------------------------------------------------------
# KROK 3: State9Bottleneck — wierny port z KHIPU-NEURAL (quantize.py)
# ---------------------------------------------------------------------

def balance_correct(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Koryguje wektor +/-1 `q` tak, żeby |sum(q)| <= 1 (warunek F4-RED:
    |N+ - N-| <= 1). Port 1:1 z khipu_neural/quantize.py — patrz tam po
    pełne uzasadnienie i weryfikację gradientową."""
    q = q.copy()
    total = int(q.sum())
    while abs(total) > 1:
        majority_sign = 1 if total > 0 else -1
        idx = np.where(q == majority_sign)[0]
        pick = idx[np.argmin(np.abs(t[idx]))]
        q[pick] = -majority_sign
        total = int(q.sum())
    return q


class State9Bottleneck:
    """x (d,) -> proj = Wq@x + bq (9,) -> t = tanh(proj) -> q = sign(t)
    -> q_corrected = balance_correct(q, t). Port 1:1 z KHIPU-NEURAL.

    Domyślnie (bez kalibracji, patrz calibrate() niżej) Wq/bq są
    ustawione jako DETERMINISTYCZNA, ustalona projekcja (nie losowa, nie
    trenowana) — 9 z góry zdefiniowanych, interpretowalnych kombinacji
    liniowych embeddingu (patrz _default_projection), żeby "twardy
    filtr" (krok 5, opcja B planu: "jeśli nie chcesz trenować") dawał
    powtarzalny, sensowny wynik od razu, bez losowej inicjalizacji."""

    def __init__(self, d_in: int, rng: np.random.Generator | None = None,
                 Wq: np.ndarray | None = None, bq: np.ndarray | None = None):
        if Wq is not None:
            self.Wq = np.asarray(Wq, dtype=float)
            self.bq = np.asarray(bq if bq is not None else np.zeros(N_AXES), dtype=float)
        elif rng is not None:
            scale = 1.0 / np.sqrt(d_in)
            self.Wq = rng.normal(0, scale, size=(N_AXES, d_in))
            self.bq = np.zeros(N_AXES)
        else:
            self.Wq = _default_projection(d_in)
            self.bq = np.zeros(N_AXES)
        self._cache = {}

    def params(self):
        return {"Wq": self.Wq, "bq": self.bq}

    def forward(self, x: np.ndarray) -> np.ndarray:
        proj = self.Wq @ x + self.bq
        t = np.tanh(proj)
        q = np.sign(t)
        q[q == 0] = 1.0
        q = balance_correct(q, t)
        self._cache = {"x": x, "proj": proj, "t": t}
        return q

    def backward(self, dL_dq: np.ndarray) -> tuple[np.ndarray, dict]:
        """STE, identyczna jak w KHIPU-NEURAL — potrzebne tylko jeśli
        wywołasz calibrate() (krok 5), nie do zwykłego użycia jako
        twardy filtr."""
        t = self._cache["t"]
        x = self._cache["x"]
        dL_dt = dL_dq
        dL_dproj = dL_dt * (1.0 - t ** 2)
        dWq = np.outer(dL_dproj, x)
        dbq = dL_dproj
        dL_dx = self.Wq.T @ dL_dproj
        return dL_dx, {"Wq": dWq, "bq": dbq}


def _default_projection(d_in: int) -> np.ndarray:
    """Ustalona (nielosowa) projekcja d_in -> 9 osi, używana gdy
    State9Bottleneck jest tworzony bez kalibracji (krok 5, opcja
    "twardy filtr"). Każda oś to prosta, interpretowalna kombinacja
    wejścia: naprzemienne +1/-1 wagi z przesuniętą fazą per oś (jak
    dyskretna transformata Walsha-Hadamarda obcięta do 9 wierszy) —
    zapewnia, że osie NIE są identyczne (różne wzorce +/-1), a jest to
    w pełni deterministyczne i odtwarzalne (bez zależności od ziarna
    losowości), co ma znaczenie dla nietrenowanego "twardego filtra":
    dwa uruchomienia na tych samych danych muszą dać ten sam wynik."""
    W = np.zeros((N_AXES, d_in))
    for k in range(N_AXES):
        for j in range(d_in):
            # wzorzec +/-1 zależny od (k, j) - wystarczy do zróżnicowania
            # osi bez uczenia; skala 1/sqrt(d_in) jak w losowej wersji
            W[k, j] = 1.0 if ((k + 1) * (j + 1)) % (k + 2) < (k + 2) / 2 else -1.0
    return W / np.sqrt(d_in)


# ---------------------------------------------------------------------
# KROK 2: embedding giełdowy pod State9Bottleneck
# ---------------------------------------------------------------------

D_EMBED = 8


def make_embedding(window: "pd.DataFrame") -> np.ndarray:
    """candle_window (OHLCV, kolumny open/high/low/close/volume, >= 2
    wiersze) -> wektor ciągły (D_EMBED,). Cechy zbudowane z istniejących
    bloków TIMDR (trm/flow/twist z timdr_core_finance.py) + podstawowe
    cechy świecowe - zamierzenie NIE nowa, niezależna definicja
    wskaźników, tylko przepakowanie tego, co repo już liczy, do jednego
    ciągłego wektora:

      [0] znormalizowany zwrot na oknie: (close[-1]-close[0])/close[0]
      [1] lokalna zmienność zwrotów (std dziennych zwrotów w oknie)
      [2] z-score wolumenu ostatniej świecy względem okna
      [3] średnie nachylenie TRM (k=5) w oknie, znormalizowane ceną
      [4] ostatnia wartość FLOW (znormalizowana ceną)
      [5] gęstość TWIST w oknie (liczba skrętów / długość okna)
      [6] średni stosunek korpusu świecy do zasięgu: (close-open)/(high-low)
      [7] średni względny zasięg świecy: (high-low)/close

    Krótkie okna (< 2*twist_window+1) dają [5]=0 (nie da się policzyć
    twist na zbyt krótkiej historii) zamiast wyjątku - embedding ma być
    zawsze zdefiniowany, nawet dla okien na granicy dostępnych danych."""
    from timdr_core_finance import trm, flow, twist

    close = window["close"].to_numpy(dtype=float)
    open_ = window["open"].to_numpy(dtype=float)
    high = window["high"].to_numpy(dtype=float)
    low = window["low"].to_numpy(dtype=float)
    volume = window["volume"].to_numpy(dtype=float)
    n = len(close)
    eps = 1e-9

    ret_total = (close[-1] - close[0]) / (close[0] + eps)

    if n >= 2:
        returns = np.diff(close) / (close[:-1] + eps)
        vol_local = float(np.std(returns))
    else:
        vol_local = 0.0

    vol_mean, vol_std = volume.mean(), volume.std()
    vol_z = float((volume[-1] - vol_mean) / (vol_std + eps))

    k = min(5, max(1, n // 2))
    trm_price = trm(close, k=k)
    if n >= 2:
        trm_slope = float(np.mean(np.diff(trm_price)) / (np.mean(close) + eps))
    else:
        trm_slope = 0.0

    flow_window = min(5, max(1, n // 2))
    flow_price = flow(trm_price, window=flow_window)
    flow_last = float(flow_price[-1] / (np.mean(close) + eps)) if n else 0.0

    twist_window = min(5, max(1, n // 4))
    if n >= 2 * twist_window + 1:
        twist_idx = twist(flow_price, window=twist_window)
        twist_density = float(len(twist_idx) / n)
    else:
        twist_density = 0.0

    body_ratio = float(np.mean((close - open_) / (high - low + eps)))
    range_ratio = float(np.mean((high - low) / (close + eps)))

    return np.array([
        ret_total, vol_local, vol_z, trm_slope,
        flow_last, twist_density, body_ratio, range_ratio,
    ], dtype=float)


# ---------------------------------------------------------------------
# KROK 4: reguła GIPU jako sygnał zgodności reżimu
# ---------------------------------------------------------------------

def regime_agreement(q_i: np.ndarray, q_j: np.ndarray) -> np.ndarray:
    """Iloczyn per-oś q_i ⊙ q_j (reguła GIPU: "ten sam znak na osi ->
    zgodność") - wektor (9,) w {-1,+1}^9. To NIE jest wartość ciągła do
    regresji - to per-oś, dyskretna zgodność/niezgodność (patrz
    ograniczenia w docstringu modułu)."""
    return q_i * q_j


def regime_agreement_score(q_i: np.ndarray, q_j: np.ndarray) -> float:
    """Skalarne podsumowanie regime_agreement: średnia po osiach, w
    [-1, 1]. +1 = identyczny kod na wszystkich 9 osiach (bardzo
    prawdopodobnie ta sama faza/reżim), -1 = maksymalnie przeciwny.
    Analogon `agree = dot(q_i, q_i+1)/9` z KHIPU-NEURAL
    (models.py::KHIPUResonanceNet)."""
    return float(np.mean(regime_agreement(q_i, q_j)))


class KHIPURegimeSignal:
    """Wygodne opakowanie: embedding dwóch sąsiednich okien -> kody
    State9 -> zgodność reżimu. To jest NOWY, OPCJONALNY sygnał - patrz
    pipeline.py: dopisywany do TimdrPacket jako `khipu_regime` TYLKO
    gdy KHIPU_BOTTLENECK_ENABLED=True, nie zastępuje istniejącego
    `resonance` (który ma inną, już ugruntowaną definicję - zgodność
    MIĘDZY RÓŻNYMI wskaźnikami w tej samej chwili, nie MIĘDZY oknami
    czasowymi - patrz timdr_core_finance.py::resonance dla różnicy)."""

    def __init__(self, bottleneck: State9Bottleneck | None = None, d_embed: int = D_EMBED):
        self.bottleneck = bottleneck or State9Bottleneck(d_in=d_embed)

    def score_series(self, ohlcv: "pd.DataFrame", window_size: int = 20, step: int = 1) -> np.ndarray:
        """Dla serii OHLCV liczy regime_agreement_score między KAŻDĄ
        parą kolejnych okien (window_size świec), krok co `step` -
        zwraca tablicę (n_windows - 1,) - analogon "resonance między
        sąsiadującymi tokenami" z KHIPU-NEURAL, tu: między sąsiadującymi
        oknami świec zamiast pojedynczymi tokenami."""
        n = len(ohlcv)
        starts = list(range(0, n - window_size + 1, step))
        if len(starts) < 2:
            return np.array([], dtype=float)

        codes = []
        for s in starts:
            emb = make_embedding(ohlcv.iloc[s:s + window_size])
            codes.append(self.bottleneck.forward(emb))

        scores = np.array([
            regime_agreement_score(codes[i], codes[i + 1])
            for i in range(len(codes) - 1)
        ], dtype=float)
        return scores


# ---------------------------------------------------------------------
# KROK 5: kalibracja (opcjonalna) — HEURYSTYCZNY bootstrap etykiet
# ---------------------------------------------------------------------

class MarketPhaseDataset:
    """Analogon khipu_neural.data.ResonanceDataset, ale zamiast sztucznych
    kategorii — PARY OKIEN z prawdziwego OHLCV, z etykietą "ta sama
    faza" zbudowaną HEURYSTYCZNIE z istniejących sygnałów TIMDR (znak
    FLOW + obecność TWIST w oknie). TO NIE JEST NIEZALEŻNA, zweryfikowana
    etykieta rynkowa - to bootstrap z tego, co repo już liczy, żeby w
    ogóle było czym kalibrować State9Bottleneck (krok 5 planu). Jeśli
    masz lepszą definicję "fazy rynku" (np. z zewnętrznej klasyfikacji
    reżimu zmienności), podmień `_phase_label()` - reszta (calibrate())
    działa na dowolnej binarnej etykiecie pary okien."""

    def __init__(self, ohlcv: "pd.DataFrame", window_size: int = 20, step: int = 5):
        self.ohlcv = ohlcv
        self.window_size = window_size
        self.step = step

    def _phase_label(self, window: "pd.DataFrame") -> int:
        from timdr_core_finance import trm, flow
        close = window["close"].to_numpy(dtype=float)
        k = min(5, max(1, len(close) // 2))
        trm_price = trm(close, k=k)
        flow_price = flow(trm_price, window=min(5, max(1, len(close) // 2)))
        flow_sign = 1 if float(np.mean(flow_price)) >= 0 else 0
        return flow_sign  # 0/1: faza spadkowa/wzrostowa wg średniego FLOW w oknie

    def pairs(self) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Zwraca listę (emb_i, emb_j, same_phase) dla kolejnych par okien."""
        n = len(self.ohlcv)
        starts = list(range(0, n - self.window_size + 1, self.step))
        out = []
        for a, b in zip(starts[:-1], starts[1:]):
            w_i = self.ohlcv.iloc[a:a + self.window_size]
            w_j = self.ohlcv.iloc[b:b + self.window_size]
            emb_i, emb_j = make_embedding(w_i), make_embedding(w_j)
            same = int(self._phase_label(w_i) == self._phase_label(w_j))
            out.append((emb_i, emb_j, same))
        return out


def calibrate(bottleneck: State9Bottleneck, dataset: MarketPhaseDataset,
              steps: int = 200, lr: float = 0.02, seed: int = 0) -> State9Bottleneck:
    """Uczy Wq/bq bottlenecku tak, żeby regime_agreement_score był WYSOKI
    dla par oznaczonych jako "ta sama faza" i NISKI dla par "inna faza" -
    prosty ręczny gradient (ten sam wzór STE co w KHIPU-NEURAL:
    d(agree)/d(q_i) = q_j (twarda wartość drugiego czynnika), potem STE
    przez tanh). Strata: 0.5*(target - agree)^2, target=+1 dla tej samej
    fazy, -1 dla różnej.

    UWAGA: etykiety pochodzą z MarketPhaseDataset._phase_label - heurystyka
    bootstrap, NIE prawdziwa etykieta rynkowa (patrz docstring klasy).
    Kalibracja na tych etykietach uczy bottleneck ODTWARZAĆ istniejący
    sygnał FLOW w postaci skwantyzowanej - to demonstruje, że mechanizm
    kalibracji DZIAŁA (jest czym trenować), nie że wynikowy sygnał niesie
    nową informację ponad to, co FLOW już dawał. Podmień etykiety na coś
    niezależnego, żeby kalibracja miała samodzielną wartość."""
    rng = np.random.default_rng(seed)
    pairs = dataset.pairs()
    if not pairs:
        return bottleneck

    for step in range(steps):
        emb_i, emb_j, label = pairs[rng.integers(0, len(pairs))]
        q_i = bottleneck.forward(emb_i)
        t_i = bottleneck._cache["t"].copy()
        q_j = bottleneck.forward(emb_j)
        t_j = bottleneck._cache["t"].copy()

        agree = float(np.mean(q_i * q_j))
        target = 1.0 if label == 1 else -1.0
        dL_dagree = -(target - agree)  # d/dagree of 0.5*(target-agree)^2

        dL_dqi = dL_dagree * q_j / N_AXES
        dL_dqj = dL_dagree * q_i / N_AXES

        # STE dla q_i/q_j: odtwarzamy cache "t"/"x" dla kazdego z osobna
        # przed wywolaniem backward (ta sama technika co w
        # KHIPU-NEURAL: forward() nadpisuje self._cache, wiec dla
        # kazdego wpisu trzeba je odtworzyc jawnie zamiast liczyc na
        # ostatni stan po drugim forward()).
        bottleneck._cache = {"x": emb_i, "proj": None, "t": t_i}
        _, grads_i = bottleneck.backward(dL_dqi)
        bottleneck._cache = {"x": emb_j, "proj": None, "t": t_j}
        _, grads_j = bottleneck.backward(dL_dqj)

        bottleneck.Wq -= lr * (grads_i["Wq"] + grads_j["Wq"])
        bottleneck.bq -= lr * (grads_i["bq"] + grads_j["bq"])

    return bottleneck
