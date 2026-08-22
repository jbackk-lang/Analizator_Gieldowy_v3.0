"""Testy dla khipu_bottleneck.py - wpięcie State9Bottleneck (KHIPU-NEURAL)
w pipeline Analizatora 3.0, wg 6-krokowego planu integracji dostarczonego
przez użytkownika. Patrz docstring khipu_bottleneck.py po pełne
uzasadnienie i ograniczenia.
"""
import numpy as np
import pandas as pd
import pytest

import khipu_bottleneck as kb
from khipu_bottleneck import (
    State9Bottleneck, balance_correct, make_embedding, regime_agreement,
    regime_agreement_score, KHIPURegimeSignal, MarketPhaseDataset, calibrate,
    regime_alerts, KHIPU_ALERT_THRESHOLD, N_AXES, D_EMBED,
)


def _make_ohlcv(n=120, seed=1, trend=0.05):
    rng = np.random.default_rng(seed)
    close = 100 + trend * np.arange(n) + np.cumsum(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.1, n)
    high = np.maximum(close, open_) + np.abs(rng.normal(0, 0.2, n))
    low = np.minimum(close, open_) - np.abs(rng.normal(0, 0.2, n))
    volume = np.abs(100000 + rng.normal(0, 5000, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


# ---------------------------------------------------------------------
# Krok 6 (wyłącznik) - NIE zakładamy konkretnej wartości domyślnej flagi
# (użytkownik może ją świadomie przełączać podczas testów na żywo na
# realnych tickerach) - test sprawdza tylko, że przełącznik NAPRAWDĘ
# działa w obie strony, cokolwiek jest akurat ustawione w pliku.
# ---------------------------------------------------------------------

def test_switch_actually_gates_khipu_regime(monkeypatch):
    assert isinstance(kb.KHIPU_BOTTLENECK_ENABLED, bool)

    import khipu_bottleneck
    from pipeline import TimdrEngine
    ohlcv = _make_ohlcv(n=100)

    monkeypatch.setattr(khipu_bottleneck, "KHIPU_BOTTLENECK_ENABLED", False)
    packet_off = TimdrEngine(ohlcv).compute_packet()
    assert packet_off.khipu_regime is None

    monkeypatch.setattr(khipu_bottleneck, "KHIPU_BOTTLENECK_ENABLED", True)
    packet_on = TimdrEngine(ohlcv).compute_packet()
    assert packet_on.khipu_regime is not None


# ---------------------------------------------------------------------
# Krok 3 - State9Bottleneck (port z KHIPU-NEURAL)
# ---------------------------------------------------------------------

def test_balance_correct_always_satisfies_constraint():
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = rng.normal(size=N_AXES)
        q = np.sign(t)
        q[q == 0] = 1.0
        q = balance_correct(q, t)
        assert abs(int(q.sum())) <= 1
        assert set(np.unique(q)).issubset({-1.0, 1.0})


def test_state9_bottleneck_forward_always_valid_state():
    rng = np.random.default_rng(5)
    bn = State9Bottleneck(d_in=D_EMBED, rng=rng)
    for _ in range(50):
        x = rng.normal(size=D_EMBED)
        q = bn.forward(x)
        assert q.shape == (N_AXES,)
        assert abs(int(q.sum())) <= 1


def test_default_projection_is_deterministic_and_reproducible():
    bn1 = State9Bottleneck(d_in=D_EMBED)
    bn2 = State9Bottleneck(d_in=D_EMBED)
    assert np.array_equal(bn1.Wq, bn2.Wq)
    x = np.random.default_rng(1).normal(size=D_EMBED)
    assert np.array_equal(bn1.forward(x), bn2.forward(x))


# ---------------------------------------------------------------------
# Krok 2 - embedding giełdowy
# ---------------------------------------------------------------------

def test_make_embedding_shape_and_finite():
    ohlcv = _make_ohlcv(n=40)
    emb = make_embedding(ohlcv.iloc[:20])
    assert emb.shape == (D_EMBED,)
    assert np.all(np.isfinite(emb))


def test_make_embedding_handles_short_window_without_crashing():
    ohlcv = _make_ohlcv(n=5)
    emb = make_embedding(ohlcv.iloc[:2])  # minimalne okno
    assert emb.shape == (D_EMBED,)
    assert np.all(np.isfinite(emb))


def test_make_embedding_reflects_trend_direction():
    """Okno z wyraznym trendem wzrostowym powinno dac dodatni zwrot [0]."""
    up = _make_ohlcv(n=30, trend=2.0, seed=2)
    down = _make_ohlcv(n=30, trend=-2.0, seed=2)
    emb_up = make_embedding(up)
    emb_down = make_embedding(down)
    assert emb_up[0] > 0
    assert emb_down[0] < 0


# ---------------------------------------------------------------------
# Krok 4 - regula GIPU / zgodnosc rezimu
# ---------------------------------------------------------------------

def test_regime_agreement_identical_codes_gives_plus_one():
    q = np.array([1, -1, 1, 1, -1, 1, -1, -1, 1], dtype=float)
    assert abs(int(q.sum())) <= 1  # sanity - poprawny F4-RED stan
    score = regime_agreement_score(q, q)
    assert score == pytest.approx(1.0)


def test_regime_agreement_opposite_codes_gives_minus_one():
    q = np.array([1, -1, 1, 1, -1, 1, -1, -1, 1], dtype=float)
    score = regime_agreement_score(q, -q)
    assert score == pytest.approx(-1.0)


def test_regime_agreement_bounded():
    rng = np.random.default_rng(9)
    bn = State9Bottleneck(d_in=D_EMBED, rng=rng)
    for _ in range(30):
        qi = bn.forward(rng.normal(size=D_EMBED))
        qj = bn.forward(rng.normal(size=D_EMBED))
        score = regime_agreement_score(qi, qj)
        assert -1.0 <= score <= 1.0


def test_khipu_regime_signal_score_series_shape_and_bounds():
    ohlcv = _make_ohlcv(n=100)
    sig = KHIPURegimeSignal()
    scores = sig.score_series(ohlcv, window_size=20, step=5)
    assert len(scores) > 0
    assert np.all(scores >= -1.0) and np.all(scores <= 1.0)


def test_khipu_regime_signal_empty_for_too_short_series():
    ohlcv = _make_ohlcv(n=15)
    sig = KHIPURegimeSignal()
    scores = sig.score_series(ohlcv, window_size=20, step=5)
    assert len(scores) == 0


def test_score_series_indexed_matches_score_series():
    """score_series() ma być cienką otoczką nad score_series_indexed() -
    same wyniki, tylko bez bar-indeksów."""
    ohlcv = _make_ohlcv(n=100)
    sig = KHIPURegimeSignal()
    scores_plain = sig.score_series(ohlcv, window_size=20, step=5)
    scores_idx, bar_idx = sig.score_series_indexed(ohlcv, window_size=20, step=5)
    assert np.array_equal(scores_plain, scores_idx)
    assert len(bar_idx) == len(scores_idx)
    # bar-indeksy muszą rosnąco mieścić się w zakresie serii wejściowej
    assert np.all(np.diff(bar_idx) > 0)
    assert np.all(bar_idx < len(ohlcv))


def test_score_series_indexed_empty_for_too_short_series():
    ohlcv = _make_ohlcv(n=15)
    sig = KHIPURegimeSignal()
    scores, bar_idx = sig.score_series_indexed(ohlcv, window_size=20, step=5)
    assert len(scores) == 0
    assert len(bar_idx) == 0


# ---------------------------------------------------------------------
# Alerty na zgodność reżimu (regime_alerts / KHIPU_ALERT_THRESHOLD)
# ---------------------------------------------------------------------

def test_regime_alerts_fires_only_below_threshold():
    scores = np.array([0.9, 0.1, -0.4, -0.6, -1.0])
    bar_idx = np.array([10, 20, 30, 40, 50])
    alerts = regime_alerts(scores, bar_idx, threshold=-0.5)
    assert [a["bar_index"] for a in alerts] == [40, 50]
    assert all(a["score"] <= -0.5 for a in alerts)
    assert all("Rozjazd reżimu KHIPU" in a["message"] for a in alerts)


def test_regime_alerts_boundary_is_inclusive():
    scores = np.array([-0.5])
    bar_idx = np.array([7])
    alerts = regime_alerts(scores, bar_idx, threshold=-0.5)
    assert len(alerts) == 1


def test_regime_alerts_empty_when_nothing_crosses_threshold():
    scores = np.array([0.9, 0.5, 0.1, -0.2])
    bar_idx = np.array([1, 2, 3, 4])
    alerts = regime_alerts(scores, bar_idx, threshold=KHIPU_ALERT_THRESHOLD)
    assert alerts == []


def test_regime_alerts_uses_module_default_threshold_when_unspecified():
    scores = np.array([KHIPU_ALERT_THRESHOLD - 0.01])
    bar_idx = np.array([5])
    alerts = regime_alerts(scores, bar_idx)
    assert len(alerts) == 1


# ---------------------------------------------------------------------
# Krok 5 - kalibracja: gradient zweryfikowany numerycznie
# ---------------------------------------------------------------------

def test_calibrate_gradient_matches_numerical():
    """Ten sam test co przy budowie modulu (linearyzowany STE, jak w
    KHIPU-NEURAL) - potwierdza, ze reczny backprop w calibrate() jest
    matematycznie poprawny, nie tylko 'wyglada sensownie'."""
    d = D_EMBED
    rng = np.random.default_rng(3)
    Wq0 = rng.normal(0, 1 / np.sqrt(d), size=(N_AXES, d))
    bq0 = np.zeros(N_AXES)
    x_i = rng.normal(size=d)
    x_j = rng.normal(size=d)
    target = 1.0

    def quantize(Wq, bq, x):
        proj = Wq @ x + bq
        t = np.tanh(proj)
        q = np.sign(t)
        q[q == 0] = 1.0
        q = balance_correct(q, t)
        return t, q

    t_i0, q_i0 = quantize(Wq0, bq0, x_i)
    t_j0, q_j0 = quantize(Wq0, bq0, x_j)

    agree = float(np.mean(q_i0 * q_j0))
    dL_dagree = -(target - agree)
    dL_dqi = dL_dagree * q_j0 / N_AXES
    dL_dqj = dL_dagree * q_i0 / N_AXES

    bn = State9Bottleneck(d_in=d, Wq=Wq0, bq=bq0)
    bn._cache = {"x": x_i, "proj": None, "t": t_i0}
    _, grads_i = bn.backward(dL_dqi)
    bn._cache = {"x": x_j, "proj": None, "t": t_j0}
    _, grads_j = bn.backward(dL_dqj)
    ana_dWq = grads_i["Wq"] + grads_j["Wq"]

    eps = 1e-5

    def loss_from_codes(qi, qj):
        a = float(np.mean(qi * qj))
        return 0.5 * (target - a) ** 2

    def numgrad(idx):
        Wp = Wq0.copy(); Wp[idx] += eps
        Wm = Wq0.copy(); Wm[idx] -= eps
        tp_i, _ = quantize(Wp, bq0, x_i)
        tm_i, _ = quantize(Wm, bq0, x_i)
        tp_j, _ = quantize(Wp, bq0, x_j)
        tm_j, _ = quantize(Wm, bq0, x_j)
        qi_p = q_i0 + (tp_i - t_i0)
        qi_m = q_i0 + (tm_i - t_i0)
        qj_p = q_j0 + (tp_j - t_j0)
        qj_m = q_j0 + (tm_j - t_j0)
        return (loss_from_codes(qi_p, qj_p) - loss_from_codes(qi_m, qj_m)) / (2 * eps)

    numg = np.zeros_like(Wq0)
    it = np.nditer(Wq0, flags=["multi_index"])
    for _ in it:
        numg[it.multi_index] = numgrad(it.multi_index)

    assert np.max(np.abs(numg - ana_dWq)) < 1e-4


def test_calibrate_runs_and_changes_projection():
    ohlcv = _make_ohlcv(n=200, seed=7)
    dataset = MarketPhaseDataset(ohlcv, window_size=20, step=5)
    bn = State9Bottleneck(d_in=D_EMBED)
    Wq_before = bn.Wq.copy()
    calibrate(bn, dataset, steps=20, lr=0.02, seed=0)
    assert not np.array_equal(Wq_before, bn.Wq)


def test_calibrate_with_no_pairs_is_noop():
    ohlcv = _make_ohlcv(n=10, seed=1)  # za krotkie na zadna pare okien
    dataset = MarketPhaseDataset(ohlcv, window_size=20, step=5)
    bn = State9Bottleneck(d_in=D_EMBED)
    Wq_before = bn.Wq.copy()
    calibrate(bn, dataset, steps=20)
    assert np.array_equal(Wq_before, bn.Wq)
