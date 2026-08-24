"""
test_selfbaseline_recovery.py -- ten sam test co w calej rodzinie repo TIMDR
(TIMDR-Crypto-Graph, universal-state-analyzer, deliverable_timdr_finanse,
TIMDR-Grid-Monitor, TIMDR-Earthquake-Core): czy anomalies()/defect()
falszywie flaguja NOWE, normalne probki po ustaniu anomalii cenowej, tylko
dlatego ze stare anomalne probki wciaz siedza w oknie referencyjnym?

Ten modul juz ma udokumentowany (test_timdr_core_finance.py, "Bug 1") i
naprawiony dokladnie ten problem dla defect(): rozrzut liczony z SAMYCH
ROZNIC (nie poziomow ceny) + jump_factor podniesiony 0.3->3.0 - dokladnie
ten sam blad, ktory przy TYM tescie znaleziono jeszcze raz, nieprzeniesiony,
w siostrzanym module deliverable_timdr_finanse/timdr_core_finance.py.

WYNIK TUTAJ: brak nowego bledu - defect() (juz naprawiony) i anomalies()
poprawnie wracaja do normy po ustaniu anomalii, na cenie skonstruowanej
jako CIAGLY random walk (bez sztucznego dodatkowego skoku przy przejsciu
anomalia->powrot, ktory dawal falszywe pozytywy w pierwszej probie tego
testu - patrz komentarz w _make_series).
"""
import numpy as np

from timdr_core_finance import anomalies, defect


def _mad_z_of_last(window_vals):
    x = np.asarray(window_vals, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        span = np.max(x) - np.min(x)
        return 0.0 if span == 0 else (x[-1] - med) / (span / 4)
    return 0.6745 * (x[-1] - med) / mad


def _make_series(rng, n_pre=60, n_anom=3, n_post=20, jump=50.0):
    """WAZNE: powrot do normy musi KONTYNUOWAC random walk od poziomu
    SPRZED anomalii, nie zaczynac nowego spaceru od stalej - inaczej sam
    test wprowadza sztuczny, dodatkowy skok w momencie przejscia (znaleziona
    pulapka przy pierwszej probie tego testu, nie blad w kodzie produkcyjnym)."""
    pre = 100 + np.cumsum(rng.normal(0, 0.3, n_pre))
    level = pre[-1]
    spike = np.full(n_anom, level + jump)
    post = level + np.cumsum(rng.normal(0, 0.3, n_post))
    return np.concatenate([pre, spike, post]), n_pre + n_anom


def test_defect_zero_falszywych_flag_po_opuszczeniu_okna():
    W = 30
    for seed in range(5):
        rng = np.random.default_rng(seed)
        full, event_end = _make_series(rng, n_pre=100, n_anom=5, n_post=150)
        idx = defect(full, window=20, jump_factor=3.0)
        post_flags = idx[idx >= event_end + 20]
        assert len(post_flags) == 0, (
            f"seed={seed}: defect() wciaz flaguje probki po opuszczeniu "
            f"okna anomalii: {list(post_flags)}"
        )


def test_anomalies_recovers_szybko_strumieniowo():
    W = 30
    for seed in range(5):
        rng = np.random.default_rng(seed)
        full, event_end = _make_series(rng)
        post_z = [
            _mad_z_of_last(full[max(0, i - W):i + 1])
            for i in range(event_end, event_end + 5)
        ]
        assert all(abs(z) < 3.0 for z in post_z), (
            f"seed={seed}: anomalies() flaguje probki tuz po evencie ({post_z})"
        )
