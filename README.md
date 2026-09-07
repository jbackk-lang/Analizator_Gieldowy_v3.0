# Analizator Giełdowy v3

Lokalne narzędzie badawczo-edukacyjne do analizy sygnałów TIMDR na
danych giełdowych (akcje, indeksy, waluty, surowce - wszystko, co
oferuje yfinance). Zbudowane od zera na bazie modułu-glue dostarczonego
przez użytkownika (`timdr_core_finance` + `AnalizatorGieldowy`) i
świadomie zaprojektowane wokół lekcji wyciągniętych z wcześniejszego
projektu **Synoptyk** (system pogodowy z tej samej rodziny TIMDR).

> **TO NIE JEST DORADZTWO INWESTYCYJNE.** Emergencja, Ufność, sharpe_n,
> winrate_n, dd_n, RSI - to wszystko statystyczne odchylenia sygnału
> względem jego własnej, lokalnej historii. Nie są to rekomendacje
> kupna/sprzedaży. Decyzje inwestycyjne podejmujesz na własną
> odpowiedzialność, najlepiej po konsultacji z licencjonowanym doradcą.

## Uruchomienie

```
run.bat
```

Skrypt sam znajdzie Pythona (`python` albo `py`), doinstaluje zależności
(`flask numpy pandas yfinance pytest`), uruchomi pełny zestaw testów i
odpali serwer + otworzy dashboard w przeglądarce pod
`http://127.0.0.1:8060`.

## Co robi analizator

1. **Silnik TIMDR** (`timdr_core_finance.py`) liczy na cenie/wolumenie
   siedem podstawowych sygnałów: `trm` (wygładzenie), `flow` (pochodna/
   momentum), `twist` (odwrócenie trendu), `rhythm` (okresowość
   wolumenu), `anomalie`, `defekt`, `rezonans` (zgodność 3 niezależnych
   sprawdzeń jednocześnie).
2. **Werdykt** (`analizator_gieldowy.py`) klasyfikuje serię jako
   "EMERGENCJA" albo "szum (brak emergencji)" na podstawie rezonansu z
   ostatnich barów, liczy Ufność (%), RSI (metoda Wildera), oraz
   wsteczny test prostej strategii opartej na rezonansie (Sharpe/
   Winrate/Drawdown, znormalizowane do 0-1 jak w oryginalnym przykładzie
   użytkownika).
3. **Kaskada przepływu kapitału** (`cascade.py`) - łańcuch
   surowce → waluty → obligacje → indeksy → sektory → akcje. Dla
   każdego wcześniejszego ogniwa sprawdza, czy miało niedawno aktywny
   rezonans, w którą stronę (na podstawie surowej zmiany ceny, nie
   wygładzonej), i jak silny był ten ruch **zmierzony faktycznym
   przepływem kapitału** (`flow_intensity = zmiana_ceny% × wolumen
   względny`), a nie z góry ustaloną "ważnością" sektora. Aktywne,
   spójne kierunkowo ogniwa podnoszą Ufność (nigdy jej nie obniżają;
   limit +20 p.p.).
4. **Samo-uczenie** (`state.py`) loguje każdą predykcję kierunku z
   horyzontem 5 barów, dopasowuje ją do późniejszej rzeczywistej ceny,
   liczy bias/MAE/trafność per horyzont z plakietką 🔴/🟠/🟢 zależną od
   liczby próbek (n≥5 = korekta ma sens).
5. **Trwały stan** (`state.py`, `StateStore`) wykrywa zmianę werdyktu
   między uruchomieniami - na dysku, przetrwa restart procesu.
6. **Cena, wolumen i trend** - dashboard pokazuje ostatnią cenę (z
   jednostką/walutą), zmianę % w okresie, wolumen (ostatni + średni) i
   linię trendu (`trm`) nałożoną na wykres ceny.

## Lekcje z Synoptyka zastosowane w v3

Projekt Synoptyk (system pogodowy) ujawnił kilka wzorców błędów typowych
dla systemów TIMDR. Wszystkie zaadresowano tu od początku, świadomie:

- **Progi adaptacyjne, nigdy stałe.** `defect()` kalibruje próg z rozstępu
  danych analizowanego okna, nie z jednej uniwersalnej stałej - plus
  bezwzględna podłoga (`min_floor_frac`), żeby próg nie zapadł się do
  zera na płaskich/rzadkich danych (dokładnie ten sam problem co przy
  opadach w Synoptyku - "prawie zawsze zero, czasem skok").
- **Cache zamiast przeliczania w kółko.** `CachedReferenceLoader`
  pobiera dane referencyjne kaskady raz dziennie (plik JSON per ticker),
  nie przy każdym wywołaniu API - unika powtórnego pobierania do 9
  dodatkowych instrumentów referencyjnych za każdym razem.
- **Nigdy `except: pass`.** Każdy błąd schematu/sieci w `data_loader.py`
  kończy się jawnym `DataLoaderError` z czytelnym komunikatem po polsku,
  nigdy cichym pustym wynikiem. Błędy pojedynczych instrumentów kaskady
  są zbierane widocznie (`fetch_errors`), nie połykane.
- **Stan na dysku, nie w pamięci.** `StateStore`/`PredictionLog` piszą
  do plików JSON/JSONL w `data/state/` i `data/predictions/` - restart
  procesu (np. crash i ponowne uruchomienie `run.bat`) nie resetuje
  wykrywania zmiany werdyktu ani logu predykcji.
- **Samo-uczenie z plakietkami widoczności.** Korekta na podstawie
  bias/MAE stosuje się tylko przy n≥5 potwierdzonych próbkach na dany
  horyzont; plakietka 🔴 "za mało danych" jest pokazywana zawsze, nie
  ukrywana - brak plakietki czytałby się jako "wszystko OK", co byłoby
  fałszywe przy braku danych.

## Uwagi techniczne (istotne przy dalszym rozwoju)

- **Kalibracja `defect()`**: próg liczony jest z rozstępu RÓŻNIC między
  kolejnymi barami (nie z rozstępu poziomów ceny - dosłowna definicja ze
  szkieletu TIMDR fałszywie flagowała ~20% czystego szumu jako "defekt",
  bo dla błądzenia losowego te dwie wielkości są tego samego rzędu).
  Aktualne stałe: `jump_factor=3.0` plus bezwzględna podłoga
  (`min_floor_frac`) na płaskich/rzadkich danych. Przy dostrajaniu
  czułości zmieniaj te dwie stałe, nie samą formę progu.
- **RSI i backtest liczone są na SUROWEJ cenie (`packet.price`), nie na
  `trm`.** `trm` (mediana krocząca k=5) to wejście dla `flow`/`twist`/
  `resonance` - do tego został zaprojektowany, ale nie nadaje się do
  RSI/backtestu (opóźnia i tłumi realną zmienność). Przy dodawaniu
  nowych sygnałów świadomie wybieraj, która seria pasuje do
  zastosowania.
- **Kaskada waży ogniwa zmierzonym przepływem kapitału
  (`_flow_intensity` = zmiana ceny % × wolumen względny), nie z góry
  przypisaną "ważnością" sektora.** `STAGE_TIMING` niesie wyłącznie
  informacyjne opóźnienie/charakter reakcji - nie wchodzi do wzoru
  wagi. Jeśli w przyszłości trzeba dostroić wpływ poszczególnych ogniw,
  rób to przez `FLOW_INTENSITY_REFERENCE` (kalibracja normalizacji), nie
  przez dodawanie stałych wag.
- **Port 8060, celowo NIE 5060.** Port 5060 (SIP) jest na liście
  "zakazanych portów" przeglądarek i `fetch()` (Node/undici) - dashboard
  uruchomiony na nim nigdy nie połączyłby się z własnym API w realnej
  przeglądarce (`ERR_UNSAFE_PORT`). Przy ewentualnej zmianie portu
  sprawdź listę zakazanych portów (m.in. 1, 7, 9, 5060, 5061, 6000,
  6666-6669, 6697).

## Struktura plików

```
analizator-gieldowy-v3/
├── timdr_core_finance.py    - silnik TIMDR (trm/flow/twist/rhythm/anomalie/defekt/rezonans)
├── ringdown.py               - ringdown_resonance(): rezonans w sensie fizycznym (oscylacyjny powrot ceny po skoku)
├── analizator_gieldowy.py   - RSI, backtest, klasyfikacja Emergencja/Ufność
├── pipeline.py               - glue: TimdrEngine, TimdrPacket, run_pipeline()
├── cascade.py                 - kaskada przepływu kapitału + samoucząca się waga (flow_intensity)
├── state.py                  - StateStore (trwały werdykt) + PredictionLog (samouczenie)
├── data_loader.py            - pobieranie OHLCV (yfinance) + dzienny cache + schemat kaskady
├── api.py                    - Flask API (port 8060) + serwowanie dashboardu
├── static/dashboard.html     - dashboard (ciemny motyw, Canvas 2D, bez CDN)
├── khipu_bottleneck.py       - opcjonalny modul KHIPU (domyslnie WYLACZONY, patrz nizej)
├── run.bat                   - instalacja zależności + testy + start serwera
├── requirements.txt
└── test_*.py                 - 119 testów pytest (w tym test_api.py, test_khipu_bottleneck.py, test_ringdown.py)
```

## Endpointy API

- `GET /` - dashboard
- `GET /api/health`
- `GET /api/analyze?ticker=AAPL&period=1y&cascade=1` - pełna analiza
- `GET /api/meta?ticker=AAPL&period=1y` - meta-warstwa TIMDR-META-DYNAMICS (patrz niżej)
- `POST /api/state/clear` `{"ticker": "AAPL"}` - czyści stan + log predykcji dla tickera

## Integracja z TIMDR-META-DYNAMICS (`/api/meta`)

DODANE: `meta_dynamics_module.py` podłącza [TIMDR-META-DYNAMICS](../TIMDR-META-DYNAMICS)
(meta-warstwę nad polem Λ-τ-ρ-J, opisującą ewolucję CAŁEGO pola sygnałów
w czasie, nie pojedynczy sygnał) do prawdziwego `TimdrPacket` z tego
repo - zamiast do `analizator3_core`, modułu, którego oryginalnie
zakładał `main.py` w TIMDR-META-DYNAMICS, ale który nigdy nie istniał.

Mapowanie Λ-τ-ρ-J -> sygnały z tego repo (decyzja projektowa, nie
jedyna możliwa - patrz uzasadnienie w docstringu `meta_dynamics_module.py`):

| Pole | Sygnał z tego repo | Uzasadnienie |
|---|---|---|
| Λ (struktura) | `packet.trm` | mediana krocząca ceny - dosłownie linia struktury/trendu |
| τ (transformacja) | `packet.flow` | tempo zmiany trm - pochodna, "transformacja w toku" |
| ρ (anomalia) | `packet.resonance` | już ciągły [0,1], zgodność 3 niezależnych sprawdzeń naraz |
| J (operator punktowy) | wolumen | surowy wolumen z OHLCV |

To DODATKOWA, eksperymentalna warstwa nad `/api/analyze` - nie zastępuje
jej i nie wpływa na Emergencja/Ufność/RSI/backtest z tamtego endpointu.
Wymaga, żeby folder `TIMDR-META-DYNAMICS` leżał jako sąsiad tego repo
(ten sam poziom katalogów); jeśli go nie ma, `/api/meta` zwraca czytelny
`501` zamiast wywalać cały proces przy starcie API.

Wcześniej jedynym sposobem użycia TIMDR-META-DYNAMICS był osobny
Tkinter GUI (`gui.py` w tamtym repo) z syntetycznymi danymi demo -
`/api/meta` to teraz realna integracja na prawdziwych danych giełdowych
z tego repo, dostępna z tego samego dashboardu/API co reszta analizy.

## Integracja z KHIPU-NEURAL (`khipu_bottleneck.py`) — opcjonalna, wyłącznik `KHIPU_BOTTLENECK_ENABLED`

Wpięcie `State9Bottleneck` z
[jbackk-lang/KHIPU-NEURAL](../KHIPU-NEURAL) w miejsce
features/embedding -> dalsza logika TRM/FLOW/TWIST, wg 6-krokowego planu
integracji. Kontrolowane wyłącznikiem `KHIPU_BOTTLENECK_ENABLED` w
`khipu_bottleneck.py` - gdy `False`, `pipeline.py`/`analizator_gieldowy.py`
zachowują się DOKŁADNIE tak jak przed dodaniem tego modułu (zero zmiany
istniejącego wyniku); testy sprawdzają działanie przełącznika w OBIE
strony (`test_switch_actually_gates_khipu_regime`), nie zakładają żadnej
konkretnej wartości domyślnej - flaga bywa świadomie przełączana podczas
testów na żywo na realnych tickerach.

Co robi, gdy włączony:
- `make_embedding(candle_window)` buduje 8-wymiarowy ciągły wektor z
  istniejących bloków TIMDR (`trm`/`flow`/`twist`) + cech świecowych.
- `State9Bottleneck` (wierny port matematyki z KHIPU-NEURAL, gradienty
  tam zweryfikowane do ~1e-11) ściska embedding do 9-osiowego,
  dyskretnego kodu ±1 z warunkiem równowagi F4-RED.
- `regime_agreement`/`regime_agreement_score` (reguła GIPU: iloczyn
  per-oś) mierzą zgodność dwóch sąsiadujących okien świecowych - NOWY
  sygnał `packet.khipu_regime`, dopisywany DO WYNIKU (`khipu_regime_last`,
  `khipu_regime_mean`), nie zastępujący istniejącego `resonance` (który
  ma inną definicję - zgodność między różnymi wskaźnikami w tej samej
  chwili, nie między oknami czasowymi).
- **Alerty rozjazdu reżimu** (`khipu_bottleneck.py::regime_alerts`,
  próg `KHIPU_ALERT_THRESHOLD = -0.5`): gdy `regime_agreement_score`
  między dwoma sąsiednimi oknami spadnie do/poniżej progu (duża
  rozbieżność kodu State9 - podejrzenie nagłej zmiany dyskretnego stanu
  rynku), dopisywane do wyniku jako `khipu_regime_alerts` (lista
  czytelnych komunikatów), `khipu_regime_alerts_idx` (odpowiadające
  bar-indeksy, do naniesienia na wykres ceny - pomarańczowe znaczniki w
  dashboardzie), `n_khipu_regime_alerts` (liczba) i
  `khipu_regime_alert_active` (czy OSTATNIE okno jest w stanie alertu).
  **`KHIPU_ALERT_THRESHOLD` to ustalona wartość heurystyczna** - nie
  wyprowadzona z rozkładu historycznych danych ani z backtestu progu
  (w odróżnieniu np. od pracy nad `EMERGENCE_CONFIDENCE_THRESHOLD`) -
  punkt startowy do dostrojenia, nie potwierdzona liczba. To DALEJ
  dyskretny sygnał zgodności stanu, NIE predykcja kierunku ani
  wielkości ruchu ceny.
- `calibrate()` pozwala douczyć projekcję na parach okien z etykietą
  "ta sama faza" - **etykiety domyślne w `MarketPhaseDataset` to
  HEURYSTYCZNY bootstrap ze znaku FLOW, NIE niezależna, zweryfikowana
  etykieta rynkowa** (nikt takiej nie dostarczył - patrz docstring
  `MarketPhaseDataset`). Kalibracja na tych etykietach demonstruje, że
  mechanizm działa, nie że sygnał wynikowy niesie nową informację ponad
  to, co `flow` już dawał.

**Granice (wprost z wniosków KHIPU-NEURAL, patrz jego README):** ten
bottleneck pomaga na zadaniach KATEGORIALNYCH ("czy te dwa stany
należą do tej samej fazy") i SZKODZI na zadaniach wymagających
precyzyjnej wartości ciągłej. Dlatego moduł nigdy nie zwraca ceny ani
innej ciągłej wielkości - tylko dyskretny sygnał zgodności (i alerty na
nim oparte). Nie używać do regresji ceny/wolumenu/odległości.

## Testy

```
python -m pytest -q
```

121/121 testów przechodzi (`timdr_core_finance`, `ringdown`,
`analizator_gieldowy` pośrednio przez `pipeline`, `pipeline`, `cascade`,
`data_loader`, `state`, `api`, `khipu_bottleneck`,
`test_selfbaseline_recovery` — czy `anomalies()`/`defect()` wracają do
normy po ustaniu anomalii cenowej; tu bez nowego błędu, bo `defect()` już
ma udokumentowaną poprawkę "Bug 1" powyżej — ten sam błąd, nieprzeniesiony,
znaleziono przy tym samym teście w siostrzanym module
`deliverable_timdr_finanse`). Wszystkie testy
`data_loader`/`api` mockują `yfinance` (brak zależności od sieci przy
testowaniu) - realne pobieranie danych giełdowych wymaga połączenia
internetowego przy faktycznym uruchomieniu.

## Rezonans w sensie fizycznym po skokach ceny (`ringdown.py`)

`timdr_core_finance.py::resonance()` to licznik koincydencji (ile z
trzech niezależnych sprawdzeń - anomalia/defekt/skręt - zgadza się naraz)
- nazwa pożyczona z fizyki, ale mechanizm inny. `ringdown.py::ringdown_resonance()`
liczy coś, co faktycznie odpowiada fizycznemu rezonansowi: dla każdego
bloku zdarzeń z `defect()` (nagły skok ceny) sprawdza, czy powrót w
stronę poziomu sprzed skoku jest OSCYLACYJNY (cena "przewahnęła" przez
ten poziom w obie strony - typowy wzorzec overreaction + korekty/
mean-reversion, czyli faktyczna synchronizacja z "otoczeniem"/poziomem
sprzed zaburzenia) czy MONOTONICZNY (permanentna przecena/przewartościowanie
- nowy poziom się utrzymuje, brak odbicia). Wynik trafia do
`wynik["price_ringdown"]` (lista dictów: `event_idx`, `is_oscillatory`,
`frequency_hz` [tu: cykle/bar, NIE Hz - patrz UWAGA w docstringu
`ringdown.py`], `damping_ratio`, `n_crossings`, itd.) oraz
`n_price_ringdown`/`n_price_ringdown_oscylacyjny`. W przeciwieństwie do
integracji KHIPU wyżej, TO NIE JEST opcjonalna zależność - pola są zawsze
obecne w wyniku (lista może być pusta, jeśli `defect()` nic nie znalazł).

Port 1:1 matematyki z `jbackk-lang/universal-state-analyzer`
(`timdr_core/ringdown.py`) - histereza Schmitta na wykrywaniu stanu (nie
doklejona po fakcie do już policzonych szczytów - pierwsza próba tak
zrobiona dawała regresję, patrz historia commitów tamtego repo),
częstotliwość liczona z mediany (nie średniej) odstępów między
przejściami. RÓŻNICA względem sieci energetycznej (TIMDR-Grid-Monitor):
cena NIE MA stałego, fizycznego punktu odniesienia jak `f_nominal=50Hz`
- `baseline` liczony jest tu domyślnie ze średniej okna PRZED zdarzeniem
(`pre_event_window=min(event_idx, 20)`, dopasowane do domyślnego okna
`defect()`), z ograniczonym zasięgiem analizy (`max_lookahead=40`), żeby
nie złapać zupełnie innego, późniejszego skoku jako część tego samego
"powrotu".

## Ograniczenia

- Backtest i wsteczne metryki (Sharpe/Winrate/DD) opisują wyłącznie
  przeszłość analizowanego okna - nie są prognozą przyszłych wyników.
- Kaskada kapitałowa to model korelacyjny (przepływ ceny/wolumenu
  między klasami aktywów), nie przyczynowy - koreluje, nie dowodzi
  przyczynowości.
- Samo-uczenie zaczyna dawać sensowne korekty dopiero po zgromadzeniu
  ≥5 potwierdzonych predykcji na dany horyzont (przy analizie raz
  dziennie to zajmuje tydzień+).
- Waluta/jednostka ceny (`currency`, `price_unit_label`) to heurystyka z
  sufiksu tickera (konwencja Yahoo Finance), NIE dane pobrane z API -
  `yfinance.download()` nie zwraca metadanych instrumentu. Dla
  nietypowych tickerów domyślnie zakłada USD.
- `ringdown_resonance()` (`ringdown.py`) zwalidowany wyłącznie na
  syntetycznym, czystym modelu tłumionego oscylatora (patrz
  universal-state-analyzer) - `noise_floor_factor=3.0`,
  `pre_event_window=20` i `max_lookahead=40` to wartości ustalone ręcznie,
  nieskalibrowane na realnych danych giełdowych. Interpretacja
  "overreaction + korekta" jest jedną z możliwych narracji dla
  oscylacyjnego powrotu ceny - metoda wykrywa WZORZEC (oscylacyjny vs
  monotoniczny powrót), nie weryfikuje przyczyny; nie została
  przetestowana pod kątem trafności predykcyjnej (czy oscylacyjny
  ringdown faktycznie poprzedza dalszy ruch ceny w jakąś stronę).
