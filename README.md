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
├── analizator_gieldowy.py   - RSI, backtest, klasyfikacja Emergencja/Ufność
├── pipeline.py               - glue: TimdrEngine, TimdrPacket, run_pipeline()
├── cascade.py                 - kaskada przepływu kapitału + samoucząca się waga (flow_intensity)
├── state.py                  - StateStore (trwały werdykt) + PredictionLog (samouczenie)
├── data_loader.py            - pobieranie OHLCV (yfinance) + dzienny cache + schemat kaskady
├── api.py                    - Flask API (port 8060) + serwowanie dashboardu
├── static/dashboard.html     - dashboard (ciemny motyw, Canvas 2D, bez CDN)
├── run.bat                   - instalacja zależności + testy + start serwera
├── requirements.txt
└── test_*.py                 - 74 testy pytest (w tym test_api.py)
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

## Testy

```
python -m pytest -q
```

74/74 testy przechodzą (`timdr_core_finance`, `analizator_gieldowy`
pośrednio przez `pipeline`, `pipeline`, `cascade`, `data_loader`, `state`,
`api`). Wszystkie testy `data_loader`/`api` mockują `yfinance` (brak
zależności od sieci przy testowaniu) - realne pobieranie danych giełdowych
wymaga połączenia internetowego przy faktycznym uruchomieniu.

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
