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

## Znalezione i naprawione błędy

Zgodnie z metodyką całego tego zestawu narzędzi (nic nie jest uznawane
za gotowe bez faktycznego uruchomienia i reprodukcji na realistycznych
danych) - poniżej każdy błąd znaleziony podczas budowy v3.

### 1. `defect()` fałszywie flagował ~20% czystego szumu jako "defekty"

Dosłowna definicja ze szkieletu TIMDR (`próg = 0.3 × rozstęp cen w
oknie`) porównywała próg skoku do rozstępu **poziomów** ceny, nie do
rozstępu **różnic** między kolejnymi barami. Dla błądzenia losowego te
dwie wielkości są tego samego rzędu - więc próg 0.3× był mniejszy niż
typowy normalny ruch, i ok. 20% czystego szumu (zweryfikowano na 500
próbkach czystego random walk) zostało błędnie oznaczone jako "defekt".
**Naprawiono**: próg liczony teraz z rozstępu różnic (nie poziomów),
`jump_factor` podniesiony do 3.0, dodana bezwzględna podłoga. Po
naprawie: fałszywe alarmy spadły do 0.2%, prawdziwe wstrzyknięte skoki
nadal wykrywane. Regresja: `test_defect_rzadki_na_zwyklym_szumie_random_walk`.

### 2. Kaskada: `is_active` liczone ze średniej rezonansu, nie ze szczytu

Rezonans to z natury krótkotrwałe zdarzenie (kilka barów). Uśrednienie
go po całym oknie 15-barowym rozcieńczało nawet wyraźny, poprawnie
wykryty szok poniżej progu (średnia=0.244 przy progu=0.333, mimo że
szczyt=0.667). **Naprawiono**: `is_active` liczone teraz z maksimum
(szczytu) w oknie, zgodnie z istniejącą semantyką `strong_idx` w
`resonance()`.

### 3. Kaskada: kierunek liczony z opóźnionego `flow`/`trm`

Kierunek ruchu w danym ogniwie liczono z `flow` (pochodnej wygładzonej
`trm`, okno k=5) dokładnie w barze szczytu rezonansu - ale `trm`
potrzebuje kilku barów PO szoku, zanim mediana ruchoma "zobaczy" nowy
poziom. Efekt: dla tego samego, jednoznacznie kierunkowego szoku różne
ogniwa łańcucha pokazywały sprzeczne kierunki (np. "Surowce: spadkowa"
vs "Energetyka/Przemysł: wzrostowa" dla tego samego szoku w górę).
**Naprawiono**: kierunek liczony teraz z surowej zmiany ceny w małym
oknie wokół szczytu (`cena[szczyt+2] - cena[szczyt-2]`), bez opóźnienia
wygładzania.

### 4. Dwa architektoniczne odrzucenia modelu kaskady (na żądanie użytkownika)

- **v1 → v2**: pierwotny łańcuch kosztowy sektorów
  (surowce→energetyka→przemysł→...→technologia) został **całkowicie
  zastąpiony** (nie uzupełniony) łańcuchem przepływu kapitału
  (surowce→waluty→obligacje→indeksy→sektory→akcje) - wyraźna decyzja
  użytkownika: "To ma ZASTĄPIĆ kaskadę sektorową".
- **v2-pierwsza-próba → v2-finalna**: pierwsza implementacja łańcucha
  kapitałowego nadal ważyła wpływ ogniw stałą tabelą "★ siła wpływu"
  (1-5 gwiazdek) - odziedziczoną koncepcyjnie z odrzuconego modelu
  sektorowego. Użytkownik to jednoznacznie odrzucił: *"mówimy o
  przepływie finansów nie szkolnej definicji ważności"*. **Naprawiono**:
  wprowadzono `_flow_intensity()` - realną, zmierzoną miarę
  (`zmiana_ceny% × wolumen_względny`) zamiast opinii o "ważności"
  sektora. `STAGE_TIMING` (dawniej `STAGE_INFLUENCE`) niesie już tylko
  informacyjne opóźnienie/charakter reakcji, nie wchodzi do wzoru wagi.

### 5. Pakiet TIMDR nie niósł surowej ceny, tylko wygładzoną

Oryginalny moduł-glue dostarczony przez użytkownika budował `TimdrPacket`
tylko z `trm` (wygładzoną ceną), `flow`, `twist` itd. - nigdy z surową
ceną. RSI i backtest liczone na wygładzonej serii dają błędne,
opóźnione wyniki. **Naprawiono**: dodano `PriceSignal` i wymagany
parametr `price_signal` do `TimdrPacket`, `TimdrEngine.compute_packet()`
przekazuje surową cenę obok wygładzonej. Regresja:
`test_packet_ma_surowa_cene_nie_tylko_wygladzona`.

### 6. `przetworz_sygnaly()` zwracał tylko liczby, nie indeksy/cenę

Podczas budowy dashboardu okazało się, że funkcja zwracała tylko
`n_anomaly`/`n_defect`/`n_twist` (liczby przez `len()`), nigdy same
indeksy ani surową serię cen - dashboard nie miałby czym narysować
markerów na wykresie ani samej linii ceny. **Naprawiono**: dodano
`anomalies_idx`, `defect_idx`, `twist_idx` (listy indeksów) i `x`
(surowa cena) do zwracanego słownika.

### 7. `/api/state/clear` - błąd systemu plików nie powinien wywalać żądania

`os.remove()` w `StateStore.clear()`/`PredictionLog.clear()` może rzucić
`OSError`/`PermissionError` (zablokowany plik, brak uprawnień). Pierwotnie
propagowało się to jako wyjątek i 500 na endpoincie. **Naprawiono**:
`try/except OSError` z widocznym ostrzeżeniem na konsoli
(`[state.py] UWAGA: ...`) i zwróceniem `False` zamiast wywalenia całego
żądania - błąd jest widoczny, ale nieusunięcie jednego pliku stanu nie
powinno być fatalne dla reszty API.

### 8. KRYTYCZNY: port 5060 jest na liście "zakazanych portów" przeglądarek

Podczas weryfikacji `dashboard.html` na żywo (Node.js + prawdziwy
`fetch()`, ta sama metoda co przy innych dashboardach TIMDR w tym
zestawie) odkryto, że **port 5060 jest zablokowany przez samą
specyfikację Fetch** (używany przez SIP, razem z 5061, 6000, 6566 i
kilkoma innymi jest na liście "bad ports"/"restricted ports"
respektowanej zarówno przez przeglądarki - Chrome zwraca
`net::ERR_UNSAFE_PORT` - jak i przez `fetch()` w Node.js/undici, z tym
samym błędem "bad port"). Dashboard uruchomiony na porcie 5060 **nigdy
nie zdołałby wykonać ani jednego zapytania do własnego API** w żadnej
prawdziwej przeglądarce - to nie był tylko artefakt tego środowiska
testowego, tylko realny błąd, który ujawniłby się dopiero po dostarczeniu
użytkownikowi. **Naprawiono**: API i dashboard przełączone na port
**8060** (bezpieczny, spoza listy zakazanych portów). Zweryfikowano
end-to-end: pełny przepływ `dashboard.html` → `fetch()` → Flask →
JSON → renderowanie kart/wykresu/kaskady/panelu uczenia działa
poprawnie na 8060.

### 9. Brakowało ceny akcji i wykresu trendu w API/dashboardzie

`przetworz_sygnaly()` zwracała `x` (surową cenę, do rysowania linii i
markerów), ale nigdy pojedynczej, czytelnej wartości "ostatnia cena" ani
jej zmiany procentowej w okresie - dashboard nie pokazywał tego jako
osobnej karty, mimo że to najbardziej podstawowa informacja, jakiej
użytkownik szuka najpierw. Wykres pokazywał samą surową cenę z
markerami anomalii/defektów, bez żadnej linii trendu - mimo że silnik
TIMDR i tak już liczy `trm` (trend reference mean, mediana krocząca
k=5) na potrzeby `flow`/`twist`/`resonance`, więc trend był policzony,
tylko nigdzie nie pokazywany. **Naprawiono**: `przetworz_sygnaly()`
zwraca teraz `last_price`, `price_change_pct` (zmiana % między
pierwszą a ostatnią barą okresu) i `trend` (pełna seria `trm`).
Dashboard dostał nową kartę "Cena" (z podpisem zmiany %) oraz drugą
linię na wykresie ceny (`drawChart` obsługuje teraz `extraLines`) -
zielona linia trendu nałożona na niebieską linię surowej ceny.
Zweryfikowano end-to-end na żywym serwerze: pole `trend` ma tyle samo
punktów co `x`, karta "Cena" renderuje się poprawnie z realną wartością
i znakiem zmiany %.

### 10. Wykres ceny "wyłamywał się" poza kartę po zmianie rozmiaru okna

`drawChart()` mierzył dostępną szerokość raz, w momencie wywołania
`analyze()`, i nigdy nie przerysowywał wykresu później. Jeśli
użytkownik zmniejszył okno przeglądarki PO analizie, canvas zachowywał
stary, szerszy rozmiar ustawiony inline w JS - a `.panel` nie miał
`overflow:hidden`, więc linia wykresu wizualnie wystawała poza kartę i
poza stronę, zamiast się przyciąć. **Naprawiono dwutorowo**: (1)
`.panel{overflow:hidden}` jako siatka bezpieczeństwa - wykres nigdy
więcej nie może wizualnie "wyłamać się" poza swoją kartę, niezależnie
od błędów w liczeniu rozmiaru; (2) dodano nasłuch na `resize` okna
(z debounce 150ms), który przerysowuje ostatni wykres z aktualnym
rozmiarem - bez ponownego zapytania do `/api/analyze` (dane są już w
pamięci, patrz `lastChartOpts`). Zweryfikowano: po wywołaniu zdarzenia
`resize` canvas jest przerysowywany, a licznik wywołań `fetch()` się
nie zmienia (potwierdza brak zbędnego zapytania sieciowego).

### 11. Cena nie miała waluty/jednostki ani wolumenu

Karta "Cena" pokazywała samą liczbę (np. `84.93`) bez informacji, w
jakiej walucie i jakiej jednostce fizycznej - dla kontraktów
terminowych (np. `CL=F` - ropa WTI) to istotna różnica (USD za baryłkę,
nie "po prostu 84.93"). Wolumen w ogóle nie docierał do API, mimo że
był w OHLCV od początku. **Naprawiono**: `cascade.py` dostał
`currency_unit_for_ticker()` - heurystykę z sufiksu tickera (`=X` pary
walutowe → kwotowane w drugiej walucie, `.WA` → PLN, `-USD` → USD,
`=F` → USD + jednostka fizyczna ze słownika dla znanych symboli
(baryłka/uncja/MMBtu/buszel), `^` → punkty indeksowe bez waluty,
domyślnie USD). **Jawnie oznaczone jako przybliżenie**: to heurystyka z
konwencji nazewnictwa Yahoo Finance, NIE autorytatywne dane pobrane z
API - `yfinance.download()` w tym pipeline zwraca tylko OHLCV, nie
metadane instrumentu; osobne zapytanie o `Ticker.info` dla każdej
analizy byłoby kolejnym wolnym wywołaniem sieciowym per żądanie (patrz
lekcja #3 ze Synoptyka - nie pobieraj bez potrzeby). `api.py` dostał
`last_volume`/`avg_volume` z tej samej ramki OHLCV, którą i tak już
pobiera. Dashboard: nowa karta "Wolumen", karta "Cena" pokazuje teraz
jednostkę (np. "116.35 USD/baryłka (ropa WTI)"). Dodano trwałe testy
regresyjne (`test_cascade.py` - 8 nowych testów `currency_unit_for_*`)
oraz **nowy plik `test_api.py`** (wcześniej nie istniał - `api.py` był
dotąd testowany tylko ręcznie; teraz ma 6 testów pytest z zamockowanym
yfinance, w tym regresję na obecność `last_price`/`last_volume`/
`currency`/`price_unit_label` w odpowiedzi).

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
- `POST /api/state/clear` `{"ticker": "AAPL"}` - czyści stan + log predykcji dla tickera

## Testy

```
python -m pytest -q
```

74/74 testy przechodzą (`timdr_core_finance`, `analizator_gieldowy`
pośrednio przez `pipeline`, `pipeline`, `cascade`, `data_loader`, `state`,
`api`). Wszystkie testy `data_loader`/`api` mockują `yfinance` (brak
zależności od sieci przy testowaniu) - realne pobieranie danych giełdowych
wymaga
połączenia internetowego przy faktycznym uruchomieniu.

## Ograniczenia

- Backtest i wsteczne metryki (Sharpe/Winrate/DD) opisują wyłącznie
  przeszłość analizowanego okna - nie są prognozą przyszłych wyników.
- Kaskada kapitałowa to model korelacyjny (przepływ ceny/wolumenu
  między klasami aktywów), nie przyczynowy - koreluje, nie dowodzi
  przyczynowości.
- Samo-uczenie zaczyna dawać sensowne korekty dopiero po zgromadzeniu
  ≥5 potwierdzonych predykcji na dany horyzont (przy analizie raz
  dziennie to zajmuje tydzień+).
