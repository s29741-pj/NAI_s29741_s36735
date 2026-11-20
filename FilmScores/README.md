# Silnik rekomendacji filmów i seriali (NAI)

**Autorzy:** Jakub Kolad, Igor Czeczot

## Problem

Grupa 14 osób oceniła filmy i seriale w skali 1–10 (`data/scores.xlsx`).
Dla wybranego użytkownika silnik:

- proponuje **5 tytułów, których nie oglądał**, a które powinny mu się spodobać (rekomendacje),
- wskazuje **5 tytułów, których powinien unikać** (antyrekomendacje),
- porównuje wyniki dla **dwóch metryk** podobieństwa: odległości euklidesowej i korelacji Pearsona (bonus),
- uzupełnia każdy tytuł danymi z **OMDb API**: gatunkiem, reżyserem, oceną IMDb i opisem (bonus).

## Jak to działa

Potok ma trzy kroki: `build_catalog.py` → `preprocess.py` → `recommender.py`.

1. **Katalog filmów z OMDb** (`build_catalog.py`). Ta sama pozycja była w ankiecie zapisywana różnie:
   po polsku lub w oryginale ("Dom z papieru" / "La Casa De Papel"), z literówkami ("Interstelar")
   albo w wariantach ("Rick & Morty" / "Rick i Morty"). Skrypt odpytuje OMDb o każdy unikalny tytuł
   i zapisuje w `data/catalog.json` dane z odpowiedzi API: `imdbID`, tytuł, rok, typ, gatunek, reżysera,
   ocenę IMDb i opis. Różne zapisy tego samego filmu scalają się, bo OMDb zwraca dla nich ten sam `imdbID`.
   OMDb nie zna większości polskich tytułów, dlatego zapytanie można ukierunkować podpowiedzią
   z `data/title_hints.json` (np. `"Dom z papieru" → "Money Heist", 2017, series`). Podpowiedź to tylko
   wskazówka wyszukiwania (tłumaczenie tytułu, rok do rozróżnienia remake'ów), a wszystkie dane o filmie
   pochodzą z OMDb. Kolejne próby, gdy podpowiedź zawiedzie: zapytanie bez roku, potem wyszukiwanie.
   Każde dopasowanie trafia do raportu `data/catalog_report.txt`, a mniej pewne są wyróżnione.
2. **Ujednolicenie ocen** (`preprocess.py`). Tytuły z ankiety są normalizowane (małe litery, bez polskich
   znaków i interpunkcji) i mapowane na `imdbID`. Gdy ktoś ocenił ten sam film dwa razy, oceny są
   uśredniane. Wynik to `data/ratings.json` w strukturze zgodnej z demo z zajęć, `{użytkownik: {film: ocena}}`,
   z `imdbID` jako kluczem filmu.
3. **Rekomendacje** (`recommender.py`). Podobieństwo użytkowników liczone jest na wspólnie ocenionych filmach:
   - *Euclidean*: `1 / (1 + RMSE)`. W demo była suma kwadratów, która rośnie z liczbą wspólnych filmów,
     więc tu użyto średniej.
   - *Pearson*: korelacja. Nie zależy od tego, czy ktoś ocenia ogólnie surowo, czy łagodnie.
   - Obie metryki są osłabiane, gdy wspólnych tytułów jest mniej niż 5 (significance weighting).
     Bez tego osoby z jednym wspólnym filmem wyglądałyby na "idealnie podobne".

   Przewidywana ocena (wzór Resnicka): `pred = średnia(u) + Σ sim·(ocena_v − średnia(v)) / (Σ sim + 0.5)`.
   Rekomendacje to tytuły z najwyższą przewidywaną oceną, antyrekomendacje z najniższą.

## Instalacja i użycie

```bash
pip install -r requirements.txt                    # openpyxl + certifi

export OMDB_API_KEY=twoj_klucz                     # Windows PowerShell: $env:OMDB_API_KEY="twoj_klucz"
python build_catalog.py                            # scores.xlsx + OMDb -> data/catalog.json + raport
python preprocess.py                               # scores.xlsx + catalog.json -> data/ratings.json

python recommender.py --list-users                 # lista użytkowników
python recommender.py --user "Jakub K., Igor C."         # obie metryki + porównanie
```

Klucz OMDb można też podać jako `--api-key KLUCZ`. Potrzebny jest tylko w kroku `build_catalog.py`.
Odpowiedzi API są zapisywane w `data/omdb_cache.json`, więc ponowne uruchomienie nie zużywa limitu
(1000 zapytań dziennie), a `catalog.json`, `ratings.json` i cache są częścią repozytorium.
Dzięki temu `recommender.py` działa bez klucza i bez internetu.

Jeśli `catalog_report.txt` wskaże niedopasowane lub błędnie dopasowane tytuły, popraw wpis
w `data/title_hints.json` i uruchom `build_catalog.py` ponownie. Zapytania już wykonane są brane z cache.

## Przykładowy wynik (fragment)

```
Użytkownik: Jakub K., Igor C. (41 ocen, średnia 7.68)

  Rekomendacje:
   Euclidean                            | Pearson
   The Shawshank Redemption (10.0)      | The Shawshank Redemption (10.0)
   Interstellar (9.7)                   | Hacksaw Ridge (10.0)
   Shrek (9.7)                          | Suits (10.0)
   Hacksaw Ridge (9.6)                  | Shrek (10.0)
   The Truman Show (9.6)                | The Truman Show (9.9)
   Wspólne pozycje: 4/5

  Antyrekomendacje:
   Euclidean                            | Pearson
   Fifty Shades of Grey (4.1)           | Fifty Shades of Grey (3.1)
   Who Killed Captain Alex? (4.6)       | Silent Night (3.2)
   Silent Night (4.6)                   | 365 Days (3.9)
   Borderlands (4.7)                    | The Human Centipede (First Seq (3.9)
   365 Days (4.8)                       | Snow White (4.0)
   Wspólne pozycje: 3/5
```

Przy każdej pozycji program wypisuje też gatunek, długość, reżysera, ocenę IMDb, link i opis z OMDb.

## Porównanie metryk: wnioski

- Listy są w dużej mierze zgodne, bo tytuły lubiane lub nielubiane przez większość grupy wygrywają przy obu metrykach.
- **Pearson** daje bardziej skrajne prognozy i chętniej promuje tytuły polecone przez jedną, bardzo podobną osobę
  (np. *Suits*). Korelacja patrzy tylko na kierunek zgodności gustu, nie na to, czy oceny są w tej samej skali.
- **Euclidean** jest ostrożniejszy. Podobieństwo nigdy nie spada do zera, więc w prognozie biorą udział
  wszyscy użytkownicy, a wynik bardziej zbliża się do opinii całej grupy.

## Struktura repozytorium

| Plik | Opis |
|---|---|
| `data/scores.xlsx` | surowe oceny z ankiety (1–10) |
| `data/title_hints.json` | podpowiedzi wyszukiwania dla OMDb (tłumaczenie tytułu, rok, typ) |
| `data/catalog.json` | katalog filmów zbudowany z OMDb, klucz = `imdbID` (generowany) |
| `data/catalog_report.txt` | raport z budowy katalogu: dopasowania, ostrzeżenia, braki (generowany) |
| `data/ratings.json` | ujednolicone oceny używane przez silnik (generowany przez `preprocess.py`) |
| `data/omdb_cache.json` | cache odpowiedzi OMDb (tworzony przez `build_catalog.py`) |
| `build_catalog.py` | budowa katalogu z OMDb |
| `preprocess.py` | wczytanie xlsx, normalizacja i ujednolicenie tytułów |
| `recommender.py` | metryki, predykcja, rekomendacje, antyrekomendacje, porównanie |
| `omdb_client.py` | klient OMDb API z cache |

## Referencje

- Demo z zajęć: `compute_scores.py`, `ratings.json`
- I. Chatterjee, [A Comparative Study of Clustering Algorithms](https://medium.com/analytics-vidhya/comparative-study-of-the-clustering-algorithms-54d1ed9ea732), Analytics Vidhya
- P. Resnick i in., *GroupLens: An Open Architecture for Collaborative Filtering of Netnews*, CSCW 1994
- J. Herlocker i in., *An Algorithmic Framework for Performing Collaborative Filtering*, SIGIR 1999
- [OMDb API](https://www.omdbapi.com/)

## Ewaluacja algorytmu (Leave-One-Out)

Ocenianie rekomendacji „na oko” nie mówi, jak dobrze algorytm przewiduje gusty. Dlatego napisaliśmy `test_recommender.py`, który sprawdza trafność predykcji na danych, które już mamy.

### Jak działa Leave-One-Out

W metodzie Leave-One-Out każda ocena jest po kolei ukrywana, a algorytm ma ją odtworzyć na podstawie pozostałych danych:

- Każdy eksperyment tymczasowo usuwa dokładnie **jedną** ocenę. Wszystkie pozostałe oceny są zachowane.
- Pearson i Euclidean dostają **identyczne** dane treningowe.
- Istniejąca funkcja `predict_ratings()` nie jest zmieniana. Próbuje odtworzyć usuniętą ocenę.
- Eksperyment powtarzamy dla każdej z 455 ocen w zbiorze (14 użytkowników).
- Gdy metryka nie daje predykcji, przypadek wlicza się do Coverage, ale jest pomijany w MAE i RMSE.

Metryki:

- **MAE** (*Mean Absolute Error*): średni bezwzględny błąd predykcji. Im mniej, tym lepiej.
- **RMSE** (*Root Mean Squared Error*): pierwiastek ze średniego kwadratu błędów. Mocniej karze duże pomyłki.
- **Coverage**: procent ukrytych ocen, które algorytm zdołał przewidzieć.

### Wyniki

Wyniki z `python3 test_recommender.py` dla aktualnej implementacji (`MIN_COMMON_PEARSON = 2`, `MIN_COMMON_EUCLIDEAN = 1`):

| Metryka | Pearson | Euclidean | Znaczenie |
|---|---:|---:|---|
| Predykcje | 235 | 270 | Liczba ocen, które algorytm zdołał przewidzieć spośród 455 prób. |
| Coverage | 51,6% | 59,3% | Odsetek prób zakończonych uzyskaniem predykcji. Im wyższy, tym lepiej. |
| MAE (średni błąd bezwzględny) | **1,478** | 1,604 | Średnia różnica między oceną rzeczywistą a przewidywaną, wyrażona w punktach skali 1–10. Im niższy wynik, tym dokładniejsze predykcje. |
| RMSE (pierwiastek średniego błędu kwadratowego) | **1,887** | 2,026 | Miara błędu, która silniej karze duże pomyłki. Im niższy wynik, tym lepiej. |
|

Przykłady z `test_results/results.csv`, wybrane tak, by pokazać zarówno trafne predykcje, jak i ograniczenia:

| Użytkownik | Film | Ocena rzeczywista | Pearson | Euclidean |
|---|---|---|---|---|
| Karol S. | Peaky Blinders | 8 | 7.94 | 7.78 |
| Karol S. | Forrest Gump | 8 | 8.10 | 8.00 |
| Krysia C. | 1670 | 5 | 5.43 | 5.01 |
| Krysia C. | One Battle After Another | 5 | 5.09 | 5.36 |
| Paweł C. | Rick and Morty | 10 | 9.09 | 7.42 |
| Marceli G. | Other Mommy | 3 | — | 6.93 |
| Paweł C. | Lucifer | 3 | — | — |
| Karol S. | The Pacific | 9 | — | — |

- Pearson trafniejszy: *Peaky Blinders*, *One Battle After Another*, *Rick and Morty*.
- Euclidean trafniejszy: *Forrest Gump*, *1670*.
- Pearson nie przewiduje, a Euclidean tak: *Other Mommy* (duży błąd, 3.93 pkt).
- Żadna metryka nie przewiduje: *Lucifer*, *The Pacific*.

### Interpretacja

- Pearson ma niższe MAE na dostępnych predykcjach (1.478 wobec 1.604).
- Euclidean ma wyższe Coverage (59.3% wobec 51.6%).
- Metody nie przewidują dokładnie tego samego podzbioru ocen. Pearson przewidział 235 ocen, a Euclidean 270. Każdą ocenę przewidzianą przez Pearsona przewidział także Euclidean, więc 35 dodatkowych przypadków dotyczy tylko Euclidean. Ogólne MAE nie jest więc ścisłym porównaniem.

Porównanie na wspólnym podzbiorze (235 ocen przewidzianych przez obie metryki):

- MAE Pearson: 1.478 (taka sama wartość jak w pełnych wynikach, bo zbiór Pearsona jest podzbiorem zbioru Euclidean),
- MAE Euclidean: 1.527.

Dlaczego Coverage jest ograniczone: zbiór ma 14 użytkowników i 455 ocen, a aż 180 z nich dotyczy filmów ocenionych tylko przez jedną osobę. Po ukryciu takiej oceny żaden inny użytkownik nie ma tego filmu, więc User-Based Collaborative Filtering nie ma z czego przewidzieć oceny. Przy tym zbiorze Coverage nie może więc przekroczyć 275/455 (ok. 60.4%).

### Powtórzenie eksperymentu

```bash
python3 test_recommender.py
```

Szczegółowe wyniki dla każdej oceny są zapisywane w `test_results/results.csv`. Skrypt nadpisuje ten plik przy każdym uruchomieniu.

