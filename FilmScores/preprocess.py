"""
Preprocessing ocen filmów: scores.xlsx -> ratings.json
=====================================================

Problem
-------
Oceny zebrane od grupy (plik ``data/scores.xlsx``) mają postać wierszy
``Imię Nazwisko | tytuł | ocena | tytuł | ocena | ...`` w skali 1–10.
Te same filmy są zapisywane na różne sposoby: tytuł polski lub oryginalny
("Dom z papieru" / "La Casa De Papel"), literówki ("Interstelar"),
różna wielkość liter ("skazani na shawshank") czy warianty zapisu
("Rick & Morty" / "Rick i Morty"). Bez ujednolicenia metryki podobieństwa
nie widzą, że dwie osoby oceniły ten sam film.

Rozwiązanie
-----------
1. Wczytanie arkusza (openpyxl) do listy ocen ``(użytkownik, tytuł, ocena)``.
2. Normalizacja tekstu tytułu (małe litery, bez polskich znaków i interpunkcji)
   i dopasowanie go do katalogu ``data/catalog.json`` zbudowanego z OMDb
   przez ``build_catalog.py``. Katalog zawiera dla każdego filmu ``imdbID``
   oraz listę zapisów tego filmu spotkanych w ankiecie (``raw_titles``).
3. Scalanie duplikatów: jeśli użytkownik ocenił ten sam film kilka razy
   (np. "Rick & Morty" i "Rick i Morty"), oceny są uśredniane.
4. Zapis do ``data/ratings.json`` w strukturze zgodnej z demo
   (``{użytkownik: {film: ocena}}``), gdzie kluczem filmu jest ``imdbID``.
   Jednoznaczny identyfikator rozwiązuje problem języka, remake'ów
   i powtarzających się tytułów.

Autorzy
-----
Jakub Kolad, Igor Czeczot

Użycie
------
    python build_catalog.py        # najpierw: katalog z OMDb
    python preprocess.py
    python preprocess.py --input data/scores.xlsx --catalog data/catalog.json \
                         --output data/ratings.json

Skrypt kończy się błędem, jeśli w arkuszu jest tytuł, którego nie ma
w katalogu (np. po dopisaniu nowych ocen). Wtedy należy uruchomić ponownie
``build_catalog.py``.
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict

import openpyxl


def normalize_title(title):
    """Sprowadza tytuł do postaci porównywalnej (klucz dopasowania).

    Zamienia na małe litery, usuwa znaki diakrytyczne (ł -> l, ś -> s),
    interpunkcję oraz nadmiarowe spacje.

    :param title: surowy tytuł (str lub liczba, np. 1670 z Excela)
    :return: znormalizowany tytuł, np. ``"Zadzwoń do Saula" -> "zadzwon do saula"``
    """
    text = str(title).strip().casefold().replace("ł", "l")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_xlsx(path):
    """Wczytuje surowe oceny z arkusza.

    Oczekiwany układ: wiersz 1 to nagłówek, każdy kolejny wiersz to
    ``Imię Nazwisko, tytuł, ocena, tytuł, ocena, ...`` (puste komórki są
    pomijane).

    :param path: ścieżka do pliku .xlsx
    :return: lista krotek ``(użytkownik, surowy_tytuł, ocena)``
    """
    sheet = openpyxl.load_workbook(path, read_only=True).active
    records = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        cells = [c for c in row if c is not None and str(c).strip() != ""]
        if not cells:
            continue
        user, rest = str(cells[0]).strip(), cells[1:]
        if len(rest) % 2:
            raise ValueError(f"Nieparzysta liczba komórek tytuł/ocena u: {user}")
        for title, rating in zip(rest[::2], rest[1::2]):
            records.append((user, str(title).strip(), float(rating)))
    return records


def build_title_index(catalog):
    """Tworzy indeks ``znormalizowany tytuł z ankiety -> imdbID``.

    :param catalog: słownik z ``catalog.json`` (``imdbID -> wpis``)
    :return: słownik ``{znormalizowany_tytuł: imdbID}``
    :raises ValueError: gdy ten sam tytuł wskazuje na dwa różne filmy
    """
    index = {}
    for imdb_id, entry in catalog.items():
        for raw in entry["raw_titles"]:
            key = normalize_title(raw)
            if index.get(key, imdb_id) != imdb_id:
                raise ValueError(f"Tytuł '{raw}' wskazuje na {index[key]} i {imdb_id}")
            index[key] = imdb_id
    return index


def unify(records, title_index):
    """Mapuje tytuły z ankiety na ``imdbID`` i scala duplikaty (średnia).

    :param records: lista ``(użytkownik, surowy_tytuł, ocena)``
    :param title_index: wynik :func:`build_title_index`
    :return: krotka ``(ratings, unmapped, duplicates)``, gdzie ``ratings`` ma
             strukturę ``{użytkownik: {imdbID: ocena}}``
    """
    collected = defaultdict(lambda: defaultdict(list))
    unmapped = set()
    for user, raw_title, rating in records:
        canonical = title_index.get(normalize_title(raw_title))
        if canonical is None:
            unmapped.add(raw_title)
            continue
        collected[user][canonical].append(rating)

    ratings, duplicates = {}, []
    for user, movies in collected.items():
        ratings[user] = {}
        for title, values in movies.items():
            if len(values) > 1:
                duplicates.append((user, title, values))
            ratings[user][title] = round(sum(values) / len(values), 2)
    return ratings, sorted(unmapped), duplicates


def main():
    """Punkt wejścia CLI."""
    parser = argparse.ArgumentParser(
        description="Ujednolicenie ocen: xlsx -> ratings.json"
    )
    parser.add_argument("--input", default="data/scores.xlsx")
    parser.add_argument("--catalog", default="data/catalog.json")
    parser.add_argument("--output", default="data/ratings.json")
    args = parser.parse_args()

    with open(args.catalog, encoding="utf-8") as f:
        catalog = json.load(f)

    records = load_xlsx(args.input)
    ratings, unmapped, duplicates = unify(records, build_title_index(catalog))

    if unmapped:
        print("Brak w katalogu (uruchom ponownie build_catalog.py):", file=sys.stderr)
        for title in unmapped:
            print(f"  - {title}", file=sys.stderr)
        sys.exit(1)

    for user, title, values in duplicates:
        print(
            f"Uwaga: {user} ocenił(a) '{catalog[title]['title']}' kilka razy {values} -> średnia"
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(ratings, f, ensure_ascii=False, indent=4)

    raw_titles = len({normalize_title(t) for _, t, _ in records})
    used = len({t for movies in ratings.values() for t in movies})
    print(
        f"Użytkownicy: {len(ratings)}, oceny: {len(records)}, "
        f"unikalne tytuły: {raw_titles} surowych -> {used} po ujednoliceniu"
    )
    print(f"Zapisano: {args.output}")


if __name__ == "__main__":
    main()
