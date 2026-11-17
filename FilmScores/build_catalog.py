"""
Budowa katalogu filmów z OMDb API: scores.xlsx -> data/catalog.json
===================================================================

Problem
-------
Ankieta zawiera 300+ różnie zapisanych tytułów (polskie, oryginalne,
z literówkami). Aby porównywać oceny różnych osób, każdy tytuł trzeba
sprowadzić do jednego, jednoznacznego filmu lub serialu. Dodatkowo chcemy
mieć o filmach informacje spoza ankiety (gatunek, reżyser, ocena IMDb, opis).

Rozwiązanie
-----------
Źródłem danych o filmie jest **OMDb API**. Skrypt:

1. wczytuje wszystkie unikalne tytuły z ``data/scores.xlsx``,
2. dla każdego odpytuje OMDb. OMDb nie zna większości polskich tytułów, więc
   zapytanie można ukierunkować podpowiedzią z ``data/title_hints.json``:
   ``"Dom z papieru": {"query": "Money Heist", "year": 2017, "type": "series"}``.
   Podpowiedź to wyłącznie *wskazówka wyszukiwania* (tłumaczenie tytułu, rok
   do rozróżnienia remake'ów). Wszystkie dane o filmie pochodzą z odpowiedzi OMDb,
3. próbuje po kolei: ``t=tytuł&y=rok&type=typ`` -> ``t=tytuł`` (bez roku) ->
   wyszukiwanie ``s=tytuł``. Dopasowania mniej pewne trafiają do raportu,
4. tytuły, które OMDb wskazało jako ten sam ``imdbID`` (np. "Rick & Morty"
   i "Rick i Morty"), są automatycznie scalane,
5. zapisuje ``data/catalog.json`` (klucz = ``imdbID``) oraz
   ``data/catalog_report.txt`` z listą wszystkich dopasowań, ostrzeżeń
   i tytułów, których nie udało się dopasować.

Wszystkie odpowiedzi API są zapisywane w ``data/omdb_cache.json``, więc
ponowne uruchomienie nie zużywa limitu zapytań.

Autorzy
-----
Jakub Kolad
Igor Czeczot

Użycie
------
    export OMDB_API_KEY=twoj_klucz            # Windows: $env:OMDB_API_KEY="twoj_klucz"
    python build_catalog.py
    python build_catalog.py --api-key KLUCZ

Jeśli raport wskaże niedopasowane tytuły lub błędne dopasowania, popraw wpis
w ``data/title_hints.json`` i uruchom skrypt ponownie. Zapytania już
wykonane są brane z cache.
"""

import argparse
import json
import re
import sys

from omdb_client import OmdbClient
from preprocess import load_xlsx, normalize_title


def parse_year(value):
    """Wyciąga rok początkowy z pola ``Year`` OMDb (np. ``"2008–2013"`` -> 2008).

    :param value: tekst z odpowiedzi OMDb
    :return: rok jako liczba albo ``None``
    """
    match = re.match(r"\d{4}", value or "")
    return int(match.group()) if match else None


def resolve(client, raw_title, hint):
    """Dopasowuje jeden tytuł z ankiety do filmu w OMDb.

    :param client: instancja :class:`OmdbClient`
    :param raw_title: tytuł z ankiety
    :param hint: podpowiedź wyszukiwania (``query``, ``year``, ``type``) lub ``{}``
    :return: krotka ``(dane_omdb, sposób_dopasowania, ostrzeżenia)``; ``dane_omdb``
             to ``None``, jeśli nie znaleziono
    """
    query = hint.get("query", raw_title)
    year, kind = hint.get("year"), hint.get("type")

    data = client.find(query, year, kind)
    if data:
        return data, "t+rok" if year else "t", []

    if year:
        data = client.find(query, None, kind)
        if data:
            return (
                data,
                "t (bez roku)",
                [f"rok w OMDb ({data.get('Year')}) różni się od podpowiedzi ({year})"],
            )

    candidates = client.search(query, year, kind) or client.search(query, None, kind)
    if candidates:
        data = client.by_id(candidates[0]["imdbID"])
        if data:
            return (
                data,
                "wyszukiwanie",
                [f"wzięto pierwszy wynik wyszukiwania z {len(candidates)} kandydatów"],
            )
    return None, None, []


def build_entry(data):
    """Wybiera z odpowiedzi OMDb pola zapisywane w katalogu.

    :param data: odpowiedź OMDb
    :return: słownik wpisu katalogu (bez listy tytułów z ankiety)
    """
    return {
        "title": data.get("Title"),
        "year": parse_year(data.get("Year")),
        "type": data.get("Type"),
        "genre": data.get("Genre"),
        "director": data.get("Director"),
        "runtime": data.get("Runtime"),
        "imdb_rating": data.get("imdbRating"),
        "plot": data.get("Plot"),
    }


def write_report(path, matches, warnings, unresolved, merged):
    """Zapisuje raport z budowy katalogu.

    :param path: ścieżka pliku raportu
    :param matches: lista ``(tytuł_z_ankiety, dane_omdb, sposób)``
    :param warnings: lista ``(tytuł_z_ankiety, ostrzeżenie)``
    :param unresolved: lista niedopasowanych tytułów
    :param merged: słownik ``imdbID -> [tytuły z ankiety]`` dla scalonych pozycji
    """
    lines = ["RAPORT BUDOWY KATALOGU", "=" * 60, ""]
    lines.append(
        f"Niedopasowane tytuły ({len(unresolved)}): dopisz podpowiedź w data/title_hints.json"
    )
    lines += [f"  - {t}" for t in unresolved] or ["  (brak)"]
    lines += ["", f"Do sprawdzenia ({len(warnings)}):"]
    lines += [f"  - {t}: {w}" for t, w in warnings] or ["  (brak)"]
    lines += ["", f"Scalone pozycje ({len(merged)}): różne zapisy tego samego filmu"]
    for imdb_id, titles in sorted(merged.items()):
        lines.append(f"  - {imdb_id}: {' | '.join(titles)}")
    lines += [
        "",
        f"Wszystkie dopasowania ({len(matches)}): tytuł z ankiety -> film w OMDb",
    ]
    for raw, data, how in sorted(matches, key=lambda m: normalize_title(m[0])):
        lines.append(
            f"  {raw}  ->  {data['Title']} ({data.get('Year')}, {data.get('Type')}) "
            f"[{data['imdbID']}] ({how})"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    """Punkt wejścia CLI."""
    parser = argparse.ArgumentParser(description="Budowa katalogu filmów z OMDb")
    parser.add_argument("--input", default="data/scores.xlsx")
    parser.add_argument("--hints", default="data/title_hints.json")
    parser.add_argument("--output", default="data/catalog.json")
    parser.add_argument("--report", default="data/catalog_report.txt")
    parser.add_argument("--api-key", help="klucz OMDb (lub zmienna OMDB_API_KEY)")
    args = parser.parse_args()

    with open(args.hints, encoding="utf-8") as f:
        hints = {normalize_title(k): v for k, v in json.load(f).items()}

    unique = {}
    for _, raw_title, _ in load_xlsx(args.input):
        unique.setdefault(normalize_title(raw_title), raw_title)

    client = OmdbClient(api_key=args.api_key)
    catalog, matches, warnings, unresolved = {}, [], [], []

    try:
        for i, (key, raw) in enumerate(unique.items(), 1):
            data, how, warns = resolve(client, raw, hints.get(key, {}))
            if data is None:
                unresolved.append(raw)
                print(f"[{i}/{len(unique)}] {raw}  ->  BRAK")
                continue
            matches.append((raw, data, how))
            warnings += [(raw, w) for w in warns]
            entry = catalog.setdefault(
                data["imdbID"], {**build_entry(data), "raw_titles": []}
            )
            entry["raw_titles"].append(raw)
            print(
                f"[{i}/{len(unique)}] {raw}  ->  {data['Title']} ({data.get('Year')}) [{how}]"
            )
    except RuntimeError as exc:
        sys.exit(
            f"\nPrzerwano: {exc}\nDotychczasowe odpowiedzi zapisano w cache, "
            f"po usunięciu problemu uruchom skrypt ponownie."
        )

    merged = {
        k: v["raw_titles"] for k, v in catalog.items() if len(v["raw_titles"]) > 1
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            dict(sorted(catalog.items(), key=lambda kv: kv[1]["title"].casefold())),
            f,
            ensure_ascii=False,
            indent=1,
        )
    write_report(args.report, matches, warnings, unresolved, merged)

    print(
        f"\nTytuły z ankiety: {len(unique)}, dopasowane: {len(matches)}, "
        f"unikalne filmy w katalogu: {len(catalog)}, "
        f"zapytań do API w tym uruchomieniu: {client.requests_made}"
    )
    print(f"Do sprawdzenia: {len(warnings)}, niedopasowane: {len(unresolved)}")
    print(f"Zapisano: {args.output}, raport: {args.report}")
    if unresolved:
        sys.exit(
            "Są niedopasowane tytuły: patrz raport. Preprocessing się nie powiedzie, "
            "dopóki nie zostaną dopasowane."
        )


if __name__ == "__main__":
    main()
