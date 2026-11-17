"""
Silnik rekomendacji filmów i seriali (collaborative filtering)
==============================================================

Problem
-------
Na podstawie ocen (1–10) wystawionych przez grupę osób zaproponować
wybranemu użytkownikowi 5 filmów/seriali, których nie oglądał, a które
powinny mu się spodobać (rekomendacje), oraz 5, których powinien unikać
(antyrekomendacje).

Rozwiązanie
-----------
User-based collaborative filtering, rozwinięcie demo z zajęć
(``compute_scores.py``):

1. Podobieństwo użytkownika do każdej innej osoby liczone jest jedną
   z dwóch metryk, wyłącznie na filmach ocenionych przez obie osoby:

   * **Euclidean**: ``1 / (1 + RMSE)`` różnic ocen. W odróżnieniu od demo
     użyto średniej kwadratów zamiast sumy, bo suma rośnie z liczbą
     wspólnych filmów i "karze" osoby, które mają ich dużo.
   * **Pearson**: współczynnik korelacji Pearsona. Mierzy zgodność gustu
     niezależnie od tego, czy ktoś ocenia ogólnie surowo, czy łagodnie.

   Oba wyniki są mnożone przez ``min(n, SHRINK) / SHRINK``, gdzie ``n``
   to liczba wspólnych filmów (tzw. significance weighting). Dzięki temu
   dwie osoby z jednym wspólnym filmem nie są uznane za "idealnie podobne".

2. Przewidywana ocena filmu ``m`` dla użytkownika ``u`` (wzór Resnicka,
   ze średnimi użytkowników jako punktem odniesienia):

       pred(u, m) = mean(u) + Σ sim(u,v) * (r(v,m) - mean(v)) / (Σ sim(u,v) + DAMPING)

   Sumy obejmują osoby z dodatnim podobieństwem, które oceniły ``m``.
   ``DAMPING`` ściąga do średniej użytkownika przewidywania oparte na
   pojedynczej, słabo podobnej osobie, więc wyżej trafiają filmy
   potwierdzone przez kilka podobnych osób.

3. Rekomendacje to filmy nieobejrzane przez użytkownika z najwyższą
   przewidywaną oceną, antyrekomendacje to te z najniższą.

4. (Bonus) Wyniki dla obu metryk są zestawiane obok siebie, a każdy film
   jest uzupełniony danymi z OMDb API (gatunek, reżyser, ocena IMDb, opis).
   Dane te pobiera ``build_catalog.py`` i zapisuje w ``data/catalog.json``,
   a filmy w ``ratings.json`` są identyfikowane przez ``imdbID``.

Autorzy
-----
Jakub Kolad
Igor Czeczot

Użycie
------
    pip install -r requirements.txt
    python build_catalog.py --api-key KLUCZ           # xlsx + OMDb -> data/catalog.json
    python preprocess.py                              # xlsx -> data/ratings.json
    python recommender.py --list-users
    python recommender.py --user "Jakub Kolad"        # obie metryki + dane o filmach
    python recommender.py --user "Jakub Kolad" --score-type Pearson

Pliki ``data/catalog.json`` i ``data/ratings.json`` są częścią repozytorium,
więc samo ``recommender.py`` działa bez klucza OMDb i bez internetu.


"""

import argparse
import json
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RATINGS_FILE = os.path.join(BASE_DIR, "data", "ratings.json")
CATALOG_FILE = os.path.join(BASE_DIR, "data", "catalog.json")

MIN_COMMON = 2  # minimalna liczba wspólnych filmów, by liczyć podobieństwo
SHRINK = 5  # od tylu wspólnych filmów podobieństwo nie jest osłabiane
DAMPING = 0.5  # ściąganie przewidywań słabo potwierdzonych do średniej
METRICS = ("Euclidean", "Pearson")


def load_json(path):
    """Wczytuje plik JSON (UTF-8).

    :param path: ścieżka do pliku
    :return: zdekodowany obiekt
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def common_movies(dataset, user1, user2):
    """Zwraca listę filmów ocenionych przez obu użytkowników.

    :param dataset: słownik ``{użytkownik: {film: ocena}}``
    :param user1: pierwszy użytkownik
    :param user2: drugi użytkownik
    :raises KeyError: gdy któregoś użytkownika nie ma w danych
    """
    for user in (user1, user2):
        if user not in dataset:
            raise KeyError(f"Nie ma użytkownika '{user}' w danych")
    return [m for m in dataset[user1] if m in dataset[user2]]


def euclidean_score(dataset, user1, user2):
    """Podobieństwo euklidesowe ``1 / (1 + RMSE)``, w zakresie (0, 1].

    :param dataset: słownik ``{użytkownik: {film: ocena}}``
    :param user1: pierwszy użytkownik
    :param user2: drugi użytkownik
    :return: krotka ``(podobieństwo, liczba_wspólnych_filmów)``; 0, gdy
             wspólnych filmów jest mniej niż ``MIN_COMMON``
    """
    common = common_movies(dataset, user1, user2)
    if len(common) < MIN_COMMON:
        return 0.0, len(common)
    mse = sum((dataset[user1][m] - dataset[user2][m]) ** 2 for m in common) / len(
        common
    )
    return 1 / (1 + math.sqrt(mse)), len(common)


def pearson_score(dataset, user1, user2):
    """Współczynnik korelacji Pearsona ocen wspólnych filmów, w zakresie [-1, 1].

    :param dataset: słownik ``{użytkownik: {film: ocena}}``
    :param user1: pierwszy użytkownik
    :param user2: drugi użytkownik
    :return: krotka ``(korelacja, liczba_wspólnych_filmów)``; 0, gdy
             wspólnych filmów jest za mało lub któraś osoba ocenia je identycznie
    """
    common = common_movies(dataset, user1, user2)
    n = len(common)
    if n < MIN_COMMON:
        return 0.0, n
    x = [dataset[user1][m] for m in common]
    y = [dataset[user2][m] for m in common]
    mean_x, mean_y = sum(x) / n, sum(y) / n
    sxy = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    sxx = sum((a - mean_x) ** 2 for a in x)
    syy = sum((b - mean_y) ** 2 for b in y)
    if sxx * syy == 0:
        return 0.0, n
    return sxy / math.sqrt(sxx * syy), n


SCORE_FUNCTIONS = {"Euclidean": euclidean_score, "Pearson": pearson_score}


def similarity(dataset, user1, user2, metric):
    """Podobieństwo osłabione dla małej liczby wspólnych filmów.

    :param dataset: słownik ``{użytkownik: {film: ocena}}``
    :param user1: pierwszy użytkownik
    :param user2: drugi użytkownik
    :param metric: ``"Euclidean"`` lub ``"Pearson"``
    :return: krotka ``(ważone_podobieństwo, surowe_podobieństwo, liczba_wspólnych)``
    """
    raw, n = SCORE_FUNCTIONS[metric](dataset, user1, user2)
    return raw * min(n, SHRINK) / SHRINK, raw, n


def similar_users(dataset, user, metric):
    """Ranking pozostałych użytkowników według podobieństwa do ``user``.

    :param dataset: słownik ``{użytkownik: {film: ocena}}``
    :param user: użytkownik docelowy
    :param metric: ``"Euclidean"`` lub ``"Pearson"``
    :return: lista ``(inny_użytkownik, ważone, surowe, wspólne)`` malejąco
    """
    scores = [
        (other, *similarity(dataset, user, other, metric))
        for other in dataset
        if other != user
    ]
    return sorted(scores, key=lambda s: s[1], reverse=True)


def clip(score, low=1.0, high=10.0):
    """Przycina przewidywaną ocenę do skali ankiety (1–10).

    :param score: przewidywana ocena
    :return: ocena z przedziału ``[low, high]``
    """
    return min(high, max(low, score))


def mean_rating(ratings):
    """Średnia ocen jednego użytkownika.

    :param ratings: słownik ``{film: ocena}``
    """
    return sum(ratings.values()) / len(ratings)


def predict_ratings(dataset, user, metric):
    """Przewiduje oceny wszystkich filmów, których ``user`` nie ocenił.

    :param dataset: słownik ``{użytkownik: {film: ocena}}``
    :param user: użytkownik docelowy
    :param metric: ``"Euclidean"`` lub ``"Pearson"``
    :return: lista ``(film, przewidywana_ocena, liczba_oceniających)``;
             ocena nie jest przycinana do skali 1–10 (robi to dopiero
             prezentacja), żeby nie tworzyć sztucznych remisów w rankingu
    """
    user_mean = mean_rating(dataset[user])
    numerators, denominators, supporters = {}, {}, {}

    for other, sim, _, _ in similar_users(dataset, user, metric):
        if sim <= 0:
            continue
        other_mean = mean_rating(dataset[other])
        for movie, rating in dataset[other].items():
            if movie in dataset[user]:
                continue
            numerators[movie] = numerators.get(movie, 0.0) + sim * (rating - other_mean)
            denominators[movie] = denominators.get(movie, 0.0) + sim
            supporters[movie] = supporters.get(movie, 0) + 1

    predictions = []
    for movie, num in numerators.items():
        pred = user_mean + num / (denominators[movie] + DAMPING)
        predictions.append((movie, pred, supporters[movie]))
    return predictions


def recommend(dataset, user, metric, top=5, titles=None):
    """Zwraca rekomendacje i antyrekomendacje dla użytkownika.

    :param dataset: słownik ``{użytkownik: {film: ocena}}``
    :param user: użytkownik docelowy
    :param metric: ``"Euclidean"`` lub ``"Pearson"``
    :param top: liczba pozycji na każdej liście
    :param titles: słownik ``imdbID -> tytuł``, używany do rozstrzygania remisów
                   alfabetycznie (domyślnie według ``imdbID``)
    :return: krotka ``(rekomendacje, antyrekomendacje)``, listy
             ``(film, przewidywana_ocena, liczba_oceniających)``
    """
    predictions = predict_ratings(dataset, user, metric)
    name = (titles or {}).get
    best = sorted(predictions, key=lambda p: (-p[1], -p[2], name(p[0], p[0])))[:top]
    worst = sorted(predictions, key=lambda p: (p[1], -p[2], name(p[0], p[0])))[:top]
    return best, worst


def print_similar(dataset, user, metric, limit=3):
    """Wypisuje najbardziej podobnych użytkowników.

    :param dataset: słownik ``{użytkownik: {film: ocena}}``
    :param user: użytkownik docelowy
    :param metric: ``"Euclidean"`` lub ``"Pearson"``
    :param limit: liczba wypisanych osób
    """
    print(f"  Najbardziej podobni ({metric}):")
    for other, weighted, raw, n in similar_users(dataset, user, metric)[:limit]:
        print(
            f"    {other:<25} {weighted:6.3f}  (surowe {raw:6.3f}, wspólne filmy: {n})"
        )


def movie_title(catalog, movie_id):
    """Zwraca tytuł filmu z katalogu (lub ``imdbID``, gdy filmu nie ma w katalogu).

    :param catalog: katalog ``catalog.json``
    :param movie_id: identyfikator ``imdbID``
    """
    return catalog.get(movie_id, {}).get("title", movie_id)


def print_list(header, items, catalog):
    """Wypisuje listę filmów z przewidywaną oceną i danymi z OMDb.

    :param header: nagłówek sekcji
    :param items: lista ``(imdbID, przewidywana_ocena, liczba_oceniających)``
    :param catalog: katalog ``catalog.json`` (dane pobrane z OMDb)
    """
    print(f"\n  {header}")
    if not items:
        print(
            "   (brak: za mało wspólnych ocen z innymi użytkownikami, by wyznaczyć podobieństwo)"
        )
    for i, (movie, pred, support) in enumerate(items, 1):
        info = catalog.get(movie, {})
        kind = "serial" if info.get("type") == "series" else "film"
        print(
            f"   {i}. {movie_title(catalog, movie)} ({info.get('year', '?')}, {kind}): "
            f"przewidywana ocena {clip(pred):.1f}/10, na podstawie {support} os."
        )
        if info:
            print(
                f"      {info.get('genre')}, {info.get('runtime')} | reżyseria: {info.get('director')} "
                f"| IMDb: {info.get('imdb_rating')}/10 | https://www.imdb.com/title/{movie}/"
            )
            print(f"      {info.get('plot')}")


def print_comparison(results, top, catalog):
    """Zestawia listy z obu metryk i liczy ich część wspólną.

    :param results: słownik ``{metryka: (rekomendacje, antyrekomendacje)}``
    :param top: długość list
    :param catalog: katalog ``catalog.json`` (tytuły filmów)
    """
    print("\n" + "=" * 78)
    print("PORÓWNANIE METRYK")
    print("=" * 78)
    for idx, label in ((0, "Rekomendacje"), (1, "Antyrekomendacje")):
        print(f"\n  {label}:")
        print(f"   {'Euclidean':<36} | {'Pearson':<36}")
        e_list, p_list = results["Euclidean"][idx], results["Pearson"][idx]
        for i in range(top):
            e = (
                f"{movie_title(catalog, e_list[i][0])[:30]} ({clip(e_list[i][1]):.1f})"
                if i < len(e_list)
                else ""
            )
            p = (
                f"{movie_title(catalog, p_list[i][0])[:30]} ({clip(p_list[i][1]):.1f})"
                if i < len(p_list)
                else ""
            )
            print(f"   {e:<36} | {p:<36}")
        shared = {m for m, _, _ in e_list} & {m for m, _, _ in p_list}
        print(
            f"   Wspólne pozycje: {len(shared)}/{top}"
            + (
                f": {', '.join(sorted(movie_title(catalog, m) for m in shared))}"
                if shared
                else ""
            )
        )


def main():
    """Punkt wejścia CLI."""
    parser = argparse.ArgumentParser(description="Silnik rekomendacji filmów")
    parser.add_argument("--user", help="użytkownik, dla którego liczymy rekomendacje")
    parser.add_argument(
        "--score-type",
        choices=[*METRICS, "both"],
        default="both",
        help="metryka podobieństwa (domyślnie obie + porównanie)",
    )
    parser.add_argument("--top", type=int, default=5, help="liczba pozycji na liście")
    parser.add_argument(
        "--list-users", action="store_true", help="wypisz dostępnych użytkowników"
    )
    parser.add_argument("--ratings", default=RATINGS_FILE, help="plik z ocenami")
    parser.add_argument(
        "--catalog", default=CATALOG_FILE, help="katalog filmów (z OMDb)"
    )
    args = parser.parse_args()

    for path, step in (
        (args.catalog, "build_catalog.py"),
        (args.ratings, "preprocess.py"),
    ):
        if not os.path.exists(path):
            sys.exit(f"Brak pliku {path}. Najpierw uruchom {step} (patrz README).")
    dataset = load_json(args.ratings)
    catalog = load_json(args.catalog)

    if args.list_users or not args.user:
        print("Dostępni użytkownicy:")
        for name in dataset:
            print(f"  - {name} ({len(dataset[name])} ocen)")
        if not args.user:
            print('\nPodaj użytkownika: python recommender.py --user "Imię Nazwisko"')
        return
    if args.user not in dataset:
        sys.exit(f"Nie ma użytkownika '{args.user}'. Użyj --list-users.")

    titles = {movie_id: entry["title"] for movie_id, entry in catalog.items()}
    metrics = METRICS if args.score_type == "both" else (args.score_type,)
    results = {}
    print(
        f"Użytkownik: {args.user} ({len(dataset[args.user])} ocen, "
        f"średnia {mean_rating(dataset[args.user]):.2f})"
    )
    for metric in metrics:
        print("\n" + "=" * 78)
        print(f"METRYKA: {metric}")
        print("=" * 78)
        print_similar(dataset, args.user, metric)
        best, worst = recommend(dataset, args.user, metric, args.top, titles)
        results[metric] = (best, worst)
        print_list(f"TOP {args.top} REKOMENDACJI", best, catalog)
        print_list(f"TOP {args.top} ANTYREKOMENDACJI", worst, catalog)

    if len(results) == 2:
        print_comparison(results, args.top, catalog)


if __name__ == "__main__":
    main()
