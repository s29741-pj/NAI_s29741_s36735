"""
Klient OMDb API (https://www.omdbapi.com/) z cache na dysku
==========================================================

Cienka warstwa nad OMDb używana przez ``build_catalog.py``. Obsługuje trzy
rodzaje zapytań:

* ``find``   : wyszukanie po tytule (parametr ``t``), opcjonalnie z rokiem i typem,
* ``search`` : lista kandydatów (parametr ``s``),
* ``by_id``  : pobranie po identyfikatorze IMDb (parametr ``i``).

Każda odpowiedź (także "nie znaleziono") jest zapisywana w
``data/omdb_cache.json`` pod kluczem utworzonym z parametrów zapytania.
Darmowy klucz ma limit 1000 zapytań dziennie, a dzięki cache kolejne
uruchomienia nie zużywają limitu i działają bez sieci. Plik cache jest
częścią repozytorium, więc wyniki są powtarzalne.

Autor
-----
Jakub Kolad

Użycie
------
    from omdb_client import OmdbClient
    client = OmdbClient(api_key="TWÓJ_KLUCZ")      # lub zmienna OMDB_API_KEY
    client.find("Dune", year=2021, type="movie")    # -> słownik z odpowiedzią lub None
"""

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request

try:
    import certifi
except ImportError:  # certifi jest opcjonalne, ale zalecane (patrz README)
    certifi = None

API_URL = "https://www.omdbapi.com/"
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "omdb_cache.json")


class OmdbClient:
    """Klient OMDb z trwałym cache w pliku JSON.

    :param api_key: klucz OMDb; domyślnie wartość zmiennej ``OMDB_API_KEY``
    :param cache_path: ścieżka pliku cache
    :param timeout: limit czasu pojedynczego zapytania (sekundy)
    """

    def __init__(self, api_key=None, cache_path=DEFAULT_CACHE, timeout=10):
        self.api_key = api_key or os.environ.get("OMDB_API_KEY")
        self.cache_path = cache_path
        self.timeout = timeout
        self.cache = self._load_cache()
        self.requests_made = 0
        # Zaufane certyfikaty z paczki certifi zamiast magazynu systemowego, który
        # na starszych instalacjach Windows bywa nieaktualny
        # ("CERTIFICATE_VERIFY_FAILED: certificate has expired").
        self.ssl_context = ssl.create_default_context(cafile=certifi.where() if certifi else None)

    def _load_cache(self):
        """Wczytuje cache z dysku (pusty słownik, jeśli plik nie istnieje)."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Zapisuje cache na dysk."""
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=1, sort_keys=True)

    def _request(self, **params):
        """Wykonuje pojedyncze zapytanie HTTP do OMDb.

        :param params: parametry zapytania (np. ``t``, ``y``, ``type``, ``i``, ``s``)
        :return: zdekodowana odpowiedź JSON
        :raises RuntimeError: przy błędzie sieci, nieprawidłowym kluczu lub
                              przekroczonym limicie zapytań
        """
        query = urllib.parse.urlencode({"apikey": self.api_key, **params})
        try:
            with urllib.request.urlopen(f"{API_URL}?{query}", timeout=self.timeout,
                                        context=self.ssl_context) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            hint = ""
            if "CERTIFICATE_VERIFY_FAILED" in str(exc):
                hint = " (zainstaluj aktualne certyfikaty: pip install --upgrade certifi)"
            raise RuntimeError(f"Błąd połączenia z OMDb: {exc}{hint}") from exc
        error = data.get("Error", "")
        if error == "Invalid API key!":
            raise RuntimeError("Nieprawidłowy klucz OMDb API")
        if error == "Request limit reached!":
            raise RuntimeError("Przekroczono dzienny limit zapytań OMDb (1000). Uruchom ponownie jutro: "
                               "dotychczasowe odpowiedzi są w cache.")
        return data

    def query(self, **params):
        """Wykonuje zapytanie z użyciem cache.

        Puste parametry (``None``, ``""``) są pomijane. Odpowiedzi "nie
        znaleziono" również trafiają do cache, żeby nie zużywać limitu.

        :param params: parametry zapytania OMDb
        :return: słownik odpowiedzi OMDb (pole ``Response`` to ``"True"``/``"False"``)
        :raises RuntimeError: gdy brak klucza i brak odpowiedzi w cache albo błąd połączenia
        """
        params = {k: v for k, v in params.items() if v not in (None, "")}
        key = urllib.parse.urlencode(sorted(params.items()))
        if key in self.cache:
            return self.cache[key]
        if not self.api_key:
            raise RuntimeError("Brak klucza OMDb API (--api-key lub zmienna OMDB_API_KEY)")
        data = self._request(**params)
        self.requests_made += 1
        self.cache[key] = data
        self._save_cache()
        return data

    def find(self, title, year=None, type=None):
        """Szuka filmu lub serialu po tytule (``t=``).

        :param title: tytuł (OMDb zna głównie tytuły angielskie i oryginalne)
        :param year: rok premiery (opcjonalnie)
        :param type: ``"movie"`` lub ``"series"`` (opcjonalnie)
        :return: słownik z danymi filmu albo ``None``, gdy nie znaleziono
        """
        data = self.query(t=title, y=year, type=type, plot="short")
        return data if data.get("Response") == "True" else None

    def search(self, title, year=None, type=None):
        """Zwraca listę kandydatów dla fragmentu tytułu (``s=``).

        :return: lista słowników (``Title``, ``Year``, ``imdbID``, ``Type``), pusta gdy brak
        """
        data = self.query(s=title, y=year, type=type)
        return data.get("Search", []) if data.get("Response") == "True" else []

    def by_id(self, imdb_id):
        """Pobiera pełne dane po identyfikatorze IMDb (``i=``).

        :param imdb_id: np. ``"tt1160419"``
        :return: słownik z danymi filmu albo ``None``
        """
        data = self.query(i=imdb_id, plot="short")
        return data if data.get("Response") == "True" else None
