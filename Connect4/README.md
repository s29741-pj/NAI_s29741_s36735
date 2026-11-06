# Connect4 – Cztery w rzędzie

Konsolowa implementacja gry **Connect4** (Cztery w rzędzie) w Pythonie, oparta na bibliotece [easyAI](https://github.com/Zulko/easyAI). Można grać człowiek vs człowiek albo człowiek vs AI (algorytm Negamax).

## Zasady gry

Pełne zasady: [Wikipedia – Czwórka w rzędzie](https://pl.wikipedia.org/wiki/Czw%C3%B3rka_w_rz%C4%99dzie) ([wersja angielska](https://en.wikipedia.org/wiki/Connect_Four)).

W skrócie:

- Plansza ma 6 wierszy i 7 kolumn.
- Gracze na zmianę wrzucają token (`@` lub `O`) do wybranej kolumny. Token spada na najniższe wolne pole.
- Wygrywa gracz, który pierwszy ułoży cztery własne tokeny obok siebie: poziomo, pionowo lub po skosie.
- Jeśli plansza zapełni się bez zwycięzcy, jest remis.

## Autorzy

- Jakub Kolad
- Igor Czeczot

## Przygotowanie środowiska

1. Zainstaluj Pythona 3.10 lub nowszego: <https://www.python.org/downloads/>.
2. (Opcjonalnie) utwórz i aktywuj wirtualne środowisko:

   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Linux / macOS
   ```

3. Zainstaluj bibliotekę easyAI:

   ```
   pip install easyAI
   ```

### Znany problem: `No module named 'easyAI.AI'`

W niektórych wersjach paczki z PyPI brakuje podpakietu `easyAI.AI`. Wtedy:

```
git clone https://github.com/Zulko/easyAI.git easyAI_repo
```

Skopiuj folder `easyAI_repo/easyAI` (ten z plikiem `__init__.py`) obok `Connect4.py`, aby powstał układ:

```
Connect4/
├── Connect4.py
└── easyAI/
    ├── __init__.py
    ├── Player.py
    ├── TwoPlayerGame.py
    └── AI/
```

Sprawdzenie, czy używana jest lokalna wersja:

```
python -c "import easyAI; print(easyAI.__file__)"
```

## Uruchomienie

```
python Connect4.py
```

Domyślnie gra człowiek (`@`, gracz 1) kontra AI (`O`, gracz 2, `Negamax(4)`).

## Sterowanie

Gracz podaje numer kolumny (1–7), do której chce wrzucić token. Niepoprawny ruch (poza zakresem lub do pełnej kolumny) jest odrzucany, a gra pyta ponownie.

Przykład planszy:

```
[ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ]
[O][ ][ ][ ][ ][ ][ ]
[@][@][ ][ ][ ][ ][ ]
```

## Zmiana trybu gry

W bloku `if __name__ == "__main__":` w `Connect4.py`:

```python
# Dwóch ludzi:
players = [Human_Player(), Human_Player()]

# Człowiek vs AI (domyślnie):
players = [Human_Player(), AI_Player(Negamax(4))]
```

Argument `Negamax` to głębokość przeszukiwania. Większa wartość oznacza silniejsze, ale wolniejsze AI.

## Jak to działa

Klasa `Connect4` dziedziczy po `TwoPlayerGame` z easyAI. Plansza to płaska lista 42 pól (indeks pola = `wiersz * 7 + kolumna`), a ruch to numer kolumny.

| Metoda | Rola |
| --- | --- |
| `possible_moves` | zwraca kolumny, które nie są pełne |
| `make_move` / `unmake_move` | wrzuca token / cofa ruch (zdejmuje najwyższy token kolumny) |
| `is_over` | koniec gry: czwórka ostatniego gracza lub pełna plansza |
| `scoring` | ocena pozycji dla AI: `-100` dla gracza, który właśnie przegrał, w innym przypadku `0` |
| `check_connect_x/y/z`, `has_four` | wykrywanie czwórek poziomo, pionowo i po skosie |

Biblioteka easyAI przełącza gracza po `make_move`, a dopiero potem wywołuje `is_over` i `scoring`. Dlatego do sprawdzania wygranej używany jest token **poprzedniego** gracza (`previous_token`).

`unmake_move` jest opcjonalne, ale przyspiesza Negamax, bo nie trzeba kopiować całej gry przy każdym ruchu.
