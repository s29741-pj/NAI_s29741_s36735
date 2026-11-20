"""Leave-one-out evaluation of the existing recommender (recommender.py).

Every rating in data/ratings.json is hidden in turn. The existing
predict_ratings() is run for Pearson and Euclidean on the same in-memory
copy of the dataset (only that one rating removed), and each available
prediction is compared with the hidden rating.

data/ratings.json is never modified. Results go to test_results/results.csv.
"""

import copy
import csv
import math
import os

import recommender
from recommender import load_json, movie_title, predict_ratings

METRICS = ("Pearson", "Euclidean")
RESULTS_FILE = os.path.join(recommender.BASE_DIR, "test_results", "results.csv")
CSV_COLUMNS = [
    "user",
    "movie_id",
    "movie_title",
    "actual_rating",
    "pearson_prediction",
    "pearson_absolute_error",
    "euclidean_prediction",
    "euclidean_absolute_error",
]


def fmt_mae_rmse(value):
    return "N/A" if value is None else f"{value:.3f}"


def main():
    original = load_json(recommender.RATINGS_FILE)
    catalog = load_json(recommender.CATALOG_FILE)

    total_ratings = sum(len(ratings) for ratings in original.values())
    excluded = 0
    rows = []
    # signed differences (actual - clipped prediction) for each metric
    diffs = {m: [] for m in METRICS}
    common = {m: [] for m in METRICS}

    for user, ratings in original.items():
        for movie, actual in ratings.items():
            # Fresh copy of the ORIGINAL dataset for every experiment.
            reduced = copy.deepcopy(original)
            del reduced[user][movie]
            if not reduced[user]:
                excluded += 1  # the recommender needs a nonempty user profile
                continue

            row = {
                "user": user,
                "movie_id": movie,
                "movie_title": movie_title(catalog, movie),
                "actual_rating": actual,
            }
            found = {}
            for metric in METRICS:
                preds = {m: p for m, p, _ in predict_ratings(reduced, user, metric)}
                key = metric.lower()
                if movie in preds:
                    found[metric] = actual - recommender.clip(preds[movie])
                    diffs[metric].append(found[metric])
                    row[f"{key}_prediction"] = f"{preds[movie]:.4f}"
                    row[f"{key}_absolute_error"] = f"{abs(found[metric]):.4f}"
                else:
                    row[f"{key}_prediction"] = ""
                    row[f"{key}_absolute_error"] = ""
            if len(found) == len(METRICS):
                for metric in METRICS:
                    common[metric].append(found[metric])
            rows.append(row)

    attempted = len(rows)

    def mae(values):
        return sum(abs(v) for v in values) / len(values) if values else None

    def rmse(values):
        return math.sqrt(sum(v * v for v in values) / len(values)) if values else None

    def coverage(n):
        return f"{100 * n / attempted:.1f}% ({n}/{attempted})"

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    line = "=" * 52
    print(line)
    print("RECOMMENDER EVALUATION - LEAVE ONE OUT")
    print(line)
    print()
    print(f"Users:              {len(original)}")
    print(f"Total ratings:      {total_ratings}")
    print(f"Test cases:         {attempted}")
    print(f"Excluded cases:     {excluded}")
    print()
    print(f"{'':<24}{'PEARSON':<16}{'EUCLIDEAN':<16}")
    print("-" * 52)
    p, e = diffs["Pearson"], diffs["Euclidean"]
    print(f"{'Predictions available':<24}{len(p):<16}{len(e):<16}")
    print(f"{'Coverage':<24}{coverage(len(p)):<16}{coverage(len(e)):<16}")
    print(f"{'MAE':<24}{fmt_mae_rmse(mae(p)):<16}{fmt_mae_rmse(mae(e)):<16}")
    print(f"{'RMSE':<24}{fmt_mae_rmse(rmse(p)):<16}{fmt_mae_rmse(rmse(e)):<16}")
    print()
    print("COMMON SUBSET")
    print()
    print(f"Predictions available: {len(common['Pearson'])}")
    print(f"Pearson MAE:   {fmt_mae_rmse(mae(common['Pearson']))}")
    print(f"Euclidean MAE: {fmt_mae_rmse(mae(common['Euclidean']))}")
    print()
    print(line)
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
