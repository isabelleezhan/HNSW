"""
Loads the real MovieLens ml-latest-small dataset (GroupLens / University of
Minnesota, https://grouplens.org/datasets/movielens/) and turns it into
feature vectors for the HNSW recommendation demo.

Dataset layout expected under DATA_DIR (see scripts/download_movielens.sh):
    movies.csv   movieId,title,genres        (9,742 movies)
    ratings.csv  userId,movieId,rating,ts     (100,836 ratings)
    tags.csv     userId,movieId,tag,ts        (3,683 user-supplied tags)

Feature vector per movie = weighted concatenation of:
    - multi-hot over the 20 official MovieLens genres
    - multi-hot over the TOP_N_TAGS most common user-supplied tags
    - normalized release year (parsed out of the title's "(YYYY)")
    - normalized average rating
    - log-scaled rating count (popularity), normalized
The whole vector is L2-normalized so squared-L2 distance in HNSWIndex
behaves like cosine distance.

Usage license: GroupLens permits use of this dataset for research/education
with attribution; see data/ml-latest-small/README.txt after downloading.
"""
import csv
import math
import os
import re
from collections import Counter, defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "ml-latest-small")

GENRES = [
    "(no genres listed)", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
    "War", "Western",
]

# Tags that describe *how a user filed the movie away* rather than what it's
# about; they'd otherwise dominate the tag vocabulary without adding signal.
TAG_STOPLIST = {"in netflix queue", "watched", "netflix", "to watch", "seen more than once"}

TOP_N_TAGS = 120
MIN_TAG_COUNT = 2

GENRE_WEIGHT = 1.0
TAG_WEIGHT = 0.8
YEAR_WEIGHT = 0.3
RATING_WEIGHT = 0.3
POPULARITY_WEIGHT = 0.2

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _require_dataset():
    movies_csv = os.path.join(DATA_DIR, "movies.csv")
    if not os.path.isfile(movies_csv):
        raise SystemExit(
            f"MovieLens dataset not found at {DATA_DIR}.\n"
            "Download it first:\n"
            "  bash scripts/download_movielens.sh"
        )


def _parse_year(title):
    m = _YEAR_RE.search(title)
    return int(m.group(1)) if m else None


def _load_movies():
    movies = []  # list of dicts, in movies.csv order
    with open(os.path.join(DATA_DIR, "movies.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            movies.append({
                "movie_id": int(row["movieId"]),
                "title": row["title"],
                "year": _parse_year(row["title"]),
                "genres": [g for g in row["genres"].split("|") if g],
            })
    return movies


def _load_top_tags():
    """Returns (tag_vocab, movie_id -> set of tags in tag_vocab)."""
    counts = Counter()
    by_movie = defaultdict(set)
    raw_by_movie = defaultdict(set)

    with open(os.path.join(DATA_DIR, "tags.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tag = row["tag"].strip().lower()
            if not tag or tag in TAG_STOPLIST:
                continue
            counts[tag] += 1
            raw_by_movie[int(row["movieId"])].add(tag)

    vocab = [tag for tag, n in counts.most_common(TOP_N_TAGS) if n >= MIN_TAG_COUNT]
    vocab_set = set(vocab)

    for movie_id, tags in raw_by_movie.items():
        by_movie[movie_id] = tags & vocab_set

    return vocab, by_movie


def _load_rating_stats():
    """Returns movie_id -> (avg_rating, num_ratings)."""
    sums = defaultdict(float)
    counts = defaultdict(int)
    with open(os.path.join(DATA_DIR, "ratings.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            movie_id = int(row["movieId"])
            sums[movie_id] += float(row["rating"])
            counts[movie_id] += 1

    stats = {}
    for movie_id, count in counts.items():
        stats[movie_id] = (sums[movie_id] / count, count)
    return stats


def vector_dim(tag_vocab):
    return len(GENRES) + len(tag_vocab) + 3  # + year + avg_rating + popularity


def _build_vector(movie, tag_vocab, tags, avg_rating, num_ratings,
                   year_range, max_num_ratings):
    vec = [0.0] * vector_dim(tag_vocab)

    for g in movie["genres"]:
        if g in GENRES:
            vec[GENRES.index(g)] = GENRE_WEIGHT

    offset = len(GENRES)
    for t in tags:
        vec[offset + tag_vocab.index(t)] = TAG_WEIGHT

    min_year, max_year = year_range
    if movie["year"] is not None and max_year > min_year:
        norm_year = (movie["year"] - min_year) / (max_year - min_year)
        vec[-3] = norm_year * YEAR_WEIGHT

    if avg_rating is not None:
        vec[-2] = (avg_rating / 5.0) * RATING_WEIGHT

    if num_ratings and max_num_ratings > 0:
        # log-scale popularity so a handful of blockbusters don't blow out the range
        norm_pop = math.log1p(num_ratings) / math.log1p(max_num_ratings)
        vec[-1] = norm_pop * POPULARITY_WEIGHT

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def build_catalog(min_ratings=1):
    """Loads the dataset and returns (catalog, tag_vocab).

    catalog is a list of dicts {movie_id, title, year, genres, tags, vector},
    indexed 0..n-1 in the order fed to HNSWIndex (this index position *is*
    the HNSW vector id). Movies with fewer than `min_ratings` ratings are
    skipped to cut out obscure/noise entries.
    """
    _require_dataset()

    movies = _load_movies()
    tag_vocab, tags_by_movie = _load_top_tags()
    rating_stats = _load_rating_stats()

    years = [m["year"] for m in movies if m["year"] is not None]
    year_range = (min(years), max(years))
    max_num_ratings = max((c for _, c in rating_stats.values()), default=0)

    catalog = []
    for movie in movies:
        avg_rating, num_ratings = rating_stats.get(movie["movie_id"], (None, 0))
        if num_ratings < min_ratings:
            continue

        tags = tags_by_movie.get(movie["movie_id"], set())
        vector = _build_vector(
            movie, tag_vocab, tags, avg_rating, num_ratings, year_range, max_num_ratings
        )
        catalog.append({
            "movie_id": movie["movie_id"],
            "title": movie["title"],
            "year": movie["year"],
            "genres": movie["genres"],
            "tags": sorted(tags),
            "avg_rating": avg_rating,
            "num_ratings": num_ratings,
            "vector": vector,
        })

    return catalog, tag_vocab
