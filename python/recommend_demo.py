#!/usr/bin/env python3
"""
Movie recommendation demo built on the hnsw-cpp HNSWIndex, using the real
MovieLens ml-latest-small dataset (GroupLens / University of Minnesota).

Every movie is embedded as a genre + user-tag + year + rating feature vector
(see movielens.py), all ~9,700 of them are inserted into an HNSWIndex, and
"recommendations" are just approximate nearest neighbors of a query movie's
vector.

First time setup:
    bash scripts/download_movielens.sh

Usage:
    python3 recommend_demo.py                       # a few example queries
    python3 recommend_demo.py --title "Inception"    # recommend for one movie
    python3 recommend_demo.py --title "Inception" -k 10
    python3 recommend_demo.py --search "matrix"       # browse titles matching a keyword
"""
import argparse
import difflib
import os
import re
import sys
import time

# Make the compiled hnsw_cpp extension importable regardless of which build
# directory it landed in (CMake builds it alongside the other targets).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _candidate in ("build-python", "build", "build-release"):
    _path = os.path.join(_REPO_ROOT, _candidate)
    if os.path.isdir(_path):
        sys.path.insert(0, _path)

try:
    import hnsw_cpp
except ImportError as exc:
    raise SystemExit(
        "Could not import the hnsw_cpp extension module. Build it first, e.g.:\n"
        "  mkdir -p build-python && cd build-python && cmake .. && make hnsw_cpp\n"
        f"(import error: {exc})"
    )

from movielens import build_catalog, vector_dim

M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 64

_ARTICLE_SUFFIX_RE = re.compile(r",\s*(the|a|an)$")
_ARTICLE_PREFIX_RE = re.compile(r"^(the|a|an)\s+")
_YEAR_SUFFIX_RE = re.compile(r"\(\d{4}\)\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_title(title):
    """'Matrix, The (1999)' -> 'matrix', so a query of 'the matrix' matches
    it too (queries carry a leading article; MovieLens titles carry a
    trailing one) without either swallowing an in-name article like
    'Godfather Trilogy'.
    """
    text = title.lower()
    text = _YEAR_SUFFIX_RE.sub("", text)
    text = _ARTICLE_SUFFIX_RE.sub("", text.strip())
    text = _ARTICLE_PREFIX_RE.sub("", text.strip())
    text = _NON_ALNUM_RE.sub(" ", text)
    return " ".join(text.split())


def build_index(catalog, dim):
    index = hnsw_cpp.HNSWIndex(dim, M, EF_CONSTRUCTION)
    for movie in catalog:
        index.add(movie["vector"])
    return index


def find_movie(catalog, query):
    """Fuzzy title lookup: exact (article/year-insensitive) match first,
    then substring match, auto-resolving ambiguity to the most-rated movie.
    """
    norm_query = _normalize_title(query)

    exact = [i for i, m in enumerate(catalog) if _normalize_title(m["title"]) == norm_query]
    candidates = exact if exact else [
        i for i, m in enumerate(catalog) if norm_query and norm_query in _normalize_title(m["title"])
    ]

    if not candidates:
        titles = [m["title"] for m in catalog]
        suggestions = difflib.get_close_matches(query, titles, n=5, cutoff=0.5)
        if suggestions:
            raise KeyError(f'No movie matching "{query}". Did you mean: {", ".join(suggestions)}?')
        raise KeyError(f'No movie matching "{query}". Try --search "<keyword>" to browse the catalog.')

    best = max(candidates, key=lambda i: catalog[i]["num_ratings"])
    if len(candidates) > 1:
        others = [catalog[i]["title"] for i in candidates if i != best][:4]
        note = f'matched "{catalog[best]["title"]}"'
        if others:
            note += f"; also matched: {', '.join(others)}"
        print(f"  ({note})")

    return best, catalog[best]


def recommend(catalog, index, title, k=5, ef_search=EF_SEARCH):
    query_idx, query_movie = find_movie(catalog, title)

    # Ask for k+1 since the query movie itself will be its own nearest neighbor.
    results = index.search(query_movie["vector"], k + 1, ef_search)

    recs = []
    for neighbor_id, dist in results:
        if neighbor_id == query_idx:
            continue
        recs.append((catalog[neighbor_id], dist))
        if len(recs) == k:
            break
    return query_movie, recs


def format_movie(movie):
    genres = "/".join(movie["genres"]) or "?"
    rating = f'{movie["avg_rating"]:.1f}★' if movie["avg_rating"] else "n/a"
    return f'{movie["title"]} [{genres}] ({rating}, {movie["num_ratings"]} ratings)'


def print_recommendations(catalog, index, title, k=5, ef_search=EF_SEARCH):
    query_movie, recs = recommend(catalog, index, title, k, ef_search)
    print(f"\nBecause you watched: {format_movie(query_movie)}")
    print("Recommended:")
    for movie, dist in recs:
        print(f"  - {format_movie(movie):<85} (distance={dist:.4f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--title", help="movie to get recommendations for")
    parser.add_argument("-k", type=int, default=5, help="number of recommendations (default: 5)")
    parser.add_argument("--ef-search", type=int, default=EF_SEARCH, help="HNSW ef_search (default: %(default)s)")
    parser.add_argument("--min-ratings", type=int, default=1, help="drop movies with fewer ratings than this (default: %(default)s)")
    parser.add_argument("--search", help="list catalog titles containing this keyword and exit")
    args = parser.parse_args()

    catalog, tag_vocab = build_catalog(min_ratings=args.min_ratings)
    dim = vector_dim(tag_vocab)

    if args.search:
        needle = args.search.lower()
        matches = [m for m in catalog if needle in m["title"].lower()]
        print(f'{len(matches)} title(s) matching "{args.search}":')
        for movie in matches[:50]:
            print(f"  {format_movie(movie)}")
        return

    t0 = time.perf_counter()
    index = build_index(catalog, dim)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"Indexed {len(catalog)} movies (dim={dim}, from MovieLens ml-latest-small) in {build_ms:.0f} ms")

    if args.title:
        try:
            print_recommendations(catalog, index, args.title, args.k, args.ef_search)
        except KeyError as exc:
            raise SystemExit(str(exc))
        return

    # No title given: show a handful of examples spanning different genres/eras.
    for example in ["Toy Story", "The Matrix", "The Godfather", "La La Land", "Get Out"]:
        try:
            print_recommendations(catalog, index, example, args.k, args.ef_search)
        except KeyError as exc:
            print(f"\n(skipping example {example!r}: {exc})")


if __name__ == "__main__":
    main()
