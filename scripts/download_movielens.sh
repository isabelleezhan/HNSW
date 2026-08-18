#!/usr/bin/env bash
# Downloads the MovieLens ml-latest-small dataset (GroupLens / University of
# Minnesota) used by python/recommend_demo.py. ~1 MB zip, no API key needed.
# See data/ml-latest-small/README.txt (after extraction) for the usage
# license: free for research/education with attribution, no redistribution
# for commercial use.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
URL="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_PATH="$DATA_DIR/ml-latest-small.zip"

mkdir -p "$DATA_DIR"

if [ -f "$DATA_DIR/ml-latest-small/movies.csv" ]; then
    echo "Already downloaded: $DATA_DIR/ml-latest-small"
    exit 0
fi

echo "Downloading $URL ..."
curl -sSL -o "$ZIP_PATH" "$URL"

echo "Extracting ..."
unzip -o -q "$ZIP_PATH" -d "$DATA_DIR"
rm -f "$ZIP_PATH"

echo "Done: $DATA_DIR/ml-latest-small"
