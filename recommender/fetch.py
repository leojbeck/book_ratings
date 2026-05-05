# Fetches book metadata from the Google Books API and caches results locally.
#
# Usage (from project root):
#   py recommender/fetch.py "Title" "Author"
#   py recommender/fetch.py --batch input_tables/want_to_read_list.csv
#   py recommender/fetch.py --batch input_tables/Reading_data_table.csv
#   py recommender/fetch.py --show "Title" "Author"   <- print cached entry

import csv
import json
import os
import sys
import time
from datetime import date
from urllib.parse import quote

import requests

# ---------- paths ----------

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(_HERE, "cache", "books.json")
API_BASE = "https://www.googleapis.com/books/v1/volumes"

# Set GOOGLE_BOOKS_API_KEY in your environment (or a .env file loaded externally).
# Without a key the anonymous shared quota is very small; a free key gives 1000 req/day.
_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY", "")

# ---------- genre mapping ----------
# Google Books categories are verbose strings like "Fiction / Science Fiction / General".
# We split on "/" and match keywords in priority order to produce up to 3 genre tags.

GENRE_MAP = [
    ("science fiction",  "SciFi"),
    ("space opera",      "SciFi"),
    ("dystopian",        "SciFi"),
    ("fantasy",          "Fantasy"),
    ("mythology",        "Fantasy"),
    ("arthurian",        "Fantasy"),
    ("mystery",          "Mystery"),
    ("detective",        "Mystery"),
    ("crime",            "Mystery"),
    ("thriller",         "Mystery"),
    ("mathematics",      "Science"),
    ("physics",          "Science"),
    ("biology",          "Science"),
    ("chemistry",        "Science"),
    ("technology",       "Science"),
    ("popular science",  "Science"),
    ("economics",        "Economics"),
    ("business",         "Economics"),
    ("finance",          "Economics"),
    ("history",          "History"),
    ("biography",        "History"),
    ("political",        "History"),
    ("social science",   "History"),
    ("drama",            "Play"),
    ("plays",            "Play"),
    ("literary",         "Fiction"),
    ("fiction",          "Fiction"),
    ("classic",          "Classic"),
    ("nonfiction",       "Nonfiction"),
    ("non-fiction",      "Nonfiction"),
    ("self-help",        "Nonfiction"),
    ("psychology",       "Nonfiction"),
]

def map_genres(raw_categories: list[str]) -> list[str]:
    """Map raw Google Books category tokens to our taxonomy; return up to 3 unique genres."""
    genres: list[str] = []
    for token in raw_categories:
        t = token.lower().strip()
        for keyword, genre in GENRE_MAP:
            if keyword in t and genre not in genres:
                genres.append(genre)
                break
        if len(genres) == 3:
            break
    return genres

# ---------- cache helpers ----------

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def cache_key(title: str, author: str) -> str:
    return f"{title.lower().strip()}|{author.lower().strip()}"

# ---------- core fetch ----------

def fetch_book(title: str, author: str = "", cache: dict | None = None,
               verbose: bool = True) -> tuple[dict | None, dict]:
    """
    Look up one book. Returns (result_dict, updated_cache).
    Hits the cache first; only calls the API on a miss.
    """
    if cache is None:
        cache = load_cache()

    key = cache_key(title, author)
    if key in cache:
        if verbose:
            print(f"  [cache]   {title}")
        return cache[key], cache

    # Build the query string manually so + stays as a literal AND operator.
    # Passing q through requests' params dict encodes + as %2B, which Google rejects.
    q = f"intitle:{quote(title, safe='')}"
    if author:
        q += f"+inauthor:{quote(author, safe='')}"
    url = f"{API_BASE}?q={q}&maxResults=5&printType=books"
    if _API_KEY:
        url += f"&key={_API_KEY}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [error]   {title}: {e}")
        return None, cache

    items = data.get("items", [])
    if not items:
        print(f"  [missing] {title}")
        return None, cache

    # Prefer an exact title match; fall back to the top result
    vi = next(
        (item["volumeInfo"] for item in items
         if item.get("volumeInfo", {}).get("title", "").lower() == title.lower()),
        items[0]["volumeInfo"]
    )

    # Categories come back as e.g. ["Fiction / Science Fiction / General"] — split on "/"
    raw_cats = vi.get("categories", [])
    tokens = [tok.strip() for cat in raw_cats for tok in cat.split("/")]

    isbn = next(
        (i["identifier"] for i in vi.get("industryIdentifiers", [])
         if i.get("type") == "ISBN_13"),
        None
    )

    # publishedDate is "1965", "1965-01", or "1965-01-01"
    pub = vi.get("publishedDate", "")
    year = int(pub[:4]) if pub and pub[:4].isdigit() else None

    result = {
        "title":          vi.get("title", title),
        "author":         ", ".join(vi.get("authors", [author] if author else [])),
        "year":           year,
        "pages":          vi.get("pageCount"),
        "description":    vi.get("description", ""),
        "raw_categories": tokens,
        "mapped_genres":  map_genres(tokens),
        "isbn":           isbn,
        "fetched_at":     str(date.today()),
    }

    cache[key] = result
    if verbose:
        genres_str = ", ".join(result["mapped_genres"]) or "—"
        print(f"  [fetched] {title}  |  {genres_str}")

    return result, cache

# ---------- batch mode ----------

def fetch_csv(csv_path: str) -> None:
    """Fetch metadata for every book in one of the project CSVs."""
    cache = load_cache()

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("Title", "").strip()]

    new_count = 0
    for row in rows:
        title  = row["Title"].strip()
        author = row.get("Author", "").strip()
        if cache_key(title, author) not in cache:
            _, cache = fetch_book(title, author, cache=cache)
            time.sleep(0.25)   # stay well under the anonymous quota
            new_count += 1
        else:
            print(f"  [cache]   {title}")

    save_cache(cache)
    print(f"\nDone — {new_count} new fetch(es), {len(cache)} total in cache.")

# ---------- public helper (for use as a module) ----------

def lookup(title: str, author: str = "") -> dict | None:
    """Single-book convenience wrapper: fetch, persist cache, return result."""
    cache = load_cache()
    result, cache = fetch_book(title, author, cache=cache)
    save_cache(cache)
    return result

# ---------- CLI ----------

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print('  py recommender/fetch.py "Title" "Author"')
        print('  py recommender/fetch.py --batch path/to/file.csv')
        print('  py recommender/fetch.py --show  "Title" "Author"')
        sys.exit(0)

    if args[0] == "--batch":
        if len(args) < 2:
            print("--batch requires a CSV path")
            sys.exit(1)
        fetch_csv(args[1])

    elif args[0] == "--show":
        title  = args[1] if len(args) > 1 else ""
        author = args[2] if len(args) > 2 else ""
        cache  = load_cache()
        entry  = cache.get(cache_key(title, author))
        if entry:
            print(json.dumps(entry, indent=2))
        else:
            print(f'No cache entry for "{title}" / "{author}"')

    else:
        title  = args[0]
        author = args[1] if len(args) > 1 else ""
        result = lookup(title, author)
        if result:
            print(json.dumps(result, indent=2))
