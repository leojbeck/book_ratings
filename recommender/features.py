# Builds feature vectors for book rating prediction.
#
# Importable: from recommender.features import load_all, build_stats, build_tfidf, build_features
# CLI:        py recommender/features.py "Title"   <- inspect features + similar books for any title

import csv
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

READ_CSV   = os.path.join(_ROOT, "input_tables", "Reading_data_table.csv")
WANT_CSV   = os.path.join(_ROOT, "input_tables", "want_to_read_list.csv")
CACHE_FILE = os.path.join(_HERE, "cache", "books.json")

# ---------- loaders ----------

def _coerce(row, has_rating):
    row["My Rating"]  = int(row["My Rating"])   if has_rating and row.get("My Rating", "").strip() else None
    row["Avg Rating"] = float(row["Avg Rating"]) if row.get("Avg Rating", "").strip() else None
    row["Pages"]      = int(row["Pages"])        if row.get("Pages", "").strip().isdigit() else None
    row["School?"]    = int(row.get("School?", 0))
    yr = row.get("Release", "").strip()
    row["Release"]    = int(yr) if yr.lstrip("-").isdigit() else None
    sn = row.get("Series Number", "").strip()
    row["Series Number"] = int(sn) if sn.isdigit() else None
    return row

def load_read_books():
    with open(READ_CSV, newline="", encoding="utf-8-sig") as f:
        return [_coerce(r, True) for r in csv.DictReader(f)
                if r["Title"].strip() and r.get("My Rating", "").strip()]

def load_want_books():
    with open(WANT_CSV, newline="", encoding="utf-8-sig") as f:
        return [_coerce(r, False) for r in csv.DictReader(f) if r["Title"].strip()]

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def cache_key(title, author=""):
    return f"{title.lower().strip()}|{author.lower().strip()}"

def _avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

# ---------- stats (precomputed from read history) ----------

def build_stats(read_books):
    """
    Compute per-genre, per-author, and per-series average ratings from read history.
    All three genre columns are folded in so Genre 2/3 tags get their own averages too.
    """
    genre_ratings  = defaultdict(list)
    author_ratings = defaultdict(list)
    series_ratings = defaultdict(list)

    for b in read_books:
        r = b["My Rating"]
        if r is None:
            continue
        for gf in ("Genre 1", "Genre 2", "Genre 3"):
            g = b.get(gf, "").strip()
            if g:
                genre_ratings[g].append(r)
        a = b["Author"].strip()
        if a:
            author_ratings[a].append(r)
        s = b.get("Series Name", "").strip()
        if s:
            series_ratings[s].append(r)

    return {
        "genre_avgs":   {g: _avg(rs) for g, rs in genre_ratings.items()},
        "author_stats": {a: {"avg": _avg(rs), "count": len(rs)} for a, rs in author_ratings.items()},
        "series_stats": {s: {"avg": _avg(rs), "count": len(rs)} for s, rs in series_ratings.items()},
        "global_avg":   _avg([b["My Rating"] for b in read_books if b["My Rating"] is not None]),
    }

# ---------- TF-IDF description similarity ----------

def build_tfidf(read_books, cache):
    """
    Fit a TF-IDF model on descriptions of read books pulled from the cache.
    Returns a model dict, or None if sklearn is unavailable or no descriptions exist.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return None

    descs, books_with_desc = [], []
    for b in read_books:
        desc = cache.get(cache_key(b["Title"], b["Author"]), {}).get("description", "").strip()
        if desc:
            descs.append(desc)
            books_with_desc.append(b)

    if not descs:
        return None

    vec    = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
    matrix = vec.fit_transform(descs)
    return {"vectorizer": vec, "matrix": matrix, "books": books_with_desc}

def desc_similarities(book, cache, tfidf_model, top_k=3):
    """
    Return top_k most similar read books by TF-IDF cosine similarity.
    Returns list of (read_book, score) sorted descending, or [] if unavailable.
    """
    if tfidf_model is None:
        return []
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return []

    desc = cache.get(cache_key(book["Title"], book.get("Author", "")), {}).get("description", "").strip()
    if not desc:
        return []

    vec  = tfidf_model["vectorizer"].transform([desc])
    sims = cosine_similarity(vec, tfidf_model["matrix"]).flatten()
    top  = sims.argsort()[::-1][:top_k]
    return [(tfidf_model["books"][i], float(sims[i])) for i in top if sims[i] > 0]

# ---------- feature builder ----------

# Ordered list used by model.py to build the numpy matrix consistently
FEATURE_NAMES = [
    "gr_avg",           # Goodreads avg rating — strong baseline
    "genre1_avg",       # user's historical avg for Genre 1
    "genre2_avg",       # user's historical avg for Genre 2 (falls back to genre1_avg)
    "author_avg",       # user's avg for this author (falls back to global avg)
    "author_count",     # books by this author already read
    "series_avg",       # user's avg for this series (falls back to global avg)
    "series_count",     # books in this series already read
    "series_number",    # position in series (0 = standalone)
    "pages",            # page count
    "release_year",     # publication year
    "desc_sim_rating",  # rating-weighted avg of top-3 most similar read books by description
    "desc_sim_max",     # highest cosine similarity score to any single read book
]

def build_features(book, stats, cache, tfidf_model=None):
    """
    Return a feature dict for one book. Never returns None values — missing entries
    fall back to the global average so the vector is always complete.
    """
    g  = stats["global_avg"]
    ga = stats["genre_avgs"]
    aa = stats["author_stats"]
    sa = stats["series_stats"]

    # Genres — chain fallback: genre2 → genre1 → global
    g1 = book.get("Genre 1", "").strip()
    g2 = book.get("Genre 2", "").strip()
    genre1_avg = ga.get(g1, g)
    genre2_avg = ga.get(g2, genre1_avg) if g2 else genre1_avg

    # Author
    author      = book.get("Author", "").strip()
    author_info = aa.get(author, {})
    author_avg  = author_info.get("avg", g)
    author_count = author_info.get("count", 0)

    # Series
    series_name   = book.get("Series Name", "").strip()
    series_number = book.get("Series Number") or 0
    series_info   = sa.get(series_name, {}) if series_name else {}
    series_avg    = series_info.get("avg", g)
    series_count  = series_info.get("count", 0)

    # Pages — CSV value preferred; fall back to cache (cache editions are often wrong)
    pages = book.get("Pages")
    if not pages:
        pages = cache.get(cache_key(book["Title"], author), {}).get("pages") or 0

    release_year = book.get("Release") or 0
    gr_avg = book.get("Avg Rating") or g

    # Description similarity
    sims = desc_similarities(book, cache, tfidf_model, top_k=3)
    if sims:
        total_sim     = sum(s for _, s in sims)
        desc_sim_rating = sum(b["My Rating"] * s for b, s in sims) / total_sim
        desc_sim_max    = sims[0][1]
    else:
        desc_sim_rating = g
        desc_sim_max    = 0.0

    return {
        "gr_avg":           gr_avg,
        "genre1_avg":       genre1_avg,
        "genre2_avg":       genre2_avg,
        "author_avg":       author_avg,
        "author_count":     author_count,
        "series_avg":       series_avg,
        "series_count":     series_count,
        "series_number":    series_number,
        "pages":            pages,
        "release_year":     release_year,
        "desc_sim_rating":  desc_sim_rating,
        "desc_sim_max":     desc_sim_max,
    }

def build_feature_matrix(books, stats, cache, tfidf_model=None):
    """Build feature dicts for a list of books. Returns (feat_dicts, ratings)."""
    feat_dicts = [build_features(b, stats, cache, tfidf_model) for b in books]
    ratings    = [b.get("My Rating") for b in books]
    return feat_dicts, ratings

# ---------- CLI ----------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py recommender/features.py \"Title\"")
        sys.exit(0)

    title = sys.argv[1].lower()

    read_books = load_read_books()
    want_books = load_want_books()
    cache      = load_cache()
    stats      = build_stats(read_books)
    tfidf      = build_tfidf(read_books, cache)

    target = next((b for b in read_books + want_books
                   if b["Title"].lower() == title), None)
    if target is None:
        print(f'Not found: "{sys.argv[1]}"')
        sys.exit(1)

    feats = build_features(target, stats, cache, tfidf)
    sims  = desc_similarities(target, cache, tfidf, top_k=3)

    rating_str = str(target["My Rating"]) if target["My Rating"] is not None else "(unread)"
    print(f"\n{target['Title']} — {target['Author']}  [{rating_str}]")
    print(f"  Genre 1: {target.get('Genre 1','')}  Genre 2: {target.get('Genre 2','')}")
    print(f"\nFeatures:")
    for name in FEATURE_NAMES:
        v = feats[name]
        print(f"  {name:<20} {v:.4f}" if isinstance(v, float) else f"  {name:<20} {v}")

    if sims:
        print(f"\nTop similar read books (description TF-IDF):")
        for b, score in sims:
            print(f"  {score:.3f}  [{b['My Rating']}/5]  {b['Title']}")
    else:
        print("\n(No description similarity — book not in cache or no description)")
