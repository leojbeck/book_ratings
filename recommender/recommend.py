# Scores the want-to-read list using the trained differential model.
# Predicted rating = gr_avg + predicted_differential, clamped to [1, 5].
#
# Usage (from project root):
#   py recommender/recommend.py
#   py recommender/recommend.py --sort gr       <- sort by GR avg instead of predicted

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (FEATURE_NAMES, build_features, build_stats, build_tfidf,
                      desc_similarities, load_cache, load_read_books, load_want_books)
from model import load_model, predict

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE = os.path.join(_ROOT, "analysis_outputs", "recommendations.txt")

# ---------- scoring ----------

def score_want_list(sort_by="predicted"):
    read_books = load_read_books()
    want_books = load_want_books()
    cache      = load_cache()
    params     = load_model()
    stats      = build_stats(read_books)
    tfidf      = build_tfidf(read_books, cache)

    results = []
    for book in want_books:
        diff, rating = predict(book, stats, cache, tfidf, params)
        sims = desc_similarities(book, cache, tfidf, top_k=3)
        results.append({
            "book":    book,
            "diff":    diff,
            "rating":  rating,
            "gr_avg":  book.get("Avg Rating") or 0.0,
            "sims":    sims,
        })

    key = "rating" if sort_by == "predicted" else "gr_avg"
    results.sort(key=lambda r: r[key], reverse=True)
    return results, stats

# ---------- formatting ----------

def _genre_str(book):
    parts = [book.get(f, "").strip() for f in ("Genre 1", "Genre 2") if book.get(f, "").strip()]
    return " / ".join(parts)

def _series_str(book):
    name = book.get("Series Name", "").strip()
    num  = book.get("Series Number")
    if name and num:
        return f"{name} #{num}"
    if name:
        return name
    return ""

def _author_context(book, stats):
    """Brief string showing prior read history with this author."""
    author = book.get("Author", "").strip()
    info   = stats["author_stats"].get(author)
    if info:
        return f"{info['count']} book(s) read, avg {info['avg']:.2f}"
    return "not yet read"

def format_results(results, stats):
    lines = []
    header = (
        f"RECOMMENDATIONS — want-to-read list scored {date.today()}\n"
        f"Model: Ridge regression on differential (my rating - GR avg)\n"
        f"{'=' * 72}"
    )
    lines.append(header)

    for rank, r in enumerate(results, 1):
        book   = r["book"]
        title  = book["Title"]
        author = book.get("Author", "").strip()
        genre  = _genre_str(book)
        series = _series_str(book)

        # Main line
        diff_sign = "+" if r["diff"] >= 0 else ""
        lines.append(
            f"\n{rank:>2}. {r['rating']:.1f}  (GR {r['gr_avg']:.2f}, diff {diff_sign}{r['diff']:.2f})"
            f"  {title}"
        )
        # Metadata line
        meta_parts = [f"{author}", f"{genre}"]
        if series:
            meta_parts.append(series)
        meta_parts.append(f"pub {book.get('Release', '?')}")
        if book.get("Pages"):
            meta_parts.append(f"{book['Pages']}p")
        lines.append(f"    {' | '.join(meta_parts)}")

        # Author context
        lines.append(f"    Author history: {_author_context(book, stats)}")

        # Similar read books
        if r["sims"]:
            sim_strs = [f"{b['Title']} [{b['My Rating']}/5, {s:.2f}]" for b, s in r["sims"]]
            lines.append(f"    Similar reads:  {',  '.join(sim_strs)}")

    lines.append(f"\n{'=' * 72}")
    return "\n".join(lines)

# ---------- CLI ----------

if __name__ == "__main__":
    sort_by = "gr" if "--sort" in sys.argv and sys.argv[sys.argv.index("--sort") + 1] == "gr" \
              else "predicted"

    results, stats = score_want_list(sort_by=sort_by)
    output = format_results(results, stats)

    # Print to console (replace non-ascii for Windows cp1252 terminals)
    print(output.encode("ascii", errors="replace").decode("ascii"))

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to analysis_outputs/recommendations.txt")
