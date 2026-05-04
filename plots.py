# Personal book ratings — output plots
# Saves all figures as PNGs to analysis_outputs/
# Run with: py plots.py

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------- load data ----------

with open(r"input_tables/Reading_data_table.csv", newline="", encoding="utf-8-sig") as f:
    books = [row for row in csv.DictReader(f) if row["Title"].strip()]

for b in books:
    b["My Rating"] = int(b["My Rating"])
    b["Avg Rating"] = float(b["Avg Rating"]) if b["Avg Rating"] else None
    b["Pages"] = int(b["Pages"]) if b["Pages"] else None
    b["School?"] = int(b["School?"])
    b["diff"] = b["My Rating"] - b["Avg Rating"] if b["Avg Rating"] else None

OUT = "analysis_outputs"

def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0

# consistent genre color map across all plots
GENRES = ["Mystery", "Fantasy", "SciFi", "Nonfiction", "Science", "Economics", "History",
          "Fiction", "Classic", "Play"]
COLORS = plt.cm.tab10.colors
GENRE_COLOR = {g: COLORS[i] for i, g in enumerate(GENRES)}

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")

# ---------- 1. Rating distribution ----------

counts = [sum(1 for b in books if b["My Rating"] == r) for r in range(1, 6)]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(range(1, 6), counts, color="#4C72B0", edgecolor="white", linewidth=0.8)
ax.bar_label(bars, padding=3, fontsize=10)
ax.set_xlabel("My Rating")
ax.set_ylabel("Number of Books")
ax.set_title("Rating Distribution")
ax.set_xticks(range(1, 6))
ax.set_xlim(0.4, 5.6)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.spines[["top", "right"]].set_visible(False)
save(fig, "1_rating_distribution.png")

# ---------- 2. My avg vs GR avg by genre ----------

genres_data = defaultdict(list)
for b in books:
    genres_data[b["Genre 1"]].append(b)

genre_order = sorted(GENRES, key=lambda g: avg([b["My Rating"] for b in genres_data[g]]))
my_avgs = [avg([b["My Rating"] for b in genres_data[g]]) for g in genre_order]
gr_avgs = [avg([b["Avg Rating"] for b in genres_data[g]]) for g in genre_order]
counts_g = [len(genres_data[g]) for g in genre_order]

y = np.arange(len(genre_order))
h = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.barh(y + h/2, my_avgs, h, label="My Rating", color="#4C72B0")
b2 = ax.barh(y - h/2, gr_avgs, h, label="GR Avg", color="#DD8452")
ax.set_yticks(y)
ax.set_yticklabels([f"{g}  (n={c})" for g, c in zip(genre_order, counts_g)])
ax.set_xlabel("Average Rating")
ax.set_title("My Rating vs Goodreads Average by Genre")
ax.set_xlim(1, 5.4)
ax.axvline(x=avg([b["My Rating"] for b in books]), color="#4C72B0", linestyle="--", linewidth=0.8, alpha=0.6)
ax.axvline(x=avg([b["Avg Rating"] for b in books]), color="#DD8452", linestyle="--", linewidth=0.8, alpha=0.6)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
save(fig, "2_genre_ratings.png")

# ---------- 3. Rating differential by genre, school vs non-school ----------

school = [b for b in books if b["School?"]]
non_school = [b for b in books if not b["School?"]]

fig, ax = plt.subplots(figsize=(8, 5))
genre_order2 = sorted(GENRES,
    key=lambda g: avg([b["diff"] for b in genres_data[g] if b["diff"] is not None]))

y = np.arange(len(genre_order2))
school_diffs = [avg([b["diff"] for b in school if b["Genre 1"] == g and b["diff"] is not None])
                for g in genre_order2]
ns_diffs = [avg([b["diff"] for b in non_school if b["Genre 1"] == g and b["diff"] is not None])
            for g in genre_order2]

# use NaN so genres with no school books don't plot a 0 bar
school_diffs = [d if any(b["Genre 1"] == g and b["School?"] for b in books) else float("nan")
                for d, g in zip(school_diffs, genre_order2)]

b1 = ax.barh(y + h/2, ns_diffs, h, label="Non-school", color="#4C72B0")
b2 = ax.barh(y - h/2, school_diffs, h, label="School", color="#C44E52")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels(genre_order2)
ax.set_xlabel("Avg Differential (My Rating − GR Avg)")
ax.set_title("Rating Differential by Genre and School Status")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
save(fig, "3_differential_genre_school.png")

# ---------- 4. My rating vs GR avg scatter, colored by genre ----------

fig, ax = plt.subplots(figsize=(7, 6))
for genre in GENRES:
    grp = [b for b in books if b["Genre 1"] == genre and b["Avg Rating"] is not None]
    if not grp:
        continue
    xs = [b["Avg Rating"] for b in grp]
    ys = [b["My Rating"] for b in grp]
    markers = ["^" if b["School?"] else "o" for b in grp]
    for x, y_val, m in zip(xs, ys, markers):
        ax.scatter(x, y_val, color=GENRE_COLOR[genre], marker=m, s=60, alpha=0.8, linewidths=0)

# legend for genres
for genre in GENRES:
    if any(b["Genre 1"] == genre for b in books):
        ax.scatter([], [], color=GENRE_COLOR[genre], label=genre, s=60)
ax.scatter([], [], color="gray", marker="o", label="Non-school", s=60)
ax.scatter([], [], color="gray", marker="^", label="School", s=60)

lo = min(b["Avg Rating"] for b in books if b["Avg Rating"])
hi = max(b["Avg Rating"] for b in books if b["Avg Rating"])
ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.7, linestyle="--", alpha=0.4, label="y = x")

ax.set_xlabel("Goodreads Avg Rating")
ax.set_ylabel("My Rating")
ax.set_title("My Rating vs Goodreads Rating")
ax.set_yticks(range(1, 6))
ax.legend(fontsize=8, framealpha=0.7, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "4_scatter_my_vs_gr.png")

# ---------- 5. Books and pages read per year ----------

by_year = defaultdict(list)
for b in books:
    by_year[b["Year Read"]].append(b)

years = sorted(by_year)
book_counts = [len(by_year[y]) for y in years]
page_totals = [sum(b["Pages"] for b in by_year[y] if b["Pages"]) for y in years]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax1.bar(years, book_counts, color="#4C72B0", edgecolor="white")
ax1.set_ylabel("Books Read")
ax1.set_title("Reading Volume by Year")
ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax1.spines[["top", "right"]].set_visible(False)

ax2.bar(years, page_totals, color="#55A868", edgecolor="white")
ax2.set_ylabel("Pages Read")
ax2.set_xlabel("Year")
ax2.spines[["top", "right"]].set_visible(False)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.xticks(rotation=45, ha="right")
fig.tight_layout()
save(fig, "5_volume_by_year.png")

# ---------- 6. Multi-book authors ----------

author_books = defaultdict(list)
for b in books:
    if b["Author"].strip():
        author_books[b["Author"]].append(b)
multi = {a: grp for a, grp in author_books.items() if len(grp) >= 2}
author_order = sorted(multi, key=lambda a: avg([b["My Rating"] for b in multi[a]]))

fig, ax = plt.subplots(figsize=(8, max(4, len(author_order) * 0.55)))
for i, author in enumerate(author_order):
    grp = multi[author]
    for b in grp:
        color = GENRE_COLOR.get(b["Genre 1"], "gray")
        ax.scatter(b["My Rating"], i, color=color, s=80, zorder=3)
    xs = [b["My Rating"] for b in grp]
    ax.plot([min(xs), max(xs)], [i, i], color="gray", linewidth=1, zorder=2)
    ax.scatter(avg(xs), i, color="black", marker="|", s=200, zorder=4, linewidths=2)

ax.set_yticks(range(len(author_order)))
ax.set_yticklabels([f"{a} (n={len(multi[a])})" for a in author_order])
ax.set_xticks(range(1, 6))
ax.set_xlabel("My Rating")
ax.set_title("Ratings Spread — Authors with 2+ Books\n(bar = avg, dots = individual books)")
ax.set_xlim(0.5, 5.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3)
save(fig, "6_multi_author_spread.png")

# ---------- 7. Ratings vs publication year ----------

# The Odyssey (~800 BC) is excluded by the 1800 cutoff intentionally
pub_books = [b for b in books if b["Avg Rating"] is not None
             and b["Release"] and 1800 <= int(b["Release"]) <= 2030]
for b in pub_books:
    b["Release"] = int(b["Release"])

fig, ax = plt.subplots(figsize=(9, 5))
for genre in GENRES:
    grp = [b for b in pub_books if b["Genre 1"] == genre]
    if not grp:
        continue
    xs = [b["Release"] for b in grp]
    ax.scatter(xs, [b["My Rating"] for b in grp],
               color=GENRE_COLOR[genre], marker="o", s=55, alpha=0.85, linewidths=0)
    ax.scatter(xs, [b["Avg Rating"] for b in grp],
               color=GENRE_COLOR[genre], marker="^", s=55, alpha=0.5, linewidths=0)

# legend: genres + series markers
for genre in GENRES:
    if any(b["Genre 1"] == genre for b in pub_books):
        ax.scatter([], [], color=GENRE_COLOR[genre], label=genre, s=55)
ax.scatter([], [], color="gray", marker="o", label="My Rating", s=55)
ax.scatter([], [], color="gray", marker="^", label="GR Avg", s=55, alpha=0.5)

# The x-axis is set to 1800-2030 to exclude outliers like The Odyssey (~800 BC) and 
# show more detail in the modern distribution
ax.set_xlim(1800, 2030)
ax.set_ylim(0.5, 5.5)
ax.set_yticks(range(1, 6))
ax.set_xlabel("Publication Year")
ax.set_ylabel("Rating")
ax.set_title("Ratings vs Publication Year\n(circles = my rating, triangles = GR avg)")
ax.legend(fontsize=8, framealpha=0.7, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "7_ratings_vs_pub_year.png")

# ---------- 8. Ratings vs length (pages) ----------

len_books = [b for b in books if b["Pages"] and b["Avg Rating"] is not None]

fig, ax = plt.subplots(figsize=(9, 5))
for genre in GENRES:
    grp = [b for b in len_books if b["Genre 1"] == genre]
    if not grp:
        continue
    xs = [b["Pages"] for b in grp]
    ax.scatter(xs, [b["My Rating"] for b in grp],
               color=GENRE_COLOR[genre], marker="o", s=55, alpha=0.85, linewidths=0)
    ax.scatter(xs, [b["Avg Rating"] for b in grp],
               color=GENRE_COLOR[genre], marker="^", s=55, alpha=0.5, linewidths=0)

for genre in GENRES:
    if any(b["Genre 1"] == genre for b in len_books):
        ax.scatter([], [], color=GENRE_COLOR[genre], label=genre, s=55)
ax.scatter([], [], color="gray", marker="o", label="My Rating", s=55)
ax.scatter([], [], color="gray", marker="^", label="GR Avg", s=55, alpha=0.5)

# Power broker (1246 pages) is an outlier that skews the x-axis, 
# so set a cutoff at 800 to show more detail in the rest of the distribution
ax.set_xlim(0, 800)
ax.set_ylim(0.5, 5.5)
ax.set_yticks(range(1, 6))
ax.set_xlabel("Page Count")
ax.set_ylabel("Rating")
ax.set_title("Ratings vs Book Length\n(circles = my rating, triangles = GR avg)")
ax.legend(fontsize=8, framealpha=0.7, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "8_ratings_vs_length.png")

# ---------- 9. Length vs publication year ----------

yr_len_books = [b for b in books if b["Pages"]
                and b["Release"] and 1800 <= int(b["Release"]) <= 2030]

fig, ax = plt.subplots(figsize=(9, 5))
for genre in GENRES:
    grp = [b for b in yr_len_books if b["Genre 1"] == genre]
    if not grp:
        continue
    ax.scatter([int(b["Release"]) for b in grp], [b["Pages"] for b in grp],
               color=GENRE_COLOR[genre], label=genre, s=55, alpha=0.85, linewidths=0)

# The x-axis is set to 1800-2030 to exclude outliers like The Odyssey (~800 BC) and 
# show more detail in the modern distribution. The y-axis is set to a max of 800 to 
# exclude outliers like Power Broker (1246 pages).
ax.set_xlim(1800, 2030)
ax.set_ylim(0, 800)
ax.set_xlabel("Publication Year")
ax.set_ylabel("Pages")
ax.set_title("Book Length vs Publication Year")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=8, framealpha=0.7, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "9_length_vs_pub_year.png")

print("\nAll plots saved to analysis_outputs/")
