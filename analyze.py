# Personal book ratings analysis
# Source data: Reading_data_table.csv
# Ratings are 1-5 (mine) vs Goodreads avg; "diff" = my rating - GR avg
# Run with: py analyze.py

import csv
from collections import defaultdict

with open(r"input_tables/Reading_data_table.csv", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    books = [row for row in reader if row["Title"].strip()]

for b in books:
    b["My Rating"] = int(b["My Rating"])
    b["Avg Rating"] = float(b["Avg Rating"]) if b["Avg Rating"] else None
    b["Pages"] = int(b["Pages"]) if b["Pages"] else None
    b["School?"] = int(b["School?"])
    b["diff"] = b["My Rating"] - b["Avg Rating"] if b["Avg Rating"] else None

total = len(books)
school = [b for b in books if b["School?"]]
non_school = [b for b in books if not b["School?"]]

def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0

def fmt(v): return f"{v:.3f}"

# Mirror all print output to a file as well
output_file = open(r"analysis_outputs/analysis_output.txt", "w", encoding="utf-8")

def p(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_file)


p("=" * 60)
p(f"OVERVIEW  ({total} books)")
p("=" * 60)
p(f"  Avg my rating:        {fmt(avg([b['My Rating'] for b in books]))}")
p(f"  Avg GR rating:        {fmt(avg([b['Avg Rating'] for b in books]))}")
p(f"  Avg differential:     {fmt(avg([b['diff'] for b in books]))}")
p(f"  Total pages read:     {sum(b['Pages'] for b in books if b['Pages']):,}")
p()

p("SCHOOL vs NON-SCHOOL")
p("-" * 60)
for label, grp in [("School", school), ("Non-school", non_school)]:
    my_avg = avg([b["My Rating"] for b in grp])
    gr_avg = avg([b["Avg Rating"] for b in grp])
    diff_avg = avg([b["diff"] for b in grp])
    p(f"  {label} ({len(grp)} books):")
    p(f"    Avg my rating:    {fmt(my_avg)}")
    p(f"    Avg GR rating:    {fmt(gr_avg)}")
    p(f"    Avg differential: {fmt(diff_avg)}")
p()

p("BY GENRE")
p("-" * 60)
genres = defaultdict(list)
for b in books:
    genres[b["Genre 1"]].append(b)

rows = []
for genre, grp in sorted(genres.items()):
    rows.append((genre, len(grp), avg([b["My Rating"] for b in grp]),
                 avg([b["Avg Rating"] for b in grp]),
                 avg([b["diff"] for b in grp])))
rows.sort(key=lambda r: r[2], reverse=True)

p(f"  {'Genre':<12} {'N':>3}  {'My Avg':>7}  {'GR Avg':>7}  {'Diff':>7}")
for genre, n, my_avg, gr_avg, diff in rows:
    p(f"  {genre:<12} {n:>3}  {my_avg:>7.3f}  {gr_avg:>7.3f}  {diff:>+7.3f}")
p()

# Break out school vs non-school differential per genre to see where the school penalty lives
p("SCHOOL BOOKS BY GENRE (differential)")
p("-" * 60)
sg = defaultdict(list)
for b in school:
    sg[b["Genre 1"]].append(b)
for genre, grp in sorted(sg.items()):
    d = avg([b["diff"] for b in grp])
    p(f"  {genre:<12}  n={len(grp)}  avg diff={d:+.3f}")
p()

p("NON-SCHOOL BOOKS BY GENRE (differential)")
p("-" * 60)
sg = defaultdict(list)
for b in non_school:
    sg[b["Genre 1"]].append(b)
for genre, grp in sorted(sg.items()):
    d = avg([b["diff"] for b in grp])
    p(f"  {genre:<12}  n={len(grp)}  avg diff={d:+.3f}")
p()

p("BOOKS PER YEAR READ")
p("-" * 60)
by_year = defaultdict(list)
for b in books:
    by_year[b["Year Read"]].append(b)
for year in sorted(by_year):
    grp = by_year[year]
    p(f"  {year}: {len(grp):>3} books  |  avg my rating: {fmt(avg([b['My Rating'] for b in grp]))}")
p()

p("RATING DISTRIBUTION (my ratings)")
p("-" * 60)
for r in range(1, 6):
    n = sum(1 for b in books if b["My Rating"] == r)
    bar = "#" * n
    p(f"  {r}/5: {bar:<30} ({n})")
p()

p("TOP 5 RATED (my rating, then GR avg as tiebreak)")
p("-" * 60)
top = sorted(books, key=lambda b: (b["My Rating"], b["Avg Rating"] or 0), reverse=True)[:5]
for b in top:
    p(f"  {b['My Rating']}/5  GR:{b['Avg Rating']}  {b['Title']} — {b['Author']}")
p()

p("BOTTOM 5 RATED")
p("-" * 60)
bot = sorted(books, key=lambda b: (b["My Rating"], b["Avg Rating"] or 0))[:5]
for b in bot:
    p(f"  {b['My Rating']}/5  GR:{b['Avg Rating']}  {b['Title']} — {b['Author']}")
p()

all_diff = sorted([b for b in books if b["diff"] is not None], key=lambda b: b["diff"], reverse=True)

p("BIGGEST POSITIVE DIFFERENTIALS (I rated higher than GR)")
p("-" * 60)
for b in all_diff[:5]:
    p(f"  {b['diff']:+.2f}  My:{b['My Rating']}  GR:{b['Avg Rating']}  {b['Title']}")
p()

p("BIGGEST NEGATIVE DIFFERENTIALS (I rated lower than GR)")
p("-" * 60)
for b in all_diff[-5:]:
    p(f"  {b['diff']:+.2f}  My:{b['My Rating']}  GR:{b['Avg Rating']}  {b['Title']}")
p()

p("AUTHORS WITH MULTIPLE BOOKS")
p("-" * 60)
authors = defaultdict(list)
for b in books:
    if b["Author"].strip():
        authors[b["Author"]].append(b)
multi = {a: grp for a, grp in authors.items() if len(grp) >= 2}
for author, grp in sorted(multi.items(), key=lambda x: avg([b["My Rating"] for b in x[1]]), reverse=True):
    titles = ", ".join(b["Title"] for b in grp)
    p(f"  {author} (n={len(grp)}, avg={fmt(avg([b['My Rating'] for b in grp]))}): {titles}")

output_file.close()
print("\nOutput saved to analysis_output.txt")
