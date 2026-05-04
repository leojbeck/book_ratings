import csv
from collections import defaultdict

with open("Reading_data_table.csv", newline="", encoding="utf-8-sig") as f:
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

print("=" * 60)
print(f"OVERVIEW  ({total} books)")
print("=" * 60)
print(f"  Avg my rating:        {fmt(avg([b['My Rating'] for b in books]))}")
print(f"  Avg GR rating:        {fmt(avg([b['Avg Rating'] for b in books]))}")
print(f"  Avg differential:     {fmt(avg([b['diff'] for b in books]))}")
print(f"  Total pages read:     {sum(b['Pages'] for b in books if b['Pages']):,}")
print()

print("SCHOOL vs NON-SCHOOL")
print("-" * 60)
for label, grp in [("School", school), ("Non-school", non_school)]:
    my_avg = avg([b["My Rating"] for b in grp])
    gr_avg = avg([b["Avg Rating"] for b in grp])
    diff_avg = avg([b["diff"] for b in grp])
    print(f"  {label} ({len(grp)} books):")
    print(f"    Avg my rating:    {fmt(my_avg)}")
    print(f"    Avg GR rating:    {fmt(gr_avg)}")
    print(f"    Avg differential: {fmt(diff_avg)}")
print()

print("BY GENRE")
print("-" * 60)
genres = defaultdict(list)
for b in books:
    genres[b["Genre"]].append(b)

rows = []
for genre, grp in sorted(genres.items()):
    rows.append((genre, len(grp), avg([b["My Rating"] for b in grp]),
                 avg([b["Avg Rating"] for b in grp]),
                 avg([b["diff"] for b in grp])))
rows.sort(key=lambda r: r[2], reverse=True)

print(f"  {'Genre':<12} {'N':>3}  {'My Avg':>7}  {'GR Avg':>7}  {'Diff':>7}")
for genre, n, my_avg, gr_avg, diff in rows:
    print(f"  {genre:<12} {n:>3}  {my_avg:>7.3f}  {gr_avg:>7.3f}  {diff:>+7.3f}")
print()

print("SCHOOL BOOKS BY GENRE (differential)")
print("-" * 60)
sg = defaultdict(list)
for b in school:
    sg[b["Genre"]].append(b)
for genre, grp in sorted(sg.items()):
    d = avg([b["diff"] for b in grp])
    print(f"  {genre:<12}  n={len(grp)}  avg diff={d:+.3f}")
print()

print("BOOKS PER YEAR READ")
print("-" * 60)
by_year = defaultdict(list)
for b in books:
    by_year[b["Year Read"]].append(b)
for year in sorted(by_year):
    grp = by_year[year]
    print(f"  {year}: {len(grp):>3} books  |  avg my rating: {fmt(avg([b['My Rating'] for b in grp]))}")
print()

print("RATING DISTRIBUTION (my ratings)")
print("-" * 60)
for r in range(1, 6):
    n = sum(1 for b in books if b["My Rating"] == r)
    bar = "#" * n
    print(f"  {r}/5: {bar:<30} ({n})")
print()

print("TOP 5 RATED (my rating, then GR avg as tiebreak)")
print("-" * 60)
top = sorted(books, key=lambda b: (b["My Rating"], b["Avg Rating"] or 0), reverse=True)[:5]
for b in top:
    print(f"  {b['My Rating']}/5  GR:{b['Avg Rating']}  {b['Title']} — {b['Author']}")
print()

print("BOTTOM 5 RATED")
print("-" * 60)
bot = sorted(books, key=lambda b: (b["My Rating"], b["Avg Rating"] or 0))[:5]
for b in bot:
    print(f"  {b['My Rating']}/5  GR:{b['Avg Rating']}  {b['Title']} — {b['Author']}")
print()

all_diff = sorted([b for b in books if b["diff"] is not None], key=lambda b: b["diff"], reverse=True)

print("BIGGEST POSITIVE DIFFERENTIALS (I rated higher than GR)")
print("-" * 60)
for b in all_diff[:5]:
    print(f"  {b['diff']:+.2f}  My:{b['My Rating']}  GR:{b['Avg Rating']}  {b['Title']}")
print()

print("BIGGEST NEGATIVE DIFFERENTIALS (I rated lower than GR)")
print("-" * 60)
for b in all_diff[-5:]:
    print(f"  {b['diff']:+.2f}  My:{b['My Rating']}  GR:{b['Avg Rating']}  {b['Title']}")
print()

print("AUTHORS WITH MULTIPLE BOOKS")
print("-" * 60)
authors = defaultdict(list)
for b in books:
    if b["Author"].strip():
        authors[b["Author"]].append(b)
multi = {a: grp for a, grp in authors.items() if len(grp) >= 2}
for author, grp in sorted(multi.items(), key=lambda x: avg([b["My Rating"] for b in x[1]]), reverse=True):
    titles = ", ".join(b["Title"] for b in grp)
    print(f"  {author} (n={len(grp)}, avg={fmt(avg([b['My Rating'] for b in grp]))}): {titles}")
