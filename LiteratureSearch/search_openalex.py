import csv
import requests
import time
from datetime import datetime, timezone

API_BASE = "https://api.openalex.org/works"
API_KEY = "API_KEY"
EMAIL = "caleb.hendren@chattanoogastate.edu"

EXTRACTION_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

biology_keywords = [
    "scRNA-seq", "scRNAseq", "bulk RNA", "transcriptomic data", "transcriptomics",
    "single-cell", "single cell", "biomarker discovery", "biomarker selection",
    "synthetic biology", "bioinformatics", "molecular biology",
    "gene networks", "gene network", "protein folding", "protein structure",
    "genomics", "genome sequencing",
    "proteomics", "metabolomics", "gene expression data", "gene expression",
    "biological sequences", "biological sequence", "protein structure prediction",
    "biological datasets", "DNA sequences", "RNA sequences", "epigenomics",
    "metagenomics", "lipidomics", "glycomics", "nutrigenomics", "RNA-seq", "RNA sequencing", "ATAC-seq", "scATAC-seq", "CITE-seq", "Hi-C", "multi-omics", "multiomics", "omics", "DNA methylation",
]

generative_ai_keywords = [
    "variational autoencoder", "variational autoencoders",
    "adversarial autoencoder",
    "generative adversarial network", "generative adversarial networks",
    "generative adversarial",
    "denoising diffusion probabilistic model", "denoising diffusion probabilistic models",
    "denoising diffusion",
    "diffusion probabilistic model",
    "score-based generative model", "score-based generative models",
    "deep generative model", "deep generative models",
    "normalizing flow", "normalizing flows",
    "generative neural network", "generative neural networks", "deep belief network", "deep belief networks", "energy-based generative", "latent diffusion", "deep generative", "generative ai", "generative artificial intelligence"
]


def or_group(terms):
    return "(" + " OR ".join(f'"{t}"' for t in terms) + ")"
SEARCH_QUERY = f"{or_group(biology_keywords)} AND {or_group(generative_ai_keywords)}"

FROM_DATE = "2015-01-01"
TO_DATE = "2025-12-31"

filter_str = (
    f"title_and_abstract.search:{SEARCH_QUERY},"
    f"from_publication_date:{FROM_DATE},"
    f"to_publication_date:{TO_DATE},"
    f"type:article,"
    f"has_abstract:true"
)

SELECT_FIELDS = ",".join([
    "id", "display_name", "publication_year",
    "abstract_inverted_index", "cited_by_count",
    "doi", "primary_location", "locations",
    "type",
])

output_file = f"openalex_genai_omics_{EXTRACTION_DATE}.csv"
fieldnames = ["Title", "Journal", "Year", "Abstract", "Cited By Count", "URL", "DOI", "Type"]


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return "N/A"
    pairs = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pairs)


def get_journal(work):
    source = ((work.get("primary_location") or {}).get("source") or {})
    name = source.get("display_name")
    if name:
        return name
    for loc in work.get("locations", []):
        loc_name = ((loc or {}).get("source") or {}).get("display_name")
        if loc_name:
            return loc_name
    return "N/A"


def fetch_all_works():
    cursor = "*"
    total = None
    fetched = 0

    while cursor:
        params = {
            "filter": filter_str,
            "select": SELECT_FIELDS,
            "per_page": 100,
            "cursor": cursor,
            "api_key": API_KEY,
            "mailto": EMAIL,
        }
        resp = requests.get(API_BASE, params=params, timeout=30)

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 5))
            print(f"  Rate limited... sleeping {retry_after}s ...")
            time.sleep(retry_after)
            continue

        resp.raise_for_status()
        data = resp.json()
        meta = data.get("meta", {})

        if total is None:
            total = meta.get("count", 0)
            print(f"Extraction date (UTC): {EXTRACTION_DATE}")
            print(f"Total matching works: {total}")

        results = data.get("results", [])
        if not results:
            break

        fetched += len(results)
        print(f"  Fetched {fetched}/{total}")

        yield from results
        cursor = meta.get("next_cursor")


if __name__ == "__main__":
    print("=== OpenAlex boolean query (title + abstract only) ===")
    print(SEARCH_QUERY)
    print(f"Filters: type=article, {FROM_DATE}..{TO_DATE}, has_abstract=true")
    print("=" * 55)

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for work in fetch_all_works():
            writer.writerow({
                "Title": work.get("display_name", "N/A"),
                "Journal": get_journal(work),
                "Year": work.get("publication_year", "N/A"),
                "Abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
                "Cited By Count": work.get("cited_by_count", 0),
                "URL": work.get("id", "N/A"),
                "DOI": work.get("doi", "N/A"),
                "Type": work.get("type", "N/A"),
            })

    print(f"Query complete. Results saved to {output_file}")

filtered_output_file = "openalex_biology_generative_ai_filtered.csv"

stats = {"total": 0, "duplicate": 0, "kept": 0}

with open(output_file, mode="r", newline="", encoding="utf-8") as infile, \
     open(filtered_output_file, mode="w", newline="", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()

    seen = set()

    for row in reader:
        stats["total"] += 1

        doi = (row.get("DOI") or "").strip().lower()
        title = (row.get("Title") or "").strip()
        dedup_key = doi if doi and doi != "N/A" else title
        if not dedup_key:
            pass
        elif dedup_key in seen:
            stats["duplicate"] += 1
            continue
        else:
            seen.add(dedup_key)

        writer.writerow(row)
        stats["kept"] += 1

print(f"Filtering complete. Results saved to {filtered_output_file}")
print(f"  Total:      {stats['total']}")
print(f"  Duplicates: {stats['duplicate']}")
print(f"  Kept:       {stats['kept']}")

import matplotlib.pyplot as plt
import csv
from collections import Counter
from datetime import datetime, timezone

plt.style.use("default")

filtered_output_file = "openalex_biology_generative_ai_filtered.csv"
output_image = "publications_by_year.png"
prov_txt_file = "figure1_provenance.txt"

EXTRACTION_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

years = []
with open(filtered_output_file, mode="r", newline="", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        year = row.get("Year", "")
        if year.isdigit():
            y = int(year)
            if 2015 <= y <= 2025:
                years.append(y)

year_counts = Counter(years)

all_years = list(range(2015, 2026))
counts = [year_counts.get(y, 0) for y in all_years]

fig = plt.figure(figsize=(10, 6.5))
bars = plt.bar(all_years, counts, color="steelblue", edgecolor="black")

for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
             ha="center", va="bottom", fontsize=10)

plt.xlabel("Year")
plt.ylabel("Number of Publications")
plt.title("Generative AI for Biological Data: Publications by Year")
plt.xticks(all_years, rotation=45)

prov_lines = [
    f"Source: OpenAlex (openalex.org); extracted {EXTRACTION_DATE}",
    "Inclusion: type=article; 2015\u20132025; abstract required; "
    "\u22651 biological-data term AND \u22651 generative-model term in title/abstract.",
]

plt.tight_layout(rect=[0, 0.10, 1, 1])
fig.text(0.5, 0.055, prov_lines[0], ha="center", va="top", fontsize=8.5, color="gray")
fig.text(0.5, 0.020, prov_lines[1], ha="center", va="top", fontsize=8,   color="gray")

plt.savefig(output_image, dpi=300, bbox_inches="tight")
plt.show()

full_query = globals().get(
    "SEARCH_QUERY",
)
with open(prov_txt_file, "w", encoding="utf-8") as f:
    f.write("Figure 1 — provenance and reproduction details\n")
    f.write("=" * 50 + "\n")
    f.write("Database:        OpenAlex (https://openalex.org), REST API\n")
    f.write(f"Extraction date: {EXTRACTION_DATE} (UTC)\n")
    f.write("Search field:    title_and_abstract.search (title + abstract only)\n")
    f.write('Text handling:   Kstem stemming + stop-word removal; '
            'double quotes = exact phrase; AND/OR uppercase boolean operators\n')
    f.write("Inclusion:       type=article; from_publication_date 2015-01-01; "
            "to_publication_date 2025-12-31; has_abstract=true\n")
    f.write("Logic:           \u22651 biological-data term AND \u22651 generative-model term\n")
    f.write("De-duplication:  by DOI (lowercased), OpenAlex Work ID fallback\n\n")
    f.write("Full Boolean query string:\n")
    f.write(full_query + "\n")

print(f"Plot saved as {output_image}")
print(f"Provenance saved as {prov_txt_file}")