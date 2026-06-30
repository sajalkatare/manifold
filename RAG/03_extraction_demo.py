import argparse
import warnings
warnings.filterwarnings("ignore")  # hide pypdf's crypto deprecation noise in class

from ragkit import extract_naive, extract_structured, quality_report


def rule(label):
    print("\n" + "=" * 74)
    print(label)
    print("=" * 74)


def main(doc):
    rule(f"DOCUMENT: {doc}")

    # ---------- 1. NAIVE ----------
    rule("1) NAIVE EXTRACTION  (pypdf — the way tutorials do it)")
    naive_pages = extract_naive(doc)
    naive_text = "\n".join(p.text for p in naive_pages)
    # Show the part that breaks most visibly: the last page (two columns) if present
    show = naive_pages[-1].text
    print(show)
    nq = quality_report(naive_text)
    print("\n--- quality flags (naive) ---")
    for problem in nq["summary"]:
        print("   [!]", problem)

    # ---------- 2. STRUCTURED ----------
    rule("2) STRUCTURED EXTRACTION  (pdfplumber — layout + tables)")
    struct_pages = extract_structured(doc)
    struct_text = "\n".join(p.text for p in struct_pages)
    print(struct_pages[-1].text)
    if struct_pages[0].tables:
        print("\n--- tables recovered on page 1 (clean rows) ---")
        for row in struct_pages[0].tables[0]:
            print("   ", row)
    sq = quality_report(struct_text)
    print("\n--- quality flags (structured) ---")
    for problem in sq["summary"]:
        print("   [ok]" if "no obvious" in problem else "   [~]", problem)

    # ---------- 3. THE VERDICT ----------
    rule("3) THE VERDICT")
    print(f"   naive  : interleave={nq['column_interleave_hits']}  "
          f"short_lines={nq['suspicious_short_lines']}  split_words={nq['split_words']}")
    print(f"   struct : interleave={sq['column_interleave_hits']}  "
          f"short_lines={sq['suspicious_short_lines']}  split_words={sq['split_words']}")
    fixed = nq["column_interleave_hits"] - sq["column_interleave_hits"]
    print()
    if fixed > 0:
        print(f"   -> Structured extraction removed {fixed} column-interleave failure(s).")
        print("      In the naive output, sick-leave and parental-leave text were")
        print("      scrambled together. That single corruption is the $15K disaster:")
        print("      a confidently WRONG chunk that every layer above will trust.")
    else:
        print("   -> Compare the two outputs above. Naive flattens structure;")
        print("      structured preserves it. Extraction is a decision, not a default.")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="messy_policy.pdf",
                    help="path to a PDF (default: the messy demo policy)")
    args = ap.parse_args()
    main(args.doc)
