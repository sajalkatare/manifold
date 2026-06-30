import os
import json
import warnings
warnings.filterwarnings("ignore")

from ragkit import (extract_structured, chunk_fixed, records_from_pages)

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
DOC = "messy_policy.pdf"


def rule(label):
    print("\n" + "=" * 74)
    print(label)
    print("=" * 74)


def main():
    pages = extract_structured(DOC)
    full_text = "\n".join(p.text for p in pages)

    # ---------- 1. FIXED-SIZE (the naive default) ----------
    rule("1) FIXED-SIZE CHUNKING  (size=120 to make the damage visible)")
    fixed = chunk_fixed(full_text, size=120, overlap=0)
    for i, c in enumerate(fixed[:4]):
        print(f"\n  chunk {i} ({len(c)} chars):")
        print("   ", repr(c))
    print("\n  [!] Notice chunks cut mid-sentence and mid-word. A chunk that ends")
    print("      '...entitled to twenty' cannot answer 'how many weeks?' on its own.")

    # ---------- 2. RECURSIVE + SECTION-AWARE (good chunks) ----------
    rule("2) SECTION-AWARE + RECURSIVE CHUNKING  (units of answerability)")
    records = records_from_pages(pages, document_type="policy", size=500, overlap=50)
    for r in records:
        head = r["content"].replace("\n", " ")[:64]
        print(f"  {r['section'][:26]:26s} p{r['page']} | {len(r['content']):3d}c | {head}...")
    print(f"\n  -> {len(records)} good chunks. Each is a self-contained unit of meaning.")

    # ---------- 3. METADATA AT CHUNK BIRTH ----------
    rule("3) METADATA AT CHUNK BIRTH  (3-chunk sample)")
    for r in records[:3]:
        print(json.dumps({
            "chunk_id": r["chunk_id"],
            "source": r["source"],
            "page": r["page"],
            "section": r["section"],
            "document_type": r["document_type"],
            "content_preview": r["content"].replace("\n", " ")[:70] + "...",
        }, indent=2))

    # ---------- 4. SAVE THE ARTIFACT ----------
    rule("4) SAVE retrieval-ready chunks -> outputs/chunks.json")
    path = os.path.join(OUT, "chunks.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"   Wrote {len(records)} chunks to {path}")
    print("   This is your Session 2 input: chunks WITH metadata, ready to embed.")
    print("\nNext (optional stretch): python 03_chunk_inspector.py")


if __name__ == "__main__":
    main()
