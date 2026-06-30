import os
import sys
import json
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from dotenv import load_dotenv
load_dotenv()
EMBED_MODEL = "text-embedding-3-small"
COLLECTION = "session2_kb"
DEFAULT_DSN = "postgresql+psycopg://rag:rag@localhost:5433/ragdb"
OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def die(msg: str):
    print(f"\n[setup] {msg}", file=sys.stderr)
    sys.exit(1)


def pg_connection() -> str:
    dsn = os.getenv("DATABASE_URL", DEFAULT_DSN)
    if dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    return dsn


def get_embedder() -> OpenAIEmbeddings:
    if not os.getenv("OPENAI_API_KEY"):
        die("OPENAI_API_KEY is not set. export it and re-run (real API, no mock).")
    return OpenAIEmbeddings(model=EMBED_MODEL)


def build_store(chunks: list[dict], emb: OpenAIEmbeddings) -> PGVector:
    docs = [Document(page_content=c["content"],
                     metadata={"source": c.get("source"), "section": c.get("section"),
                               "chunk_id": c.get("chunk_id"),
                               "document_type": c.get("document_type")})
            for c in chunks]
    try:
        store = PGVector(embeddings=emb, collection_name=COLLECTION,
                         connection=pg_connection(), use_jsonb=True,
                         pre_delete_collection=True)
    except Exception as e:
        die(f"cannot connect to Postgres ({e}). Start it with: docker compose up -d")
    # No custom ids: langchain_pg_embedding ids are global across collections, so
    # reusing chunk_ids would collide with 02's collection. chunk_id is in metadata.
    store.add_documents(docs)
    return store


# ----------------------------- metrics ---------------------------------------
def precision_at_k(hit_ids, relevant_ids, k=5):
    topk = hit_ids[:k]
    return sum(1 for cid in topk if cid in relevant_ids) / min(k, len(topk)) if topk else 0.0


def hit_at_k(hit_ids, relevant_ids, k=5):
    return int(any(cid in relevant_ids for cid in hit_ids[:k]))


def recall_at_k(hit_ids, relevant_ids, k=5):
    if not relevant_ids:
        return 0.0
    return len(set(hit_ids[:k]) & relevant_ids) / len(relevant_ids)


def main():
    if not os.path.exists("RAG\\data\\chunks.json"):
        die("RAG\\data\\chunks.json not found. Run: python build_corpus.py")
    chunks = json.load(open("RAG\\data\\chunks.json", encoding="utf-8"))
    golden = json.load(open("RAG\\data\\golden_queries.json", encoding="utf-8"))

    emb = get_embedder()
    store = build_store(chunks, emb)
    retriever = store.as_retriever(search_kwargs={"k": 5})   # LangChain retriever

    print("=" * 72)
    print(f"RETRIEVAL SCORECARD   embedder=openai:{EMBED_MODEL}  store=pgvector(langchain)")
    print("=" * 72)
    print(f"{'Hit@5':>5} {'R@5':>5} {'P@5':>5}   query")
    rows, P = [], dict(hit=0, rec=0.0, prec=0.0)
    for g in golden:
        hits = retriever.invoke(g["query"])
        ids = [h.metadata["chunk_id"] for h in hits]
        rel = set(g["relevant"])
        h5, r5, p5 = hit_at_k(ids, rel), recall_at_k(ids, rel), precision_at_k(ids, rel)
        P["hit"] += h5; P["rec"] += r5; P["prec"] += p5
        print(f"{h5:>5} {r5:>5.2f} {p5:>5.2f}   {g['query'][:46]}")
        rows.append({"query": g["query"], "top5": ids,
                     "top5_sections": [h.metadata["section"] for h in hits],
                     "relevant": list(rel), "hit@5": h5,
                     "recall@5": round(r5, 3), "precision@5": round(p5, 3)})
    n = len(golden)
    summary = {"embedder": f"openai:{EMBED_MODEL}", "store": "pgvector(langchain)", "n_queries": n,
               "mean_hit@5": round(P["hit"] / n, 3),
               "mean_recall@5": round(P["rec"] / n, 3),
               "mean_precision@5": round(P["prec"] / n, 3)}
    print("-" * 72)
    print(f"MEAN  Hit@5={summary['mean_hit@5']}  "
          f"Recall@5={summary['mean_recall@5']}  "
          f"Precision@5={summary['mean_precision@5']}")

    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(OUT, "scorecard.json"), "w"), indent=2)
    with open(os.path.join(OUT, "scorecard.md"), "w") as f:
        f.write(f"# Retrieval Scorecard\n\nembedder: `openai:{EMBED_MODEL}` · "
                f"store: `pgvector (LangChain)`\n\n")
        f.write("| Hit@5 | Recall@5 | Precision@5 | Query |\n|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['hit@5']} | {r['recall@5']:.2f} | {r['precision@5']:.2f} | {r['query']} |\n")
        f.write(f"\n**Mean** Hit@5 {summary['mean_hit@5']} · "
                f"Recall@5 {summary['mean_recall@5']} · Precision@5 {summary['mean_precision@5']}\n")
    print(f"\nWrote outputs/scorecard.json and outputs/scorecard.md")



if __name__ == "__main__":
    main()
