import sys

QUERY = "How much time off do new parents get?"
HF_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

HYBRID_CANDIDATES = [
    {
        "section": "Core Hours",
        "text": "Core collaboration hours are 11:00 AM to 4:00 PM local time.",
        "hybrid_rank": 1,
    },
    {
        "section": "Parental Leave",
        "text": "Employees are entitled to 26 weeks of parental leave per child.",
        "hybrid_rank": 2,
    },
    {
        "section": "Earned Leave",
        "text": "Earned leave accrues at 18 days per year after the first year.",
        "hybrid_rank": 3,
    },
    {
        "section": "Wellness",
        "text": "The wellness program reimburses gym memberships up to 500 USD.",
        "hybrid_rank": 4,
    },
]


def die(msg: str):
    print(f"\n[setup] {msg}", file=sys.stderr)
    sys.exit(1)


def load_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        die("sentence-transformers is not installed. Run: pip install sentence-transformers")
    return CrossEncoder(HF_MODEL)


def rerank(query: str, candidates: list[dict], reranker) -> list[tuple[dict, float]]:
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    return sorted(zip(candidates, scores), key=lambda pair: -float(pair[1]))


def show_hybrid(label, items):
    print(f"\n{label}:")
    for rank, item in enumerate(items, 1):
        print(f"  {rank}. {item['section']:16s}")


def show_reranked(label, scored):
    print(f"\n{label}:")
    for rank, (item, score) in enumerate(scored, 1):
        print(f"  {rank}. {item['section']:16s}  score={float(score):.4f}")


print("=" * 60)
print("Retrieve wide, rerank narrow")
print("=" * 60)
print(f"\nQuery: {QUERY!r}")

show_hybrid("hybrid recall", HYBRID_CANDIDATES)

reranker = load_cross_encoder()
show_reranked("after rerank", rerank(QUERY, HYBRID_CANDIDATES, reranker))
