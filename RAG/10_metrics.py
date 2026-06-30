def precision_at_k(hit_ids, relevant_ids, k=5):
    topk = hit_ids[:k]
    return sum(1 for cid in topk if cid in relevant_ids) / min(k, len(topk)) if topk else 0.0


def hit_at_k(hit_ids, relevant_ids, k=5):
    return int(any(cid in relevant_ids for cid in hit_ids[:k]))


def recall_at_k(hit_ids, relevant_ids, k=5):
    if not relevant_ids:
        return 0.0
    return len(set(hit_ids[:k]) & relevant_ids) / len(relevant_ids)


print("=" * 60)
print("WARM-UP 4 — measuring retrieval (no API, no DB)")
print("=" * 60)

retrieved = ["c7", "c3", "c1", "c9", "c2"]   # what the retriever returned (ranked)
relevant = {"c3"}                            # the ONE chunk that truly answers it

print(f"\nretrieved top-5 : {retrieved}")
print(f"relevant (truth): {relevant}\n")

print(f"Hit@5       = {hit_at_k(retrieved, relevant)}      (did a relevant chunk show up? 1=yes)")
print(f"Recall@5    = {recall_at_k(retrieved, relevant):.2f}   (fraction of relevant chunks found)")
print(f"Precision@5 = {precision_at_k(retrieved, relevant):.2f}   (fraction of the top-5 that are relevant)")

print("\nWhy is Precision@5 only 0.20 when retrieval clearly WORKED?")
print("Because there's just 1 relevant chunk — best possible is 1/5 = 0.20.")
print("That's the metric, not a failure. Always read Precision@5 next to Hit@5.")

