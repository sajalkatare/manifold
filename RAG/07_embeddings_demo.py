import os
import sys
import json
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

def get_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_chunks(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"{path} not found. Run: python build_corpus.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    emb = get_embedder()
    print(f"ACTIVE EMBEDDER (LangChain): openai:{EMBED_MODEL}  (dim={EMBED_DIM})")

    chunks = load_chunks("RAG\\data\\chunks.json")
    texts = [c["content"] for c in chunks]

    # 1) an embedding is just numbers — LangChain: embed_query(text) -> list[float]
    print("1) AN EMBEDDING IS A VECTOR")
    v = np.array(emb.embed_query(texts[0]), dtype=np.float32)
    print("chunk:", texts[0][:70], "...")
    print(f"-> vector of {len(v)} numbers. first 8: {np.round(v[:8], 3)}")

    # 2) cosine similarity — LangChain: embed_documents(list) -> list[list[float]]
    print("2) COSINE SIMILARITY  (query vs chunks)")
    query = "How long is parental leave?"
    qv = np.array(emb.embed_query(query), dtype=np.float32) # query embedding
    mat = np.array(emb.embed_documents(texts), dtype=np.float32) # generate the emebddings for full text
    sims = np.array([cosine(qv, row) for row in mat]) # calculate the cosine similarity between the query embedding aˀnd the full text embeddings
    order = np.argsort(-sims)[:5] # sort and selct top 5
    print(f"query: {query!r}\n")
    for i in order:
        print(f"   cos={sims[i]:.3f}  [{chunks[i]['section']:20s}] {texts[i][:54]}...")

    # 3) similar != answers
    print("3) 'SIMILAR' IS NOT ALWAYS 'ANSWERS THE QUESTION'")
    top = chunks[order[0]]
    print(f"top match: [{top['section']}] (cosine {sims[order[0]]:.3f})")


if __name__ == "__main__":
    main()
