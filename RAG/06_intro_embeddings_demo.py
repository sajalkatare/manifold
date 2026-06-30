from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

c1 = "What is the leave policy for new parents?"
c2 = "Parental policy is for 2 weeks"
c3 = "Machine learning is a subfield of artificial intelligence"
c4 = "Cat"

emb1 = embeddings.embed_query(c1)
emb2 = embeddings.embed_query(c2)
emb3 = embeddings.embed_query(c3)
emb4 = embeddings.embed_query(c4)
print("--------------------------------")
print(f"Size of embedding 1: {len(emb1)}")
print(emb1)
print("--------------------------------")
print(f"Size of embedding 2: {len(emb2)}")
print(emb2)
print("--------------------------------")
print(f"Size of embedding 3: {len(emb3)}")
print(emb3)
print("--------------------------------")
print(f"Size of embedding 4: {len(emb4)}")
print(emb4)
print("--------------------------------")
# calculate the similarity between the embeddings
def calculate_similarity(emb1, emb2):
    return np.array(emb1).dot(np.array(emb2))/(np.linalg.norm(emb1)*np.linalg.norm(emb2))

print(f"Similarity between embedding 1 and embedding 2: {calculate_similarity(emb1, emb2)}")
print(f"Similarity between embedding 1 and embedding 3: {calculate_similarity(emb1, emb3)}")
print(f"Similarity between embedding 2 and embedding 3: {calculate_similarity(emb2, emb3)}")