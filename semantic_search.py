from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

# -----------------------------
# 1. Load Documents
# -----------------------------
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

documents = [doc.strip() for doc in documents if doc.strip() != ""]
print("Total documents loaded:", len(documents))

# -----------------------------
# 2. Load Model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 3. Create or Load FAISS Index
# -----------------------------
index_file = "faiss_index.bin"

if os.path.exists(index_file):
    print("Loading existing FAISS index...")
    index = faiss.read_index(index_file)
else:
    print("Creating new FAISS index...")
    doc_embeddings = model.encode(documents)
    doc_embeddings = np.array(doc_embeddings).astype("float32")

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)

    faiss.write_index(index, index_file)
    print("FAISS index saved to disk!")

print("Documents indexed:", index.ntotal)

# -----------------------------
# 4. Search Function
# -----------------------------
def search(query, k=3):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    distances, indices = index.search(query_vector, k)

    print("\nTop Results:")
    for rank, i in enumerate(indices[0]):
        print(f"{rank+1}. {documents[i]}")
        print(f"   Distance: {distances[0][rank]:.4f}")

# -----------------------------
# 5. Chat-Style Loop
# -----------------------------
print("\nSemantic Search Ready! Type 'exit' to quit.\n")

while True:
    user_query = input("You: ")

    if user_query.lower() == "exit":
        print("Exiting Semantic Search. Goodbye 👋")
        break

    search(user_query)
