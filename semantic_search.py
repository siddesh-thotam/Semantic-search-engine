from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

# -----------------------------
# 1. Load and Clean Book
# -----------------------------
def load_book(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Remove Gutenberg header/footer
    start_marker = "*** START"
    end_marker = "*** END"

    start_index = text.find(start_marker)
    end_index = text.find(end_marker)

    if start_index != -1 and end_index != -1:
        text = text[start_index:end_index]

    return text


# -----------------------------
# 2. Chunk Text
# -----------------------------
def chunk_text(text, chunk_size=400):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -----------------------------
# 3. Load Book + Chunk It
# -----------------------------
book_text = load_book("documents.txt")
documents = chunk_text(book_text, chunk_size=400)

print("Total chunks created:", len(documents))


# -----------------------------
# 4. Load Model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 5. Create or Load FAISS Index (Cosine Similarity)
# -----------------------------
index_file = "faiss_index.bin"

if os.path.exists(index_file):
    print("Loading existing FAISS index...")
    index = faiss.read_index(index_file)
else:
    print("Creating new FAISS index...")

    doc_embeddings = model.encode(documents)
    doc_embeddings = np.array(doc_embeddings).astype("float32")

    faiss.normalize_L2(doc_embeddings)

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    index.add(doc_embeddings)
    faiss.write_index(index, index_file)

    print("FAISS index saved to disk!")

print("Documents indexed:", index.ntotal)


# -----------------------------
# 6. Search Function
# -----------------------------
def search(query, k=5, threshold=0.4):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    faiss.normalize_L2(query_vector)

    similarities, indices = index.search(query_vector, k)

    print("\nTop Results:\n")

    found = False

    for rank, i in enumerate(indices[0]):
        score = similarities[0][rank]

        if score >= threshold:
            found = True
            percentage = score * 100
            print(f"Result {rank+1} (Similarity: {percentage:.2f}%):\n")
            print(documents[i][:500])
            print("\n" + "-"*80 + "\n")

    if not found:
        print("No relevant results found.")


# -----------------------------
# 7. Chat Loop
# -----------------------------
print("\nSemantic Search Ready (Book Mode)!")
print("Type 'exit' to quit.\n")

while True:
    user_query = input("You: ")

    if user_query.lower() == "exit":
        print(" Exiting Semantic Search. ")
        break

    search(user_query)
