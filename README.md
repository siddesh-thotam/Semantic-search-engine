# 🔍 Semantic Search Engine (AI Student Handbook Edition)

A beginner-to-intermediate level Semantic Search Engine built using:

- SentenceTransformers
- FAISS (Cosine Similarity)
- Python
- PDF text extraction (pypdf)

This project demonstrates how modern AI systems retrieve relevant information using vector embeddings and similarity search.

---

# 🚀 Features

✅ Load large documents (PDF → text)  
✅ Automatic text chunking  
✅ Cosine similarity search  
✅ Similarity threshold filtering  
✅ Persistent FAISS index (no rebuilding every time)  
✅ Chat-style interactive search loop  

---

# 🧠 How It Works

1. PDF is converted to plain text.
2. Text is split into chunks.
3. Each chunk is converted into embeddings.
4. Embeddings are stored in FAISS index.
5. User query is converted into embedding.
6. Cosine similarity is used to retrieve top relevant chunks.

---

# 📦 Installation Guide

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/siddesh-thotam/Semantic-search-engine.git
cd Semantic-search-engine

2️⃣ Create Virtual Environment (Recommended)

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

3️⃣ Install Dependencies

pip install sentence-transformers faiss-cpu numpy pypdf

📄 How To Use
Step 1: Add Your PDF

Place your AI PDF file in the project folder.

Example:

AI_Student_Handbook.pdf

Step 2: Convert PDF to Text
Run:
python pdf_to_text.py

This creates:
documents.txt

Step 3: Delete Old FAISS Index (If Replacing Documents)
del faiss_index.bin

Step 4: Run Semantic Search
python semantic_search.py

💬 Using the Search

After running:

Semantic Search Ready!
Type 'exit' to quit.


Example queries:

What is artificial intelligence?

Explain supervised learning

What is NLP?

Applications of AI

Difference between AI and ML

Type:

exit


to quit the program.

📊 Similarity Scores

Results are shown as:

Similarity: 67.42%


Higher percentage = more relevant.

Threshold filtering removes weak matches automatically.

📁 Project Structure

Semantic-search-engine/
│
├── semantic_search.py
├── pdf_to_text.py
├── documents.txt
├── README.md
├── faiss_index.bin (auto-generated)
└── venv/ (ignored)

🛠 Technologies Used

SentenceTransformers (all-MiniLM-L6-v2)

FAISS (Inner Product for Cosine Similarity)

NumPy

PyPDF

Python 3.9+

