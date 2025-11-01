from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os

# -----------------------------
# 1. Load all notes
# -----------------------------
NOTES_FOLDER = "notes"
documents = []

for file_name in os.listdir(NOTES_FOLDER):
    if file_name.endswith((".txt", ".md")):
        with open(os.path.join(NOTES_FOLDER, file_name), "r", encoding="utf-8") as f:
            documents.append(Document(
                page_content=f.read(),
                metadata={"source": file_name}
            ))

# -----------------------------
# 2. Split texts into chunks
# -----------------------------
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
split_docs = [
    Document(page_content=chunk, metadata=doc.metadata)
    for doc in documents
    for chunk in text_splitter.split_text(doc.page_content)
]

print(f"Total document chunks: {len(split_docs)}")
for doc in split_docs[:3]:
    print(doc)

# -----------------------------
# 3. Create embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# 4. Save/load FAISS index
# -----------------------------
INDEX_PATH = "faiss_index"

if os.path.exists(INDEX_PATH):
    # Load existing FAISS index
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings)
    print("FAISS index loaded from disk.")
else:
    # Create new FAISS index and save it
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print("FAISS index created and saved.")

# -----------------------------
# 5. Query function
# -----------------------------
def query_notes(query, k=3):
    """
    Query the FAISS vectorstore for top-k relevant chunks.
    """
    results = vectorstore.similarity_search(query, k=k)
    response = "\n---\n".join([doc.page_content for doc in results])
    return response

# -----------------------------
# 6. Quick interactive test
# -----------------------------
if __name__ == "__main__":
    print("Type your query (or 'exit' to quit):")
    while True:
        q = input("Ask something: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        print(query_notes(q))
