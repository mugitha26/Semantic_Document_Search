import os
import glob
import pandas as pd

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "chroma_db")
DOCS_PATH = os.path.join(BASE_DIR, "dataset", "documents")
QUERIES_PATH = os.path.join(BASE_DIR, "dataset", "queries.csv")

RESULTS_PATH = os.path.join(
    BASE_DIR, "dataset", "search_results.csv"
)

SCORES_PATH = os.path.join(
    BASE_DIR, "dataset", "similarity_scores.csv"
)

COLLECTION_NAME = "education_documents"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Load embedding model
print("\nLoading Sentence Transformer model...")

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME
)


# Connect to Week 11 ChromaDB
print("\nConnecting to Week 11 ChromaDB...")

if not os.path.exists(
    os.path.join(DB_PATH, "chroma.sqlite3")
):
    raise FileNotFoundError(
        "Week 11 ChromaDB files not found. "
        "Copy the Week 11 database into chroma_db."
    )

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DB_PATH
)


# Load existing document IDs
existing_data = vector_store.get()
existing_ids = set(existing_data["ids"])


# Add 10 new documents
print("\nAdding new Education documents...")

document_files = glob.glob(
    os.path.join(DOCS_PATH, "*.txt")
)

new_documents = []
new_ids = []

for file_path in document_files:

    file_name = os.path.basename(file_path)
    document_id = os.path.splitext(file_name)[0]

    if document_id in existing_ids:
        continue

    with open(
        file_path, "r", encoding="utf-8"
    ) as file:
        content = file.read()

    document = Document(
        page_content=content,
        metadata={
            "title": document_id,
            "category": "Education",
            "source": file_name
        }
    )

    new_documents.append(document)
    new_ids.append(document_id)


if new_documents:
    vector_store.add_documents(
        documents=new_documents,
        ids=new_ids
    )

    print(len(new_documents), "new documents added.")

else:
    print("Documents already exist in ChromaDB.")


# Load all documents for keyword comparison
stored_data = vector_store.get()

all_ids = stored_data["ids"]
all_texts = stored_data["documents"]
all_metadatas = stored_data["metadatas"]


# Read 15 queries
queries_df = pd.read_csv(QUERIES_PATH)

search_results = []
similarity_scores = []


# Perform search for every query
print("\nPerforming semantic search...")

for _, row in queries_df.iterrows():

    query_id = row["query_id"]
    query = str(row["query"])

    print("\nQuery:", query)

    # Convert query into vector embedding
    query_embedding = embeddings.embed_query(query)

    print(
        "Query embedding generated. Vector size:",
        len(query_embedding)
    )

    # Semantic search - Top 5
    semantic_results = (
        vector_store.similarity_search_with_relevance_scores(
            query,
            k=5
        )
    )

    print("\nTop 5 Semantic Search Results:")

    for rank, (doc, score) in enumerate(
        semantic_results, start=1
    ):

        title = doc.metadata.get("title", "Unknown")

        print(
            rank,
            title,
            "Similarity:",
            round(float(score), 4)
        )

        search_results.append({
            "query_id": query_id,
            "query": query,
            "method": "Semantic Search",
            "rank": rank,
            "document": title,
            "content": doc.page_content,
            "similarity_score": round(float(score), 4)
        })

        similarity_scores.append({
            "query_id": query_id,
            "query": query,
            "document": title,
            "similarity_score": round(float(score), 4)
        })

    # Keyword-based search
    query_words = set(query.lower().split())

    keyword_results = []

    for doc_id, content, metadata in zip(
        all_ids, all_texts, all_metadatas
    ):

        document_words = set(content.lower().split())

        keyword_score = len(
            query_words.intersection(document_words)
        )

        keyword_results.append(
            (keyword_score, doc_id, content, metadata)
        )

    keyword_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print("\nTop 5 Keyword Search Results:")

    for rank, item in enumerate(
        keyword_results[:5], start=1
    ):

        keyword_score, doc_id, content, metadata = item

        title = metadata.get("title", doc_id)

        print(rank, title, "Keyword score:", keyword_score)

        search_results.append({
            "query_id": query_id,
            "query": query,
            "method": "Keyword Search",
            "rank": rank,
            "document": title,
            "content": content,
            "similarity_score": ""
        })


# Save search results
results_df = pd.DataFrame(search_results)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# Save similarity scores
scores_df = pd.DataFrame(similarity_scores)

scores_df.to_csv(
    SCORES_PATH,
    index=False
)


print("\n--------------------------------")
print("SEMANTIC SEARCH COMPLETED")
print("--------------------------------")

print("\nSearch results saved:", RESULTS_PATH)

print("Similarity scores saved:", SCORES_PATH)


# Accept manual query from keyboard
while True:

    user_query = input(
        "\nEnter your question (type exit to stop): "
    )

    if user_query.lower().strip() == "exit":
        print("Search application closed.")
        break

    if not user_query.strip():
        print("Please enter a question.")
        continue

    user_results = (
        vector_store.similarity_search_with_relevance_scores(
            user_query,
            k=5
        )
    )

    print("\nTop 5 Relevant Documents:")

    for rank, (doc, score) in enumerate(
        user_results, start=1
    ):

        print("\nRank:", rank)

        print(
            "Document:",
            doc.metadata.get("title", "Unknown")
        )

        print("Similarity Score:", round(float(score), 4))

        print("Content:", doc.page_content)