# Week 12 - Semantic Document Search

## Task Title
Building a Semantic Document Search System using LangChain and ChromaDB.

## Domain
Education

## Objective
- Understand semantic search using vector embeddings.
- Retrieve relevant documents based on semantic similarity.
- Integrate LangChain with ChromaDB.
- Build a semantic search application for RAG.
- Prepare documents for Week 13 question answering.

## Technologies Used
- Python
- LangChain
- ChromaDB
- Sentence Transformers
- Hugging Face
- Pandas
- Jupyter Notebook

## Embedding Model
all-MiniLM-L6-v2

## Dataset
The project contains 10 new Education documents and 15 user queries.

## Installation

Install the required libraries:

pip install -r requirements.txt

## Database Setup

Copy the Week 11 ChromaDB database files into the Week 12 chroma_db folder.

## Execution

Run:

python semantic_search.py

## Search Methods
1. Semantic Search
2. Keyword-Based Search

## Output Files
- search_results.csv
- similarity_scores.csv

## Conclusion
Semantic search retrieves relevant educational documents by understanding the meaning of user queries. It improves information retrieval compared to traditional keyword matching and prepares the retrieval component for a RAG application.

## Observations

1. Semantic search retrieves documents based on the meaning of queries.
2. Keyword search retrieves documents based on matching words.
3. Semantic search can identify relevant documents even when different words are used.
4. Top-5 results help users find useful educational information.
5. Similarity scores indicate how relevant the retrieved documents are to the query.

## Final Result

The semantic document search system was successfully implemented using LangChain and ChromaDB. The system retrieves the Top-5 relevant Education documents, compares semantic search with keyword search, and saves the results in CSV files.