<h1>RAG BASED TRAVEL AI</h1>
This project implements a local, retrieval‑augmented travel recommendation system for Indian destinations using a Flask web application, FAISS vector search, and LangChain’s Ollama integration. The application loads a curated travel.csv dataset of places across India, generates dense embeddings for the description field via OllamaEmbeddings with the nomic-embed-text model, and indexes them in a FAISS IndexFlatL2 structure for efficient similarity search at query time. User queries are submitted through a chat-style web interface, embedded into the same vector space, and used to retrieve the most relevant rows from the CSV, which are then compiled into a contextual prompt for ChatOllama with the phi3 model to produce personalized, context-aware travel recommendations. The entire system runs locally with Ollama, requires only a single app.py file plus the CSV data source, and is designed as a lightweight, extensible template for building domain-specific RAG applications on top of structured datasets.

<h1>Tech-Stack</h1>

1.Backend framework: Flask (Python web framework for HTTP routing and serving the web UI).

2.Programming language: Python for all server, RAG, and data processing logic.

3.Data layer: pandas to load and manage the travel.csv dataset (pd.read_csv).

4.Vector database: FAISS IndexFlatL2 for efficient similarity search over embeddings.

5.Numerical computing: NumPy for array operations and converting embeddings to float32.

6.LLM + embeddings integration: LangChain langchain_ollama with OllamaEmbeddings and ChatOllama.

7.Model runtime: Ollama for running nomic-embed-text (embeddings) and phi3 (chat LLM) locally.

8.Frontend stack: Single-page HTML + CSS + vanilla JavaScript (with fetch to call the /chat API).

