"""
build_index.py — one-time (re)build of the ChromaDB vector store.

Scrapes the MS-ADS program pages, chunks the text, embeds with OpenAI, and
persists the vectors to ./chroma_db so the chatbot has content to retrieve.

Run once locally (and whenever the source pages change):
    python build_index.py
"""

from rag_chain import URLS, scrape_all_pages, chunk_documents, build_vectorstore

if __name__ == "__main__":
    print("1/3  Scraping program pages …")
    raw_data = scrape_all_pages(URLS)

    print("2/3  Chunking documents …")
    chunks = chunk_documents(raw_data)

    print("3/3  Embedding & persisting to ./chroma_db …")
    build_vectorstore(chunks)

    print("\n✅ Index built. You can now run:  streamlit run app.py")
