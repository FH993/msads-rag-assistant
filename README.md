# msads-rag-assistant

# UChicago MS-ADS RAG Assistant 🎓

> A retrieval-augmented (RAG) chatbot that answers questions about the University of
> Chicago **MS in Applied Data Science** program, grounding every answer in official
> program web pages and citing its sources.

**🔗 Live demo:** [https://msads-rag-assistant.streamlit.app](#)
**📺 Walkthrough (30s):** [add a Loom or GIF link]

<!-- Add a screenshot or GIF here once deployed: -->
<!-- ![demo](docs/demo.gif) -->

---

## What it does

Ask any question about the MS-ADS program — admissions, curriculum, capstone, tuition,
career outcomes — and get a concise, factual answer drawn from the program's own pages.
Every response includes an expandable **"View retrieved sources"** panel showing the
exact sections and URLs used, so answers are transparent and verifiable rather than
hallucinated.

## How it works

```
14 official MS-ADS web pages
        │  scrape (BeautifulSoup) → clean text
        ▼
RecursiveCharacterTextSplitter  (chunk_size=500, overlap=50)
        │  embed (OpenAI text-embedding-3-small)
        ▼
ChromaDB vector store  (persisted to ./chroma_db)
        │  similarity search (top-k = 4)
        ▼
GPT-4o-mini  +  retrieved context  →  grounded answer + cited sources
```

The assistant is instructed to answer **only** from retrieved context and to fall back
to "please visit the program website" when no relevant context is found — preventing
fabricated program details.

## Tech stack

`Python` · `Streamlit` · `LangChain` · `ChromaDB` · `OpenAI (GPT-4o-mini + text-embedding-3-small)` · `BeautifulSoup`

## Project structure

```
msads-rag-assistant/
├── app.py              # Streamlit chat UI
├── rag_chain.py        # Scraping, chunking, embedding, retrieval + generation
├── chroma_db/          # Persisted vector store (built once)
├── requirements.txt
├── .streamlit/
│   └── config.toml     # Theme
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt

# Provide your OpenAI key (either works):
export OPENAI_API_KEY=sk-...
# or create .streamlit/secrets.toml with:  OPENAI_API_KEY = "sk-..."

streamlit run app.py
```

> **First run:** if `chroma_db/` is empty, build the vector store once by scraping and
> embedding the program pages (run the build step in `rag_chain.py`). After that the
> store is reused on every launch.

## Evaluation

Retrieval and answer quality were spot-checked against a small ground-truth set of
program questions (core courses, admission requirements, capstone) defined in
`build_eval_dataset()`. [Add your results / observations here — e.g. "answers matched
reference content on N/3 questions."]

## Team

[Your names] — University of Chicago, MS in Applied Data Science, [course / quarter].

## License

MIT
