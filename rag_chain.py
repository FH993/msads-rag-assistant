# =============================================================================
# RAG Pipeline — MS in Applied Data Science @ UChicago
# =============================================================================

import os
import json
import requests
from bs4 import BeautifulSoup
from datasets import Dataset
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ.setdefault("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])
except Exception:
    pass

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Provide it via an environment variable "
        "or .streamlit/secrets.toml — do not hardcode it in the source."
    )
# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "msads_rag"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50

URLS = {
    "General Info":      "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/",
    "In-Person program": "https://datascience.uchicago.edu/education/masters-programs/in-person-program/",
    "Online Program":    "https://datascience.uchicago.edu/education/masters-programs/online-program/",
    "Capstone":          "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/capstone-projects/",
    "Curriculum":        "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/course-progressions/",
    "Admission":         "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/how-to-apply/",
    "Upcoming Events":   "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/events-deadlines/",
    "Finance":           "https://datascience.uchicago.edu/education/tuition-fees-aid/",
    "Student Diversity": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/our-students/",
    "International":     "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/international-students/",
    "Faculty":           "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/instructors-staff/",
    "FAQs":              "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/faqs/",
    "Explore campus":    "https://datascience.uchicago.edu/explore-the-ms-ads-campus/",
    "Career outcome":    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/career-outcomes/"
}

# =============================================================================
#  Web Scraping
# =============================================================================

def scrape_all_pages(urls: dict, save_path: str = "data.json") -> list[dict]:
    all_data = []

    for section, url in urls.items():
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()

        main = soup.find("main")
        if main:
            soup = main

        elements = soup.find_all(["h1", "h2", "h3", "p", "li"])

        seen = set()
        clean_texts = []
        for el in elements:
            t = el.get_text(" ", strip=True)
            if len(t) > 40 and t not in seen:
                seen.add(t)
                clean_texts.append(t)

        all_data.append({
            "section": section,
            "url":     url,
            "content": "\n\n".join(clean_texts)
        })

    with open(save_path, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"✅ Scraped {len(all_data)} sections → saved to {save_path}")
    return all_data


# =============================================================================
# Chunking with Langchain
# =============================================================================

def chunk_documents(raw_data: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []
    for item in raw_data:
        if not item["content"].strip():
            print(f"⚠️  Skipping empty section: {item['section']}")
            continue

        doc = Document(
            page_content=item["content"],
            metadata={"section": item["section"], "url": item["url"]}
        )
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
        print(f"✅  {item['section']}: {len(chunks)} chunks")

    print(f"\n🎉 Total chunks: {len(all_chunks)}")
    return all_chunks


# =============================================================================
#  Embedding & Vector Store
# =============================================================================

def build_vectorstore(chunks: list[Document]) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH
    )

    print(f"🎉 Stored {vectorstore._collection.count()} chunks in ChromaDB at '{CHROMA_PATH}'")
    return vectorstore


def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vectorstore


# =============================================================================
# RAG: Retrieval + Generation
# =============================================================================

# Load shared components (used by both get_answer() and evaluation)
vectorstore = load_vectorstore()
retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})
llm         = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_answer(user_input: str) -> tuple[str, list]:
    """
    Main RAG function — used by app.py and evaluation.
    Returns (answer string, list of retrieved Document objects).
    """
    retrieved_docs = retriever.invoke(user_input)

    context_text = "\n\n".join([
        f"[{doc.metadata.get('section', 'Unknown')}]\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    prompt = f"""
You are a helpful assistant for the MS in Applied Data Science program
at the University of Chicago.

Use the provided context to answer the question as best as you can.
If the context contains relevant information, summarize and use it to form your answer.
Only if the context has absolutely no relevant information at all, say:
"I don't have that information, please visit the program website."
Keep the answer concise and factual.

Context:
{context_text}

Question:
{user_input}

Answer:
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content, retrieved_docs


# =============================================================================
#  Evaluation Dataset
# =============================================================================

def build_eval_dataset() -> Dataset:
    questions = [
        "What are the core courses in the MS in Applied Data Science program?",
        "What are the admission requirements for the MS in Applied Data Science program?",
        "Can you provide information about the capstone project?"
    ]

    reference_answers = [
        "The core courses in the MS in Applied Data Science program include Machine Learning, Data Engineering Platforms, Statistical Inference, and Applied Data Science.",
        "Applicants need a bachelor's degree in a related field, with coursework in programming, statistics, and mathematics. The application also requires a personal statement, letters of recommendation, and a resume.",
        "The capstone project is a key component of the MS in Applied Data Science program, where students work on real-world problems, applying their learned skills to develop data-driven solutions."
    ]

    answers  = []
    contexts = []

    for q in questions:
        answer, retrieved_docs = get_answer(q)
        answers.append(answer)
        contexts.append([doc.page_content for doc in retrieved_docs])

    dataset = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": reference_answers
    })

    return dataset, questions, answers, contexts


# =============================================================================
# MAIN — Run evaluation and print results
# =============================================================================

if __name__ == "__main__":
    dataset, questions, answers, contexts = build_eval_dataset()

    for i in range(len(questions)):
        print(f"\n{'='*80}")
        print(f"Question {i+1}: {questions[i]}")
        print(f"\nGenerated Answer:\n{answers[i]}")
        print(f"\nTop Retrieved Contexts:")
        for j, ctx in enumerate(contexts[i][:2]):
            print(f"\n  Context {j+1}:\n  {ctx[:300]}")
