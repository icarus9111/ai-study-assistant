import re
from collections import defaultdict

import fitz
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Study Assistant")
st.caption("Ask questions about your lecture PDFs and find the most relevant passages.")

def extract_pdf(uploaded_file):
    """Extract selectable text from each PDF page."""
    document = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
    pages = []
    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_number, "text": text})
    document.close()
    return pages


def create_chunks(pages, size=1000, overlap=150):
    chunks = []
    for page in pages:
        text = re.sub(r"\s+", " ", page["text"]).strip()
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"page": page["page"], "text": chunk})
            if end == len(text):
                break
            start = end - overlap
    return chunks


def split_sentences(pages):
    sentences = []
    for page in pages:
        parts = re.split(r"(?<=[.!?])\s+", page["text"])
        for part in parts:
            part = re.sub(r"\s+", " ", part).strip()
            if len(part.split()) >= 5:
                sentences.append({"page": page["page"], "text": part})
    return sentences


def retrieve(question, chunks, vectorizer, matrix, top_k=8):
    page_match = re.search(r"\bpage\s+(\d+)\b", question.lower())
    if page_match:
        page_number = int(page_match.group(1))
        return [item for item in chunks if item["page"] == page_number][:top_k]

    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).flatten()
    ranked_indices = scores.argsort()[::-1]
    results = []
    for index in ranked_indices:
        if scores[index] <= 0:
            continue
        result = dict(chunks[index])
        result["score"] = float(scores[index])
        results.append(result)
        if len(results) == top_k:
            break
    return results



def rank_sentences(question, results, sentence_vectorizer, limit=6):
    candidates = []
    seen = set()

    for source in results:
        for sentence in re.split(r"(?<=[.!?])\s+", source["text"]):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            key = sentence.lower()

            if len(sentence.split()) >= 5 and key not in seen:
                seen.add(key)
                candidates.append({
                    "page": source["page"],
                    "text": sentence
                })

    if not candidates:
        return []

    # Fit the vectorizer before transforming the question
    sentence_matrix = sentence_vectorizer.fit_transform(
        [item["text"] for item in candidates]
    )

    query_vector = sentence_vectorizer.transform([question])

    scores = cosine_similarity(
        query_vector,
        sentence_matrix
    ).flatten()

    ranked = []

    for index in scores.argsort()[::-1]:
        if scores[index] <= 0:
            continue

        item = dict(candidates[index])
        item["score"] = float(scores[index])
        ranked.append(item)

        if len(ranked) == limit:
            break

    return ranked


def answer_question(question, results, sentence_vectorizer):
    if not results:
        return "I couldn't find relevant text in the uploaded document."

    page_match = re.search(
        r"\b(?:summarize|summary|explain)\s+page\s+(\d+)\b",
        question.lower(),
    )
    if page_match:
        sentences = []
        for source in results:
            for part in re.split(r"(?<=[.!?])\s+", source["text"]):
                part = re.sub(r"\s+", " ", part).strip()
                if len(part.split()) >= 5 and part not in sentences:
                    sentences.append(part)
        if not sentences:
            return "No readable content was found on that page."
        return "\n\n".join(f"- {sentence}" for sentence in sentences[:12])

    ranked = rank_sentences(question, results, sentence_vectorizer)
    if not ranked:
        return (
            "I couldn't find a sufficiently relevant answer in the retrieved "
            "passages. Try using keywords from your notes."
        )
    return "\n\n".join(
        f"- {item['text']} [Page {item['page']}]"
        for item in ranked
    )


def summarize_document(sentences, vectorizer):
    if not sentences:
        return "No readable text was extracted from this PDF."

    texts = [item["text"] for item in sentences]
    matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(matrix)
    centrality = similarity.sum(axis=1)

    ranked_indices = centrality.argsort()[::-1]
    selected = []
    selected_pages = set()

    # Prefer coverage across different pages before filling remaining places.
    for index in ranked_indices:
        page = sentences[index]["page"]
        if page not in selected_pages:
            selected.append(index)
            selected_pages.add(page)
        if len(selected) >= 15:
            break

    if len(selected) < 15:
        for index in ranked_indices:
            if index not in selected:
                selected.append(index)
            if len(selected) >= 15:
                break

    selected.sort(key=lambda i: (sentences[i]["page"], i))
    return "\n\n".join(
        f"- {sentences[i]['text']} [Page {sentences[i]['page']}]"
        for i in selected
    )


if "pages" not in st.session_state:
    st.session_state.pages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "history" not in st.session_state:
    st.session_state.history = []
if "document_name" not in st.session_state:
    st.session_state.document_name = ""


with st.sidebar:
    st.header("Your document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.button("Process PDF", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Extracting and indexing your PDF..."):
                try:
                    pages = extract_pdf(uploaded_file)
                    chunks = create_chunks(pages)
                    if not chunks:
                        st.session_state.pages = []
                        st.session_state.chunks = []
                        st.error(
                            "No selectable text was found. This may be a scanned "
                            "PDF and require OCR."
                        )
                    else:
                        st.session_state.pages = pages
                        st.session_state.chunks = chunks
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.history = []
                        st.success(
                            f"Processed {len(pages)} readable pages "
                            f"into {len(chunks)} text chunks."
                        )
                except Exception as error:
                    st.error(f"Could not process this PDF: {error}")

    if st.session_state.pages:
        st.caption(f"Loaded: {st.session_state.document_name}")
        if st.button("Clear document and chat", use_container_width=True):
            st.session_state.pages = []
            st.session_state.chunks = []
            st.session_state.history = []
            st.session_state.document_name = ""
            st.rerun()


if not st.session_state.pages:
    st.info("Upload a text-based PDF from the sidebar, then select **Process PDF**.")
    st.markdown(
        """
        **What this app does**
        - Extracts text from PDF pages.
        - Finds passages using TF-IDF keyword matching.
        - Shows relevant source sentences with page references.
        - Creates a basic extractive overview of the document.
        """
    )
else:
    pages = st.session_state.pages
    chunks = st.session_state.chunks

    st.success(
        f"Document ready: **{st.session_state.document_name}** "
        f"({len(pages)} readable pages)"
    )

    if st.button("Generate document summary"):
        summary_vectorizer = TfidfVectorizer(stop_words="english")
        summary = summarize_document(
            split_sentences(pages),
            summary_vectorizer,
        )
        st.subheader("Document overview")
        st.markdown(summary)

    st.divider()
    st.subheader("Ask your document")

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What is ICT? or Summarize page 4",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        corpus = [chunk["text"] for chunk in chunks]
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus)
        sentence_vectorizer = TfidfVectorizer(stop_words="english")

        results = retrieve(question.strip(), chunks, vectorizer, matrix)
        answer = answer_question(
            question.strip(),
            results,
            sentence_vectorizer,
        )
        st.session_state.history.append(
            {
                "question": question.strip(),
                "answer": answer,
                "sources": results,
            }
        )

    for item in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            with st.expander("Retrieved source passages"):
                if item["sources"]:
                    for source in item["sources"]:
                        st.markdown(
                            f"**Page {source['page']}**\n\n{source['text']}"
                        )
                else:
                    st.write("No source passages were retrieved.")

st.divider()
st.caption(
    "AI Study Assistant v1.0 | Python · Streamlit · PyMuPDF · TF-IDF"
)
