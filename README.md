# AI Study Assistant — PDF Question Answering System

A lightweight Python application that helps students explore lecture notes in PDF format. Upload a PDF, ask a question, retrieve relevant passages, and view the page references that support the result.

## Features

- Upload and process text-based PDF documents.
- Extract page-by-page text with **PyMuPDF**.
- Split extracted text into overlapping chunks.
- Retrieve relevant passages using **TF-IDF** and cosine similarity.
- Rank source sentences against a question.
- Request a basic extractive overview of the document.
- Display retrieved passages and PDF page references.
- Run locally without an API key or paid AI service.

## Technology

- Python
- Streamlit
- PyMuPDF
- scikit-learn
- TF-IDF vectorization
- Cosine similarity

## How it works

1. PyMuPDF extracts selectable text from each PDF page.
2. The text is divided into overlapping chunks.
3. TF-IDF converts chunks and questions into numerical vectors.
4. Cosine similarity ranks chunks by lexical similarity to the question.
5. Relevant sentences are selected from the retrieved passages.
6. The app displays the extracted sentences with their page references.

## Run locally

### 1. Clone the repository

```powershell
git clone https://github.com/icarus9111/ai-study-assistant.git
cd ai-study-assistant
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Start the application

```powershell
python -m streamlit run app.py
```

Streamlit will provide a local address to open in your browser.

## Project structure

```text
ai-study-assistant/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Important limitations

This version is a **retrieval and extractive NLP application**, not a generative chatbot or large language model.

- It selects and organizes sentences already present in the PDF; it does not independently generate explanations.
- TF-IDF relies on word overlap. It may miss relevant passages when a question uses very different wording from the document.
- Scanned PDFs or image-only pages may contain no selectable text. OCR is not included in this version.
- PDF extraction can introduce broken sentences, unusual spacing, or other formatting artifacts.
- The document overview is extractive and may not read like a polished human-written summary.
- Page references correspond to the PDF's page numbering, starting at page 1.

## Privacy

PDFs are processed locally by this application. The project does not send document contents to an external AI API. Do not commit private lecture files, personal documents, credentials, or API keys to GitHub.

## Future improvements

- OCR support for scanned PDFs.
- Better semantic retrieval with embeddings.
- More robust sentence and paragraph handling.
- Optional LLM-based answers with clear source citations.
- Automated evaluation using a set of test questions.

## Project status

**Version 1.0 — lightweight local PDF retrieval and extractive question answering.**
