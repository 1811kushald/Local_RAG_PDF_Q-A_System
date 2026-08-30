# Local RAG PDF Q&A System

A simple web application that lets you upload a PDF and ask natural-language questions about it — powered by a locally-hosted Retrieval-Augmented Generation (RAG) pipeline. No paid APIs, no external LLM calls — everything runs on your own machine.

## Overview

Upload any PDF through the web interface, and the app will read it, break it into chunks, and index it for search. You can then ask questions in plain English, and the app retrieves the most relevant parts of the document and generates an answer using a locally-hosted language model.

## Features

- Upload a PDF directly from the browser
- Ask unlimited questions about the uploaded document
- Fully local pipeline — no external API keys or paid services
- Answers grounded only in the document's actual content
- Simple, minimal web interface built with Django

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | Django |
| PDF parsing | PyPDF / LangChain document loaders |
| Text chunking | LangChain text splitters |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS |
| Language model | Local Hugging Face model (`Qwen2.5-0.5B-Instruct`) |
| Orchestration | LangChain |

## How It Works

```
 Upload PDF
     |
     v
 Parse & split into text chunks
     |
     v
 Convert chunks into embeddings
     |
     v
 Store embeddings in a FAISS vector index
     |
     v
 Ask a question
     |
     v
 Retrieve the most relevant chunks
     |
     v
 Local LLM generates an answer using those chunks
     |
     v
 Answer shown on the web page
```

The heavy setup work (loading the PDF, chunking, embedding, indexing) happens once when a PDF is uploaded. Answering questions afterward is fast, since it only needs to search the existing index and generate a response.

## Screenshots

*(Add screenshots of the interface here)*

**Upload Page**

<!-- ![Upload Page](screenshots/upload.png) -->

**Q&A Page**

<!-- ![Q&A Page](screenshots/qa.png) -->

## Setup & Installation

1. Clone the repository
   ```bash
   git clone https://github.com/1811kushald/Local_RAG_PDF_Q-A_System.git
   cd Local_RAG_PDF_Q-A_System
   ```

2. Create a virtual environment and install dependencies
   ```bash
   python -m venv venv
   venv\Scripts\activate       # Windows
   pip install -r requirements.txt
   ```

3. Run database migrations
   ```bash
   python manage.py migrate
   ```

4. Start the server
   ```bash
   python manage.py runserver
   ```

5. Open the app in your browser
   ```
   http://localhost:8000/
   ```

> **Note:** On first run, the app downloads the required Hugging Face models (~1 GB total) to your local cache. This only happens once — later runs load instantly from the local cache.

## Usage

1. Upload a PDF from the home page.
2. Wait for the document to finish indexing.
3. Type a question in the input box and get an answer generated from the document's content.

## Project Structure

```
ragsite/
├── manage.py
├── ragsite/          # Django project settings
├── ragapp/
│   ├── models.py     # PDF document model
│   ├── views.py       # Upload / ask / remove logic
│   ├── rag_engine.py  # Core RAG pipeline (embeddings, FAISS, LLM)
│   └── templates/
├── requirements.txt
└── .gitignore
```

## Future Improvements

- Support multiple PDFs per user with authentication
- Faster inference using quantized models (GGUF / llama.cpp)
- Background indexing with Celery for large PDFs
- Support for other document formats (Word, TXT)



## Setup
1. `pip install -r requirements.txt`
2. `python manage.py migrate`
3. `python manage.py runserver`

Note: on first run, the app downloads two Hugging Face models
(~1GB total: Qwen2.5-0.5B-Instruct + all-MiniLM-L6-v2 embeddings)
into your local Hugging Face cache. This can take several minutes
depending on connection speed, but only happens once.

