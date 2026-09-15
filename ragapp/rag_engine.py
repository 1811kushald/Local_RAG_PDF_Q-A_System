import os
from django.conf import settings
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ── Groq Model Configuration ──────────────────────────────────────────────────
def _get_model():
    return getattr(settings, "GROQ_MODEL", "") or os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")

_SYSTEM_PROMPT = (
    "You are a document assistant. Answer using ONLY the context below. "
    "Always format your answer as a bullet list using '- ' at the start "
    "of each line. Maximum 6 bullet points. Each bullet is one concise "
    "sentence. No paragraphs, no repeating the same point twice. "
    "Do not add commentary, disclaimers, or phrases like 'let me know'. "
    "If nothing relevant is found, reply exactly: "
    "'Not found in the document.'"
)

# ── Singletons ────────────────────────────────────────────────────────────────
_embeddings = None
_groq_client = None
_index_cache: dict = {}    # { index_name: FAISS }


def _get_groq_client():
    """Retrieve or initialize the Groq client instance."""
    global _groq_client
    if _groq_client is None:
        api_key = getattr(settings, "GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. Please set your key in the .env file or environment."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _get_embeddings():
    """Retrieve or lazily initialize the embedding model."""
    global _embeddings
    if _embeddings is None:
        load_models()
    return _embeddings


def load_models():
    """Called once when the Django server starts (see apps.py).

    Loads only the lightweight embedding model into memory (~200MB RAM).
    Text generation is handled externally via Groq API.
    """
    global _embeddings
    if _embeddings is not None:
        return

    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding model ready. Text generation powered by Groq API.")


def build_index_from_pdf(pdf_path, index_name):
    """Run once per uploaded PDF. Saves index to disk and warms in-memory cache."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    vector_store = FAISS.from_documents(chunks, _get_embeddings())

    index_dir = os.path.join(settings.MEDIA_ROOT, "vectorstores", index_name)
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)
    _index_cache[index_name] = vector_store


def invalidate_cache(index_name):
    """Remove a vector store from the in-memory cache."""
    _index_cache.pop(index_name, None)


def answer_question(index_name, question, k=5):
    """Non-streaming query: retrieves context via MMR and calls Groq API."""
    if index_name not in _index_cache:
        index_dir = os.path.join(settings.MEDIA_ROOT, "vectorstores", index_name)
        _index_cache[index_name] = FAISS.load_local(
            index_dir, _get_embeddings(), allow_dangerous_deserialization=True
        )
    vector_store = _index_cache[index_name]

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.7},
    )
    context_docs = retriever.invoke(question)
    context_text = "\n\n".join(d.page_content for d in context_docs)

    client = _get_groq_client()
    completion = client.chat.completions.create(
        model=_get_model(),
        messages=[
            {
                "role": "system",
                "content": f"{_SYSTEM_PROMPT}\n\nContext:\n{context_text}",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content or ""


def stream_answer(index_name: str, question: str, k: int = 5):
    """Generator that yields raw text token-fragments streamed live from Groq API.

    Compatible with Django's StreamingHttpResponse + Server-Sent Events.
    """
    if index_name not in _index_cache:
        index_dir = os.path.join(settings.MEDIA_ROOT, "vectorstores", index_name)
        _index_cache[index_name] = FAISS.load_local(
            index_dir, _get_embeddings(), allow_dangerous_deserialization=True
        )
    vector_store = _index_cache[index_name]

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.7},
    )
    context_docs = retriever.invoke(question)
    context_text = "\n\n".join(d.page_content for d in context_docs)

    client = _get_groq_client()
    stream = client.chat.completions.create(
        model=_get_model(),
        messages=[
            {
                "role": "system",
                "content": f"{_SYSTEM_PROMPT}\n\nContext:\n{context_text}",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta