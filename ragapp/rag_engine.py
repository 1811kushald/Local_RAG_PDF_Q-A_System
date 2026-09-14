import os
from django.conf import settings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ── generation constants ──────────────────────────────────────────────────────
_MAX_NEW_TOKENS    = 220   # per-round token cap; continuation loop handles overflow
_MAX_CONTINUATIONS = 2     # maximum extra rounds after the first answer

# ── singletons ────────────────────────────────────────────────────────────────
_embeddings  = None
_llm         = None
_tokenizer   = None        # stored at startup; used by _looks_truncated() and _continue_answer()
_index_cache: dict = {}    # { index_name: FAISS } — avoids disk reload on every question


def load_models():
    """Called once when the Django server starts. See apps.py."""
    global _embeddings, _llm, _tokenizer
    if _embeddings is not None:
        return

    print("Loading embedding model...")
    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Loading local LLM...")
    model_id  = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    _tokenizer = tokenizer   # expose globally so helpers can re-tokenise for truncation detection

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=_MAX_NEW_TOKENS,   # single source of truth
        do_sample=True,
        temperature=0.3,                  # lower = more factual, less rambling in RAG
        repetition_penalty=1.15,          # discourages repeated phrases / duplicate bullets
        return_full_text=False,
    )
    _llm = HuggingFacePipeline(pipeline=pipe)
    print("Models ready.")


def _looks_truncated(text: str, cap: int = _MAX_NEW_TOKENS, margin: int = 5) -> bool:
    """Return True if *text* appears to have been cut off at the token cap.

    Two independent signals are checked — either is enough to return True:
      (a) Re-tokenising the output gives a count within *margin* of *cap*:
          strong sign the pipeline stopped because it ran out of budget.
      (b) The last non-whitespace character is not terminal punctuation:
          sign the model was mid-thought when generation halted.
    """
    if not text or not text.strip():
        return False
    tokens     = _tokenizer.encode(text, add_special_tokens=False)
    near_cap   = len(tokens) >= cap - margin
    last_char  = text.rstrip()[-1]
    open_ended = last_char not in {'.', '!', '?', ':', '"', "'"}
    return near_cap or open_ended


# System instruction used exclusively for continuation calls.
_CONTINUATION_SYSTEM = (
    "You are a document assistant. Use ONLY the provided context. "
    "Continue the partial bullet list below in the same '- ' format. "
    "One point per line. Do NOT repeat any bullet already written. "
    "Stop when the answer is complete. No commentary or disclaimers."
)


def _continue_answer(question: str, context_text: str, partial_answer: str) -> str:
    """Ask the LLM to continue a truncated bullet-list answer.

    Bypasses the retrieval chain (context is already known). Uses
    apply_chat_template so Qwen's <|im_start|> chat tokens are correctly
    inserted before the string is passed to _llm.invoke().

    Returns the continuation text stripped of leading/trailing whitespace,
    or an empty string if the model produced nothing useful.
    """
    messages = [
        {"role": "system", "content": _CONTINUATION_SYSTEM},
        {"role": "user", "content": (
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Partial answer already given (do not repeat any of this):\n{partial_answer}\n\n"
            "Continue ONLY the remaining bullet points:"
        )},
    ]
    formatted = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = _llm.invoke(formatted)
    return result.strip() if isinstance(result, str) else ""


def build_index_from_pdf(pdf_path, index_name):
    """Run once per uploaded PDF. Slow (~seconds). Saves index to disk."""
    loader    = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(documents)

    vector_store = FAISS.from_documents(chunks, _embeddings)

    index_dir = os.path.join(settings.MEDIA_ROOT, "vectorstores", index_name)
    vector_store.save_local(index_dir)
    _index_cache[index_name] = vector_store   # warm cache — first question hits memory, not disk


def invalidate_cache(index_name):
    """Remove a vector store from the in-memory cache.
    Call this before deleting a PDF so stale FAISS objects don't linger in memory.
    Safe to call even when the key is absent (e.g. after a server restart).
    """
    _index_cache.pop(index_name, None)


def answer_question(index_name, question, k=3):
    """Run on every question. Fast — hits the in-memory cache; only falls back to disk on a cache miss.

    After the main chain call, a detect-truncation-then-continue loop fires only
    when the answer looks cut off — short answers pay zero extra cost.
    """
    if index_name not in _index_cache:
        # Cache miss: server was restarted or cache was invalidated — reload from disk once.
        index_dir = os.path.join(settings.MEDIA_ROOT, "vectorstores", index_name)
        _index_cache[index_name] = FAISS.load_local(
            index_dir, _embeddings, allow_dangerous_deserialization=True
        )
    vector_store = _index_cache[index_name]

    # MMR retriever: fetches 10 candidates and returns the k most diverse ones,
    # preventing near-duplicate chunks from flooding the context.
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 10, "lambda_mult": 0.7},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a document assistant. Answer using ONLY the context below. "
         "Always format your answer as a bullet list using '- ' at the start "
         "of each line. Maximum 6 bullet points. Each bullet is one concise "
         "sentence. No paragraphs, no repeating the same point twice. "
         "Do not add commentary, disclaimers, or phrases like 'let me know'. "
         "If nothing relevant is found, reply exactly: "
         "'Not found in the document.'\n\n"
         "Context:\n{context}"),
        ("human", "{input}"),
    ])

    doc_chain = create_stuff_documents_chain(_llm, prompt)
    qa_chain  = create_retrieval_chain(retriever, doc_chain)

    response     = qa_chain.invoke({"input": question})
    answer       = response["answer"]
    context_text = "\n\n".join(d.page_content for d in response["context"])

    # ── detect-truncation-then-continue loop ─────────────────────────────────
    # Fast path: _looks_truncated() returns False → loop never entered, zero overhead.
    # Slow path: fires only for answers that genuinely hit the token cap.
    rounds = 0
    while _looks_truncated(answer) and rounds < _MAX_CONTINUATIONS:
        continuation = _continue_answer(question, context_text, answer)
        if not continuation:
            break
        answer  = answer.rstrip() + "\n" + continuation
        rounds += 1

    return answer