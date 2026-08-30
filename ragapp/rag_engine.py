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

_embeddings = None
_llm = None


def load_models():
    """Called once when the Django server starts. See apps.py."""
    global _embeddings, _llm
    if _embeddings is not None:
        return

    print("Loading embedding model...")
    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Loading local LLM...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        return_full_text=False,
    )
    _llm = HuggingFacePipeline(pipeline=pipe)
    print("Models ready.")


def build_index_from_pdf(pdf_path, index_name):
    """Run once per uploaded PDF. Slow (~seconds). Saves index to disk."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    vector_store = FAISS.from_documents(chunks, _embeddings)

    index_dir = os.path.join(settings.MEDIA_ROOT, "vectorstores", index_name)
    vector_store.save_local(index_dir)


def answer_question(index_name, question, k=3):
    """Run on every question. Fast — just loads a saved index + generates."""
    index_dir = os.path.join(settings.MEDIA_ROOT, "vectorstores", index_name)
    vector_store = FAISS.load_local(
        index_dir, _embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document assistant. Answer using ONLY the context below. "
     "Always format your answer as a bullet list using '- ' at the start "
     "of each line — one clear point per line, no paragraphs, no repeating "
     "the same point twice. Do not add commentary, disclaimers, or phrases "
     "like 'let me know'. If nothing relevant is found, reply exactly: "
     "'Not found in the document.'\n\n"
     "Context:\n{context}"),
    ("human", "{input}"),
    ])

    doc_chain = create_stuff_documents_chain(_llm, prompt)
    qa_chain = create_retrieval_chain(retriever, doc_chain)

    response = qa_chain.invoke({"input": question})
    return response["answer"]