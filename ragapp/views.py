import shutil
from django.shortcuts import render, redirect
from .models import UploadedPDF
from . import rag_engine



def _parse_answer_points(raw_answer):
    """Split the LLM's raw answer into clean bullet points for display."""
    if not raw_answer:
        return None

    lines = [line.strip(" -•\t") for line in raw_answer.split("\n")]
    points = [line for line in lines if line]

    return points if points else None


def ask_view(request):

    doc = UploadedPDF.objects.order_by("-uploaded_at").first()

    answer_points = None
    question = None
    error = None

    # -------------------------
    # UPLOAD PDF
    # -------------------------
    if request.method == "POST" and "upload" in request.POST:

        file = request.FILES.get("file")

        if not file:
            error = "Please select a PDF file."

        else:
            doc = UploadedPDF.objects.create(
                file=file,
                original_name=file.name
            )

            try:
                rag_engine.build_index_from_pdf(
                    doc.file.path,
                    doc.index_name()
                )

                doc.index_ready = True
                doc.save()

            except Exception as e:
                doc.file.delete(save=False)
                doc.delete()
                doc = None
                error = f"Failed to process PDF: {e}"

    # -------------------------
    # ASK QUESTION
    # -------------------------
    elif request.method == "POST" and "ask" in request.POST:

        if doc is None:
            error = "Please upload a PDF first."

        elif not doc.index_ready:
            error = "PDF is not ready yet."

        else:
            question = request.POST.get("question", "").strip()

            if question:
                raw_answer = rag_engine.answer_question(
                    doc.index_name(),
                    question
                )
                answer_points = _parse_answer_points(raw_answer)

    # -------------------------
    # REMOVE PDF
    # -------------------------
    elif request.method == "POST" and "remove" in request.POST:

        if doc:
            index_name = doc.index_name()

            # 1. Evict from in-memory cache before any deletion.
            rag_engine.invalidate_cache(index_name)

            # 2. Delete the uploaded PDF file from disk.
            doc.file.delete(save=False)

            # 3. Delete the FAISS vectorstore directory from disk.
            from django.conf import settings
            import os
            vectorstore_dir = os.path.join(settings.MEDIA_ROOT, "vectorstores", index_name)
            if os.path.isdir(vectorstore_dir):
                shutil.rmtree(vectorstore_dir)

            # 4. Delete the database record.
            doc.delete()

        return redirect("ask")


    return render(request, "ragapp/ask.html", {
        "doc": doc,
        "question": question,
        "answer_points": answer_points,
        "error": error,
    })