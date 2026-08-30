# Local_RAG_PDF_Q&A_System 

## Setup
1. `pip install -r requirements.txt`
2. `python manage.py migrate`
3. `python manage.py runserver`

Note: on first run, the app downloads two Hugging Face models
(~1GB total: Qwen2.5-0.5B-Instruct + all-MiniLM-L6-v2 embeddings)
into your local Hugging Face cache. This can take several minutes
depending on connection speed, but only happens once.

