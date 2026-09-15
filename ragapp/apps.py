from django.apps import AppConfig


class RagappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ragapp"

    def ready(self):
        # Embeddings are loaded lazily on first document upload or query via rag_engine._get_embeddings().
        # This keeps boot time < 0.2s and prevents memory spikes during Gunicorn startup on Render.
        pass