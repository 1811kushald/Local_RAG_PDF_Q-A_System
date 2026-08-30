import os
from django.apps import AppConfig


class RagappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ragapp"

    def ready(self):
        # Avoid double-loading under the dev server's autoreloader,
        # and avoid loading during `manage.py migrate` / `makemigrations`.
        import sys
        if "runserver" not in sys.argv:
            return
        if os.environ.get("RUN_MAIN") != "true":
            return

        from . import rag_engine
        rag_engine.load_models()