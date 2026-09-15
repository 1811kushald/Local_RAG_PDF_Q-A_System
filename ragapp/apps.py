import os
import sys
from django.apps import AppConfig


class RagappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ragapp"

    def ready(self):
        # Skip model loading during administrative commands
        if any(cmd in sys.argv for cmd in ["migrate", "makemigrations", "collectstatic", "check"]):
            return

        # Under development runserver, only load in the main child process
        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return

        from . import rag_engine
        rag_engine.load_models()