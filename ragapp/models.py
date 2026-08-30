import uuid
from django.db import models

class UploadedPDF(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file = models.FileField(upload_to="pdfs/")
    original_name = models.CharField(max_length=255, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    index_ready = models.BooleanField(default=False)

    def index_name(self):
        return str(self.id)