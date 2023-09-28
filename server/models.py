from django.db import models
import uuid

class MainObject(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

class Interaction(models.Model):
    main_object = models.ForeignKey(MainObject, related_name='interactions', on_delete=models.CASCADE)
    request = models.TextField()
    response = models.TextField()