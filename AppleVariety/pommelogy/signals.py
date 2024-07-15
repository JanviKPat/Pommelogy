from django.db.models.signals import post_migrate
from django.dispatch import receiver
from .models import AppleVariety

@receiver(post_migrate)
def create_initial_apple_varieties(sender, **kwargs):
    if not AppleVariety.objects.exists():
        AppleVariety.objects.create(name='Not an Apple', description='Image uploaded is not of an Apple', variety_id="NA")
