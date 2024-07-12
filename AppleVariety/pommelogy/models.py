from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError

class CustomUser(AbstractUser):
    # Add custom fields if necessary
    pass
class MyModel(models.Model):
    name = models.CharField(max_length=100)

    class Meta:
        permissions = [
            ('can_view', 'Can view my model'),
            ('can_edit', 'Can edit my model'),
        ]

class AppleVariety(models.Model):
    name = models.CharField(max_length=100)
    # variety_id = models.CharField(max_length=100, unique=True, editable=False)
    variety_id = models.CharField(max_length=100, unique=True)
    image = models.ImageField()
    description = models.TextField()
    visual_characteristics = models.TextField()
    taste_profile = models.TextField()
    culinary_uses = models.TextField()

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if self.pk is not None:
            original = AppleVariety.objects.get(pk=self.pk)
            if original.variety_id != self.variety_id:
                raise ValidationError("You cannot change the Variety ID of an existing Apple.")
        super().save(*args, **kwargs)

class ImageIdentify(models.Model):

    # user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    image = models.ImageField()
    identify_date = models.DateTimeField(auto_now_add=True)
    identified_variety = models.CharField(max_length=10)

    def __str__(self):
        return f'{self.identify_date}'