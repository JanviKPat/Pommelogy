import json
from django.core.management.base import BaseCommand
from your_app.models import AppleVariety

class Command(BaseCommand):
    help = 'Load apple varieties from JSON file into the database'

    def handle(self, *args, **kwargs):
        with open('AppleVariety/management/commands/apple_variety_db.json', 'r') as file:
            apple_varieties = json.load(file)

            for variety in apple_varieties:
                AppleVariety.objects.update_or_create(
                    variety_id=variety['variety_id'],
                    defaults={
                        'name': variety['name'],
                        'description': variety['description'],
                        'visual_characteristics': variety['visual_characteristics'],
                        'taste_profile': variety['taste_profile'],
                        'culinary_uses': variety['culinary_uses'],
                    }
                )
            self.stdout.write(self.style.SUCCESS('Successfully loaded apple varieties into the database'))
