from rest_framework import serializers
from django.contrib.auth.models import User
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import AppleVariety, ImageIdentify

class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['username'] = user.username
        return token

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email')


class AppleVarietySerializer(serializers.ModelSerializer):
    class Meta:
        model = AppleVariety
        fields = '__all__'


class ImageUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageIdentify
        fields = '__all__'