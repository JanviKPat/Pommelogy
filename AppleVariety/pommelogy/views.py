from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import generics
from .serializers import MyTokenObtainPairSerializer, UserSerializer
from django.contrib.auth.models import User

class MyObtainTokenPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer

class UserDetailView(generics.RetrieveAPIView):
    serializer_class = UserSerializer
    queryset = User.objects.all()
