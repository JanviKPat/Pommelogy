# import tensorflow as tf
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import generics, viewsets
from .serializers import MyTokenObtainPairSerializer, UserSerializer, AppleVarietySerializer, ImageUploadSerializer
from django.contrib.auth.models import User
from .models import AppleVariety, CustomUser, ImageIdentify
from rest_framework.decorators import action
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

from tensorflow.keras.models import load_model

MODEL_PATH = r'Q:\projects\apple_variety\AppleVariety\Pommelogy\AppleVariety\ml_model\try_pommelogy.keras'

import keras
_in = keras.layers.Input(shape=(8, 8, 3))
_out = keras.layers.Conv2D(3, 3)(_in)
model = keras.Model(inputs=_in, outputs=_out)

# modelpath = f'model.keras'
model.save(MODEL_PATH)

def load_apple_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

model = load_apple_model()

if model is not None:
    print("Model loaded successfully!")
else:
    print("Failed to load model.")


class AppleVarietyViewSet(viewsets.ModelViewSet):
    queryset = AppleVariety.objects.all()
    serializer_class = AppleVarietySerializer

class MyObtainTokenPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer

class UserDetailView(generics.RetrieveAPIView):
    serializer_class = UserSerializer
    queryset = CustomUser.objects.all()

class AppleVarietyViewSet(viewsets.ModelViewSet):
    queryset = AppleVariety.objects.all()
    serializer_class = AppleVarietySerializer

class ImageUploadViewSet(viewsets.ModelViewSet):
    queryset = ImageIdentify.objects.all()
    serializer_class = ImageUploadSerializer
    #
    # def perform_create(self, serializer):
    #     serializer.save(user=self.request.user)

        # Load and preprocess the image
    # def preprocess_image(image_path):
    #     img_size = (256, 256)
    #     img = load_img(image_path, target_size=img_size)  # Load the image
    #     img_array = img_to_array(img)  # Convert the image to an array
    #     img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension
    #     img_array = img_array / 255.0  # Normalize the image to the [0, 1] range
    #     return img_array
    @action(detail=True, methods=['post'])
    def predict(self, request, pk=None):
        # pass
        print("11111111111111111111111111111\n\n\n\n\n\n\n")
        image_upload = self.get_object()
        image_path = image_upload.image.path
        # img = Image.open(image_path)

        img_size = (256, 256)
        img = load_img(image_path, target_size=img_size)  # Load the image
        img_array = img_to_array(img)  # Convert the image to an array
        img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension
        predictions = img_array / 255.0  # Normalize the image to the [0, 1] range


        # Preprocess the uploaded image
        # preprocessed_img = preprocess_image(image_path)
        # predictions = model.predict(preprocessed_img)
        print(model)
        print("-" * 20)
        # Get the predicted class
        print(predictions)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")