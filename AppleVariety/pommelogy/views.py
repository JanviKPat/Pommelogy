# import tensorflow as tf
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import generics, viewsets
from .serializers import MyTokenObtainPairSerializer, UserSerializer, AppleVarietySerializer, ImageUploadSerializer
from django.contrib.auth.models import User
from .models import AppleVariety, CustomUser, ImageIdentify
from rest_framework.decorators import action
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from rest_framework.response import Response
from collections import Counter
from pathlib import Path
from django.core.files.uploadedfile import InMemoryUploadedFile
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from tensorflow.keras import models
loaded_model = None
model_dir = Path('ml_model')
MODEL_PATH = model_dir / 'fresh_try_pommelogy.keras'

def load_apple_model():
    global loaded_model
    if loaded_model is not None:
        print("Model already loaded.")
        return loaded_model

    try:
        # Recreate the model architecture
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(20, activation='softmax'))

        # Compile the model (optional, but recommended if you plan to continue training)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Load the weights into the model
        # weights_before = model.get_weights()
        model.load_weights(MODEL_PATH)
        # weights_after = model.get_weights()
        # # Set the global variable
        # weights_diff = [np.any(weight_before != weight_after) for weight_before, weight_after in
        #                 zip(weights_before, weights_after)]
        # if any(weights_diff):
        #     print("Weights have been updated.")
        # else:
        #     print("Weights have not changed.")
        # # breakpoint()
        loaded_model = model
        print("Model loaded successfully!")

        return loaded_model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Call the function to load the model
model = load_apple_model()


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

    def retrieve(self, request, *args, **kwargs):
        variety_id = kwargs.get('pk')
        queryset = self.get_queryset().filter(variety_id=variety_id)
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)


class ImageUploadViewSet(viewsets.ModelViewSet):
    queryset = ImageIdentify.objects.all()
    serializer_class = ImageUploadSerializer

    @action(detail=False, methods=['post'])
    def predict_apple(self, request, *args, **kwargs):
        class_names = ["apple", "braeburn", "crimson_snow", "fuji", "gala", "golden_1", "golden_2", "golden_delicious",
                       "granny_smith", "gravensteins_banks", "hit", "honey_crisp", "mcintosh", "pink_lady", "red_1",
                       "red_2", "red_3", "red_delicious", "red_yellow", "ruby"]

        image_file = request.FILES['image']

        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=400)

        # Ensure the uploaded file is of type InMemoryUploadedFile
        if not isinstance(image_file, InMemoryUploadedFile):
            return Response({"error": "Uploaded file is not an image"}, status=400)

        try:
            # Read the image file
            image = Image.open(image_file)

            # Preprocess the image

            target_size = (model.input_shape[1], model.input_shape[2])
            image = image.resize(target_size)

            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            # img_array /= 255.0

            # Make a prediction
            predictions = model.predict(img_array)

            predicted_class_index = np.argmax(predictions)
            confidence = np.max(predictions)
            print(predictions)
            print(predicted_class_index)
            # if confidence * 100 <= 30:
            #     return Response({
            #         'predicted_class': None,
            #         'confidence': None,
            #         'apple_variety': "Not an apple"
            #     })

        except Exception as e:
            return Response({'error': str(e)})

        try:
            predicted_class = class_names[predicted_class_index - 1]
            apple_variety = AppleVariety.objects.get(variety_id=predicted_class)
        except AppleVariety.DoesNotExist:
            return Response({'error': 'Apple variety not found'}, status=404)

        apple_variety_serializer = AppleVarietySerializer(apple_variety)

        return Response({
            'predicted_class': predicted_class,
            'confidence': float(confidence) *100,
            'apple_variety': apple_variety_serializer.data
        })