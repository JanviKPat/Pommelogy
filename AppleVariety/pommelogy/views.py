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
# model.add(Dense(10, activation='softmax'))

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
        # if not self.get_object():
        #     pass
        class_names = ['Variety1', 'Variety2', 'Variety3', 'Variety4', 'Variety5',
                       'Variety6', 'Variety7', 'Variety8', 'Variety9', 'Variety10']


        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=400)

        image_file = request.FILES['image']
        expected_img_size = (8, 8)
        img_size = (256, 256)

        # Save the image to a temporary location if necessary
        from PIL import Image
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            for chunk in image_file.chunks():
                temp_file.write(chunk)
            temp_file.flush()
            temp_file.seek(0)
            image_path = temp_file.name

        # Load and preprocess the image
        img = load_img(image_path, target_size=expected_img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = img_array / 255.0

        try:
            predictions = model.predict(preprocessed_img)
            # print(predictions)
            # breakpoint()
        except Exception as e:
            return Response({'error': 'Not an apple'})

        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        print(predicted_class_idx)
        # if predicted_class_idx < len(class_names):
        #     predicted_class = class_names[predicted_class_idx]
        # else:
        #     return Response({'error': 'Predicted class index out of range'}, status=400)

            # Clean up temporary file
        predicted_class = "red_yellow"
        try:
            apple_variety = AppleVariety.objects.get(variety_id=predicted_class)
        except AppleVariety.DoesNotExist:
            return Response({'error': 'Apple variety not found'}, status=404)

        temp_file.close()

        # Assuming you have a serializer for AppleVariety
        from .serializers import AppleVarietySerializer
        apple_variety_serializer = AppleVarietySerializer(apple_variety)

        return Response({
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'apple_variety': apple_variety_serializer.data
        })