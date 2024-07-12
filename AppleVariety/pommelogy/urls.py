from django.urls import path, include
from .views import MyObtainTokenPairView, UserDetailView, AppleVarietyViewSet, ImageUploadViewSet
from rest_framework_simplejwt.views import TokenRefreshView
from rest_framework.routers import DefaultRouter
# from .views import ImageUploadViewSet

# router = DefaultRouter()
# router.register(r'predict', ImageUploadViewSet, basename='predict_apple_1')


urlpatterns = [
    # path('api/', include(router.urls)),
    path('api/token/', MyObtainTokenPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/user/<int:pk>/', UserDetailView.as_view(), name='user_detail'),
    # path('api/apples/', AppleVarietyViewSet.as_view({'get': 'list'}), name='apple_detail'),
    path('api/apples/<str:pk>/', AppleVarietyViewSet.as_view({'get': 'retrieve'}), name='apple_detail'),
    path('api/predict/', ImageUploadViewSet.as_view({'post': 'predict_apple'}), name='predict_apple'),
]
