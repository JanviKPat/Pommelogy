from django.urls import path
from .views import MyObtainTokenPairView, UserDetailView
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path('api/token/', MyObtainTokenPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/user/<int:pk>/', UserDetailView.as_view(), name='user_detail'),
]
