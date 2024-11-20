from django.urls import path
from .views import VideoListView, UploadView
urlpatterns = [
    path('show/', VideoListView.as_view(), name='video_list'),
    path('', UploadView.as_view(), name='upload')
]