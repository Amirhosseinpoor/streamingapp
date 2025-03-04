from django.urls import path
from .views import VideoListView, UploadView, TrimingView, trim_video
from django.views.generic import TemplateView

urlpatterns = [
    path('show/', VideoListView.as_view(), name='video_list'),
    path('', UploadView.as_view(), name='upload'),
    path('pwa/', TemplateView.as_view(template_name='pwa/index.html'), name='pwa'),
    path('triming/', TrimingView.as_view(), name='triming'),

    # مسیر برای API برش ویدیو
    path('trim-video/<int:video_id>/', trim_video, name='trim_video'),
]
