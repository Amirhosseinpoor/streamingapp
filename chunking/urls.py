from django.urls import path
from .views import VideoListView, UploadView
from django.views.generic import TemplateView
urlpatterns = [
    path('show/', VideoListView.as_view(), name='video_list'),
    path('', UploadView.as_view(), name='upload'),
    path('pwa', TemplateView.as_view(template_name='pwa/index.html'), name='pwa'),

]