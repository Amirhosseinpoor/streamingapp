from .models import VideoModels
from django.forms import ModelForm

class VideoForm(ModelForm):
    class Meta:
        model = VideoModels
        fields = ['title', 'video_file']