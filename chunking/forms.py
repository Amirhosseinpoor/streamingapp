
from django import forms
from .models import VideoModels
from django.forms import ModelForm

class VideoForm(ModelForm):
    start_time = forms.IntegerField(min_value=0, required=True, help_text="Start time in seconds")
    end_time = forms.IntegerField(min_value=0, required=True, help_text="End time in seconds")

    class Meta:
        model = VideoModels
        fields = ['title', 'video_file', 'start_time', 'end_time']