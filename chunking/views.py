from django.shortcuts import render
from django.views.generic import ListView, CreateView
from .models import VideoModels
from .forms import VideoForm
from .tasks import convert_to_hls


class VideoListView(ListView):
    model = VideoModels
    template_name = 'videofair.html'
    context_object_name = 'videos'


class UploadView(CreateView):
    model = VideoModels
    form_class = VideoForm
    template_name = 'upload.html'
    success_url = 'show'

    def form_valid(self, form):
        video = form.save(commit=False)
        video.start_time = int(self.request.POST.get('start_time'))
        video.end_time = int(self.request.POST.get('end_time'))
        video.save()
        convert_to_hls.delay(video.id)
        return super().form_valid(form)


from django.shortcuts import render

# Create your views here.
