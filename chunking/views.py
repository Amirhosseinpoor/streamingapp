from django.shortcuts import render
from django.views.generic import ListView, CreateView, TemplateView
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
    success_url = 'triming'
    def form_valid(self, form):
        video = form.save(commit=False)
        video.save()
        return super(UploadView, self).form_valid(form)





class TrimingView(TemplateView):
    template_name = 'trimizing.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['video'] = VideoModels.objects.last()  # دریافت آخرین ویدیو
        return context







from django.shortcuts import render

# Create your views here.
