from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import ListView, CreateView
from django.http import JsonResponse
from .models import VideoModels
from .forms import VideoForm, VideoTrimingForm
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


class TrimingView(CreateView):
    model = VideoModels
    form_class = VideoTrimingForm
    template_name = 'trimizing.html'
    success_url = 'show'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['video'] = VideoModels.objects.last()  # آخرین ویدیو را برای نمایش ارسال می‌کنیم
        return context


def trim_video(request, video_id):
    """ این ویو یک ویدیو را دریافت کرده و آن را برش می‌دهد """
    if request.method == "POST":
        video = get_object_or_404(VideoModels, id=video_id)
        start_time = int(request.POST.get("start_time", 0))
        end_time = int(request.POST.get("end_time", 0))

        if start_time >= end_time or start_time < 0:
            return JsonResponse({"error": "Invalid start and end times"}, status=400)

        video.start_time = start_time
        video.end_time = end_time
        video.save()

        convert_to_hls.delay(video.id)  # ارسال تسک به Celery
        return JsonResponse({"message": "Video trimming started successfully!"}, status=200)

    return JsonResponse({"error": "Invalid request"}, status=400)
