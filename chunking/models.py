from django.db import models
from django.utils.timezone import now


class VideoModels(models.Model):
    title = models.CharField(max_length=255)
    video_file = models.FileField(upload_to='videos/')
    hls_ready = models.BooleanField(default=False)

    upload_time = models.DateTimeField(default=now, verbose_name="Upload Time")
    start_time = models.IntegerField(default=0, verbose_name="Start Time")
    end_time = models.IntegerField(default=0, verbose_name="End Time")
    def __str__(self):
        return self.title

    def get_hls_url(self):
        return f"/media/videos/output/{self.title}/output.m3u8"





# Create your models here.