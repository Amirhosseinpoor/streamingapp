from django.db import models



class VideoModels(models.Model):
    title = models.CharField(max_length=255)
    video_file = models.FileField(upload_to='videos/')
    hls_ready = models.BooleanField(default=False)

    def get_hls_url(self):
        return f"/media/videos/output/{self.title}/output.m3u8"

# Create your models here.