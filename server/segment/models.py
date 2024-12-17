from django.db import models
from datetime import datetime
# Create your models here.
class Video(models.Model):
    path = models.TextField()
    segmentedPath = models.TextField()
    errorFrame = models.TextField()
    process = models.ForeignKey(
        "ProcessDB", on_delete=models.CASCADE,blank=True,
        null=True)
    # eyeCenter = models.TextField()
    # needleCenter = models.TextField()
    # operators = models.TextField()
    
class ProcessDB(models.Model):
    processID = models.TextField(null = True,blank = True)
    result = models.TextField(null = True,blank = True)
    startTime = models.DateTimeField(default = datetime.now)
    endTime = models.DateTimeField(default = datetime.now)
    pass


class AnnotateVideo(models.Model):
    path = models.TextField()
    time = models.DateTimeField(default= datetime.now)
    duration = models.PositiveIntegerField(default=1)
    pass

class Section(models.Model):
    path = models.TextField()
    duration = models.PositiveIntegerField(default=1)
    data = models.TextField()


class Annotate(models.Model):
    videoID  =models.ForeignKey("AnnotateVideo",on_delete=models.CASCADE,blank=True,
        null=True)
    events = models.TextField(null=True,blank=True)
    comments = models.TextField(null=True,blank=True)
    sections = models.TextField(null=True,blank=True)
    pass