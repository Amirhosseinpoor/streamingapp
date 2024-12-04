from django.contrib import admin

# Register your models here.

from django.contrib import admin
from .models import VideoModels
@admin.register(VideoModels)
class CustomAdmin(admin.ModelAdmin):
    list_display = ('title', 'upload_time')
    ordering = ('-upload_time',)
