from django.apps import AppConfig
import os
from servicelocator.lookup import global_lookup
from .process import ProcessManager
class SegmentConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'segment'
    def ready(self) -> None:
        print("Starting UP The segment application")
        try:
            os.makedirs("videos/")
        except:
            pass
        try:
            os.makedirs("videosSegmented/")
        except:
            pass
        try:
            os.makedirs(f"Results/")
        except:
            pass
        global_lookup.add(ProcessManager,ProcessManager())
        return super().ready()