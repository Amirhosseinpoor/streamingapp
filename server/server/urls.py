"""
URL configuration for server project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from segment.views import isReady,result,upload,download,models,downloadLib,library,getVideoLength,annotate,speachToText

urlpatterns = [
    path("admin/", admin.site.urls),
    path("video/isReady/",isReady),
    path("video/result/",result),
    path("video/upload/",upload),
    path("video/download/",download),
    path("video/models/",models),
    path("video/library/",library),
    path("video/downloadLib/",downloadLib),
    path("video/time/",getVideoLength),
    path("video/annotate/",annotate),
    path("video/voiceToText/",speachToText)
]