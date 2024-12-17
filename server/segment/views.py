from django.shortcuts import render,HttpResponse
from django.http import HttpRequest
from segment.models import Video,Annotate,AnnotateVideo,Section
from ultralytics import YOLO
from servicelocator.lookup import global_lookup
import json
from django.views.decorators.csrf import csrf_exempt
import os
from .process import ProcessManager,SegmentProcess,Process
from video import MODELS
import glob
import datetime
from django.http import HttpResponse
from wsgiref.util import FileWrapper
BEST_MODEL = "Yolov8"
import random
import assemblyai as aai

import cv2
# print uuid.uuid4()
def handle_uploaded_file(f,path):
    with open(path, "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)
        destination.close()
# Create your views here.
def library(request:HttpRequest):
    if(request.method == 'GET'):
        libs = glob.glob("lib/*.mp4")
        return HttpResponse(json.dumps(libs))
    pass
def downloadLib(request:HttpRequest):
    if(request.method == "GET"):
        filename = request.GET.get('File',None)
        # content = FileWrapper(filename)
        print(f"fILENAME {filename}")
        with open(filename, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="video/mp4")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(filename)
            return response
        # response = HttpResponse(fh.read(), content_type="video/mp4")
        response['Content-Disposition'] = 'inline; filename=' + filename
        return response
    pass
@csrf_exempt 
def upload(request:HttpRequest):
    # if request.method == "POST":
        print(f"Requet method : {request.method}")
        # form = UploadFileForm(request.POST, request.FILES)
        # if form.is_valid():
        v = Video.objects.last()
        if(v == None):
            pk =0
        else:
            pk = v.pk
        pk = pk + 1
        
        path = f"Results/Mamad/ID {pk}/video.mp4"
        savePath = f"Results/Mamad/ID {pk}/segmentedVideo.mp4"
        resultPath = f"Results/Mamad/ID {pk}"
        try:
            os.makedirs(f"Results/Mamad/ID {pk}/")
        except:
            pass
    
        model = request.headers.get("Model",None)
        
        if(model == "BEST"):
            model = BEST_MODEL
        print(f"Model for segmentation {model} {pk}")
        masks = request.headers.get("Mask",None)
        promptTime = request.headers.get("Time",None)
        handle_uploaded_file(request.FILES["Video"],path)
        print(f"Prompt mask in requesting for general segmenting {pk} {masks} ")
        process = SegmentProcess(model,videoPath=path,savePath=savePath,resultPath=resultPath,promptTime=promptTime,promptMask=masks)
        process.start()
        return HttpResponse(json.dumps({"ID":process.id}),)
        pass
def isReady(request:HttpRequest):
    if(request.method == "GET"):
        id = request.headers.get('ID',None)
        print(f"ID recivig for isReady {id}")
        manager:ProcessManager = global_lookup.lookup(ProcessManager)
        process:Process = manager.getID(id)
        if(process.isDoneCompletly()):
            return HttpResponse(json.dumps({"Status":"Done"}),)
            pass
        elif(process.isError()):
            return HttpResponse(json.dumps({"Status":"Error"}),)
        elif(process.isRunning()):
            return HttpResponse(json.dumps({"Status":"Run"}),)
        return HttpResponse(json.dumps({"Status":"UNKNOWN"}),)
    pass
def result(request:HttpRequest):
    if(request.method == "GET"):
        id = request.headers.get('ID',None)
        manager:ProcessManager = global_lookup.lookup(ProcessManager)
        process:Process = manager.getID(id)
        print(f"ID recivig for result {id}")
        if(process.isDoneCompletly()):
            print(f"ID DONE {id}")
            print(f"Precess List before get result {manager.processList}")
            result = process.getResult()[2]
            print(f"Precess List {manager.processList}")
            # print(f"ID DONE ")
            if(result != None):
                return HttpResponse(json.dumps({"Data":result},),)
        res = HttpResponse(json.dumps({"Data":"NOT FOUND"}),)
        res.status_code  =404
        return res
    pass
def download(request:HttpRequest):
    if(request.method == "GET"):
        print(f"BODY : {request.body.decode()} Get : {request.GET.items()}")
        id = request.GET.get('ID',None)
        manager:ProcessManager = global_lookup.lookup(ProcessManager)
        process:Process = manager.getID(id)
        print(f"ID dowload {id}")
        if(process.isDoneCompletly()):
            print(f"ID DONE {id}")
            savePath = process.savePath
            with open(savePath, 'rb') as fh:
                response = HttpResponse(fh.read(), content_type="video/mp4")
                response['Content-Disposition'] = 'inline; filename=' + os.path.basename(savePath)
                return response
            return 
        res = HttpResponse(json.dumps({"Data":"NOT FOUND"}),)
        res.status_code  =404
        return res
    pass
def models(request:HttpRequest):
    if(request.method == "GET"):
        return HttpResponse(json.dumps(MODELS))
def calculate(request:HttpRequest):
    if(request.method == "GET"):
        id = request.headers.get('ID',None)
        data = request.headers.get("Data",None)
        
@csrf_exempt
def getVideoLength(request:HttpRequest):
    path = f"VideoDuration/{int(random.random() * 10000)}.mp4"
    try:
        os.makedirs(f"VideoDuration/")
    except:
        pass
    handle_uploaded_file(request.FILES["Video"],path)
    data  = cv2.VideoCapture(path)
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = data.get(cv2.CAP_PROP_FPS) 
    seconds = int(frames / fps) 
    return HttpResponse(json.dumps({"Time":seconds}))
    pass

@csrf_exempt
def annotate(request:HttpRequest):
    v = AnnotateVideo.objects.last()
    if(v == None):
        pk =0
    else:
        pk = v.pk
    pk = pk + 1
    path = f"Annotations/Video/{pk}/video.mp4"
    try:
        os.makedirs(f"Annotations/Video/{pk}/")
    except:
        pass
    handle_uploaded_file(request.FILES["Video"],path)
    data  = cv2.VideoCapture(path)
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = data.get(cv2.CAP_PROP_FPS) 
    seconds = round(frames / fps) 
    aData = json.loads(request.headers.get("Data",None))
    av = AnnotateVideo(path = path ,duration = seconds)

    av.save()
    sections = aData["Sections"]
    sectionIDs = []
    for section in sections:
        v = Section.objects.last()
        if(v == None):
            pkSection =0
        else:
            pkSection = v.pk
        startTime = int(section["Start"])
        endTime = int(section["End"])
        video = videoSplit(path,endTime,startTime,size=(640,640))
        sectionPath =f"Annotations/Video/{pk}/{pkSection}.mp4"
        writer = cv2.VideoWriter(
            sectionPath, cv2.VideoWriter.fourcc(*"XVID"), fps, (640, 640)
        )
        for v in video:
            writer.write(v)
        writer.release()
        """ startTime - endTime"""
        sec = Section(path=sectionPath,duration =12,data = json.dumps(section))
        sec.save()
        sectionIDs.append(pkSection)
    a = Annotate(videoID = av,events =aData["Events"],comments = aData["Comments"],sections = json.dumps(sectionIDs))
    a.save()
    data.release()
    return HttpResponse(json.dumps({"Result":"OK"}))

def speachToText(request:HttpRequest):
    print("Speach To Text")
    path = f"Annotations/Voice/{random.Random().randint(0,1000)}.wav"
    try:
        os.makedirs(f"Annotations/Voice/")
    except:
        pass
    handle_uploaded_file(request.FILES["Voice"],path)
    aai.settings.api_key = "50ebd5adffcf4d4f98bae7a7542ad070"
    transcriber = aai.Transcriber()
    print("Speach To pre Send")
    transcript = transcriber.transcribe(f"{path}")
    print("Speach To Done")
    return HttpResponse(json.dumps({"Result":f"{transcript.text}"}))



def videoSplit(path,endTime:int,startTime:int,size):
    print(f"End time {endTime} Start time {startTime}")
    cap  = cv2.VideoCapture(path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) 
    ret , frame = cap.read()
    time = 0
    if(endTime>=time>=startTime):
        frame = cv2.resize(frame,size)
        frames.append(frame)
        pass
    
    while(ret):
        time = time + 1/fps
        ret , frame = cap.read()
        print(f"End time {endTime} Start time {startTime}  Time {time} bool {endTime>=time>=startTime}")
        if(endTime>=time>=startTime):
            frame = cv2.resize(frame,size)
            frames.append(frame)
        pass    
    cap.release()
    return frames