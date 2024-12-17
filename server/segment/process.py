import numpy as np
from op.operator_handler import OperatorHandler
from plot.plot_handler import OperatorPlotHandler
import threading
import traceback
from ultralytics import YOLO
from video import YoloV8Segmenter, CatractSegmenter,VideoSaver,GENERAL_INDEX,MODELS,index12,index9,NineClassSegmenter,YOLO_V8_INDEX,YOLO_V8_151_INDEX,YOLO_V8_TIP_INDEX,YoloV8NeedleTipSegmenter,MamadSegmenter
import uuid
from servicelocator.lookup import global_lookup
import time
import json
import datetime
class Process ():
    id:str
    processName:str
    running :bool = True
    error : bool = False
    result:object
    @staticmethod
    def fromDB(p):
        pro = Process()
        pro.id = p.processID
        pro.running = False
        pro.error = False
        pro.result = json.loads(p.result)
        pro.startTime = p.startTime
        pro.endTime = p.endTime
        return pro
    def __init__(self,processName = "Process_Default") -> None:
        self.id = None
        self.processName = processName
        self.running = False
        self.error = False
        manager = global_lookup.lookup(ProcessManager)
        manager.addProcess(self)
        self.result = None
        pass
    def setResult(self,r):
        self.result = r
    def getResult(self):
        return self.result
    def isDoneCompletly(self):
        return not self.isRunning() and not self.isError()
    def protectiveRun(self):
        self.running = True
        self.startTime = datetime.datetime.now()
        try:
           
            self.run()
            
        except:
            self.error = True
            traceback.print_exc()
        self.endTime = datetime.datetime.now()
        self.running = False
        self.onEnd()
    def start(self):
        self.thread = threading.Thread(target=self.protectiveRun)
        self.thread.start()
    def run(self):
        pass
    def isRunning(self):
        return self.running
    def isError(self):
        return self.error
    def __str__(self) -> str:
        return f"{self.id} : {self.processName}"
    def onEnd(self):
       
        from segment.models import ProcessDB
        self.processDB = ProcessDB(processID = self.id,result =json.dumps( self.getResult()),endTime =datetime.datetime.now(),startTime = self.startTime )
        self.processDB.save()
        manager = global_lookup.lookup(ProcessManager)
        manager.removeID(self.id)
        pass
class SegmentProcess(Process):
    @staticmethod
    def fromDB(p,v):
        pro = SegmentProcess("","","","")
        pro.id = p.processID
        pro.running = False
        pro.error = False
        pro.result = json.loads(p.result)
        pro.startTime = p.startTime
        pro.endTime = p.endTime
        pro.videoPath = v.path
        pro.savePath = v.segmentedPath
        return pro
    def onEnd(self):
        super().onEnd()
        from segment.models import Video
        m = Video(path = self.videoPath,segmentedPath = self.savePath,process = self.processDB,errorFrame = json.dumps(self.badFrames))
        m.save()
    def __init__(self,modelName,videoPath,savePath, resultPath,processName="Video Segment",promptTime = 0,promptMask = {}) -> None:
        self.modelName= modelName
        self.videoPath = videoPath
        self.savePath = savePath
        self.resultPath = resultPath
        self.saver = VideoSaver(savePath)
        if(modelName != ""):
            if(modelName in MODELS ):
                i = MODELS.index(modelName)
                if(i == YOLO_V8_151_INDEX):
                    model = YOLO("best+151.pt", task="segment")
                    self.segmneter = YoloV8Segmenter(videoPath, model, self.saver)
                elif(i == YOLO_V8_INDEX):
                    model = YOLO("bestIman.pt", task="segment")
                    self.segmneter = YoloV8Segmenter(videoPath, model, self.saver)
                elif(i ==YOLO_V8_TIP_INDEX):
                    model = YOLO("bestTip.pt", task="segment")
                    self.segmneter = YoloV8Segmenter(videoPath, model, self.saver)
                elif(i == GENERAL_INDEX):
                    # model = YOLO("bestTip.pt", task="segment")
                    # self.segmneter = YoloV8NeedleTipSegmenter(videoPath, model, self.saver)
                    self.segmneter = MamadSegmenter(videoPath,self.saver,promptTime,promptMask)
                elif(i == index12):
                    # model = YOLO("bestTip.pt", task="segment")
                    # self.segmneter = YoloV8NeedleTipSegmenter(videoPath, model, self.saver)
                    print("12 class Debugger")
                    model = YOLO("best12.pt", task="segment")
                    self.segmneter =CatractSegmenter (videoPath, model, self.saver)
                elif(i == index9):
                    # model = YOLO("bestTip.pt", task="segment")
                    # self.segmneter = YoloV8NeedleTipSegmenter(videoPath, model, self.saver)
                    print("9 class Debugger")
                    model = YOLO("9class.pt", task="segment")
                    self.segmneter =NineClassSegmenter (videoPath, model, self.saver)
                super().__init__( processName)
            else:
                raise Exception("Model not found")
    def run(self):
        if(type(self.segmneter) == CatractSegmenter):
            objs,f,timeSerie,timeC = self.segmneter.segmentVideo()
            needleCenters = []
            eyeCenter = []
            needleBB = []
            eyeBB = []
            eyes = obj["Cannula"]
            needles = obj["Phaco Handpiece"]
            timeSerie = []
            for i,p in enumerate(needles):
                eye = eyes[i]
                eyeCenter.append(eye["Center"])
                eyeBB.append(eye["BB"])
                needleCenters.append(p["Center"])
                needleBB.append(p["BB"])
                timeSerie.append(i * (1/25))

            handler = OperatorHandler(
                self.resultPath + "",needleCenters, eyeCenter, np.array(releativeNeedle), timeSerie,needleBB,eyeBB,tips,timeC
            )
            opPlHandler = OperatorPlotHandler(
                operatorhandler=handler,
                path=self.resultPath,
                timeSerie=timeSerie,
            )
            opPlHandler.save()
            m,hm,send= handler.doOperators()
            self.setResult([m,hm,send])        
            self.saver.end()
        elif(type(self.segmneter) != MamadSegmenter):
            eye,needle,releativeNeedle,f, timeSerie,timeC,self.badFrames,tips = self.segmneter.segmentVideo()
            # print(f"Tips for segmmmenting {tips}")
            obj = {"Eye": eye, "Needle": needle, "RelativeNeedle": releativeNeedle}
            needleCenters = []
            eyeCenter = []
            needleBB = []
            eyeBB = []
            for i, n in enumerate(needle):
                print("Name for Proccessing Must be needle : ",n["Name"])
                if(n != None):
                    needleCenters.append((n["Center"]))
                    needleBB.append((n["BB"]))
                
            for n in eye:
                print("Name for Proccessing Must be eye : ",n["Name"])
                if(n != None):
                    eyeCenter.append((n["Center"]))
                    eyeBB.append((n["BB"]))
            
            handler = OperatorHandler(
                self.resultPath + "",needleCenters, eyeCenter, np.array(releativeNeedle), timeSerie,eyeBB,needleBB,tips,timeC
            )
            # opPlHandler = OperatorPlotHandler(
            #     operatorhandler=handler,
            #     path=self.resultPath,
            #     timeSerie=timeSerie,
            # )
            # opPlHandler.save()
            m,hm,send= handler.doOperators()
            self.setResult([m,hm,send])        
            self.saver.end()
        else:
            obj_dict = self.segmneter.segmentVideo()
            self.badFrames = []
            self.setResult([obj_dict,[],[]])
            self.saver.end()
        return super().run()
    
class ProcessManager():
    processList:list[Process]
    def __init__(self) -> None:
        self.processList = []
        pass
    def getID(self,id:str) ->Process:
        for p in self.processList:
            if(p.id == id):
                return p
        else:
            from .models import ProcessDB,Video
            p = ProcessDB.objects.get(processID = id)
            v = Video.objects.get(process = p)
            if(v != None):
                process = SegmentProcess.fromDB(p,v)
            else:    
                process = Process.fromDB(p)
            return process
    def addProcess(self,proces):
        if(proces.id == None):
            id = uuid.uuid1().__str__()
            proces.id = id
        self.processList.append(proces)
    def removeID(self,id:str):
        for p in self.processList:
            if(p.id == id):
                self.processList.remove(p)
                return
    
    
    