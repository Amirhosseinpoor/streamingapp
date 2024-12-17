from ultralytics import YOLO
import cv2 as cv
import numpy as np
import random
from PIL import Image
import time
import torch
import os
from ForAll import FPS
import math
import gc
import glob
import json
from inference_core import InferenceCore
from model.network import XMem
from palette import color_map
from utils import *
MODELS = ["YoloV8+151","Yolov8","Yolob8-Tip","GENERAL","12Class","9Class"]
YOLO_V8_151_INDEX = 0
YOLO_V8_INDEX = 1
YOLO_V8_TIP_INDEX = 2
GENERAL_INDEX = 3
index12 = 4
index9 = 5
# b = torch.cuda.is_available()
# torch.cuda.get_device_name()
# # a = torch.get_device()
# torch.device("cuda:0")
# print(f"Torch setted video {b}")


# model = YOLO('bestIman.pt',)
# path = 'ID 438.mp4'
# #
#
#

class VideoSaver:
    writer: cv.VideoWriter
    path: str

    def __init__(self, path) -> None:
        self.path = path
        self.writer = cv.VideoWriter(
            self.path, cv.VideoWriter.fourcc(*"XVID"), 10, (640, 640)
        )
        # self.writer.open(self.path)
        pass

    def saveFrame(self, frame):
        self.writer.write(frame)

    def end(self):
        self.writer.release()

    pass

class ImageSegmenter:
    pathToImage: str
    savePath:str
    def __init__(self,imagePath:str,savePath:str) -> None:
        self.pathToImage = imagePath
        self.savePath = savePath
        pass
    def segment(self, image):
        pass
    def segmentImage(self, ):
        img = cv.imread(self.pathToImage)
        img,data,processTime,modelTime = self.segment(img)
        print(f"Timings modelTime {modelTime} Process Time {processTime}")
        dir = ""
        if(self.savePath.count("\\") != 0):
            splits = self.savePath.split("\\")[:-1]
            for s in splits:
                dir = dir + s + "\\"
        if(dir != "" and not os.path.exists(dir)):
            os.makedirs(dir)
        cv.imwrite(self.savePath,img)
        pass
    # saver: VideoSaver

class VideoSegmenter:
    pathToVideo: str
    saver: VideoSaver
    def __init__(self, path, videoSaver) -> None:
        self.pathToVideo = path
        # self.model = model
        self.saver = videoSaver
        self.cap = cv.VideoCapture(self.pathToVideo)
        pass
    def segmentVideo(self):

        while self.cap.isOpened():
            ret, rawImage = self.cap.read()
            video_object_dict = {}
            if ret:
                image, data= self.segment(rawImage)

                for d in data:
                    center = d["Center"]
                    center[0] = str(int(center[0]))
                    center[1] = str(int(center[1]))
                    name = d["Name"]
                    mask = []
                    for p in d["Mask"]:
                        mask.append([str(int(p[0])),str(int(p[1]))])
                        pass
                    print ("Center : ",center)
                    print("Mask : ",mask)

                    if(not name  in video_object_dict):
                        video_object_dict[name] = [{"Mask":mask,"Center":center}]
                    else:
                        video_object_dict[name].append({"Mask":mask,"Center":center})
                for key in video_object_dict:
                    preCenter = None
                    dict_list = video_object_dict[key]
                    for dic in dict_list:
                        center = dic["Center"]
                        if(preCenter!=None):
                            cv.line(image,center,preCenter,(150,150,150),thickness=2)
                        else:
                            cv.circle(image,center,3,(200,200,200),thickness=-1)
                        pass
                    pass
                self.saver.writer.write(image)
            else:
                self.cap.release()
        endVideoTime = time.time()
        self.cap.release()
        self.saver.writer.write(image)
        self.saver.writer.release()
        print("video_object_dict : ",video_object_dict)
        return video_object_dict

class PromptingSegmenter(VideoSegmenter):
    def __init__(self, path, videoSaver,promptTime) -> None:
        self.promptTime = int(promptTime)
        self.badFrames = []
        super().__init__(path, videoSaver)
    def segmentVideo(self):
        fps = self.cap.get(cv.CAP_PROP_FPS)
        frameNumber = int(fps * (self.promptTime / 1000))
        i = -1
        video_object_dict = {}
        while self.cap.isOpened():
            ret, rawImage = self.cap.read()
            i = i+1
            if(i<frameNumber):
                continue

            if ret:
                image, data= self.segment(rawImage)
                # cv.imshow("Image",image)
                for d in data:
                    center = d["Center"]
                    center[0] = str(int(center[0]))
                    center[1] = str(int(center[1]))
                    name = d["Name"]
                    mask = []
                    for p in d["Mask"]:
                        mask.append([str(int(p[0])),str(int(p[1]))])
                        pass


                    if(not name  in video_object_dict):
                        video_object_dict[name] = [{"Mask":mask,"Center":center}]
                    else:
                        video_object_dict[name].append({"Mask":mask,"Center":center})
                for key in video_object_dict:
                    preCenter = None
                    dict_list = video_object_dict[key]
                    for dic in dict_list:
                        center = dic["Center"]
                        center[0] = int(center[0])
                        center[1] = int(center[1])
                        if(preCenter!=None):
                            cv.line(image,center,preCenter,(150,150,150),thickness=2)
                        else:
                            cv.circle(image,center,3,(200,200,200),thickness=-1)
                        preCenter = center
                    pass
                print(f"Image Segmented and Writed {i}")
                # cv.imshow("MOS",image)
                # cv.waitKey(30)
                self.saver.writer.write(image)
                i = i + 1
            else:
                self.cap.release()
        endVideoTime = time.time()
        self.cap.release()
        self.saver.writer.write(image)
        self.saver.writer.release()
        print("Mamad End")
        return video_object_dict


class EyeSegmenter(VideoSegmenter):
    pathToVideo: str
    saver: VideoSaver
    def __init__(self, path, videoSaver) -> None:
        self.pathToVideo = path
        # self.model = model
        self.saver = videoSaver
        self.cap = cv.VideoCapture(self.pathToVideo)
        pass
    def printLines(self, f, dicts):
        preX = None
        preY = None
        for d in dicts:
            if d != None:
                center = d["Center"]
                x, y = center
                if preX == None and preY == None:
                    f = cv.circle(
                        f, center=center, radius=2, color=(0, 0, 255), thickness=-1
                    )
                else:
                    f = cv.line(f, (preX, preY), (x, y), (255, 0, 0), 1)
                preX = x
                preY = y
        return f
    def printLinesForTip(self,f,tips):
        preX = None
        preY = None
        print(f"draw tips {len(tips)} {tips[-1]}")
        for center in tips:
            if(type(center) != type(None)):
                # print(f"Needle center {center}")
                x = center[0]
                y = center[1]
                if preX == None and preY == None:
                    # print("Circle in tips")
                    f = cv.circle(
                        f, center=center, radius=2, color=(0, 0, 255), thickness=-1
                    )
                else:
                    # print("Circle in line in tips")
                    # f = cv.circle(
                    #     f, center=center, radius=2, color=(0, 255, 0), thickness=-1
                    # )
                    f = cv.line(f, (preX, preY), (x, y), (255, 0, 0), 1)

                preX = x
                preY = y
        if preX != None and preY != None:
                    # print("Circle in tips")
            f = cv.circle(
                f, center=center, radius=2, color=(0, 255, 0), thickness=-1
            )
        return f

    def randomColor(self, masks):
        seed = random.Random()
        colors = []
        for m in masks:
            c = (
                int(seed.random() * 255),
                int(seed.random() * 255),
                int(seed.random() * 255),
            )
            colors.append(c)
        return colors
    def fitLineToNeedle(self,img,needle):
        mask = needle["Mask"]
        center = needle["Center"]
        [vx,vy,x,y] = cv.fitLine(np.array(mask),cv.DIST_L2, 0, 0.01, 0.01)

        center2 = np.array(center) + np.array([int(200*vx[0]),int(200*vy[0])])
        p2 = np.array([int(x[0]),int(y[0])]) + np.array([int(-20000*vx[0]),int(-20000*vy[0])])
        p1 = np.array([int(x[0]),int(y[0])]) + np.array([int(+20000*vx[0]),int(+20000*vy[0])])
        # print(f"vx {vx} vy {vy} x {x} y {y} center {center} center2 {center2} p2 {p2}")
        # img = cv.line(img, p2,  p1, (150, 255, 150), 2)
        # img = cv.line(img, center2, np.array(center), (255, 0, 0), 2)
        # img = cv.line(img,ellipse, (0,0,255), 3)
        # let lefty = Math.round((-x * vy / vx) + y);
        # let righty = Math.round(((src.cols - x) * vy / vx) + y);
        # let point1 = new cv.Point(src.cols - 1, righty);
        # let point2 = new cv.Point(0, lefty);
        # t0 = (0-fit_line[3])/fit_line[1]
        # t1 = (img.shape[0]-fit_line[3])/fit_line[1]
        # lefty =    int( (-x * vy / vx) +y)
        # righty =    int( ((src.cols - x) * vy / vx) + y)
        # point1 =    int( (-x * vy / vx) +y)
        # point2 =    int( (-x * vy / vx) +y)
        # # plug into the line formula to find the two endpoints, p0 and p1
        # # to plot, we need pixel locations so convert to int
        # p0 = (fit_line[2:4] + (t0 * fit_line[0:2])).astype(np.uint32)
        # p1 = (fit_line[2:4] + (t1 * fit_line[0:2])).astype(np.uint32)
        # cv.line(img, tuple(p0.ravel()), tuple(p1.ravel()), (0, 255, 0), 10)
        return img,vx[0],vy[0]
    def fitElipseToIris(self,img,iris):
        # print(f"fitElipseToIris input arguments {iris}")
        mask = iris["Mask"]
        ellipse = cv.fitEllipse(np.array(mask),)
        img = cv.ellipse(img,ellipse, (0,0,255), 3)

        return img,ellipse
    def projectOnline(self,point,vx,vy,p0):
        x = p0[0]*vy**2 + point[0]*vx**2 + vx*vy*(point[1] - p0[1])
        y = (x-point[0])*(1/-vy)*(vx) + point[1]
        return int(x),int(y)
    def calculatePoint(self,vx,vy,needle,eye,img):
        mask = needle["Mask"]
        center = eye["Center"]
        # print(f"Needle mask {mask}")
        if(vx*vy<=0):
            min_max = [np.inf,0]
            max_min = [0,np.inf]
            for x,y in mask:
                # img = cv.circle(img,(int(x),int(y)),5,color=(0,0,0),thickness=-1)
                if(x<min_max[0]):
                    min_max[0] = int(x)
                if(y>min_max[1]):
                    min_max[1] = int(y)
                if(x>max_min[0]):
                    max_min[0] = int(x)
                if(y<max_min[1]):
                    max_min[1] = int(y)
            dist1 = np.linalg.norm(np.array(min_max)-np.array(center))
            dist2 = np.linalg.norm(np.array(max_min)-np.array(center))
            if(dist1<dist2):
                return min_max,max_min
            else:
                return max_min,min_max
            # return min_max,max_min,img
        else:
            max_max = [0,0]
            min_min = [np.inf,np.inf]
            for x,y in mask:
                # img = cv.circle(img,(int(x),int(y)),5,color=(0,0,0),thickness=-1)
                if(x<min_min[0]):
                    min_min[0] = int(x)
                if(y<min_min[1]):
                    min_min[1] = int(y)
                if(x>max_max[0]):
                    max_max[0] = int(x)
                if(y>max_max[1]):
                    max_max[1] = int(y)
        # el_center = ellipsis[0]
            dist1 = np.linalg.norm(np.array(max_max)-np.array(center))
            dist2 = np.linalg.norm(np.array(min_min)-np.array(center))
            if(dist1<dist2):
                return max_max,min_min
            else:
                return min_min,max_max
        pass
    def occlustionDetection(self,maxLength,lenght):
        if(0.8 * maxLength<lenght):
            return True
        return False
    def crossLineOnCircle(self,image,tip,eye,radius,vx,vy,):
        deltaY =  tip[1] - eye[1]
        m = (vy/vx)
        k = deltaY - m*tip[0]
        try:
            halfDelta = math.sqrt(math.pow((k*m - eye[0]),2) - (1 + math.pow(m,2))*(k**2 - radius ** 2 + eye[0] ** 2))
            if(halfDelta>=0):
                x1 = int((eye[0] - k*m + halfDelta)/(1+m**2))
                y1 =int(self.line(vx,vy,tip,x1))
                x2 = int((eye[0] - k*m - halfDelta)/(1+m**2))
                y2 =int(self.line(vx,vy,tip,x2))
                # cv.line(image,(x1,y1),tip,(0,0,0),thickness=2)
                # cv.line(image,(x2,y2),tip,(0,0,0),thickness=2)
                cv.circle(image,eye,radius=6,color=(200,200,200),thickness=-1)
                cv.circle(image,(x1,y1),radius=6,color=(100,100,100),thickness=-1)
                cv.circle(image,(x2,y2),radius=6,color=(100,100,100),thickness=-1)
            else:
                vx = tip[0] - eye[0]
                vy = tip[1] - eye[1]
                self.crossLineOnCircle(image,tip,eye,radius,vx,vy)
        except:
            pass
        return image
    def line(self,vx,vy,tip,x):
        return (vy/vx)*(x-tip[0]) + tip[1]

    def fitCircleToTips(self,image,tips,eyes,vx,vy,end,maxLenght):
        # print(f"Tips {tips}")
        sum_distance = 0
        i = 0
        lastEye = None
        lastTip = None
        for tip,eye in zip(tips,eyes):
            if(tip!=None and eye != None):
                distance = abs(tip[0] - eye["Center"][0])**2 + abs(tip[1] - eye["Center"][1])**2
                distance = math.sqrt(distance)
                sum_distance += distance
                lastEye = eye["Center"]
                lastTip = tip
                i+=1
            pass

        distance =int( sum_distance/i)
        # print(lastEye)
        image = self.crossLineOnCircle(image,lastTip,lastEye,distance,vx,vy)
        image = cv.circle(image,lastEye,distance,(0,0,0),thickness=3)

        endX = end[0]
        endY = end[1]
        imageShapeX = np.shape(image)[0]
        imageShapeY = np.shape(image)[1]

        endXCondition = 0<=endX<=5*(imageShapeX/100) or 95*(imageShapeX/100)<=endX<=imageShapeX/100
        endYCondition = 0<=endY<=5*(imageShapeY/100) or 95*(imageShapeY/100)<=endY<=imageShapeY/100
        tipX = lastTip[0]
        tipY = lastTip[1]
        if(not (endXCondition or endYCondition)):
            deltaX  = abs(maxLenght * (vx/math.sqrt(vx**2+vy**2)))
            deltaY = abs(maxLenght * (vy/math.sqrt(vx**2+vy**2)))
            tipX = int(endX + deltaX) if(endX <= lastTip[0]) else int(endX - deltaX)
            tipY = int(endY + deltaY) if(endY <= lastTip[1]) else int(endY - deltaY)

            image = cv.circle(image,(tipX,tipY),radius=6,color=(200,0,200),thickness=-1)
            pass

        # ellipse = cv.fitEllipse(np.array(tips),)
        # img = cv.ellipse(image,ellipse, (0,0,0), 3)
        return image,(tipX,tipY)
    def segmentVideo(self):
        f = -1
        needles = []
        eyes = []
        relativeNeedle = []
        i = 0
        badFrames = []
        timeSeries = np.array([])
        tips = []
        filTips = []
        lenghts = []
        startVideoTime = time.time()
        maxLength = 0
        while self.cap.isOpened():
            ret, rawImage = self.cap.read()
            # cv.imshow("Raw Image",rawImage)
            if ret:
                start = time.time()
                image, data,processTime,modelTime = self.segment(rawImage)
                print(f"Model time {modelTime}")
                segmentTime = time.time()
                eyeDict = None
                needleDict = None
                for d in data:
                    print(f"Detected name {d['Name']}")
                    if d["Name"] == "Eye":
                        if eyeDict == None:
                            eyeDict = d
                        else:
                            if float(d["Conf"]) > float(eyeDict["Conf"]):
                                eyeDict = d
                    elif d["Name"] == "Needle":
                        if needleDict == None:
                            needleDict = d
                        else:
                            if float(d["Conf"]) > float(needleDict["Conf"]):
                                needleDict = d
                # print(f"EyeDict {eyeDict == None} {len(eyes)} needleDict {needleDict == None} {len(needles)}")
                # print(f"EyeDict {eyeDict} needleDict {needleDict}")
                # time.sleep(1000)
                eyes.append(eyeDict)
                needles.append(needleDict)
                if(needleDict == None or eyeDict == None):
                    badFrames.append(i)
                else:
                    image,vx,vy = self.fitLineToNeedle(image,needleDict)

                if(needleDict!=None and eyeDict != None):
                    tip,end= self.calculatePoint(vx,vy,needleDict,eyeDict,image)

                    tip = self.projectOnline(tip,vx,vy,needleDict["Center"])
                    end = self.projectOnline(end,vx,vy,needleDict["Center"])
                    lenght = (tip[0] - end[0]) ** 2 + (tip[1] - end[1]) ** 2
                    lenght = int(lenght ** 0.5)
                    notOcc = True
                    if(lenght > maxLength):
                        maxLength = lenght
                    else:
                        if(len(lenghts) > 0):
                            notOcc = self.occlustionDetection(lenghts[-1],lenght)
                            if(not notOcc):

                                lenght = lenghts[-1]
                    lenghts.append(lenght)
                    # print(f"Tip {tip} end {end} {needleDict}")
                    tips.append(tip)

                    image = cv.circle(image,tip,5,color=(0,255,0),thickness=-1)
                    image = cv.circle(image,end,5,color=(0,0,0),thickness=-1)
                    if(not notOcc):
                        image = cv.putText(
                            image,
                            f"Occlusion : {lenght} {lenghts[-2]}",
                            (20, 40),
                            cv.FONT_HERSHEY_COMPLEX,
                            0.5,
                            (0, 0, 0),
                        )
                        if(not os.path.exists("Occlusion/")):
                            os.makedirs("Occlusion/")
                            pass
                        cv.imwrite(f"Occlusion/{random.Random().randint(0,10000)}.jpg",rawImage)
                    else:
                        image = cv.putText(
                            image,
                            f"{lenght}",
                            (20, 40),
                            cv.FONT_HERSHEY_COMPLEX,
                            0.5,
                            (0, 0, 0),
                        )
                        if(not os.path.exists("no Occlusion/")):
                            os.makedirs("no Occlusion/")
                            pass
                        cv.imwrite(f"no Occlusion/{random.Random().randint(0,10000)}.jpg",rawImage)
                else:
                    tips.append(None)
                    filTips.append(None)
                    # image = cv.circle(image,p2,5,color=(0,0,0),thickness=-1
                stop = time.time()
                try:
                    if f == -1:
                        f = image
                except:
                    pass
                cv.putText(
                    image,
                    f"Computation Time (ms) {(stop - start) * 1000:0.0f} FPS {1/(stop-start):0.3f}",
                    (20, 20),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                )

                cv.putText(
                    image,
                    f"Semgent Time (ms) {(segmentTime - start) * 1000 : 0.0f} FPS {1/(segmentTime-start):0.3f}",
                    (20, 60),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                )
                if(modelTime != 0):
                    cv.putText(
                        image,
                        f"Model Time (ms) {(modelTime) * 1000 : 0.0f} FPS {1/(modelTime):0.3f}",
                        (20, 80),
                        cv.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 0),
                    )
                if(processTime !=0):

                    cv.putText(
                        image,
                        f"Process Time (ms) {(processTime) * 1000 : 0.0f} FPS {1/(processTime):0.3f}",
                        (20, 100),
                        cv.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 0),
                    )
                # cv.imshow('frame',image)
                timeSeries = np.append(timeSeries, float(i * 1 / 25))
                self.saver.writer.write(image)
                # a = cv.waitKey(25)
                # if(a == ord('X')):
                #     break
                i = i + 1
                cv.waitKey(0)
            else:
                self.cap.release()
        endVideoTime = time.time()
        self.cap.release()


        # cv.destroyAllWindows()
        # if(not os.path.exists("BADS\\")):
        #     os.makedirs("BADS\\")
        #     pass
        # index = len(glob.glob("BADS\\*")) + 1
        # if(not os.path.exists(f"BADS\\{index}\\")):
        #     os.makedirs(f"BADS\\{index}\\")
        #     pass
        # for i,bad in enumerate(badFrames):
        #     print(f"{i} {len(badFrames)}in Bads")
        #     cv.imwrite(f"BADS\\{index}\\{i}.jpg",bad)
        #     pass
        tips = self.correctTips(tips)
        needles = self.correct(needles)
        eyes = self.correct(eyes)
        print(f"Eye {len(eyes)} Needle {len(needles)}")
        relativeNeedle = []
        for e,n in zip(eyes,needles):
            eCenter = e["Center"]
            nCenter = n["Center"]
            n["Center"] = [nCenter[0] - eCenter[0],nCenter[1] - eCenter[1]]
            relativeNeedle.append(n)
            pass
        self.saver.writer.write(image)
        self.saver.writer.release()
        print(f"Eyes {eyes}")
        return eyes, needles, relativeNeedle, f, timeSeries,endVideoTime-startVideoTime,badFrames,tips
    def segment(self):

        pass
    def correct(self,eyes):
        fuckedUp = []
        for i,n in enumerate(eyes):
            if(type(n) == type(None) ):
                fuckedUp.append(i)
                pass
        for i in fuckedUp:

            a = self.predictBackwardForward(eyes,i)
            eyes[i] = a
        return eyes
    def predictBackwardForward(self,eyes,i):
        forwardIndex = i+1
        backwardIndex = i-1
        forward = True
        backward = True
        while(forward or backward):
            endForward = forwardIndex >= len(eyes)
            endBackward = backwardIndex <0
            if(endForward):
                forward = False
                forNeedle = eyes[-1]
            else:
                forNeedle = eyes[forwardIndex]
            if(endBackward):
                backward = False
                backNeedle = eyes[0]
            else:
                backNeedle = eyes[backwardIndex]
            if(type(forNeedle)!= type(None) and type(backNeedle) != type(None)):
                a = self.computeEye(forNeedle,backNeedle)
                return a
            if(type(forNeedle) == type(None) and endForward and type(backNeedle) != type(None)):
                return backNeedle
            if (type(backNeedle) == type(None) and endBackward and type(forNeedle) != type(None)):
                return forNeedle
            forwardIndex  =  forwardIndex  + 1
            backwardIndex = backwardIndex  - 1
            pass
        pass
    def computeEye(self,forEye,backEye):
        # {
    #                 "Name": ("Eye" if classes.cpu().numpy()[i] == 0 else "Needle"),
    #                 "Conf": f"{confs.cpu().numpy()[i]}",
    #                 "Center": center,
    #                 "BB":bbs[i].cpu().numpy(),
    #                 "Mask":masksArray[i]
    #             }
        a = {}
        a["Name"] = forEye["Name"]
        a["Conf"] = (float(forEye["Conf"]) + float(backEye["Conf"]))/2
        a["Center"] = [(float(backEye["Center"][0]) + float(forEye["Center"][0]))/2
                       ,(float(backEye["Center"][1]) + float(backEye["Center"][1]))/2]
        a["BB"] = (forEye["BB"] + backEye["BB"])/2
        a["Mask"] = forEye["Mask"]
        return a
    def correctTips(self,tips,b = False):
        fuckedUp = []
        for i,n in enumerate(tips):
            if(type(n) == type(None) ):
                fuckedUp.append(i)
                pass
        for i in fuckedUp:

            a = self.predictTips(tips,i)
            tips[i] = a
        return tips
    def predictTips(self,needles,i):
        forwardIndex = i+1
        backwardIndex = i-1
        forward = True
        backward = True
        while(forward or backward):
            endForward = forwardIndex >= len(needles)
            endBackward = backwardIndex <0
            if(endForward):
                forward = False
                forNeedle = needles[-1]
            else:
                forNeedle = needles[forwardIndex]
            if(endBackward):
                backward = False
                backNeedle = needles[0]
            else:
                backNeedle = needles[backwardIndex]
            if(type(forNeedle)!= type(None) and type(backNeedle) != type(None)):
                a = self.computeNeedle(forNeedle,backNeedle)
                return a
            if(type(forNeedle) == type(None) and endForward and not endBackward and backNeedle!= None):
                return backNeedle
            if (type(backNeedle) == type(None) and endBackward and not endForward and forNeedle!= None):
                return forNeedle
            elif(endBackward and endForward):
                return [0,0]
            forwardIndex = forwardIndex + 1
            backwardIndex = backwardIndex  -1
            pass
        pass
    def computeNeedle(self,a,b):
        c = [0,0]
        c[0] = a[0] + b[0]
        c[0] =int( c[0]/2)
        c[1] = a[1] + b[1]
        c[1] =int (c[1]/2)
        # print(f"Computing needles {bb1} {bb2} {mask1} {mask2} {center1} {center2}")
        return c

class YoloV8Segmenter(EyeSegmenter):
    pathToVideo: str
    saver: VideoSaver
    def __init__(self, path,model, videoSaver) -> None:
        self.model = model
        super().__init__(path, videoSaver)
    def segment(self, image):
        startTime = time.time()
        image = cv.resize(image, (640, 640))
        results = self.model(image, stream=False)

        modelTime = time.time()
        data = []
        for r in results:
            if(r!=None):
                masks = r.masks  # Masks object for segmentation masks outputs
                # probs = result.probs
                if(masks!=None):
                    confs = r.boxes.conf
                    classes = r.boxes.cls
                    bbss = r.boxes.xyxy
                    xy = masks.xy
                    centers = []
                    bbs = []
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    image = np.asarray(Image.fromarray(im_array[..., ::-1]))
                    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                    masksArray = []
                    for cordinates,bb in zip(xy,bbss):
                        i = 0
                        xCenter = 0
                        yCenter = 0
                        mask = []
                        for x, y in cordinates:
                            mask.append([x,y])
                            xCenter = xCenter + x
                            yCenter = yCenter + y
                            i = i + 1
                        xCenter = xCenter / i
                        yCenter = yCenter / i
                        centers.append([int(xCenter), int(yCenter)])
                        bbs.append(bb)
                        masksArray.append(mask)
                    data = []
                    for i, center in enumerate(centers):
                        data.append(
                            {
                                "Name": ("Eye" if classes.cpu().numpy()[i] == 0 else "Needle"),
                                "Conf": f"{confs.cpu().numpy()[i]}",
                                "Center": center,
                                "BB":bbs[i].cpu().numpy(),
                                "Mask":masksArray[i]
                            }
                        )
                    continue

        processTime = time.time()
        print(f"Start time {startTime} {modelTime} {processTime}")
        return image, data, processTime - startTime,modelTime - startTime
class YoloV8NeedleTipSegmenter(EyeSegmenter):
    pathToVideo: str
    saver: VideoSaver
    def __init__(self, path,model, videoSaver) -> None:
        self.model = model
        super().__init__(path, videoSaver)
    def segment(self, image):
        image = cv.resize(image, (640, 640))
        results = self.model(image, stream=True)
        data = []
        for r in results:
            masks = r.masks  # Masks object for segmentation masks outputs
            # probs = result.probs
            confs = r.boxes.conf
            classes = r.boxes.cls
            bbss = r.boxes.xyxy
            bbs = []
            xy = masks.xy
            centers = []
            masks = []
            im_array = r.plot()  # plot a BGR numpy array of predictions
            image = np.asarray(Image.fromarray(im_array[..., ::-1]))
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            for cordinates,bb in zip(xy,bbss):
                i = 0
                xCenter = 0
                yCenter = 0
                mask = []
                for x, y in cordinates:
                    mask.append([x,y])
                    xCenter = xCenter + x
                    yCenter = yCenter + y
                    i = i + 1
                xCenter = xCenter / i
                yCenter = yCenter / i
                centers.append([int(xCenter), int(yCenter)])
                bbs.append(bb)
                masks.append(mask)

            # # Line thickness of 2 px
            # thickness = 2
            # color = self.randomColor(xy)
            # # print(color)
            # for pts,c in zip(xy,color):
            #     pts = pts.astype(np.int32)
            #     image = cv.polylines(image,[pts],True,c,thickness)

            data = []
            for i, center in enumerate(centers):
                data.append(
                    {
                        "Name": ("Eye" if classes.cpu().numpy()[i] == 0 else "Needle"),
                        "Conf": f"{confs.cpu().numpy()[i]}",
                        "Center": center,
                        "BB":bbs[i],
                        "Mask":masks[i]
                    }
                )
        return image, data,1,1

    pass

class YoloV8ImageSegmenter(ImageSegmenter):
    pathToImage: str
    savePath:str
    def __init__(self,model,imagePath:str,savePath:str) -> None:
        self.pathToImage = imagePath
        self.savePath = savePath
        self.model = model
        pass
    pass
    def segment(self, image):

        image = cv.resize(image, (640, 640))
        startTime = time.perf_counter()
        results = self.model(image, stream=True)

        # torch.cuda.synchronize()
        # print(results)
        modelTime = time.perf_counter()
        a = modelTime - startTime
        data = []
        for r in results:
            masks = r.masks  # Masks object for segmentation masks outputs
            # probs = result.probs
            confs = r.boxes.conf
            classes = r.boxes.cls
            bbss = r.boxes.xyxy
            xy = masks.xy
            centers = []
            bbs = []
            im_array = r.plot()  # plot a BGR numpy array of predictions
            image = np.asarray(Image.fromarray(im_array[..., ::-1]))
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            masksArray = []
            for cordinates,bb in zip(xy,bbss):
                i = 0
                xCenter = 0
                yCenter = 0
                mask = []
                for x, y in cordinates:
                    mask.append([x,y])
                    xCenter = xCenter + x
                    yCenter = yCenter + y
                    i = i + 1
                xCenter = xCenter / i
                yCenter = yCenter / i
                centers.append([int(xCenter), int(yCenter)])
                bbs.append(bb)
                masksArray.append(mask)
            # # Line thickness of 2 px
            # thickness = 2
            # color = self.randomColor(xy)
            # # print(color)
            # for pts,c in zip(xy,color):
            #     pts = pts.astype(np.int32)
            #     image = cv.polylines(image,[pts],True,c,thickness)

            data = []
            for i, center in enumerate(centers):
                data.append(
                    {
                        "Name": ("Eye" if classes.cpu().numpy()[i] == 0 else "Needle"),
                        "Conf": f"{confs.cpu().numpy()[i]}",
                        "Center": center,
                        "BB":bbs[i].cpu().numpy(),
                        "Mask":masksArray[i]
                    }
                )

class MamadSegmenter(PromptingSegmenter):
    def __init__(self,path,videoSaver,promptTime,promptMask)->None:
        self.promptMask=json.loads(promptMask)
        print(f"Prompt mask for segmenting in dictionary {self.promptMask}")
        a = {}
        for k in self.promptMask:
            int_v = []
            v = self.promptMask[k]
            for u in v:
                int_v.append([float(u[0]),float(u[1])])
            a[int(k)+1] = np.array(int_v)
        self.promptMask = a
        self.is_init_frame=True
        self.num_obj=len(self.promptMask)
        # self.processor.set_all_labels(list(range(1, self.num_obj+1)))
        super().__init__(path,videoSaver,promptTime)
        config = {
            'top_k': 30,
            'mem_every': 5,
            'deep_update_every': -1,
            'enable_long_term': True,
            'enable_long_term_count_usage': True,
            'num_prototypes': 128,
            'min_mid_term_frames': 5,
            'max_mid_term_frames': 10,
            'max_long_term_elements': 10000,
        }
        self.device = "cuda"
        network = XMem(config, './saves/XMem.pth', map_location=self.device).to(self.device).eval()
        self.processor = InferenceCore(network, config=config)
        self.processor.set_all_labels(list(range(1, self.num_obj+1)))
        self.i = 0
    def segment(self, image):
        # torch.cuda.empty_cache()
        # self.processor.clear_memory()
        self.i = self.i + 1
        image=cv.resize(image,(640,640))
        h,w,_=image.shape

        # for k in self.promptMask:
        #     v = self.promptMask[k]
        #     for u in v:
        #         cv.circle(image,u,radius=2,thickness=-1,color=(255,255,0))
        # cv.imshow("First Prompt",image)
        # cv.waitKey(0)

        image_torch,_= image_to_torch(image, self.device)
        with torch.no_grad():
            if self.is_init_frame==True:
                a = {}
                print(f"Prompt mask for segmenting in dictionary {self.promptMask}")
                for k in self.promptMask:
                    int_v = []
                    v = self.promptMask[k]
                    for u in v:
                        int_v.append([int(float(u[0]) * w),int(float(u[1]) * h)])
                    a[int(k)] = np.array(int_v)
                self.promptMask = a
                self.processor.clear_memory()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                # print('object number',self.num_obj)
                mask_torch=contour_to_one_hot_torch(self.promptMask,self.num_obj+1,h,w).to(self.device)
                # print('mask_torch_shape=',np.shape(mask_torch))
                pred = self.processor.step(image_torch, mask_torch[1:])
                self.is_init_frame=False
                # self.promptMask=None
                del mask_torch
            else:
                pred=self.processor.step(image_torch)

        current_contour=torch_prob_to_contour(pred,1e-1)
        del pred
        data=[]
        for class_num in current_contour:
            mask = np.zeros_like(image)
            cv.fillPoly(mask,[current_contour[class_num]],color_map[int(class_num)])
            alpha = 0.5  # Transparency factor
            image = cv.addWeighted(image, 1 - alpha, mask, alpha, 0)
            i = 0
            xCenter = 0
            yCenter = 0
            for point in current_contour[class_num]:
                xCenter = xCenter + point[0]
                yCenter = yCenter + point[1]
                i = i + 1
            xCenter = xCenter / i
            yCenter = yCenter / i

            data.append(
                            {
                                "Name": str(class_num),
                                "Conf": "1",
                                "Center":[int(xCenter), int(yCenter)],
                                "Mask":current_contour[class_num]
                            }
                        )
        torch.cuda.empty_cache()

        # del image_torch
        # del pred
        # time.sleep(0.5)
        image=cv.resize(image,(640,640))
        return image,data

class NineClassSegmenter(EyeSegmenter):
    pathToVideo: str
    saver: VideoSaver
    def __init__(self, path,model, videoSaver) -> None:
        self.model = model
        super().__init__(path, videoSaver)
    def segment(self, image):
        startTime = time.time()
        image = cv.resize(image, (640, 640))
        results = self.model(image, stream=False)
        names = self.model.names
        modelTime = time.time()
        data = []
        for r in results:

            if(r!=None):
                # r.show()
                masks = r.masks  # Masks object for segmentation masks outputs
                # probs = result.probs
                if(masks!=None):
                    confs = r.boxes.conf
                    classes = r.boxes.cls
                    bbss = r.boxes.xyxy
                    xy = masks.xy
                    centers = []
                    bbs = []
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    image = np.asarray(Image.fromarray(im_array[..., ::-1]))
                    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                    # cv.imshow("Test",image)
                    cv.waitKey(0)
                    masksArray = []
                    for cordinates,bb in zip(xy,bbss):

                        i = 0
                        xCenter = 0
                        yCenter = 0
                        mask = []
                        for x, y in cordinates:
                            mask.append([x,y])
                            xCenter = xCenter + x
                            yCenter = yCenter + y
                            i = i + 1
                        xCenter = xCenter / i
                        yCenter = yCenter / i
                        centers.append([int(xCenter), int(yCenter)])
                        bbs.append(bb)
                        masksArray.append(mask)
                    data = []
                    for i, center in enumerate(centers):
                        name = names[classes.cpu().numpy()[i]]
                        print(f"Detected name {name}")
                        if(name == "Cornea" or name == "Instrument" or name == "Cap-Forceps"):

                            center = (int((bbs[i].cpu().numpy()[0] + bbs[i].cpu().numpy()[2])/2),int((bbs[i].cpu().numpy()[1] + bbs[i].cpu().numpy()[3])/2))
                            print(f"{name} BB {bbs[i].cpu().numpy()} Center: {center}")
                            # time.sleep(100000)
                            data.append(
                                {
                                    "Name": ("Eye" if name == "Cornea" else "Needle"),
                                    "Conf": f"{confs.cpu().numpy()[i]}",
                                    "Center": center,
                                    "BB":bbs[i].cpu().numpy(),
                                    "Mask":masksArray[i]
                                }
                            )
        processTime = time.time()
        print(f"Start time {startTime} {modelTime} {processTime}")
        return image, data, processTime - startTime,modelTime - startTime

class MultiEyeSegmenter(VideoSegmenter):
    pathToVideo: str
    saver: VideoSaver
    def __init__(self, path, videoSaver) -> None:
        self.pathToVideo = path
        # self.model = model
        self.saver = videoSaver
        self.cap = cv.VideoCapture(self.pathToVideo)
        pass
    def printLineForObject(self,object,image,color):
        preCenter = None
        for o in object:
            center = o["Center"]
            if(preCenter == None):
                preCenter = center
                image = cv.circle(image,center,3,(255,0,0),-1)
            else:
                image = cv.line(image,center,preCenter,color,2)
                preCenter = center
            pass
        return image
    def fitLineToNeedle(self,img,needle):
        mask = needle["Mask"]
        center = needle["Center"]
        [vx,vy,x,y] = cv.fitLine(np.array(mask),cv.DIST_L2, 0, 0.01, 0.01)

        center2 = np.array(center) + np.array([int(200*vx[0]),int(200*vy[0])])
        p2 = np.array([int(x[0]),int(y[0])]) + np.array([int(-20000*vx[0]),int(-20000*vy[0])])
        p1 = np.array([int(x[0]),int(y[0])]) + np.array([int(+20000*vx[0]),int(+20000*vy[0])])
        # print(f"vx {vx} vy {vy} x {x} y {y} center {center} center2 {center2} p2 {p2}")
        # img = cv.line(img, p2,  p1, (150, 255, 150), 2)
        # img = cv.line(img, center2, np.array(center), (255, 0, 0), 2)
        # img = cv.line(img,ellipse, (0,0,255), 3)
        # let lefty = Math.round((-x * vy / vx) + y);
        # let righty = Math.round(((src.cols - x) * vy / vx) + y);
        # let point1 = new cv.Point(src.cols - 1, righty);
        # let point2 = new cv.Point(0, lefty);
        # t0 = (0-fit_line[3])/fit_line[1]
        # t1 = (img.shape[0]-fit_line[3])/fit_line[1]
        # lefty =    int( (-x * vy / vx) +y)
        # righty =    int( ((src.cols - x) * vy / vx) + y)
        # point1 =    int( (-x * vy / vx) +y)
        # point2 =    int( (-x * vy / vx) +y)
        # # plug into the line formula to find the two endpoints, p0 and p1
        # # to plot, we need pixel locations so convert to int
        # p0 = (fit_line[2:4] + (t0 * fit_line[0:2])).astype(np.uint32)
        # p1 = (fit_line[2:4] + (t1 * fit_line[0:2])).astype(np.uint32)
        # cv.line(img, tuple(p0.ravel()), tuple(p1.ravel()), (0, 255, 0), 10)
        return img,vx[0],vy[0]
    def projectOnline(self,point,vx,vy,p0):
        x = p0[0]*vy**2 + point[0]*vx**2 + vx*vy*(point[1] - p0[1])
        y = (x-point[0])*(1/-vy)*(vx) + point[1]
        return int(x),int(y)
    def calculatePoint(self,vx,vy,needle,eye,img):
        mask = needle["Mask"]
        center = eye["Center"]
        # print(f"Needle mask {mask}")
        if(vx*vy<=0):
            min_max = [np.inf,0]
            max_min = [0,np.inf]
            for x,y in mask:
                # img = cv.circle(img,(int(x),int(y)),5,color=(0,0,0),thickness=-1)
                if(x<min_max[0]):
                    min_max[0] = int(x)
                if(y>min_max[1]):
                    min_max[1] = int(y)
                if(x>max_min[0]):
                    max_min[0] = int(x)
                if(y<max_min[1]):
                    max_min[1] = int(y)
            dist1 = np.linalg.norm(np.array(min_max)-np.array(center))
            dist2 = np.linalg.norm(np.array(max_min)-np.array(center))
            if(dist1<dist2):
                return min_max,max_min
            else:
                return max_min,min_max
            # return min_max,max_min,img
        else:
            max_max = [0,0]
            min_min = [np.inf,np.inf]
            for x,y in mask:
                # img = cv.circle(img,(int(x),int(y)),5,color=(0,0,0),thickness=-1)
                if(x<min_min[0]):
                    min_min[0] = int(x)
                if(y<min_min[1]):
                    min_min[1] = int(y)
                if(x>max_max[0]):
                    max_max[0] = int(x)
                if(y>max_max[1]):
                    max_max[1] = int(y)
        # el_center = ellipsis[0]
            dist1 = np.linalg.norm(np.array(max_max)-np.array(center))
            dist2 = np.linalg.norm(np.array(min_min)-np.array(center))
            if(dist1<dist2):
                return max_max,min_min
            else:
                return min_min,max_max
        pass

    def segmentVideo(self):
        f = -1
        i = 0
        objDicts = {}
        colors = {}
        tips = []
        timeSeries = np.array([])
        startVideoTime = time.time()
        maxLength = 0
        while self.cap.isOpened():
            ret, rawImage = self.cap.read()

            if ret:
                start = time.time()
                image, data,processTime,modelTime = self.segment(rawImage)
                print(f"Model time {modelTime}")
                segmentTime = time.time()
                objDict = {}
                for d in data:
                    name = d["Name"]
                    if(name in objDict):
                        if(float(d["Conf"]) > float(objDict[name]["Conf"])):
                            objDict[name] = d
                    else:
                        objDict[name] = d

                for k in objDict:
                    print("Object name: ",k)
                    if(k == "I-A Handpiece"):
                        print("PrintLine for: ",k)
                        if("Pupil" in objDict):
                            image,vx,vy = self.fitLineToNeedle(image,objDict[k])
                            print("Calculating end and Start: ",k)
                            tip ,end = self.calculatePoint(vx,vy,objDict[k],objDict["Pupil"],image)
                            # tip = self.projectOnline(tip,vx,vy,objDict[k]["Center"])
                            # end = self.projectOnline(end,vx,vy,objDict[k]["Center"])
                            objDict[k]["Center"] = tip
                            image = cv.circle(image,tip,3,color=(0,0,0),thickness=-1)
                            image = cv.circle(image,end,3,color=(0,0,0),thickness=-1)
                            pass
                    if(k in objDicts):
                        objDicts[k].append(objDict[k])

                    else:
                        objDicts[k] = [objDict[k]]
                        colors[k] = (int(random.random() * 255),int(random.random() * 255),int(random.random() * 255))
                    if(k == "I-A Handpiece"):
                        image = self.printLineForObject(objDicts[k],image,(0,0,0))



                # if eyeDict != None:
                # eyes.append(eyeDict)
                # # if needleDict != None:
                # needles.append(needleDict)
                # if(needleDict == None):
                #     badFrames.append(i)
                # else:

                    # if(initalNeedle != None):
                    #     image = cv.line(image,(initalNeedle["Center"]),needleDict["Center"],color=(0,0,255),thickness=1)
                    # initalNeedle = needleDict
                # relativeNeedle = []
                # for e,n in zip(eyes,needles):
                #     eCenter = e["Center"]
                #     nCenter = n["Center"]
                #     n["Center"] = [nCenter[0] - eCenter[0],nCenter[1] - eCenter[1]]
                #     relativeNeedle.append(needleDict)
                #     pass
                # image = self.printLines(image, needles)

                # image,ellipsis = self.fitElipseToIris(image,eyeDict)

                # if(needleDict!=None and eyeDict != None):
                #     ntip,end= self.calculatePoint(vx,vy,needleDict,eyeDict,image)
                #     tip = self.projectOnline(tip,vx,vy,needleDict["Center"])
                #     end = self.projectOnline(end,vx,vy,needleDict["Center"])
                #     lenght = (tip[0] - end[0]) ** 2 + (tip[1] - end[1]) ** 2
                #     print(f"Length { (tip[0] - end[0]) ** 2} {(tip[1] - end[1]) ** 2} {lenght} {lenght**0.5} {maxLength}")
                #     lenght = int(lenght ** 0.5)
                #     notOcc = True
                #     if(lenght > maxLength):
                #         maxLegth = lenght
                #     else:
                #         if(len(lenghts) > 0):
                #             notOcc = self.occlustionDetection(lenghts[-1],lenght)
                #             if(not notOcc):

                #                 lenght = lenghts[-1]
                #     newTip1 = np.array(end) + np.array([int(lenght*vx), int(lenght*vy)])
                #     newTip2 = np.array(end) - np.array([int(lenght*vx), int(lenght*vy)])
                #     lenghts.append(lenght)
                #     tips.append(tip)

                #     # image = self.printLinesForTip(image,tips)
                #     if(len(lenghts)>1):
                #         l = np.mean(lenghts[-5:])
                #     else:
                #         l = np.mean(lenghts)
                #     image ,filteredTip= self.fitCircleToTips(image,tips,eyes,vx,vy,end,l)
                #     filTips.append(filteredTip)
                #     print(f"Tip  {newTip1} {tip} {end} {lenght} {vx} {vy}")
                #     # image = cv.circle(image,newTip1,5,color=(0,255,0),thickness=-1)
                #     image = cv.circle(image,tip,5,color=(0,255,0),thickness=-1)
                #     # image = cv.circle(image,tip,5,color=(0,0,0),thickness=-1)
                #     image = cv.circle(image,end,5,color=(0,0,0),thickness=-1)
                #     if(not notOcc):
                #         image = cv.putText(
                #             image,
                #             f"Occlusion : {lenght} {lenghts[-2]}",
                #             (20, 40),
                #             cv.FONT_HERSHEY_COMPLEX,
                #             0.5,
                #             (0, 0, 0),
                #         )
                #         if(not os.path.exists("Occlusion/")):
                #             os.makedirs("Occlusion/")
                #             pass
                #         cv.imwrite(f"Occlusion/{random.Random().randint(0,10000)}.jpg",rawImage)
                #     else:
                #         image = cv.putText(
                #             image,
                #             f"{lenght}",
                #             (20, 40),
                #             cv.FONT_HERSHEY_COMPLEX,
                #             0.5,
                #             (0, 0, 0),
                #         )
                #         if(not os.path.exists("no Occlusion/")):
                #             os.makedirs("no Occlusion/")
                #             pass
                #         cv.imwrite(f"no Occlusion/{random.Random().randint(0,10000)}.jpg",rawImage)
                # else:
                #     tips.append(None)
                #     filTips.append(None)
                    # image = cv.circle(image,p2,5,color=(0,0,0),thickness=-1)
                # ((295.416259765625, 346.8917236328125), (481.7784423828125, 497.65838623046875), 125.09139251708984)
                # center_coordinates, axesLength, angle, startAngle, endAngle
                size = np.shape(image)

                # for d in data:
                #     image = cv.circle(image, center=d["Center"], radius=2, color=(0, 0, 255), thickness=-1)

                stop = time.time()
                try:
                    if f == -1:
                        f = image
                except:
                    pass
                cv.putText(
                    image,
                    f"Computation Time (ms) {(stop - start) * 1000:0.0f} FPS {1/(stop-start):0.3f}",
                    (20, 20),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                )

                cv.putText(
                    image,
                    f"Semgent Time (ms) {(segmentTime - start) * 1000 : 0.0f} FPS {1/(segmentTime-start):0.3f}",
                    (20, 60),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                )
                if(modelTime != 0):
                    cv.putText(
                        image,
                        f"Model Time (ms) {(modelTime) * 1000 : 0.0f} FPS {1/(modelTime):0.3f}",
                        (20, 80),
                        cv.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 0),
                    )
                if(processTime !=0):

                    cv.putText(
                        image,
                        f"Process Time (ms) {(processTime) * 1000 : 0.0f} FPS {1/(processTime):0.3f}",
                        (20, 100),
                        cv.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 0),
                    )
                # cv.imshow('frame',image)
                timeSeries = np.append(timeSeries, float(i * 1 / 25))
                self.saver.writer.write(image)
                # a = cv.waitKey(25)
                # if(a == ord('X')):
                #     break
                i = i + 1
            else:
                self.cap.release()
        endVideoTime = time.time()
        self.cap.release()


        # cv.destroyAllWindows()
        # if(not os.path.exists("BADS\\")):
        #     os.makedirs("BADS\\")
        #     pass
        # index = len(glob.glob("BADS\\*")) + 1
        # if(not os.path.exists(f"BADS\\{index}\\")):
        #     os.makedirs(f"BADS\\{index}\\")
        #     pass
        # for i,bad in enumerate(badFrames):
        #     print(f"{i} {len(badFrames)}in Bads")
        #     cv.imwrite(f"BADS\\{index}\\{i}.jpg",bad)
        #     pass
        return objDicts, f, timeSeries,endVideoTime-startVideoTime

    def segment(self):

        pass

class CatractSegmenter(MultiEyeSegmenter):
    pathToVideo: str
    saver: VideoSaver
    def __init__(self, path,model, videoSaver) -> None:
        self.model = model
        super().__init__(path, videoSaver)
    def segment(self, image):
        image = cv.resize(image, (640, 640))
        results = self.model(image, stream=True)
        data = []
        for r in results:
            masks = r.masks  # Masks object for segmentation masks outputs
            # probs = result.probs
            confs = r.boxes.conf
            classes = r.boxes.cls
            bbss = r.boxes.xyxy
            bbs = []
            xy = masks.xy
            centers = []
            masks = []
            im_array = r.plot()  # plot a BGR numpy array of predictions
            image = np.asarray(Image.fromarray(im_array[..., ::-1]))
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            for cordinates,bb in zip(xy,bbss):
                i = 0
                xCenter = 0
                yCenter = 0
                mask = []
                for x, y in cordinates:
                    mask.append([x,y])
                    xCenter = xCenter + x
                    yCenter = yCenter + y
                    i = i + 1
                xCenter = xCenter / i
                yCenter = yCenter / i
                centers.append([int(xCenter), int(yCenter)])
                bbs.append(bb)
                masks.append(mask)

            # # Line thickness of 2 px
            # thickness = 2
            # color = self.randomColor(xy)
            # # print(color)
            # for pts,c in zip(xy,color):
            #     pts = pts.astype(np.int32)
            #     image = cv.polylines(image,[pts],True,c,thickness)

            data = []
            names = ['Cannula', 'Cap Cystotome', 'Cap Forceps', 'Cornea', 'Forceps', 'I-A Handpiece', 'Lens Injector', 'Phaco Handpiece', 'Primary Knife', 'Pupil', 'Second Instrument', 'Secondary Knife']
            for i, center in enumerate(centers):
                name = ""
                index = classes.cpu().numpy()[i];
                name = names[int(index)]
                data.append(
                    {
                        "Name": name,
                        "Conf": f"{confs.cpu().numpy()[i]}",
                        "Center": center,
                        "BB":bbs[i],
                        "Mask":masks[i]
                    }
                )
        return image, data,1,1

    pass


if __name__ == "__main__":
    print("Testing")
    for item in glob.glob("test\\*.jpg"):
        name = item.split("\\")[-1].split(".")[0]
        newName = name + "_segmented"
        newItem = item.replace(name,newName)
        imageSeg = YoloV8ImageSegmenter(YOLO("best+151.pt"),item,newItem)
        imageSeg.segmentImage()
    # saver = VideoSaver("ID 438 out.mp4")
    # videoSeg = YoloV8Segmenter(
    #     "C:\\Users\\Mosi\\Documents\\GitHub\\Aras-Farabi-v2\\server\\ID 438.mp4",
    #     YOLO(
    #         "C:\\Users\\Mosi\\Documents\\GitHub\\Aras-Farabi-v2\\server\\bestIman.pt",
    #         task="segment",
    #     ),
    #     saver,
    # )
    # eye, needles, relatives, f ,ts= videoSeg.segmentVideo()
    # saver.end()
