import cv2 as cv
import numpy as np
import pandas as pd
import time
IRIS_INDEX = 1
NEEDLE_INDEX = 0
class SegmentResult:
    name:str
    conf:str
    mask:np.ndarray
    boundingBox:np.ndarray
    tip:np.ndarray = None
    def __init__(self,name:str,conf:str,mask:np.ndarray,boundingBox:np.ndarray) -> None:
        self.name = name
        self.conf = conf
        self.mask = mask
        self.boundingBox = boundingBox
        pass

class CalculatorResult:
    name:str
    def __init__(self,name:str) -> None:
        self.name = name
        pass
    pass
class NumericResult(CalculatorResult):
    result:float
    name:str
    def __init__(self,name:str,result:float) -> None:
    
        self.result = result
        super().__init__(name)
    pass
class XYResult(CalculatorResult):
    x:list[float]
    y:list[float]
    name:str
    def __init__(self,name:str,x:list[float],y:list[float]) -> None:
        self.x = x
        self.y = y
        super().__init__(name)
class Calculator:

    def __init__(self) -> None:
        self.done = False
        pass
    def reCalculate(self):
        self.done = False
        self.error = False
        self.go()
        pass
    def preCalculate(self):
        self.startTime = time.time()
        pass
    def afterCalculate(self):
        self.endTime = time.time()
        pass
    def calculate(self) -> list[CalculatorResult]:
        pass
    def setResult(self,a):
        self.result = a
    def getResult(self):
        return self.result
    def go(self):
        if(not self.done):
            self.start()
        if(not self.error):
           return self.getResult()
        return None
    def start(self):
        self.done =False
        try:
            self.preCalculate()
            self.setResult(self.calculate())
            self.afterCalculate()
            self.done = True
        except:
            self.done = True
            self.error = True
        pass


class ResultFilter:
    def filterResult(self,result:list[SegmentResult]) -> list[SegmentResult]:
        return result

class RealTimeVideoSegmenter:
    handler : ResultFilter | None
    def __init__(self,handler:ResultFilter|None = None) -> None:
        self.handler = handler
        pass
    def nextImage(self) -> np.ndarray|None:
        pass
    def segmentImage(self,image:np.ndarray) -> list[SegmentResult]:
        pass
    def setResult(self,result):
        self.result = result
        pass
    def calculateOverResult(self,result):
        return result
    def getResult(self):
        return self.result
    def segmentVideo(self):
        image = self.nextImage()
        if(type(image) != type(None)):
            result = self.segmentImage(image)
            if(self.handler != None):
                result = self.handler.filterResult(result)
                result = self.calculateOverResult(result)
                self.setResult(result)
                pass
            pass
        pass
class PositionCalculator(Calculator):
    positions: np.ndarray
    def __init__(self,) -> None:
        self.positions = None
        super().__init__()
    def calculate(self,segmentResult:list[SegmentResult]) -> list[CalculatorResult]:
        iris = segmentResult[IRIS_INDEX]
        needle = segmentResult[NEEDLE_INDEX]
        if(needle != None and iris != None):
            # abs(eBB[0] - eBB[2])
            # deltaYEye =abs( eBB[1] - eBB[3])
            xC = iris.boundingBox[0] + iris.boundingBox[2]
            yC = iris.boundingBox[1] + iris.boundingBox[3]
            xC = xC/2
            yC = yC/2
            dx = iris.boundingBox[0] - iris.boundingBox[2]
            dy = iris.boundingBox[1] - iris.boundingBox[3]

            position = np.array([[int((needle.tip[0] - xC)/dx),int((needle[1]-yC)/dy)]])
            if(type(self.positions) != type(None)):
                np.concatenate((self.positions,position))
            else:
                self.positions = position
            return [XYResult("Position",position[:,0],position[:,1])]    
        return None
    pass


class MainRealtTimeVideoSegmenter(RealTimeVideoSegmenter):
    def __init__(self, handler: ResultFilter | None = None) -> None:
        super().__init__(handler)
    def calculateOverResult(self, result):
        
        return super().calculateOverResult(result)
    def segmentVideo(self):
        return super().segmentVideo()