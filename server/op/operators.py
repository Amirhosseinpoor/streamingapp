from ultralytics import YOLO
import cv2 as cv
import numpy as np
import random
from PIL import Image
import time as ti
import torch
from ForAll import FPS
import pandas as pd
import math
b = torch.cuda.is_available()
# a = torch.get_device()
torch.device("cuda:0")
print(f"Torch setted in operators {b}")
XY = 1
TIME_SERIES = 2
NUMERIC = 3
class Operator:
    result: object
    operated: bool = False
    name : str = "Operator"
    type : int = 0
    hide : bool = False
    def __init__(self) -> None:
        self.operator = False
        self.result = None
        # self.name = "Operator"
       
        pass

    def operate(self):
        self.operated = True
        pass
    def timingOperate(self):
        startTime = ti.time()
        self.operate()
        endTime = ti.time()
        self.time = endTime - startTime
        pass
    def getResult(self):
        if self.operated and type(self.result) != type(None):
            return self.result
        else:
            self.timingOperate()
            return self.result
    def getSerlizableResult(self) ->list:
        self.getResult()
        return None
    def do(self):
        self.operated = False
        self.result = None
        self.operate()

class PrimaryOperator(Operator):
    needleCenters: np.ndarray
    eyeCenters: np.ndarray
    needleBB: np.ndarray
    eyeBB: np.ndarray
    tips:list
    time:float
    def __init__(self, needleCenters: np.ndarray,
                  eyeCenters: np.ndarray,
                  needleBB:np.ndarray,eyeBB:np.ndarray,tips:list,time:float) -> None:
        self.needleCenters = needleCenters
        self.eyeCenters = eyeCenters
        self.needleBB = needleBB
        self.eyeBB = eyeBB
        self.time = time
        self.tips = tips
        super().__init__()
        
        pass

    def operate(self):
        return super().operate()

class SecondryOperator(Operator):
    operators: tuple[Operator]

    def __init__(self, *operator: tuple[Operator]) -> None:
        self.operators = operator
        super().__init__()

    def operate(self):
        return super().operate()

    pass

class LoggerOperator(SecondryOperator):
    operators: tuple[Operator]
    path:str
    def __init__(self,path:str, *operator: tuple[Operator]) -> None:
        self.operators = operator
        self.path = path
        super().__init__(*operator)

    def operate(self):
        return super().operate()
    def getSerlizableResult(self) -> list:
        return super().getSerlizableResult()
    pass

class RelativeVelocity(SecondryOperator):
    hide = False
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Relative Velocity"
        self.type = TIME_SERIES
        super().__init__(*operator)
    def getSerlizableResult(self) -> list:
        result = self.getResult()
        serlizebleList = []
        x = {}
        y = {}
        absData = {}
        v_x = []
        v_y = []
        absV = []
        for v in result:
            vx = v[0]
            vy = v[1]
            # print(vx,vy)
            
            # print(vx,vy)
            
            absV.append(f"{math.sqrt(vx**2 + vy**2):0.3f}")
            vx = f"{vx:0.3f}"
            vy = f"{vy:0.3f}"
            v_x.append(vx)
            v_y.append(vy)
            # time.sleep(-1)
        absData["Name"] = "1:0:Speed"
        absData["Type"] = TIME_SERIES
        absData["Data"] = absV
        x["Name"] = "1:1:Velocity X"
        x["Type"] = TIME_SERIES
        y["Name"] = "1:2:Velocity Y"
        y["Type"] = TIME_SERIES
        x["Data"] = v_x
        y["Data"] = v_y
        serlizebleList.append(absData)
        serlizebleList.append(x)
        serlizebleList.append(y)
        
        return serlizebleList
    def operate(self):
        result = self.operators[0].getResult()
        preX = None
        preY = None
        lastV  = None
        print(f"In relative velocity {len(result)}")
        for pos in result:
            x = pos[0]
            y = pos[1]
            if preX == None and preY == None:
                preX = x
                preY = y
                velocity = np.zeros(shape=(1, 2), dtype=np.float16)
            else:
                v_x = (x - preX) * FPS
                v_y = (y - preY) * FPS
                preX = x
                preY = y
                if(v_x != None and v_y != None):
                    lastV = [v_x,v_y]
                    velocity = np.concatenate(
                        (velocity, np.expand_dims(lastV, axis=0))
                    )
                # velocity = np.append(velocity,[[v_x,v_y]],axis=1)
                elif(lastV != None):
                        velocity = np.concatenate(
                            (velocity, np.expand_dims(lastV, axis=0))
                        )
                else:
                    velocity = np.concatenate(
                            (velocity, np.expand_dims([0.0], axis=0))
                        )
        self.result = velocity
        return super().operate()
    pass

class RelativeAcceleration(SecondryOperator):
    hide = False
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Relative Accelration"
        self.type = TIME_SERIES
        super().__init__(*operator)
    def getSerlizableResult(self) -> list:
        result = self.getResult()
        serlizebleList = []
        x = {}
        y = {}
        absData = {}
        v_x = []
        v_y = []
        absV = []
        for v in result:
            vx = v[0]
            vy = v[1]
            # print(vx,vy)
           
            absV.append(f"{math.sqrt(vx**2 + vy**2):0.3f}")
            vx = f"{vx:0.3f}"
            vy = f"{vy:0.3f}"
            # print(vx,vy)
            v_x.append(vx)
            v_y.append(vy)
            # time.sleep(-1)
        absData["Name"] = "2:0:Acceleration"
        absData["Type"] = TIME_SERIES
        absData["Data"] = absV
        x["Name"] = "2:1:Acceleration X"
        x["Type"] = TIME_SERIES
        y["Name"] = "2:2:Acceleration Y"
        y["Type"] = TIME_SERIES
        x["Data"] = v_x
        y["Data"] = v_y

        serlizebleList.append(absData)
        serlizebleList.append(x)
        serlizebleList.append(y)
        
        return serlizebleList
    def operate(self):
        result = self.operators[0].getResult()
        preX = None
        preY = None
        lastV  = None
        for pos in result:
            x = pos[0]
            y = pos[1]
            if preX == None and preY == None:
                preX = x
                preY = y
                velocity = np.zeros(shape=(1, 2), dtype=np.float16)
            else:
                v_x = (x - preX) * FPS
                v_y = (y - preY) * FPS
                preX = x
                preY = y
                if(v_x != None and v_y != None):
                    lastV = [v_x,v_y]
                    velocity = np.concatenate(
                        (velocity, np.expand_dims(lastV, axis=0))
                    )
                # velocity = np.append(velocity,[[v_x,v_y]],axis=1)
                elif(lastV != None):
                        velocity = np.concatenate(
                            (velocity, np.expand_dims(lastV, axis=0))
                        )
                else:
                    velocity = np.concatenate(
                            (velocity, np.expand_dims([0.0], axis=0))
                        )
        self.result = velocity
        return super().operate()

class RelativeJerk(SecondryOperator):
    hide = False
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Relative Jerk"
        self.type = TIME_SERIES
        super().__init__(*operator)
    def getSerlizableResult(self) -> list:
        result = self.getResult()
        serlizebleList = []
        x = {}
        y = {}
        absData = {}
        v_x = []
        v_y = []
        absV = []
        for v in result:
            vx = v[0]
            vy = v[1]
            # print(vx,vy)
            
            absV.append(f"{math.sqrt(vx**2 + vy**2):0.3f}")
            vx = f"{vx:0.3f}"
            vy = f"{vy:0.3f}"
            # print(vx,vy)
            v_x.append(vx)
            v_y.append(vy)
            # time.sleep(-1)
        absData["Name"] = "3:0:Jerk"
        absData["Type"] = TIME_SERIES
        absData["Data"] = absV
        x["Name"] = "3:1:Jerk X"
        x["Type"] = TIME_SERIES
        y["Name"] = "3:2:Jerk Y"
        y["Type"] = TIME_SERIES
        x["Data"] = v_x
        y["Data"] = v_y
        serlizebleList.append(absData)
        serlizebleList.append(x)
        serlizebleList.append(y)
        
        return serlizebleList
    def operate(self):
        result = self.operators[0].getResult()
        preX = None
        preY = None
        lastV  = None
        for pos in result:
            x = pos[0]
            y = pos[1]
            if preX == None and preY == None:
                preX = x
                preY = y
                velocity = np.zeros(shape=(1, 2), dtype=np.float16)
            else:
                v_x = (x - preX) * FPS
                v_y = (y - preY) * FPS
                preX = x
                preY = y
                if(v_x != None and v_y != None):
                    lastV = [v_x,v_y]
                    velocity = np.concatenate(
                        (velocity, np.expand_dims(lastV, axis=0))
                    )
                # velocity = np.append(velocity,[[v_x,v_y]],axis=1)
                elif(lastV != None):
                        velocity = np.concatenate(
                            (velocity, np.expand_dims(lastV, axis=0))
                        )
                else:
                    velocity = np.concatenate(
                            (velocity, np.expand_dims([0.0], axis=0))
                        )
        self.result = velocity
        return super().operate()

class NeedleVelocityOperator(PrimaryOperator):

    hide = True
    def __init__(self, needleCenters: np.ndarray, eyeCenters: np.ndarray,needleBB:np.ndarray,eyeBB:np.ndarray,tips:list,time:float) -> None:
        self.name = "Velocity"
        self.type = TIME_SERIES
        super().__init__(needleCenters, eyeCenters,needleBB,eyeBB,tips,time)
    def getSerlizableResult(self) -> list:
        result = self.getResult()
        serlizebleList = []
        x = {}
        y = {}
        v_x = []
        v_y = []
        
        for v in result:
            vx = v[0]
            vy = v[1]
            # print(vx,vy)
            vx = f"{vx:0.3f}"
            vy = f"{vy:0.3f}"
            # print(vx,vy)
            v_x.append(vx)
            v_y.append(vy)
            # time.sleep(-1)
        x["Name"] = "Velocity X"
        x["Type"] = TIME_SERIES
        y["Name"] = "Velocity Y"
        y["Type"] = TIME_SERIES
        x["Data"] = v_x
        y["Data"] = v_y
        serlizebleList.append(x)
        serlizebleList.append(y)
        return serlizebleList
    def operate(self):
        velocity = np.array([[]])
        preX = None
        preY = None
        lastV  = None
        for x, y in self.needleCenters:
            if preX == None and preY == None:
                preX = x
                preY = y
                velocity = np.zeros(shape=(1, 2), dtype=np.float16)
            else:
                v_x = (x - preX) * FPS
                v_y = (y - preY) * FPS
                preX = x
                preY = y
                if(v_x != None and v_y != None):
                    lastV = [v_x,v_y]
                    velocity = np.concatenate(
                        (velocity, np.expand_dims(lastV, axis=0))
                    )
                # velocity = np.append(velocity,[[v_x,v_y]],axis=1)
                elif(lastV != None):
                        velocity = np.concatenate(
                            (velocity, np.expand_dims(lastV, axis=0))
                        )
                else:
                    velocity = np.concatenate(
                            (velocity, np.expand_dims([0.0], axis=0))
                        )
        self.result = velocity
        return super().operate()

    pass

class NeedleAccelrationOperation(SecondryOperator):
    hide = True
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Accelartion"
        self.type = TIME_SERIES
        super().__init__(*operator)
    def getSerlizableResult(self):
        result = self.getResult()
        serlizebleList = []
        x = {}
        y = {}
        v_x = []
        v_y = []
        
        for v in result:
            vx = v[0]
            vy = v[1]
            # print(vx,vy)
            vx = f"{vx:0.3f}"
            vy = f"{vy:0.3f}"
            # print(vx,vy)
            v_x.append(vx)
            v_y.append(vy)
            # time.sleep(-1)
        x["Name"] = "Accelartion X"
        x["Type"] = TIME_SERIES
        y["Name"] = "Accelartion Y"
        y["Type"] = TIME_SERIES
        x["Data"] = v_x
        y["Data"] = v_y
        x["Time"] = f"{self.time:0.3f}"
        y["Time"] = f"{self.time:0.3f}"
        serlizebleList.append(x)
        serlizebleList.append(y)
        return serlizebleList
    def operate(self):
        velocityOperator = self.operators[0]
        velocity = velocityOperator.getResult()
        preX = None
        preY = None
        lastV = None
        acc = np.array([[]])
        for x, y in velocity:
            if preX == None and preY == None:
                preX = x
                preY = y
                acc = np.zeros(shape=(1, 2), dtype=np.float16)
            else:
                a_x = (x - preX) * FPS
                a_y = (y - preY) * FPS
                preX = x
                preY = y
                if(a_x != None and a_y != None):
                    lastV = [a_x,a_y]
                    acc = np.concatenate(
                        (acc, np.expand_dims(lastV, axis=0))
                    )
                # velocity = np.append(velocity,[[v_x,v_y]],axis=1)
                elif(lastV != None):
                    acc = np.concatenate(
                        (acc, np.expand_dims(lastV, axis=0))
                    )
                else:
                    acc = np.concatenate(
                            (acc, np.expand_dims([0.0], axis=0))
                        )
        self.result = acc
        return super().operate()

class NeedleJerkOperator(SecondryOperator):
    hide = True
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Jerk"
        self.type = TIME_SERIES
        super().__init__(*operator)
    def getSerlizableResult(self):
        result = self.getResult()
        serlizebleList = []
        x = {}
        y = {}
        v_x = []
        v_y = []
        
        for v in result:
            vx = v[0]
            vy = v[1]
            # print(vx,vy)
            vx = f"{vx:0.3f}"
            vy = f"{vy:0.3f}"
            # print(vx,vy)
            v_x.append(vx)
            v_y.append(vy)
            # time.sleep(-1)
        x["Name"] = "Jerk X"
        x["Type"] = TIME_SERIES
        y["Name"] = "Jerk Y"
        y["Type"] = TIME_SERIES
        x["Data"] = v_x
        y["Data"] = v_y
        x["Time"] = f"{self.time:0.3f}"
        y["Time"] = f"{self.time:0.3f}"
        serlizebleList.append(x)
        serlizebleList.append(y)
        return serlizebleList
    def operate(self):
        velocityOperator = self.operators[0]
        velocity = velocityOperator.getResult()
        preX = None
        preY = None
        lastV = None
        acc = np.array([[]])
        for x, y in velocity:
            if preX == None and preY == None:
                preX = x
                preY = y
                acc = np.zeros(shape=(1, 2), dtype=np.float16)
            else:
                a_x = (x - preX) * FPS
                a_y = (y - preY) * FPS
                preX = x
                preY = y
                if(a_x != None and a_y != None):
                    lastV = [a_x,a_y]
                    acc = np.concatenate(
                        (acc, np.expand_dims(lastV, axis=0))
                    )
                # velocity = np.append(velocity,[[v_x,v_y]],axis=1)
                elif(lastV != None):
                    acc = np.concatenate(
                        (acc, np.expand_dims(lastV, axis=0))
                    )
                else:
                    acc = np.concatenate(
                            (acc, np.expand_dims([0.0], axis=0))
                        )
        self.result = acc
        return super().operate()

class Fluidity(SecondryOperator):
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Fluidity"
        self.type = NUMERIC
        super().__init__(*operator)
    def operate(self):
        velocityOperator = self.operators[0]
        velocity = velocityOperator.getResult()
        fluidity = 0
        for x,y in velocity:
             fluidity += ((x ** 2 + y ** 2) ** .5)
        # for i in range(len(vx) - 1):
        #     fluidity += ((vx[i] ** 2 + vy[i] ** 2) ** .5)

        self.result = fluidity / len(velocity)
        return super().operate()
    def getSerlizableResult(self):
        
        result = self.getResult()
        serlizebleList = []
        x = {}
        x["Name"] = "Motion Fluidity"
        x["Type"] = NUMERIC
        x["Data"] = f"{result:0.3f}"
        x["Time"] = f"{self.time:0.3f}"
        x["Max"] = f"{1000}"
        x["Min"] = f"{0}"
        serlizebleList.append(x)
        return serlizebleList

class Economy(SecondryOperator):
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Economy"
        self.type = NUMERIC
        super().__init__(*operator)
    def operate(self):
        velocityOperator = self.operators[0]
        velocity = velocityOperator.getResult()
        economy = 0
        for x,y in velocity:
            economy += ((x ** 2 + y ** 2))
        # for i in range(len(vx) - 1):
        #     fluidity += ((vx[i] ** 2 + vy[i] ** 2) ** .5)

        self.result = economy
        return super().operate()
    def getSerlizableResult(self):
        result = self.getResult()
        serlizebleList = []
        x = {}
        x["Name"] = "Motion Energy"
        x["Type"] = NUMERIC
        x["Data"] = f"{result:0.3f}"
        x["Time"] = f"{self.time:0.3f}"
        x["Max"] = f"{1000}"
        x["Min"] = f"{0}"
        serlizebleList.append(x)
        return serlizebleList

class VelocitySignChanges(SecondryOperator):
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Velocity Sign Changes"
        self.type = NUMERIC
        super().__init__(*operator)
    def operate(self):
        velocityOperator = self.operators[0]
        velocity = velocityOperator.getResult()
        preX = None
        preY = None
        num = 0
        for x,y in velocity:
            if(preX == None and preY == None):
                preX = x
                preY = y
                pass
            else:
                if x * preX < 0:
                    num += 1
                if y * preY < 0:
                    num += 1
                preX = x
                preY = y
        # for i in range(len(vx) - 1):
        #     fluidity += ((vx[i] ** 2 + vy[i] ** 2) ** .5)

        self.result = num
        return super().operate()
    def getSerlizableResult(self):
        result = self.getResult()
        serlizebleList = []
        x = {}
        x["Name"] = "Back-and-forth Movements"
        x["Type"] = NUMERIC
        x["Data"] = f"{result:0.3f}"
        x["Time"] = f"{self.time:0.3f}"
        x["Max"] = f"{1000}"
        x["Min"] = f"{0}"
        serlizebleList.append(x)
        return serlizebleList

class UnsafeEntrance(SecondryOperator):
    hide : bool = False
    
    def __init__(self,*operator: tuple[Operator]) -> None:
        self.warningThreshould = 0.33
        self.name = "Unsafe Area Entrance"
        super().__init__(*operator)
    def operate(self):
        relative = self.operators[0].getResult()
        num = 0
        for x,y in relative:
            if(abs(x)>self.warningThreshould or abs(y)>self.warningThreshould):
                num += 1
        self.result = num
        return super().operate()
    def getSerlizableResult(self):
        result = self.getResult()
        serlizebleList = []
        x = {}
        x["Name"] = "Unsafe Area Entrance"
        x["Type"] = NUMERIC
        x["Data"] = f"{result:0.3f}"
        x["Time"] = f"{self.time:0.3f}"
        x["Max"] = f"{1000}"
        x["Min"] = f"{0}"
        serlizebleList.append(x)
        return serlizebleList

class RelativeTrajectory(PrimaryOperator):
    hide : bool = True
    def __init__(self, needleCenters: np.ndarray, eyeCenters: np.ndarray,needleBB:np.ndarray,eyeBB:np.ndarray,tips:list,time:float) -> None:
        self.name = "Relative Trajectory"
        self.type = XY
        super().__init__(needleCenters, eyeCenters,needleBB,eyeBB,tips,time)
    def operate(self):
        a = []
        
        for needle,eye in zip(self.tips,self.eyeCenters):
            x,y = needle
            x1,y1 = eye
            a.append([x-x1,y-y1])
            # print(f"Needle centers {x},{y} eye centers {x1},{y1}")
            pass
        self.result =  a
        print(f"In relative Trajectory {len(self.tips)} {len(self.eyeCenters)} {len(a)}")
        return super().operate()
    def getSerlizableResult(self):
        result = self.getResult()
        serlizebleList = []
        ax = []
        ay =[]
        for x,y in self.result:
            ax.append(f"{x:0.3f}")
            ay.append(f"{y:0.3f}")
        a = {}
        a["Name"] = "Relative Trajectory"
        a["Type"] = XY
        a["Data"] = {"X":ax,"Y":ay,"XLabel":"Relative Trj. X","YLabel":"Relative Trj. Y"}
        a["Time"] = f"{self.time:0.3f}"
        # y["Time"] = f"{self.time:0.3f}"
        serlizebleList.append(a)
        return serlizebleList
    pass

class Trajectory(PrimaryOperator):
    hide : bool = False
    def __init__(self, needleCenters: np.ndarray, eyeCenters: np.ndarray,needleBB:np.ndarray,eyeBB:np.ndarray,tips:list,time:float) -> None:
        self.name = "Trajectory"
        self.type = XY
        super().__init__(needleCenters, eyeCenters,needleBB,eyeBB,tips,time)
    def operate(self):
        self.result =  self.tips
        return super().operate()
    def getSerlizableResult(self):
        result = self.getResult()
        serlizebleList = []
        ax = []
        ay =[]
        for x,y in self.result:
            ax.append(f"{x:0.3f}")
            ay.append(f"{y:0.3f}")
        a = {}
        a["Name"] = "0:0:Trajectory"
        a["Type"] = XY
        a["Data"] = {"X":ax,"Y":ay,"XLabel":"Trj. X","YLabel":"Trj. Y"}
        a["Time"] = f"{self.time:0.3f}"
        # y["Time"] = f"{self.time:0.3f}"
        serlizebleList.append(a)
        return serlizebleList

class Iris(PrimaryOperator):
    hide : bool = True
    def __init__(self, needleCenters: np.ndarray, eyeCenters: np.ndarray,needleBB:np.ndarray,eyeBB:np.ndarray,tips:list,time:float) -> None:
        self.name = "Trajectory"
        self.type = XY
        super().__init__(needleCenters, eyeCenters,needleBB,eyeBB,tips,time)
    def operate(self):
        self.result =  self.eyeCenters
        return super().operate()
    def getSerlizableResult(self):
        result = self.getResult()
        serlizebleList = []
        ax = []
        ay =[]
        for x,y in self.result:
            ax.append(f"{x:0.3f}")
            ay.append(f"{y:0.3f}")
        a = {}
        a["Name"] = "Iris"
        a["Type"] = XY
        a["Data"] = {"X":ax,"Y":ay}
        a["Time"] = f"{self.time:0.3f}"
        # y["Time"] = f"{self.time:0.3f}"
        serlizebleList.append(a)
        return serlizebleList

class CurvatureOperator(SecondryOperator):
    hide:bool = False
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Curvature"
        self.type = NUMERIC
        super().__init__(*operator)
    def operate(self):
        velocity = self.operators[0].getResult()
        acc = self.operators[1].getResult()
        curvature_sum = 0
        for (vx,vy),(ax,ay) in zip(velocity,acc):
            denominator_curvature = math.sqrt(vx**2 + vy**2)**3
            curvature_sum += (math.sqrt(vx**2 + vy**2) * math.sqrt(ax**2 + ay**2)) / denominator_curvature if denominator_curvature != 0 else 0
        self.result =curvature_sum
        return super().operate()
    def getSerlizableResult(self):
        
        result = self.getResult()
        serlizebleList = []
        x = {}
        x["Name"] = "Curvature"
        x["Type"] = NUMERIC
        x["Data"] = f"{result:0.3f}"
        x["Time"] = f"{self.time:0.3f}"
        x["Max"] = f"{1000}"
        x["Min"] = f"{0}"
        serlizebleList.append(x)
        return serlizebleList
    pass

class AllTimeOperator(SecondryOperator):
    hide : bool = True
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "All Time"
        self.type = NUMERIC
        super().__init__(*operator)
    def operate(self):
        # velocityOperator = self.operators[0]
        # velocity = velocityOperator.getResult()
        # fluidity = 0
        # for x,y in velocity:
        #      fluidity += ((x ** 2 + y ** 2) ** .5)
        time = 0
        for op in self.operators:
        
            op.getResult()
            print(f"Time {op.time}")
            time +=op.time 
        # for i in range(len(vx) - 1):
        #     fluidity += ((vx[i] ** 2 + vy[i] ** 2) ** .5)

        self.result = time
        return super().operate()
    def getSerlizableResult(self):
        result = self.getResult()
        serlizebleList = []
        x = {}
        x["Name"] = "All Time"
        x["Type"] = NUMERIC
        x["Data"] = f"{result:0.3f}"
        x["Time"] = f"{self.time:0.3f}"
        x["Max"] = f"{1000}"
        x["Min"] = f"{0}"
        serlizebleList.append(x)
        return serlizebleList

class TimeOperator(PrimaryOperator):
    hide : bool = True
    def __init__(self, needleCenters: np.ndarray, eyeCenters: np.ndarray,needleBB:np.ndarray,eyeBB:np.ndarray,tips:list,time:float) -> None:
        self.name = "Time"
        self.type = NUMERIC
        super().__init__(needleCenters, eyeCenters,needleBB,eyeBB,tips,time)
    def operate(self):
        self.result =  self.time
        return super().operate()
    def getSerlizableResult(self):
        result = self.getResult()
        serlizebleList = []
        # ax = []
        # ay =[]
        # for x,y in self.result:
        #     ax.append(f"{x:0.3f}")
        #     ay.append(f"{y:0.3f}")
        a = {}
        a["Name"] = "Time"
        a["Type"] = NUMERIC
        a["Data"] = f"{result:0.3f}"
        a["Time"] = f"{self.time:0.3f}"
        a["Max"] = f"{1000}"
        a["Min"] = f"{0}"
        serlizebleList.append(a)
        return serlizebleList

class SpeedPeakOperator(SecondryOperator):
    hide :bool = False
    def __init__(self ,*operator: tuple[Operator]) -> None:
        self.name = "Speed peak"
        self.type = NUMERIC
        self.percentage_of_peak_speed = 0.5
        super().__init__(*operator)
    def operate(self):
        speed = self.operators[0].getResult()
        peakSpeed = float("-inf")
        time_exceed_percentage_of_peak=0
        for s in speed:
            peakSpeed = max(peakSpeed, s)
            if s > self.percentage_of_peak_speed * peakSpeed:
                time_exceed_percentage_of_peak += 1
        self.result = time_exceed_percentage_of_peak
        return super().operate()

class RationalPositions(PrimaryOperator):
    hide : bool = False
    def __init__(self, needleCenters: np.ndarray, eyeCenters: np.ndarray,needleBB:np.ndarray,eyeBB:np.ndarray,tips:list,time:float) -> None:
        self.name = "Rational Positions"
        self.type = XY
        super().__init__(needleCenters, eyeCenters,needleBB,eyeBB,tips,time)
    def operate(self):
        # startTime = 
        a = []
        gt = self.eyeBB[0]
        for needle,eye,nBB,eBB in zip(self.tips,self.eyeCenters,self.needleBB,self.eyeBB):
            x,y = needle
            x1,y1 = eye
            deltaXEye = abs(gt[0] - gt[2])
            deltaYEye =abs( gt[1] - gt[3])
            x1 = (gt[0] + gt[2])/2
            y1 = (gt[1] + gt[3])/2
            a.append([(x-x1)/deltaXEye,(y-y1)/deltaYEye])
            # print(f"Needle centers {x},{y} eye centers {x1},{y1} eye BB {eBB[0]}, {eBB[1]}, {eBB[2]}, {eBB[3]}")
            self.result = a
            pass
        return super().operate()
    def getSerlizableResult(self):
        result = self.getResult()
        serlizebleList = []
        ax = []
        ay =[]
        for x,y in self.result:
            ax.append(f"{x:0.3f}")
            ay.append(f"{y:0.3f}")
        a = {}
        a["Name"] = "0:0:Relative Trajectory"
        a["Type"] = XY
        a["Data"] = {"X":ax,"Y":ay,"XLabel":"Rational Trj. X","YLabel":"Rational Trj. Y"}
        a["Time"] = f"{self.time:0.3f}"
        # a["Max"] = f"{1000}"
        # a["Min"] = f"{0}"
        serlizebleList.append(a)
        return serlizebleList

class NeedleBBOperator(PrimaryOperator):
    hide : bool = True
    def __init__(self, needleCenters: np.ndarray, eyeCenters: np.ndarray,needleBB:np.ndarray,eyeBB:np.ndarray,tips:list,time:float) -> None:
        self.name = "Needle Bounding Boxes"
        self.type = XY
        super().__init__(needleCenters, eyeCenters,needleBB,eyeBB,tips,time)
    def operate(self):
        # startTime = 
        self.result = self.needleBB
        return super().operate()
    def getSerlizableResult(self) -> list:
        return super().getSerlizableResult()

class IrisBBOperator(PrimaryOperator):
    hide : bool = True
    def __init__(self, needleCenters: np.ndarray, eyeCenters: np.ndarray,needleBB:np.ndarray,eyeBB:np.ndarray,tips:list,time:float) -> None:
        self.name = "Iris Bounding Boxes"
        self.type = XY
        super().__init__(needleCenters, eyeCenters,needleBB,eyeBB,tips,time)
    def operate(self):
        # startTime = 
        self.result = self.eyeBB
        return super().operate()
    def getSerlizableResult(self) -> list:
        return super().getSerlizableResult()

class PathLengthOperator(SecondryOperator):
    hide : bool = False
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Path length"
        self.type = NUMERIC
        super().__init__(*operator)
    def operate(self):
        relativeTrajectory  = (self.operators[0].getResult())
        # relativeTrajectoryX = relativeTrajectory[:,0]
        # relativeTrajectoryY = relativeTrajectory[:,1]
        sum = 0
        for x,y in relativeTrajectory:
            sum += math.sqrt(x**2 + y**2)
        self.result = sum
        return super().operate()
    def getSerlizableResult(self) -> list:
        result = self.getResult()
        serlizebleList = []
        # ax = []
        # ay =[]
        # for x,y in self.result:
        #     ax.append(f"{x:0.3f}")
        #     ay.append(f"{y:0.3f}")
        a = {}
        a["Name"] = "Path"
        a["Type"] = NUMERIC
        a["Data"] = f"{result:0.3f}"
        a["Time"] = f"{self.time:0.3f}"
        a["Max"] = f"{1000}"
        a["Min"] = f"{0}"
        serlizebleList.append(a)
        return serlizebleList
    pass

class SpeedOperator(SecondryOperator):
    hide : bool = True
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Speed"
        self.type = TIME_SERIES
        super().__init__(*operator)
    def operate(self):
        relativeSpeed  = (self.operators[0].getResult())
        speed = []
        for x,y in relativeSpeed:
            a = math.sqrt(x**2 + y**2)
            speed.append(a)
        self.result = speed
        return super().operate()
    pass

class MicroscopeCentralityOperator(SecondryOperator):
    hide : bool = False
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Microscope Centerality"
        self.type = NUMERIC
        super().__init__(*operator)
    def operate(self):
        needleTrajectory = (self.operators[0].getResult())
        irisTrajectory  = (self.operators[1].getResult())
        firstFrame = irisTrajectory[0]
        centrality = 0
        for x,y in needleTrajectory:
            centrality += math.sqrt((x-firstFrame[0])**2 + (y-firstFrame[1])**2)
        self.result = centrality
        return super().operate()
    def getSerlizableResult(self):
        
        result = self.getResult()
        serlizebleList = []
        x = {}
        x["Name"] = "Microscope Centerality"
        x["Type"] = NUMERIC
        x["Data"] = f"{result:0.3f}"
        x["Time"] = f"{self.time:0.3f}"
        x["Max"] = f"{1000}"
        x["Min"] = f"{0}"
        # y["Time"] = f"{self.time:0.3f}"
        serlizebleList.append(x)
        return serlizebleList
    pass

class DistanceOperator(SecondryOperator):
    hide : bool = False
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Distance"
        self.type = NUMERIC
        super().__init__(*operator)
    def operate(self):
        relativeTrajectory = self.operators[0].getResult()
        distance = 0
        for x,y in relativeTrajectory:
            distance += math.sqrt(x**2 + y**2)
        self.result = distance 
        return super().operate()
    pass

class SmoothnessOperator(SecondryOperator):
    hide : bool = False   
    def __init__(self, *operator: tuple[Operator]) -> None:
        self.name = "Smootheness"
        self.type = NUMERIC
        super().__init__(*operator)
    def operate(self):
        relativeTrajectory = self.operators[0].getResult()
        distance = 0
        for x,y in relativeTrajectory:
            distance += math.sqrt(x**2 + y**2) 
        self.result = distance
        return super().operate()
    def getSerlizableResult(self):
        
        result = self.getResult()
        serlizebleList = []
        x = {}
        x["Name"] = "Smoothness"
        x["Type"] = NUMERIC
        x["Data"] = f"{result:0.3f}"
        x["Time"] = f"{self.time:0.3f}"
        x["Max"] = f"{1000}"
        x["Min"] = f"{0}"
        # y["Time"] = f"{self.time:0.3f}"
        serlizebleList.append(x)
        return serlizebleList
    pass

class CSVLoggerOperator(LoggerOperator):
    hide : bool = True
    def __init__(self, path:str,*operator: tuple[Operator]) -> None:
        super().__init__(path,*operator)
    def operate(self):
        velocityX = np.array(np.array(self.operators[0].getResult())[:,0],dtype=object)
        velocityY = np.array(np.array(self.operators[0].getResult())[:,1],dtype=object)
        accX =      np.array(np.array(self.operators[1].getResult())[:,0],dtype=object)
        accY =      np.array(np.array(self.operators[1].getResult())[:,1],dtype=object)
        jerkX =     np.array(np.array(self.operators[2].getResult())[:,0],dtype=object)
        jerkY =     np.array(np.array(self.operators[2].getResult())[:,1],dtype=object)



        relativeVelocityX = np.array(np.array(self.operators[3].getResult())[:,0],dtype=object)
        relativeVelocityY = np.array(np.array(self.operators[3].getResult())[:,1],dtype=object)
        relativeAccX =      np.array(np.array(self.operators[4].getResult())[:,0],dtype=object)
        relativeAccY =      np.array(np.array(self.operators[4].getResult())[:,1],dtype=object)
        relativeJerkX =     np.array(np.array(self.operators[5].getResult())[:,0],dtype=object)
        relativeJerkY =     np.array(np.array(self.operators[5].getResult())[:,1],dtype=object)

        
        # print(f"Needle BB Operator in CSV Logger {str(type(self.operators[6]))} {str(self.operators[6].getResult())}")
        needleBB0 = np.array(np.array(self.operators[6].getResult())[:,0],dtype=object)
        needleBB1 = np.array(np.array(self.operators[6].getResult())[:,1],dtype=object)
        needleBB2 = np.array(np.array(self.operators[6].getResult())[:,2],dtype=object)
        needleBB3 = np.array(np.array(self.operators[6].getResult())[:,3],dtype=object)
        eyeBB0 =    np.array(np.array(self.operators[7].getResult())[:,0],dtype=object)
        eyeBB1 =    np.array(np.array(self.operators[7].getResult())[:,1],dtype=object)
        eyeBB2 =    np.array(np.array(self.operators[7].getResult())[:,2],dtype=object)
        eyeBB3 =    np.array(np.array(self.operators[7].getResult())[:,3],dtype=object)
        
        relativeTrajectoryX = np.array(np.array(self.operators[8].getResult())[:,0],dtype=object)
        relativeTrajectoryY = np.array(np.array(self.operators[8].getResult())[:,1],dtype=object)
        trajectoryX =         np.array(np.array(self.operators[9].getResult())[:,0],dtype=object)
        trajectoryY =         np.array(np.array(self.operators[9].getResult())[:,1],dtype=object)
        irisTrajectoryX =     np.array(np.array(self.operators[10].getResult())[:,0],dtype=object)
        irisTrajectoryY =     np.array(np.array(self.operators[10].getResult())[:,1],dtype=object)

        distance =np.array([self.operators[11].getResult()],dtype=object)
        micr = np.array([self.operators[12].getResult()],dtype=object)
        pathLength = np.array([self.operators[13].getResult()],dtype=object)
        speedOperator = np.array(self.operators[14].getResult(),dtype=object)
        curvature =  np.array([self.operators[15].getResult()],dtype=object)
        speedPeak = np.array([self.operators[16].getResult()],dtype=object)
        smoothnessOperator = np.array([self.operators[17].getResult()],dtype=object)     

        smoothnessOperator = np.pad(smoothnessOperator, [0,np.shape(trajectoryX)[0] - 1], mode='constant', constant_values=0)
        speedPeak = np.pad(speedPeak, [0,np.shape(trajectoryX)[0] - 1], mode='constant', constant_values=0)
        pathLength = np.pad(pathLength, [0,np.shape(trajectoryX)[0] - 1], mode='constant', constant_values=0)
        micr = np.pad(micr, [0,np.shape(trajectoryX)[0] - 1], mode='constant', constant_values=0)
        distance = np.pad(distance, [0,np.shape(trajectoryX)[0] - 1], mode='constant', constant_values=0)
        curvature = np.pad(curvature, [0,np.shape(trajectoryX)[0] - 1], mode='constant', constant_values=0)

        distance =           np.insert(distance,0,"Distance")
        micr =           np.insert(micr,0,"Microscope")
        pathLength =           np.insert(pathLength,0,"Path length")
        speedOperator =           np.insert(speedOperator,0,"Speed")
        curvature =           np.insert(curvature,0,"Curvature")
        speedPeak =           np.insert(speedPeak,0,"Speed Peak")
        smoothnessOperator =           np.insert(smoothnessOperator,0,"Smoothness")
        
        velocityX =           np.insert(velocityX,0,"Velocity X")
        velocityY =           np.insert(velocityY,0,"Velocity Y")
        accX =                np.insert(accX,0,"Acc. X")
        accY =                np.insert(accY,0,"Acc. Y")
        jerkX =               np.insert(jerkX,0,"Jerk X")
        jerkY =               np.insert(jerkY,0,"Jerk Y")
        
        relativeVelocityX =           np.insert(relativeVelocityX,0,"VX")
        relativeVelocityY =           np.insert(relativeVelocityY,0,"VY")
        relativeAccX =                np.insert(relativeAccX,0,"AX")
        relativeAccY =                np.insert(relativeAccY,0,"AY")
        relativeJerkX =               np.insert(relativeJerkX,0,"JX")
        relativeJerkY =               np.insert(relativeJerkY,0,"JY")

        needleBB0 = np.insert(needleBB0,0,"Needle X0")
        needleBB1 = np.insert(needleBB1,0,"Needle Y0")
        needleBB2 = np.insert(needleBB2,0,"Needle X1")
        needleBB3 = np.insert(needleBB3,0,"Needle Y1")

        eyeBB0 = np.insert(eyeBB0,0,"Eye X0")
        eyeBB1 = np.insert(eyeBB1,0,"Eye Y0")
        eyeBB2 = np.insert(eyeBB2,0,"Eye X1")
        eyeBB3 = np.insert(eyeBB3,0,"Eye Y1")

        relativeTrajectoryX = np.insert(relativeTrajectoryX,0,"XX")
        relativeTrajectoryY = np.insert(relativeTrajectoryY,0,"YY")
        
        trajectoryX = np.insert(trajectoryX,0,"Trajectory X")
        trajectoryY = np.insert(trajectoryY,0,"Trajectory Y")
        
        irisTrajectoryX = np.insert(irisTrajectoryX,0,"Iris Trajectory X")
        irisTrajectoryY = np.insert(irisTrajectoryY,0,"Iris Trajectory Y")
        print(f"Shape in logger 1.{np.shape(irisTrajectoryX)} 2.{np.shape(speedPeak)} 3.{np.shape(smoothnessOperator)} 4.{np.shape(smoothnessOperator)} 5.{np.shape(speedOperator)} 6.{np.shape(micr)} 7.{np.shape(curvature)} 8.{np.shape(distance)} " )


        c = np.transpose(np.concatenate([np.array([velocityX]),np.array([velocityY])
                                         ,np.array([accX]),np.array([accY])
                                         ,np.array([jerkX]),np.array([jerkY]),
                                          np.array([relativeVelocityX]),np.array([relativeVelocityY])
                                         ,np.array([relativeAccX]),np.array([relativeAccY])
                                         ,np.array([relativeJerkX]),np.array([relativeJerkY])
                                         ,np.array([needleBB0]),np.array([needleBB1])
                                         ,np.array([needleBB2]),np.array([needleBB3])
                                         ,np.array([eyeBB0]),np.array([eyeBB1])
                                         ,np.array([eyeBB2]),np.array([eyeBB3])
                                         ,np.array([relativeTrajectoryX]),np.array([relativeTrajectoryY])
                                         ,np.array([trajectoryX]),np.array([trajectoryY])
                                         ,np.array([irisTrajectoryX]),np.array([irisTrajectoryY]),np.array([smoothnessOperator])
                                         ,np.array([speedPeak]),np.array([curvature])
                                         ,np.array([speedOperator]),np.array([pathLength])
                                         ,np.array([micr]),np.array([distance])],axis=0))
        pd.DataFrame(c).to_csv(self.path + "\\Log.csv") 
        return super().operate()
    def getSerlizableResult(self) -> list:
        return super().getSerlizableResult()