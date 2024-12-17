from op.operators import *
import numpy as np
TRAJECTORY_INDEX = 0
RELATIVE_TRAJECTORY_INDEX = 1
RATIONAL_TRAJECTORY_INDEX = 2
VELOCITY_INDEX = 3
ACC_INDEX = 4

UNSAFE_INDEX = 5

TIME_INDEX = 6
ALL_TIME_OPERATOR_INDEX = 7
IRIS = 8
JERK_INDEX = 9
NEEDLE_BB = 10
IRIS_BB = 11

RELATIVE_VELOCITY_INDEX = 12
RELATIVE_ACC_INDEX = 13
RELATIVE_JERK_INDEX = 14
SMOOTHNESS_INDEX = 15
DISTANCE_INDEX = 16
MIC_CENTREALITY_INDEX = 17
PATH_LENGHT_INDEX = 18
SPEED_INDEX = 19
CURVATURE_INDEX = 20
SPEED_PEAK_INDEX = 21
FLUIDITY_INDEX = 22
ECHONOMY_INDEX = 23
VELOCTY_SIGN_CHANGES_INDEX = 24
LOGGER = 25

class OperatorHandler:
    eye: list
    eyeBB:list
    needlBB:list
    needle: list
    relative: list
    timeSerie: np.ndarray
    tips:list
    operators: list[Operator]

    def __init__(
        self,path:str, eye: list, needle: list, relative: list, timeSeries: np.ndarray,eyeBB:list,needleBB:list,tips:list,time:float
    ) -> None:
        self.path = path
        self.eye = eye
        self.needle = needle
        self.relative = relative
        self.timeSerie = timeSeries
        self.eyeBB = eyeBB
        self.needleBB = needleBB
        self.time = time
        self.tips = tips
        # print(f"Tips for operators {self.tips}")
        self.operators = []
        self.operators.insert(
            TRAJECTORY_INDEX,Trajectory(needleCenters=self.needle, eyeCenters=self.eye,needleBB=self.needleBB,eyeBB=self.eyeBB,tips = self.tips,time=self.time),
        )
        self.operators.insert(
            RELATIVE_TRAJECTORY_INDEX,RelativeTrajectory(needleCenters=self.needle, eyeCenters=self.eye,needleBB=self.needleBB,eyeBB=self.eyeBB,tips = self.tips,time=self.time),
        )
        self.operators.insert(
            RATIONAL_TRAJECTORY_INDEX,RationalPositions(needleCenters=self.needle, eyeCenters=self.eye,needleBB=self.needleBB,eyeBB=self.eyeBB,tips = self.tips,time=self.time),
        )
        self.operators.insert(
            VELOCITY_INDEX,
            NeedleVelocityOperator(needleCenters=self.needle, eyeCenters=self.eye,needleBB=self.needleBB,eyeBB=self.eyeBB,tips = self.tips,time=self.time),
        )
        self.operators.insert(
            ACC_INDEX, NeedleAccelrationOperation(self.operators[VELOCITY_INDEX])
        )
        
        self.operators.insert(
            UNSAFE_INDEX, UnsafeEntrance(self.operators[RATIONAL_TRAJECTORY_INDEX]),
        )
        
        
        self.operators.insert(
            TIME_INDEX,TimeOperator(needleCenters=self.needle, eyeCenters=self.eye,needleBB=self.needleBB,eyeBB=self.eyeBB,tips = self.tips,time=self.time),
        )
        self.operators.insert(
            ALL_TIME_OPERATOR_INDEX,AllTimeOperator(*self.operators),
        )
        self.operators.insert(
            IRIS,Iris(needleCenters=self.needle, eyeCenters=self.eye,needleBB=self.needleBB,eyeBB=self.eyeBB,tips = self.tips,time=self.time),
        )
        self.operators.insert(
            JERK_INDEX, NeedleJerkOperator(self.operators[ACC_INDEX])
        )
        self.operators.insert(
            NEEDLE_BB, NeedleBBOperator(needleCenters=self.needle, eyeCenters=self.eye,needleBB=self.needleBB,eyeBB=self.eyeBB,tips = self.tips,time=self.time)
        )
        self.operators.insert(
            IRIS_BB, IrisBBOperator(needleCenters=self.needle, eyeCenters=self.eye,needleBB=self.needleBB,eyeBB=self.eyeBB,tips = self.tips,time=self.time)
        )
        
        self.operators.insert(RELATIVE_VELOCITY_INDEX,RelativeVelocity(self.operators[RATIONAL_TRAJECTORY_INDEX]))
        self.operators.insert(RELATIVE_ACC_INDEX,RelativeAcceleration(self.operators[RELATIVE_VELOCITY_INDEX]))
        self.operators.insert(RELATIVE_JERK_INDEX,RelativeJerk(self.operators[RELATIVE_ACC_INDEX]))
        self.operators.insert(SMOOTHNESS_INDEX,SmoothnessOperator(self.operators[RELATIVE_JERK_INDEX]))
        self.operators.insert(DISTANCE_INDEX,DistanceOperator(self.operators[RATIONAL_TRAJECTORY_INDEX]))
        self.operators.insert(MIC_CENTREALITY_INDEX,MicroscopeCentralityOperator(self.operators[TRAJECTORY_INDEX],self.operators[IRIS]))
        self.operators.insert(PATH_LENGHT_INDEX,PathLengthOperator(self.operators[RELATIVE_TRAJECTORY_INDEX]))
        self.operators.insert(SPEED_INDEX,SpeedOperator(self.operators[RELATIVE_VELOCITY_INDEX]))
        self.operators.insert(CURVATURE_INDEX,CurvatureOperator(self.operators[RELATIVE_VELOCITY_INDEX],self.operators[RELATIVE_ACC_INDEX]))
        self.operators.insert(SPEED_PEAK_INDEX,SpeedPeakOperator(self.operators[SPEED_INDEX]))
        self.operators.insert(
            FLUIDITY_INDEX, Fluidity(self.operators[RELATIVE_VELOCITY_INDEX])
        )
        self.operators.insert(
            ECHONOMY_INDEX, Economy(self.operators[RELATIVE_VELOCITY_INDEX])
        )
        self.operators.insert(
            VELOCTY_SIGN_CHANGES_INDEX, VelocitySignChanges(self.operators[RELATIVE_VELOCITY_INDEX])
        )
        self.operators.insert(
            LOGGER, CSVLoggerOperator(self.path
                                      ,self.operators[VELOCITY_INDEX]#0
                                      ,self.operators[ACC_INDEX]#1
                                      ,self.operators[JERK_INDEX]#2
                                      ,self.operators[RELATIVE_VELOCITY_INDEX]#3
                                      ,self.operators[RELATIVE_ACC_INDEX]#4
                                      ,self.operators[RELATIVE_JERK_INDEX]#5
                                      ,self.operators[NEEDLE_BB]#6
                                      ,self.operators[IRIS_BB]#7
                                      ,self.operators[RATIONAL_TRAJECTORY_INDEX]#8
                                      ,self.operators[TRAJECTORY_INDEX]#9
                                      ,self.operators[IRIS]#10
                                      ,self.operators[DISTANCE_INDEX]#11
                                      ,self.operators[MIC_CENTREALITY_INDEX]#12
                                      ,self.operators[PATH_LENGHT_INDEX]#13
                                      ,self.operators[SPEED_INDEX]#14
                                      ,self.operators[CURVATURE_INDEX]#15
                                      ,self.operators[SPEED_PEAK_INDEX]#16
                                      ,self.operators[SMOOTHNESS_INDEX]#17
                                      ))
        
        pass

    def doOperators(self):
        map = []
        hiddenMap = []
        toSend = []
        time = []
        # allMap = []
        for t in self.timeSerie:
            time.append(f"{t:0.3f}")
        for op in self.operators:
            result = op.getSerlizableResult()
            if(type(result) != type(None)):
                if(op.hide):
                    hiddenMap.extend(result)
                else:
                    map.extend(result)
            # allMap.extend(result)
                map.append(time)
                hiddenMap.append(time)
        mainPos = self.operators[TRAJECTORY_INDEX].getSerlizableResult()
        position = self.operators[RATIONAL_TRAJECTORY_INDEX].getSerlizableResult()
        velocity = self.operators[RELATIVE_VELOCITY_INDEX].getSerlizableResult()
        acc = self.operators[RELATIVE_ACC_INDEX].getSerlizableResult()
        jerk = self.operators[RELATIVE_JERK_INDEX].getSerlizableResult()
        pathLength = self.operators[PATH_LENGHT_INDEX].getSerlizableResult()
        signChange = self.operators[VELOCTY_SIGN_CHANGES_INDEX].getSerlizableResult()
        smoothness = self.operators[SMOOTHNESS_INDEX].getSerlizableResult()
        curvature = self.operators[CURVATURE_INDEX].getSerlizableResult()
        fluidity = self.operators[FLUIDITY_INDEX].getSerlizableResult()
        ecnonomy = self.operators[ECHONOMY_INDEX].getSerlizableResult()
        centerality = self.operators[MIC_CENTREALITY_INDEX].getSerlizableResult()
        # toSend = [position,velocity,acc,jerk,pathLength,signChange,smoothness,curvature,fluidity,ecnonomy,centerality,time]
        toSend.extend(position)
        toSend.extend(velocity)
        toSend.extend(acc)
        toSend.extend(jerk)
        toSend.extend(pathLength)
        toSend.extend(signChange)
        toSend.extend(smoothness)
        toSend.extend(curvature)
        toSend.extend(fluidity)
        toSend.extend(ecnonomy)
        toSend.extend(centerality)
        toSend.extend(mainPos)
        toSend.append(time)
        # duration = self.operators[TIME_INDEX].getSerlizableResult()
        return map,hiddenMap,toSend