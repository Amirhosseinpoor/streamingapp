import cv2 as cv
import numpy as np
import ultralytics
from ultralytics import YOLO
from PIL import Image
videoAddress = "lib\Good.mp4"
modelAddress = "best+151.pt"
cap = cv.VideoCapture(videoAddress)
model = YOLO(modelAddress, task="segment")
bufferLength = 100
FPS = 25

def calculatePoint(vx,vy,needle,eye,img):
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

def fitLineToNeedle(img,needle):
        mask = needle["Mask"]
        [vx,vy,x,y] = cv.fitLine(np.array(mask),cv.DIST_L2, 0, 0.01, 0.01)
        return img,vx[0],vy[0]


def segmentImage(image):
    results = model(rawImage)
    for r in results:
        if(r!=None):
            mask = r.masks  # Masks object for segmentation masks outputs
            # probs = result.probs
            if(mask!=None):
                confs = r.boxes.conf
                classes = r.boxes.cls
                bbss = r.boxes.xyxy
                xy = mask.xy
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
                bestNeedle = None
                bestEye = None
                for i, center in enumerate(centers):
                    needle = classes.cpu().numpy()[i] != 0
                    conf = confs.cpu().numpy()[i]
                    bb = bbs[i].cpu().numpy()
                    mask = masksArray[i]
                    
                    if(needle):
                        if(bestNeedle == None):
                            bestNeedle = {"Conf":conf,"BB":bb,"Mask":mask,"Center":center}
                        else:
                            if(bestNeedle["Conf"]<conf):
                                bestNeedle = {"Conf":conf,"BB":bb,"Mask":mask,"Center":center}
                        pass
                    else:
                        if(bestEye == None):
                            bestEye = {"Conf":conf,"BB":bb,"Mask":mask,"Center":center}
                        else:
                            if(bestEye["Conf"]<conf):
                                bestEye = {"Conf":conf,"BB":bb,"Mask":mask,"Center":center}
                img,vx,vy = fitLineToNeedle(image,bestNeedle)
                if(bestNeedle!=None and bestEye != None):
                    tip,end = calculatePoint(vx,vy,bestNeedle,bestEye,img)
                    return tip,bestEye["Center"]
                else:
                    return None,None


def backAndForth(tips:list):
    preX = None
    preY = None
    vxs = []
    vys = []
    for tip in tips:
        x = tip[0]
        y = tip[1]
        if(preX == None):
            vx = 0
            vy = 0
            pass
        else:
            
            vx = (x - preX) * FPS
            vy = (y - preY) * FPS
        vxs.append(vx)
        vys.append(vy)
        preX = x
        preY = y
    preVx = None
    preVy = None
    result = 0
    for vx,vy in zip(vxs,vys):
        if(preVx != None):
            if(vx * preVx <0 or vy*preVy<0):
                result+=1
        preVx = vx
        preVy = vy
        pass
    
    return result

def motionFluidity(tips:list):
    preX = None
    preY = None
    vxs = []
    vys = []
    for tip in tips:
        x = tip[0]
        y = tip[1]
        if(preX == None):
            vx = 0
            vy = 0
            pass
        else:
            vx = (x - preX) * FPS
            vy = (y - preY) * FPS
        vxs.append(vx)
        vys.append(vy)
        preX = x
        preY = y
    result = 0
    for vx,vy in zip(vxs,vys):
        result += (vx*vx + vy*vy)**0.5
        pass
    result = result/len(tips)
    return result
def smoothness(tips:list):
    preX = None
    preY = None
    vxs = []
    vys = []
    for tip in tips:
        x = tip[0]
        y = tip[1]
        if(preX == None):
            vx = 0
            vy = 0
            pass
        else:
            vx = (x - preX) * FPS
            vy = (y - preY) * FPS
        vxs.append(vx)
        vys.append(vy)
        preX = x
        preY = y

    preX = None
    preY = None
    axs = []
    ays = []
    for tip in zip(vxs,vys):
        x = tip[0]
        y = tip[1]
        if(preX == None):
            vx = 0
            vy = 0
            pass
        else:
            vx = (x - preX) * FPS
            vy = (y - preY) * FPS
        axs.append(vx)
        ays.append(vy)
        preX = x
        preY = y

    preX = None
    preY = None
    jxs = []
    jys = []
    for tip in zip(axs,ays):
        x = tip[0]
        y = tip[1]
        if(preX == None):
            vx = 0
            vy = 0
            pass
        else:
            vx = (x - preX) * FPS
            vy = (y - preY) * FPS
        jxs.append(vx)
        jys.append(vy)
        preX = x
        preY = y
    result = 0
    for vx,vy in zip(jxs,jys):
        result += (vx*vx + vy*vy)**0.5
        pass
    result = result/len(tips)
    return result
    pass
needleTipPositions = []
eyePositions = []
while cap.isOpened():
    ret, rawImage  = cap.read()
    
    tip,eye = segmentImage(rawImage)
    if(tip == None or eye == None):
        continue
    tip = [tip[0]-eye[0],tip[1]-eye[1]]
    needleTipPositions.append(tip)
    eyePositions.append(eye)
    if(len(needleTipPositions)>bufferLength):
        needleTipPositions.pop(0)
    if(len(eyePositions)>bufferLength):
        eyePositions.pop(0)
    a =  smoothness(needleTipPositions)
    b =  motionFluidity(needleTipPositions)
    c =  backAndForth(needleTipPositions)
    print(f"Smoothness: -> {a:0.2f}")
    print(f"Fluidity: -> {b:0.2f}")
    print(f"Back and Forth: -> {c:0.2f}")
    


    