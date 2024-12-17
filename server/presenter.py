import ultralytics
from ultralytics import YOLO
import cv2 as cv
import numpy as np
from PIL import Image

image = cv.imread("Present\image.png")
model = YOLO("best+151.pt")
def fitLineToNeedle(img,needle):
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
    
def projectOnline(point,vx,vy,p0):
        x = p0[0]*vy**2 + point[0]*vx**2 + vx*vy*(point[1] - p0[1])
        y = (x-point[0])*(1/-vy)*(vx) + point[1]
        return int(x),int(y)
def findMaxMaxMinMin(imagedPoints):
    minminPoint = [10000000,10000000]
    maxmaxPoint = [0,0]
    for p in imagedPoints:
        if(p[0]<minminPoint[0] and p[1]<minminPoint[1]):
            minminPoint = p
    for p in imagedPoints:
        if(p[0]>maxmaxPoint[0] and p[1]>maxmaxPoint[1]):
            maxmaxPoint = p

    return minminPoint,maxmaxPoint

def findMinMaxMaxMin(imagedPoints):
    minMaxPoint = [10000000,0]
    maxMinPoint = [0,10000000]
    for p in imagedPoints:
        if(p[0]<minMaxPoint[0] and p[1]>minMaxPoint[1]):
            minMaxPoint = p
    for p in imagedPoints:
        if(p[0]>maxMinPoint[0] and p[1]<maxMinPoint[1]):
            maxMinPoint = p

    return maxMinPoint,minMaxPoint
def segment(image,model):
    results = model(image)
    image = cv.resize(image, (640, 640))
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
        return data,image

data,image = segment(image,model)
cv.imwrite("Present/stage2.png",image)
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
print(f"Needle dict {d}")
image,vx,vy = fitLineToNeedle(image,needleDict)
c = [int(needleDict["Center"][0] + 1000*[vx/((vx**2+vy**2)**0.5),vy/((vx**2+vy**2)**0.5)][0]) , int(needleDict["Center"][1] + 1000*[vx/((vx**2+vy**2)**0.5),vy/((vx**2+vy**2)**0.5)][1])]
c2 = [int(needleDict["Center"][0] - 1000*[vx/((vx**2+vy**2)**0.5),vy/((vx**2+vy**2)**0.5)][0]) , int(needleDict["Center"][1]  - 1000*[vx/((vx**2+vy**2)**0.5),vy/((vx**2+vy**2)**0.5)][1])]


image = cv.line(image,c2,c,(0,0,0),2)
cv.imwrite("Present/stage3.png",image)
eyeCenter = eyeDict["Center"]
imagedPoints = []
for point in needleDict["Mask"]:
    point = [int(point[0]),int(point[1])]
    imagedPoint = projectOnline(point,vx,vy,needleDict["Center"])
    image = cv.circle(image,point,3,color=(255,255,0),thickness=-1)
    image = cv.circle(image,imagedPoint,3,color=(0,255,0),thickness=-1)
    image = cv.line(image,point,imagedPoint,color=(100,100,100),thickness=2)
    imagedPoints.append(imagedPoint)
cv.imwrite("Present/stage4.png",image)
candidate = None
minDis = 100000000000000
direction = vx*vy>0
if(direction):
    imagedPoints = findMaxMaxMinMin(imagedPoints)
else:
    imagedPoints = findMinMaxMaxMin(imagedPoints)


for point in imagedPoints:
    distance = ((point[0] - eyeCenter[0])**2 + (point[1] - eyeCenter[1])**2)**0.5
    if(distance<=minDis):
        candidate = point
        minDis = distance
    pass

image = cv.circle(image,candidate,10,color=(200,200,200),thickness=-1)

cv.imwrite("Present/stage5.png",image)
