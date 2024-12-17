import glob
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import json
from ultralytics import YOLO
model  = YOLO("best+151.pt")
    
def feedToModel(image):
    image = cv.resize(image, (640, 640))
    results = model(image, stream=True)
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
                mask.append([int(x),int(y)])
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
    return data
occluded = []
notOccluded = []

for item in glob.glob("Occlusion/*"):
    image = cv.imread(item)
    
    data = feedToModel(image)
    bestNeedle = None
    for d in data:
        a = float(d["Conf"])
        if bestNeedle == None:
            bestNeedle = d
        else:
            if(a > float(bestNeedle["Conf"])):
                bestNeedle = d
                pass
    occluded.append(json.dumps(bestNeedle["Mask"])) 
    pass

for item in glob.glob("no Occlusion/*"):
    image = cv.imread(item)
    
    data = feedToModel(image)
    bestNeedle = None
    for d in data:
        a = float(d["Conf"])
        if bestNeedle == None:
            bestNeedle = d
        else:
            if(a > float(bestNeedle["Conf"])):
                bestNeedle = d
                pass
    notOccluded.append(json.dumps(bestNeedle["Mask"])) 
    pass
notOccluded = np.array(notOccluded)
occluded = np.array(occluded)
pd.DataFrame(occluded).to_csv("occludead.csv")
pd.DataFrame(notOccluded).to_csv("notOccluded.csv")