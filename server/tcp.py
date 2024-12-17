import socket
import json
from video import YoloV8Segmenter, VideoSaver
from ultralytics import YOLO
import cv2 as cv
import sys
import pickle
import traceback
import time
import matplotlib.pyplot as plt
import os
from ForAll import FPS
import shutil
import numpy as np
from op.operator_handler import OperatorHandler
from plot.plot_handler import OperatorPlotHandler

plt.style.use("_mpl-gallery")

host = socket.gethostbyaddr("127.0.0.1")
# print(str(host))
# host = host[2]
port = 6100
serverSocket = socket.socket()
serverSocket.bind(("127.0.0.1", port))
serverSocket.listen(100)

# accept new connection


def proccess(conn: socket.socket, data: str):
    data = json.loads(data)
    path: str = data["Video"]
    model = data["Model"]
    savePath: str = data["SaveVideo"]
    pickeSave: str = data["SavePickle"]
    resultDirectory: str = data["Result"]
    print(f"model directory : {model}")
    model = YOLO(model, task="segment")
    saver = VideoSaver(savePath)
    try:
        segmneter = YoloV8Segmenter(path, model, saver)
        conn.send(
            json.dumps(
                {"Message": "Model and Video loaded start segmenting ...", "Type": 0}
            ).encode()
        )
    except:
        traceback.print_exc()
        conn.send(
            json.dumps(
                {"Message": "Unable to load model or video ...", "Type": 1}
            ).encode()
        )
        return
    try:
        eye, needle, releativeNeedle, f, timeSerie = segmneter.segmentVideo()
        conn.send(
            json.dumps(
                {"Message": "Video segmented successfully ...", "Type": 2}
            ).encode()
        )
    except:
        traceback.print_exc()
        conn.send(
            json.dumps({"Message": "Unable to segment Video", "Type": 3}).encode()
        )
        return
    try:
        saver.end()
        conn.send(
            json.dumps(
                {"Message": "Segmented video saved successfully ...", "Type": 4}
            ).encode()
        )
    except:
        traceback.print_exc()
        conn.send(
            json.dumps(
                {"Message": "Unable to save segmented Video", "Type": 5}
            ).encode()
        )
        return
    obj = {"Eye": eye, "Needle": needle, "RelativeNeedle": releativeNeedle}
    needleCenters = []
    eyeCenter = []
    for i, n in enumerate(needle):
        # print(n)
        needleCenters.append((n["Center"]))
    for n in eye:
        # print(n)
        eyeCenter.append((n["Center"]))

    # print(f"timeSerie size : {np.shape(timeSerie)}")
    handler = OperatorHandler(
        needleCenters, eyeCenter, np.array(releativeNeedle), timeSerie
    )
    opPlHandler = OperatorPlotHandler(
        operatorhandler=handler,
        path=resultDirectory,
        timeSerie=timeSerie,
    )
    opPlHandler.save()

    try:
        with open(pickeSave, "w") as f:
            f.write(json.dumps(obj))
            f.close()
            conn.send(
                json.dumps(
                    {"Message": "Segmention data saved successfully", "Type": 6}
                ).encode()
            )

    except:
        traceback.print_exc()
        conn.send(
            json.dumps({"Message": "Unable to save in pickle file", "Type": 7}).encode()
        )
        return
    time.sleep(2)
    conn.send(json.dumps({"Message": "Done", "Type": 8}).encode())

    return


print("Starting server")
while True:
    # receive data stream. it won't accept data packet greater than 1024 bytes
    conn, address = serverSocket.accept()
    print("Connection from: " + str(address))
    while True:
        # writer = cv.VideoWriter()
        # writer.write(segmentedVideo)
        try:
            data = conn.recv(1024).decode()
            # print(f"Data :{data}")
            proccess(conn, data)

        except:
            traceback.print_exc()
            sys.exit(0)
            pass
        # send data to the client

        # conn.close()  # close the connection
