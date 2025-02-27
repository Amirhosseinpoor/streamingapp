    import os
    import cv2 as cv
    import time
    import subprocess
    import numpy as np
    from PIL import Image
    from celery import shared_task
    from ultralytics import YOLO



    MODELS = ["YoloV8+151","Yolov8","Yolob8-Tip","GENERAL","12Class","9Class"]
    YOLO_V8_151_INDEX = 0
    YOLO_V8_INDEX = 1
    YOLO_V8_TIP_INDEX = 2
    GENERAL_INDEX = 3
    index12 = 4
    index9 = 5

    class VideoSaver:

        def __init__(self, path):
            self.path = path
            fourcc = cv.VideoWriter_fourcc(*"XVID")
            self.writer = cv.VideoWriter(self.path, fourcc, 10, (640, 640))

        def saveFrame(self, frame):
            self.writer.write(frame)

        def end(self):
            self.writer.release()

    class VideoSegmenter:
        pathToVideo: str
        saver: VideoSaver
        def __init__(self, path, videoSaver) -> None:
            self.pathToVideo = path
            self.saver = videoSaver
            self.cap = cv.VideoCapture(self.pathToVideo)

        def segment(self, frame):
            return frame, [], 0, 0

        def segmentVideo(self):
            video_object_dict = {}
            while self.cap.isOpened():
                ret, rawImage = self.cap.read()
                if not ret:
                    break

                image, data, processTime, modelTime = self.segment(rawImage)
                frame_objects = {}

                for d in data:
                    center = d["Center"]
                    center[0] = str(int(center[0]))
                    center[1] = str(int(center[1]))
                    name = d["Name"]
                    mask = [[str(int(p[0])), str(int(p[1]))] for p in d["Mask"]]

                    if name not in frame_objects:
                        frame_objects[name] = [{"Mask": mask, "Center": center}]
                    else:
                        frame_objects[name].append({"Mask": mask, "Center": center})

                for key, objs in frame_objects.items():
                    prev_center = None
                    for obj in objs:
                        c = obj["Center"]
                        c_tuple = (int(c[0]), int(c[1]))
                        if prev_center is not None:
                            cv.line(image, c_tuple, prev_center, (150, 150, 150), 2)
                        else:
                            cv.circle(image, c_tuple, 3, (200, 200, 200), -1)
                        prev_center = c_tuple

                self.saver.saveFrame(image)

                for k, v in frame_objects.items():
                    if k not in video_object_dict:
                        video_object_dict[k] = v
                    else:
                        video_object_dict[k].extend(v)

            self.cap.release()
            self.saver.end()
            return video_object_dict

    class YoloV8Segmenter(VideoSegmenter):
        def __init__(self, path, model, videoSaver) -> None:
            super().__init__(path, videoSaver)
            self.model = model

        def segment(self, image):
            startTime = time.time()
            image_resized = cv.resize(image, (640, 640))
            results = self.model(image_resized, stream=False)

            modelTime = time.time()
            data = []
            if results:
                for r in results:
                    if r and r.masks:
                        confs = r.boxes.conf
                        classes = r.boxes.cls
                        bbss = r.boxes.xyxy
                        xy = r.masks.xy
                        im_array = r.plot()
                        image = np.asarray(Image.fromarray(im_array[..., ::-1]))
                        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

                        centers, bbs, masksArray = [], [], []
                        for coords,bb in zip(xy, bbss):
                            xCenter = yCenter = 0
                            mask = []
                            for x, y in coords:
                                mask.append([x, y])
                                xCenter += x
                                yCenter += y
                            xCenter /= len(coords)
                            yCenter /= len(coords)
                            centers.append([int(xCenter), int(yCenter)])
                            bbs.append(bb)
                            masksArray.append(mask)

                        data = []
                        for i, center in enumerate(centers):
                            name = "Eye" if classes.cpu().numpy()[i] == 0 else "Needle"
                            data.append({
                                "Name": name,
                                "Conf": f"{confs.cpu().numpy()[i]}",
                                "Center": center,
                                "BB": bbs[i].cpu().numpy(),
                                "Mask": masksArray[i]
                            })

            processTime = time.time()
            return image, data, (processTime - startTime), (modelTime - startTime)


    @shared_task
    def convert_to_hls(video_id):
        from .models import VideoModels

        video = VideoModels.objects.get(id=video_id)
        input_file = video.video_file.path
        video_title_safe = "".join(x if x.isalnum() else "_" for x in video.title)
        output_dir = os.path.join(os.path.dirname(input_file), 'output', video_title_safe)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        temp_output_path = os.path.join(output_dir, "processed_video.avi")
        hls_output_path = os.path.join(output_dir, "output.m3u8")

        saver = VideoSaver(temp_output_path)
        model = YOLO("server/best+151.pt")
        segmenter = YoloV8Segmenter(input_file, model, saver)

        segmenter.segmentVideo()
        print("amir")

        command = f"ffmpeg -i {temp_output_path} -hls_time 10 -hls_playlist_type vod {hls_output_path}"
        subprocess.run(command, shell=True)

        video.hls_ready = True
        video.save()

