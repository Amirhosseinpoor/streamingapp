import os
import cv2 as cv
import time
import subprocess
import numpy as np
from PIL import Image
from celery import shared_task
from ultralytics import YOLO


class VideoSaver:
    def __init__(self, path, fps, frame_size=(640, 640)):
        self.path = path
        fourcc = cv.VideoWriter_fourcc(*"XVID")
        self.writer = cv.VideoWriter(self.path, fourcc, fps, frame_size)
        self.frame_size = frame_size

    def saveFrame(self, frame):
        resized_frame = cv.resize(frame, self.frame_size)
        self.writer.write(resized_frame)

    def end(self):
        self.writer.release()


class VideoSegmenter:
    def __init__(self, path, videoSaver, start_detection, end_detection):
        self.pathToVideo = path
        self.saver = videoSaver
        self.cap = cv.VideoCapture(self.pathToVideo)
        self.start_detection = start_detection
        self.end_detection = end_detection
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS))

    def segment(self, frame):
        return frame, [], 0, 0

    def segmentVideo(self):
        video_object_dict = {}
        frame_count = 0
        total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        print(f"FPS: {self.fps}, Duration: {total_frames / self.fps} sec")
        total_start_time = time.time()
        while self.cap.isOpened():
            ret, rawImage = self.cap.read()
            if not ret:
                break

            current_time = frame_count / self.fps
            print(f"Frame {frame_count}/{total_frames}: Time {current_time}s")

            if self.start_detection <= current_time <= self.end_detection:
                image, data, processTime, modelTime = self.segment(rawImage)
            else:
                image, data = rawImage, []

            self.saver.saveFrame(image)

            frame_objects = {}
            for d in data:
                center = [str(int(c)) for c in d["Center"]]
                name = d["Name"]
                mask = [[str(int(p[0])), str(int(p[1]))] for p in d["Mask"]]

                if name not in frame_objects:
                    frame_objects[name] = [{"Mask": mask, "Center": center}]
                else:
                    frame_objects[name].append({"Mask": mask, "Center": center})

            for k, v in frame_objects.items():
                if k not in video_object_dict:
                    video_object_dict[k] = v
                else:
                    video_object_dict[k].extend(v)

            frame_count += 1
        total_time = time.time() - total_start_time
        print(f"Total Process Time: {total_time:.2f}s")
        self.cap.release()
        self.saver.end()
        return video_object_dict


class YoloV8Segmenter(VideoSegmenter):
    def __init__(self, path, model, videoSaver, start_detection, end_detection) -> None:
        super().__init__(path, videoSaver, start_detection, end_detection)
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
                    confs = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()
                    bbss = r.boxes.xyxy.cpu().numpy()
                    xy = r.masks.xy
                    im_array = r.plot()
                    image = np.asarray(Image.fromarray(im_array[..., ::-1]))
                    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                    centers, masksArray = [], []

                    for coords in xy:
                        xCenter = sum(x for x, _ in coords) / len(coords)
                        yCenter = sum(y for _, y in coords) / len(coords)
                        mask = [[x, y] for x, y in coords]
                        centers.append([int(xCenter), int(yCenter)])
                        masksArray.append(mask)

                    for i, center in enumerate(centers):
                        name = "Eye" if classes[i] == 0 else "Needle"
                        data.append({
                            "Name": name,
                            "Conf": f"{confs[i]:.2f}",
                            "Center": center,
                            "BB": bbss[i],
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

    cap = cv.VideoCapture(input_file)
    fps = int(cap.get(cv.CAP_PROP_FPS)) if cap.isOpened() else 10
    cap.release()

    saver = VideoSaver(path=temp_output_path, fps=fps)
    model = YOLO("server/bestIman.pt")
    segmenter = YoloV8Segmenter(path=input_file, model=model, videoSaver=saver, start_detection=video.start_time,
                                end_detection=video.end_time)

    segmenter.segmentVideo()

    command = f"ffmpeg -i {temp_output_path}  -hls_time 10 -hls_playlist_type vod {hls_output_path}"
    subprocess.run(command, shell=True, check=True)

    video.hls_ready = True
    video.save()
