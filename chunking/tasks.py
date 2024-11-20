import os
import subprocess
import cv2
from celery import shared_task
from .models import VideoModels

@shared_task
def convert_to_hls(video_id):
    video = VideoModels.objects.get(id=video_id)
    input_file = video.video_file.path


    video_title_safe = "".join(
        x if x.isalnum() else "_" for x in video.title)  # Replace special characters with underscores
    output_dir = os.path.join(os.path.dirname(input_file), 'output', video_title_safe)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    temp_output_path = os.path.join(output_dir, "processed_video.avi")


    cap = cv2.VideoCapture(input_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))


    top_left = (30, 30)
    bottom_right = (300, 300)
    rectangle_color = (0, 255, 0)
    thickness = 4
    name = "Amirrrrrr"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    text_color = (255, 255, 255)
    text_thickness = 2


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.rectangle(frame, top_left, bottom_right, rectangle_color, thickness)
        text_position = (top_left[0], bottom_right[1] + 40)
        cv2.putText(frame, name, text_position, font, font_scale, text_color, text_thickness, cv2.LINE_AA)
        output.write(frame)

    cap.release()
    output.release()


    hls_output_path = os.path.join(output_dir, "output.m3u8")
    command = f"ffmpeg -i {temp_output_path} -hls_time 10 -hls_playlist_type vod {hls_output_path}"
    subprocess.run(command, shell=True)

    video.hls_ready = True
    video.save()