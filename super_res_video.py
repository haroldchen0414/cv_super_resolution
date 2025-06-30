# -*- coding: utf-8 -*-
# author: haroldchen0414

from imutils.video import VideoStream
import imutils
import time
import cv2
import re
import os

modelPath = os.path.join("models", "LapSRN_x8.pb")
modelName = modelPath.split(os.path.sep)[-1].split("_")[0].lower()
modelScale = int(re.findall("_x(\d+)\.pb", modelPath)[0])

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(modelPath)
sr.setModel(modelName, modelScale)

# 调出摄像头
def online_video():
    vs = VideoStream(src=0).start()
    # 摄像头预热
    time.sleep(2)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=300)
        upscaled = sr.upsample(frame)
        cv2.imshow("Original", frame)
        cv2.imshow("High Res", upscaled)        
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break 

        cv2.destroyAllWindows()
        vs.stop()
    
# 处理视频
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        exit()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("视频总帧数: {}".format(nFrames))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("high_res_video.mp4", fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        upscaled = sr.upsample(frame)
        out.write(upscaled)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("视频处理完成")

#online_video()
#process_video("video.mp4")