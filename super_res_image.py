# -*- coding: utf-8 -*-
# author: haroldchen0414

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

image = cv2.imread("test.jpg")
image = cv2.resize(image, (60, 60))
print("原始图像宽:{}, 高:{}".format(image.shape[1], image.shape[0]))

startTIme = time.time()
upscaled = sr.upsample(image)
endTime = time.time()
print("耗时:{:.6f}".format(endTime - startTIme))
print("新图像图像宽:{}, 高:{}".format(upscaled.shape[1], upscaled.shape[0]))
cv2.imwrite("high_res.jpg", upscaled)