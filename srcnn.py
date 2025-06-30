# -*- coding: utf-8 -*-
# author: haroldchen0414

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from utils.hdf5datasetwriter import HDF5DatasetWriter
from utils.hdf5datasetgenerator import HDF5DatasetGenerator
from imutils import paths
from PIL import Image
import numpy as np
import shutil
import random
import PIL
import cv2
import os

class SRCNN:
    def __init__(self):
        self.datasetPath = "ukbench100"
        self.outputPath = "output"
        self.images = os.path.join(self.outputPath, "images")
        self.labels = os.path.join(self.outputPath, "labels")
        self.inputsDB = os.path.join(self.outputPath, "inputs.hdf5")
        self.outputsDB = os.path.join(self.outputPath, "outputs.hdf5")
        self.modelPath = os.path.join(self.outputPath, "srcnn.model")
        self.plotPath = os.path.join(self.outputPath, "plot.png")

        self.batchSize = 128
        self.epochs = 10
        self.scale = 2.0
        self.inputDim = 33
        self.labelSize = 21
        self.pad = int((self.inputDim - self.labelSize) / 2.0)
        self.stride = 14 # 33

    def build_dateset(self):
        for p in [self.images, self.labels]:
            if not os.path.exists(p):
                os.makedirs(p)
        
        imagePaths = list(paths.list_images(self.datasetPath))
        random.shuffle(imagePaths)
        total = 0

        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            (h, w) = image.shape[:2]
            w -= int(w % self.scale)
            h -= int(h % self.scale)
            image = image[0:h, 0:w]

            lowW = int(w * (1.0 / self.scale))
            lowH = int(h * (1.0 / self.scale))
            highW = int(lowW * (self.scale / 1.0))
            highH = int(lowH * (self.scale / 1.0))

            scaled = np.array(Image.fromarray(image).resize((lowW, lowH), resample=PIL.Image.BICUBIC))
            scaled = np.array(Image.fromarray(scaled).resize((highW, highH), resample=PIL.Image.BICUBIC))

            for y in range(0, h - self.inputDim + 1, self.stride):
                for x in range(0, w - self.inputDim + 1, self.stride):
                    crop = scaled[y:y + self.inputDim, x:x + self.inputDim]

                    target = image[y + self.pad:y + self.pad + self.labelSize, x + self.pad:x + self.pad + self.labelSize]
                    cropPath = os.path.sep.join([self.images, "{}.png".format(total)])
                    targetPath = os.path.sep.join([self.labels, "{}.png".format(total)])
                    cv2.imwrite(cropPath, crop)
                    cv2.imwrite(targetPath, target)
                    total += 1

        inputPaths = sorted(list(paths.list_images(self.images)))
        outputPaths = sorted(list(paths.list_images(self.labels)))

        inputWriter = HDF5DatasetWriter((len(inputPaths), self.inputDim, self.inputDim, 3), self.inputsDB)
        outputWriter = HDF5DatasetWriter((len(outputPaths), self.labelSize, self.labelSize, 3), self.outputsDB)

        for (inputPath, outputPath) in zip(inputPaths, outputPaths):
            inputImage = cv2.imread(inputPath)
            outputImage = cv2.imread(outputPath)
            inputWriter.add([inputImage], [-1])
            outputWriter.add([outputImage], [-1])

        inputWriter.close()
        outputWriter.close()

        shutil.rmtree(self.images)
        shutil.rmtree(self.labels)

    def build(self, width, height, depth):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(64, (9, 9), kernel_initializer="he_normal", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (1, 1), kernel_initializer="he_normal"))
        model.add(Activation("relu"))
        model.add(Conv2D(depth, (5, 5), kernel_initializer="he_normal"))
        model.add(Activation("relu"))

        return model
    
    def super_res_generator(self, input_data_gen, target_data_gen):
        while True:
            inputData = next(input_data_gen)[0]
            targetData = next(target_data_gen)[0]

            yield (inputData, targetData)

    def train(self):
        inputs = HDF5DatasetGenerator(self.inputsDB, self.batchSize)
        targets = HDF5DatasetGenerator(self.outputsDB, self.batchSize)

        opt = Adam(learning_rate=1e-3, decay=1e-3 / self.epochs)
        model = self.build(width=self.inputDim, height=self.inputDim, depth=3)
        model.compile(loss="mse", optimizer=opt)

        H = model.fit_generator(self.super_res_generator(inputs.generator(), targets.generator()), steps_per_epoch=inputs.numImages // self.batchSize, epochs=self.epochs, verbose=1)

        model.save(self.modelPath, overwrite=True)

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), H.history["loss"], label="loss")
        plt.title("Loss on super resolution training")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.plotPath)
        inputs.close()
        targets.close()

    def high_res(self, image_path, baseline_path="baseline.jpg", output_path="high_res.jpg"):
        model = load_model(self.modelPath)
        image = cv2.imread(image_path)
        (h, w) = image.shape[:2]
        w -= int(w % self.scale)
        h -= int(h % self.scale)
        image = image[0:h, 0:w]        

        lowW = int(w * (1.0 / self.scale))
        lowH = int(h * (1.0 / self.scale))
        highW = int(lowW * (self.scale / 1.0))
        highH = int(lowH * (self.scale / 1.0))

        scaled = np.array(Image.fromarray(image).resize((lowW, lowH), resample=PIL.Image.BICUBIC))
        scaled = np.array(Image.fromarray(scaled).resize((highW, highH), resample=PIL.Image.BICUBIC))
        cv2.imwrite(baseline_path, scaled)

        output = np.zeros(scaled.shape)
        (h, w) = output.shape[:2]

        for y in range(0, h - self.inputDim + 1, self.labelSize):
            for x in range(0, w - self.inputDim + 1, self.labelSize):
                crop = scaled[y:y + self.inputDim, x:x + self.inputDim].astype("float32")

                P = model.predict(np.expand_dims(crop, axis=0))
                P = P.reshape((self.labelSize, self.labelSize, 3))
                output[y + self.pad:y + self.pad + self.labelSize, x + self.pad:x + self.pad + self.labelSize] = P
                
        output = output[self.pad:h - ((h % self.inputDim) + self.pad), self.pad:w - ((w % self.inputDim) + self.pad)]
        output = np.clip(output, 0, 255).astype("uint8")

        cv2.imwrite(output_path, output)   

srcnn = SRCNN()
srcnn.build_dateset()
srcnn.train()
srcnn.high_res("test.jpg")