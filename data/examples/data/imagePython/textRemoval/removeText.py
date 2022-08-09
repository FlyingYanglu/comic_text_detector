import cv2
import numpy as np

originalImage = cv2.imread("shino/shino1.png")

textMaskImage = cv2.imread("textMask.png", 0)
textMaskImageInverse = cv2.bitwise_not(textMaskImage)

cv2.imwrite("textMaskImageInverse.png", textMaskImageInverse)

print("check shape of original image and mask", originalImage.shape)

for rowIndex, row in enumerate(textMaskImage):
    for pixelIndex, pixel in enumerate(row):
        if (pixel.tolist() == [0, 0, 0]):
            originalImage[rowIndex, pixelIndex] = [255, 255, 255]
        else:
            notBlackPixel = True

imageWithoutText = cv2.inpaint(originalImage, textMaskImageInverse, 150, cv2.INPAINT_NS)

cv2.imwrite("output.png", imageWithoutText)

