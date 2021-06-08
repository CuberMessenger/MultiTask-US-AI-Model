import os
import cv2
import glob
import argparse
import numpy as np
from multiprocessing import Process
from skimage import exposure

def FindBox(img):
    rows = np.any(img, axis = 1)
    cols = np.any(img, axis = 0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    return int(top), int(bottom), int(left), int(right)

def CenterResize(image, mask, expandRate):
    top, bottom, left, right = FindBox(mask)
    height = bottom - top + 1
    width = right - left + 1

    expandedHeight = round(height * expandRate)
    expandedWidth = round(width * expandRate)

    if expandedHeight >= expandedWidth:
        sideLength = expandedHeight
        verticalStart = top - round((expandedHeight - height) / 2)
        horzontalStart = left - round((expandedWidth - width) / 2 + (expandedHeight - expandedWidth) / 2)
    else:
        sideLength = expandedWidth
        verticalStart = top - round((expandedHeight - height) / 2 + (expandedWidth - expandedHeight) / 2)
        horzontalStart = left - round((expandedWidth - width) / 2)

    verticalStart = max(verticalStart, 0)
    horzontalStart = max(horzontalStart, 0)

    return np.copy(image[verticalStart:verticalStart + sideLength, horzontalStart:horzontalStart + sideLength]),\
           np.copy(mask[verticalStart:verticalStart + sideLength, horzontalStart:horzontalStart + sideLength])

def Preprocess(start, end, dataIndexList, outputSize, args):
    for dataIndex in dataIndexList[start:end]:
        print(dataIndex)
        image = cv2.imread(os.path.join(args.DataFolderPath, dataIndex + "_Original.jpg"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(args.DataFolderPath, dataIndex + "_Mask.jpg"), cv2.IMREAD_GRAYSCALE)
        mask[mask > 200] = 255#magic value, some 1~10 value arround the roi,some 240+~254 value make the 255 roi have some glitch
        mask[mask != 255] = 0

        height, width = image.shape
        sideLength = max(width, height)

        answerImage = np.zeros((sideLength, sideLength), dtype = np.uint8)
        answerMask = np.zeros((sideLength, sideLength), dtype = np.uint8)
        if width >= height:
            delta = (width - height) // 2
            answerImage[delta:delta + height, :] = image
            answerMask[delta:delta + height, :] = mask
        else:
            delta = (height - width) // 2
            answerImage[:, delta:delta + width] = image
            answerMask[:, delta:delta + width] = mask

        image = cv2.resize(answerImage, (outputSize, outputSize))
        mask = cv2.resize(answerMask, (outputSize, outputSize))

        cv2.imwrite(os.path.join(args.SaveFolderPath, dataIndex + "_Image.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(os.path.join(args.SaveFolderPath, dataIndex + "_Mask.jpg"), mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--DataFolderPath", help = "define the data folder path", type = str)
    parser.add_argument("--SaveFolderPath", help = "define the save folder path", type = str)
    args = parser.parse_args()

    target = Preprocess
    outputSize = 256

    args.SaveFolderPath = os.path.join(args.SaveFolderPath, "Preprocessed" + str(outputSize) + "Original")
    os.mkdir(args.SaveFolderPath)

    rawIndexes = glob.glob(os.path.join(args.DataFolderPath, "*_Mask.jpg"))
    dataIndexList = [os.path.split(rawIndex)[-1].split("_")[0] for rawIndex in rawIndexes]

    # target(0, len(dataIndexList), dataIndexList, outputSize, args)
    l = len(dataIndexList)
    nop = 10
    step = l // nop
    ps = []
    ps.append(Process(target = target, args = (step * (nop - 1), l, dataIndexList, outputSize, args)))
    ps[-1].start()
    for i in range(nop - 1):
        ps.append(Process(target = target, args = (i * step, i * step + step, dataIndexList, outputSize, args)))
        ps[-1].start()

    for t in ps:
        t.join()
