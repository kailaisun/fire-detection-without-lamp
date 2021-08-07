import cv2
import os
import re
import numpy as np
from shutil import copyfile
import os
import math
# os.environ["CUDA_VISIBLE_DEVICES"]='1'

def boxtobox(label,conf,x,y,w,h):
    xleft=int(x-w/2)
    xright=int(x+w/2)
    ytop=int(y-h/2)
    ybot=int(y+h/2)
    box=[label,conf,xleft,ytop,xright,ybot]
    return box


def seg(img,xleft,ytop,xright,ybot):
    if xright<=xleft or ybot<=ytop:
        return img
    height,width,channel=img.shape

    if width<=xright or height<=ybot:
        return img
    x_center=(xleft+xright)/2.0
    y_center=(ytop+ybot)/2.0
    w=xright-xleft
    h=ybot-ytop

    roi=img[ytop:ybot,xleft:xright]
    rect=[max(int(x_center-math.sqrt(2)*w/2.0+0.5),0),max(int(y_center-math.sqrt(2)*h/2.0+0.5),0),
          min(int(x_center+math.sqrt(2)*w/2.0+0.5),width),min(int(y_center+math.sqrt(2)*h/2.0+0.5),height)]
    #xleft,ytop,xright,ybot

    #相对坐标
    location=(xleft-rect[0],ytop-rect[1],xright-rect[0],ybot-rect[1])

    rectimg=img[rect[1]:rect[3],rect[0]:rect[2]]

    mask = np.zeros(rectimg.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)


    # print(rect,w,x_center,width)
    cv2.grabCut(rectimg, mask, location, bgdModel, fgdModel, 5,  cv2.GC_INIT_WITH_RECT)  # 函数返回值为mask,bgdModel,fgdModel
    mask2 = np.where((mask == 2) | (mask == 0), 1, 0).astype('uint8')  # 0和2做背景
    rectimg = rectimg * mask2[:, :, np.newaxis]  # 使用蒙板来获取前景区域

    for i in range(rect[2]-rect[0]):
        for j in range(rect[3]-rect[1]):
            try:
                img[rect[1]+j,rect[0]+i]=rectimg[j,i]
            except:
                print('no replace')

    # rectimg.copyTo(imageROI,mask2)
    # cv2.imshow('1', roi)
    # cv2.imshow('0', img)
    # cv2.waitKey(0)
    return img