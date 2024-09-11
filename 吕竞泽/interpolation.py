import cv2
import numpy as np
from matplotlib import pyplot as plt
def nearest(img):
    hight,wight,channel = img.shape
    empyt_pic = np.zeros((800,800,channel),np.uint8)
    sh,sw = 800/hight,800/wight
    for i in range(800):
        for j in range(800):
            x = int(i/sw + 0.5)
            y = int(j/sh + 0.5)
            empyt_pic[i,j] = img[x,y]
    return empyt_pic

def bilinear(img,out_dim):
    src_h,src_w,channel = img.shape
    dst_h,dst_w = out_dim[0],out_dim[1]
    if dst_h == src_h and dst_w == src_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_h,scale_w = float(src_h)/dst_h,float(src_w)/dst_w
    for channels in range(channel):
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                #中心对齐
                src_x = (dst_x + 0.5)*scale_w - 0.5
                src_y = (dst_y + 0.5)*scale_h - 0.5
                #找寻四个参考点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1,src_w-1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0+1,src_h-1)

                #套公式
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,channels] + (src_x - src_x0) * img[src_y0,src_x1,channels]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,channels] + (src_x - src_x0) * img[src_y1,src_x1,channels]
                dst_img[dst_y,dst_x,channels] = int((src_y1-src_y)*temp0 + (src_y - src_y0)*temp1)
    return dst_img
def histogram_gray(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.figure()  # 新建一个图像
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")  # X轴标签
    plt.ylabel("# of Pixels")  # Y轴标签
    plt.plot(hist)
    plt.xlim([0, 256])  # 设置x坐标轴范围
    plt.show()
def histogram_RGB(img):
    colors = ('b','g','r')
    chans = cv2.split(img)
    plt.figure()  # 新建一个图像
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")  # X轴标签
    plt.ylabel("# of Pixels")  # Y轴标签

    for(chan,co) in zip(chans,colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist,color=co)
        plt.xlim([0, 256])  # 设置x坐标轴范围
    plt.show()
def histogram_equal(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dist = cv2.equalizeHist(gray)
    hist = cv2.calcHist([dist], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Histogram Equalization")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()
    cv2.imshow("Histogram Equalization", np.hstack([dist, gray]))
def histogram_equalRGB(img):
    (b,g,r) = cv2.split(img)
    eb = cv2.equalizeHist(b)
    eg = cv2.equalizeHist(g)
    er = cv2.equalizeHist(r)
    img_out = cv2.merge((eb, eg, er))
    cv2.imshow("Histogram Equalization", img_out)


