
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

'''
图像拼接
'''


# 采集特征点
def mathch2h(im1, im2):
    orb = cv2.ORB_create(100)
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)                 # 检测特征点
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)       # 暴力匹配
    matches = bf.match(des1, des2)
    pts1 = []
    pts2 = []
    for m in range(len(matches)):
            pts1.append(kp1[matches[m].queryIdx].pt)             # 添加匹配点对对应关键点的图像坐标
            pts2.append(kp2[matches[m].trainIdx].pt)
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC)          # 计算基础矩阵F，八点法
    # 寻则内部点
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    p1 = pts1.reshape(len(pts1) * 2, 1)
    p2 = pts2.reshape(len(pts2) * 2, 1)

    H = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    return H


def img_mosaic(piclist):
    C = np.array([[0.5, 0, 1000],
                  [0, 0.5, 1000],
                  [0, 0, 1]])
    n = len(piclist)
    size = piclist[0].shape
    mid = piclist[int(n/2)].copy()   # 取中间视角作为最终视角
    result = cv2.warpAffine(mid, np.array([[0.5, 0, 1000], [0, 0.5, 1000]]),
                         (size[1], size[0]))
    for x in range(n):
        if x!=int(n/2):
            H, extra = mathch2h(piclist[x], mid)
            H = C@H
            piclist[x] = cv2.warpPerspective(piclist[x], H, (size[1], size[0]))
            for i in range(size[0]):
                for j in range(size[1]):
                    flag = (piclist[x][i, j, :] != np.array([0, 0, 0]))  # 黑
                    if flag.any():
                        result[i, j] = piclist[x][i, j]
    return result


if __name__ == '__main__':
    img1 = cv2.imread('D:/pic/mosaic/l1.jpg')
    img2 = cv2.imread('D:/pic/mosaic/l2.jpg')
    img3 = cv2.imread('D:/pic/mosaic/l3.jpg')
    imglist = [img1, img2, img3]
    size = img1.shape      # 3000,4000,3
    combine = img_mosaic(imglist)
    combine = cv2.resize(combine, (2 * size[0], size[1]))
    plt.imshow(combine)
    plt.show()