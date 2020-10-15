import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import json
from mpl_toolkits.mplot3d import Axes3D

'''
binocular_stereovision
双目立体
'''


class stereoCameral(object):
    def __init__(self):
        # 左相机内参数
        self.cam_matrix_left = np.array([[2490.82379, 0., 1560.38459], [0., 2490.07678, 1220.46872], [0., 0., 1.]])
        # 右相机内参数
        self.cam_matrix_right = np.array([[2420.77875, 0., 1530.22330], [0., 2420.27426, 1170.63536], [0., 0., 1.]])
        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.02712, -0.03795, -0.00409, 0.00526, 0.00000]])
        self.distortion_r = np.array([[-0.03348, 0.08901, -0.00327, 0.00330, 0.00000]])

        # 旋转矩阵
        om = np.array([-0.00320, -0.00163, -0.00069])
        self.R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
        # 平移矩阵
        self.T = np.array([-90.24602, 3.17981, -19.44558])


# 畸变校正(左右分别校正)
def undistortimg(img, mtx, dist):
    img = cv2.imread('D:/pic/bd1.jpg')
    imgh, imgw = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (imgw, imgh), 1, (imgw, imgh))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, rw, rh = roi
    dst = dst[y:y + rh, x:x + rw]
    return dst


# 极线校正(立体校正)

def getRectifyTransform(height, width, config, flag=1):
    if flag:
        # 读取矩阵参数
        left_K = config.cam_matrix_left
        right_K = config.cam_matrix_right
        left_distortion = config.distortion_l
        right_distortion = config.distortion_r
        R = config.R
        T = config.T
    else:
        left_K = np.array(config['M1'])
        right_K = np.array(config['M2'])
        left_distortion = np.array(config['dist1'])
        right_distortion = np.array(config['dist2'])
        R = np.array(config['R'])
        T = np.array(config['T'])
        print(left_K, '\n')
        print(right_K, '\n')
        print(left_distortion, '\n')
        print(right_distortion, '\n')
        print(R, '\n')
        print(T, '\n')

    # 计算校正变换
    if type(height) != "int" or type(width) != "int":
        height = int(height)
        width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正


def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    return rectifyed_img1, rectifyed_img2


# 视差计算
def sgbm(imgL, imgR):
    # SGBM参数设置
    blockSize = 8
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=64,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    # 计算视差图
    disp = stereo.compute(imgL, imgR)
    disp = np.divide(disp.astype(np.float32), 16.)  # 除以16得到真实视差图
    return disp


# 计算点云
def threeD(disp, Q):
    # 计算像素点的3D坐标（左相机坐标系下）
    points_3d = cv2.reprojectImageTo3D(disp, Q)

    points_3d = points_3d.reshape(points_3d.shape[0] * points_3d.shape[1], 3)
    return points_3d
    # X = points_3d[:, 0]
    # Y = points_3d[:, 1]
    # Z = points_3d[:, 2]

    # #选择并删除错误的点
    # remove_idx1 = np.where(Z <= 0)
    # remove_idx2 = np.where(Z > 15000)
    # remove_idx3 = np.where(X > 10000)
    # remove_idx4 = np.where(X < -10000)
    # remove_idx5 = np.where(Y > 10000)
    # remove_idx6 = np.where(Y < -10000)
    # remove_idx = np.hstack(
    #     (remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))
    #
    # points_3d = np.delete(points_3d, remove_idx, 0)
    #
    # #计算目标点（这里我选择的是目标区域的中位数，可根据实际情况选取）
    # if points_3d.any():
    #     x = np.median(points_3d[:, 0])
    #     y = np.median(points_3d[:, 1])
    #     z = np.median(points_3d[:, 2])
    #     targetPoint = [x, y, z]
    # else:
    #     targetPoint = [0, 0, -1]#无法识别目标区域
    #
    # return targetPoint


# 立体匹配，计算视差图
def sgbm(iml_rectified, imr_rectified):
    window_size = 9  # 匹配块大小3-11
    min_disp = 16  # 最大视差
    num_disp = 192 - min_disp  # 16的整数倍
    blockSize = window_size
    uniquenessRatio = 5  # 一般5-15
    speckleRange = 1  # 连接组件的最大尺寸1-2
    speckleWindowSize = 50  # 平滑视差区域的最大尺寸50-200
    disp12MaxDiff = 200  # 左右视察检查允许最大差异
    P1 = 600  # 控制视差平滑度的参数
    P2 = 2400  # 越大视差越平滑
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size,
                                   uniquenessRatio=uniquenessRatio, speckleRange=speckleRange,
                                   speckleWindowSize=speckleWindowSize, disp12MaxDiff=disp12MaxDiff, P1=P1, P2=P2)
    disp = stereo.compute(iml_rectified, imr_rectified).astype(np.float32) / 16.0
    # 转换为单通道图片
    disp = cv2.normalize(disp, disp, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return disp


def pcshow(mode=None, points=None, path=None):
    if mode == 'fromfile':
        # 打开点云数据文件
        if path != None:
            path = 'RawXYZ.xyz'
        with open(path, 'r') as f:
            point = f.read()
        # 数据预处理
        l1 = point.replace('\n', ',')
        # 将数据以“，”进行分割
        l2 = l1.split(',')
        l2.pop()
        # print(l2)
        # 将数据转为矩阵
        m1 = np.array(l2[0:160000])
        print(len(m1))
        # 变形
        m2 = m1.reshape(40000, 4)
        print(m2)
        m3 = []
        for each in m2:
            each_line = list(map(lambda x: float(x), each))
            m3.append(each_line)
        m4 = np.array(m3)
    else:
        m4 = points  # (n,3)
        # 列表解析x,y,z的坐标
        x = [k[0] for k in m4]
        y = [k[1] for k in m4]
        z = [k[2] for k in m4]
        # 开始绘图
        fig = plt.figure(dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        plt.title('point cloud')
        ax.scatter(x, y, z, c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')
        # ax.axis('scaled')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()


# 预处理图片
def inipic(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    w, h = img.shape[0:2]
    img = cv2.resize(img, (int(w / 4), int(h / 6)))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img


with open('D:/pic/stereoimages/bino.txt', 'r') as f:
    js = f.read()
    myconfig = json.loads(js)
config = stereoCameral()  # 获取标定参数

imgLG = inipic('D:/pic/deep/deep1.jpg')
imgRG = inipic('D:/pic/deep/deep2.jpg')
height, width = imgLG.shape[0:2]

imgLG = undistortimg(imgLG, np.array(config.cam_matrix_left), np.array(config.distortion_l))
imgRG = undistortimg(imgRG, np.array(config.cam_matrix_right), np.array(config.distortion_r))

map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config, 1)  # 计算基础矩阵
# Q深度差异映射函数

iml_rectified, imr_rectified = rectifyImage(imgLG, imgRG, map1x, map1y, map2x, map2y)  # 校正图像

disp = sgbm(iml_rectified, imr_rectified)  # 计算深度图

plt.subplot(131), plt.imshow(iml_rectified, cmap='gray')
plt.subplot(132), plt.imshow(imr_rectified, cmap='gray')
plt.subplot(133), plt.imshow(disp, cmap='gray')
plt.show()

# 计算深度，保存点云
# points = threeD(disp, Q)#计算目标点的3D坐标（左相机坐标系下）
# 点云显示 pcl(point cloud library)
# pcshow(points=points)

#ctrl+alt+l格式化代码
