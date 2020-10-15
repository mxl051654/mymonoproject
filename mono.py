import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import json
import open3d


def scaleimg(img, sh, sw):
    w, h = img.shape[0:2]
    imgc = img.copy()
    imgc = cv2.resize(imgc, (int(w / sw), int(h / sh)))
    return imgc


def writetofile(dict, path):
    for index, item in enumerate(dict):
        print(index, ":", item, ':', type(dict[item]), ":", dict[item], ' ok\n')
        dict[item] = np.array(dict[item])
        dict[item] = dict[item].tolist()
    js = json.dumps(dict)
    with open(path, 'w') as f:
        f.write(js)
        print("参数已成功保存到文件")


def readfromfile(path):
    with open(path, 'r') as f:
        js = f.read()
        mydict = json.loads(js)
    return mydict


# 标定板尺寸
w = 8
h = 6


#   单目相机标定


def monocameracalibration(path):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    imglist = glob.glob('D:/pic/stereoimages/*.jpg')
    print("总共", len(imglist), "张图片\n")
    for i, fname in enumerate(imglist):
        img = cv2.imread(imglist[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
        if ret:
            objpoints.append(objp)
            # 提取亚像素焦点
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # img = cv2.drawChessboardCorners(img, (8,6), corners2,ret)
        print("第", i, "张图片已完成角点提取\n")
        if i == 5: break
    #  保存结果到文件
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dict = {'ret': ret, 'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
    writetofile(dict, path)


# 根据标定数据，校正图片
def undistortimgs(imgspath, mtx, dist):  # 配置路径，图片路径
    # 获取目录下所有匹配文件
    pathlist = glob.glob(imgspath)
    imglist = []
    for i, fname in enumerate(pathlist):
        img = cv2.imread(pathlist[i])
        h, w = img.shape[:2]
        # retrieve only sensible pixels alpha=0 ,越接近0，，越小
        # keep all the original image pixels if there is valuable information in the corners alpha=1
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]  # 裁剪
        imglist.append(dst)
        # imglist.append(img)
    n = len(imglist)
    print("畸变校正完毕\n总共", n, "张图片")
    return imglist, n, h, w  # 返回校正好的图片


def draw(img, cornors, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img


def pose_estimation(configpath):
    config = readfromfile(configpath);
    mtx = np.array(config['mtx'])
    dist = np.array(config['dist'])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    # 得到假设的棋盘面三维坐标点（x，y，0）
    axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                       [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])
    imglist = glob.glob('D:/pic/stereoimages/*.jpg')
    print("总共", len(imglist), "张图片\n")
    for i, fname in enumerate(imglist):
        img = cv2.imread(imglist[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # solvepnp 已知三维点（通过前面图像序列求得的坐标）和(下一张图像中)对应的二维点（角点匹配）,计算R,t
            # 即rotation vector,translation vector
            # 这里三维点为假设的三维棋盘坐标，对应二维坐标为图像中的角点,后两个为标定得到的内参
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            # project 3D points to image plane
            # 三维点axis，R，t,内参，畸变参数,输出imgpts为空间点在图像上的投影点
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = draw(img, corners2, imgpts)
            simg = scaleimg(img, 4, 4)  # 缩小为1/4
            cv2.imshow('img', simg)
            k = cv2.waitKey(0) & 0xFF
            # if k == ord('s'):
            #     cv2.imwrite(fname[:6] + '.png', img)
            if i == 5: break
        print("位姿估计完毕")


# ----------------------------------------------------------------------------------------------------------------


def draw_polarlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    print(img1.shape)
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def epipolar_geometric(img1, img2, K):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    # orb = cv2.xfeatures2d.SIFT_create()
    # 获取角点及描述符
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    print("角点数量：", len(kp1), len(kp2))
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.match(des1, des2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    print("匹配点对数量：", len(matches))
    good = []
    pts1 = []
    pts2 = []
    # 筛选匹配点对
    # for i in range(len(matches)):
    #     print(matches[i].distance)
    for i, m in enumerate(matches):
        # if m.distance < 0.8 * n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        # print(i, kp2[m.trainIdx].pt, kp2[m.trainIdx].pt, "\n")
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    #  !!!特征点匹配精度有待提高
    # 计算基本矩阵
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    E, mask = cv2.findEssentialMat(pts1, pts2, K)
    # E = np.transpose(K, (1, 0)) @ F @ K
    ret, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    # 绘制极线
    # We select only inlier points
    # pts1 = pts1[mask.ravel() == 1]
    # pts2 = pts2[mask.ravel() == 1]
    #
    # # Find epilines corresponding to points in right image (second image) and
    # # drawing its lines on left image
    # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    # lines1 = lines1.reshape(-1, 3)
    # img5, img6 = draw_polarlines(img1_gray, img2_gray, lines1, pts1, pts2)
    # # Find epilines corresponding to points in left image (first image) and
    # # drawing its lines on right image
    # lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    # lines2 = lines2.reshape(-1, 3)
    # img3, img4 = draw_polarlines(img2_gray, img1_gray, lines2, pts2, pts1)
    # plt.subplot(121), plt.imshow(img5)
    # plt.subplot(122), plt.imshow(img3)
    # plt.show()
    return F, R, t


# 从本质矩阵解出R，t
# def testRt(R,t):
#     p = R @ np.array([[1], [1], [1]]) + t
#     print("p",p)
#     return p
# def solveF(F, K):
#     E = np.transpose(K, (1, 0)) @ F @ K
#     print("E:", E)
#     U, Z, V = np.linalg.svd(E)
#     V=np.transpose(V,(1,0))
#     print("U", U, "\nZ", Z, "\nV", V)
#     t = np.array(U[:, 2]).reshape((3,1))
#     W = np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]])
#     R1 = U @ W @ V
#     R2 = U @ np.transpose(W, (1, 0)) @ V
#
#     if testRt(R1, t)[2, 0] > 0:
#         print("R:", R1, "\nt", t)
#         return R1, t
#     elif testRt(R2, t)[2, 0] > 0:
#         print("R:", R2, "\nt", t)
#         return R2, t
#     elif testRt(R1, -t)[2, 0] > 0:
#         print("R:", R1, "\nt", -t)
#         return R1, -t
#     elif testRt(R2, -t)[2, 0] > 0:
#         print("R:", R2, "\nt", -t)
#         return R2, -t
#     else:
#         return R1,t


# 极线校正(立体校正)
# def getRectifyTransform(height, width, config):
def getRectifyTransform(height, width, mtx, dist, R, t):
    # left_K = np.array(config['M1'])
    # right_K = left_K
    # left_distortion = np.array(config['dist1'])
    # right_distortion = left_distortion
    # R = np.array(config['R'])
    # T = np.array(config['T'])
    height = int(height)
    width = int(width)
    # alpha越接近0,图片越小
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx, dist, mtx, dist, (width, height),
                                                      R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.7)
    # print("HEight:", height, "\nWidth:", width)
    # print("\nroi1:", roi1, "\nroi2", roi2)
    # initUndistortRectifyMap同时考虑畸变和对极几何
    dist = dist / 10;
    map1x, map1y = cv2.initUndistortRectifyMap(mtx, dist, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx, dist, R2, P2, (width, height), cv2.CV_32FC1)
    # 返回map数组为图像坐标的一一对应映射
    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    #                         原图，映射函数，插值方式
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT)
    return rectifyed_img1, rectifyed_img2


# 立体匹配，计算视差图
def sgbm(iml_rectified, imr_rectified):
    # min_disp = 16  # 最大视差
    # num_disp = 80 - min_disp  # 16的整数倍
    # blockSize = 8  # 匹配块大小3-11
    # uniquenessRatio = 10  # 一般5-15
    # speckleRange = 100  # 连接组件的最大尺寸1-2
    # speckleWindowSize = 100   # 100  # 平滑视差区域的最大尺寸50-200
    # disp12MaxDiff = -1  # 20 # 左右视察检查允许最大差异
    # P1 = 8*3*8  # 控制视差平滑度的参数
    # P2 = 32*3*8  # 越大视差越平滑
    # stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=blockSize,
    #                                uniquenessRatio=uniquenessRatio, speckleRange=speckleRange,
    #                                speckleWindowSize=speckleWindowSize, disp12MaxDiff=disp12MaxDiff,
    #                                P1=P1, P2=P2,  mode=cv2.STEREO_SGBM_MODE_HH)
    blockSize = 5
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
    disp = stereo.compute(iml_rectified, imr_rectified).astype(np.float32)
    # disp = np.divide(disp.astype(np.float32), 256.)
    # 转换为单通道图片
    disp = cv2.normalize(disp, disp, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # 获得视差图像是CV_16S类型的，这样的视差图的每个像素值由一个16bit表示，
    # 其中低位的4位存储的是视差值得小数部分，所以真实视差值应该是该值除以16。
    # 在进行映射后应该乘以16，以获得毫米级真实位置。
    return disp


def threeD(disp, Q):
    # 计算像素点的3D坐标（左相机坐标系下）
    points_3d = cv2.reprojectImageTo3D(disp, Q)

    # f = 10
    # b = 100
    # cx = 1000
    # cy = 1400
    # points_3d = np.ones([disparity.shape[0], disparity.shape[1], 3])
    # for x in range(disparity.shape[0]):
    #     for y in range(disparity.shape[1]):
    #         w = -points_3d[2] / b
    #         points_3d[0] = x / w
    #         points_3d[1] = y / w
    #         points_3d[2] = f / w

    print("point_3d:", points_3d.shape)
    points_3d = points_3d.reshape(points_3d.shape[0] * points_3d.shape[1], 3)
    return points_3d


# def pcshow(mode=None, points=None, path=None):
#     if mode == 'fromfile':
#         with open(path, 'r') as f:
#             point = f.read()
#         l1 = point.replace('\n', ',')
#         l2 = l1.split(',')
#         l2.pop()
#         # 将数据转为矩阵
#         m1 = np.array(l2[0:160000])
#         m2 = m1.reshape(40000, 4)
#         m3 = []
#         for each in m2:
#             each_line = list(map(lambda x: float(x), each))
#             m3.append(each_line)
#         m4 = np.array(m3)
#     else:
#         m4 = points  # (n,3)
#     x = [k[0] for k in m4]
#     y = [k[1] for k in m4]
#     z = [k[2] for k in m4]
#     fig = plt.figure(dpi=120)
#     ax = fig.add_subplot(111, projection='3d')
#     plt.title('point cloud')
#     # 绘制散点图
#     ax.scatter(z, x, y, c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#     plt.show()


if __name__ == "__main__":
    # 单目相机标定
    path = 'D:/pic/stereoimages/mono.txt'  # 参数文件地址
    # monocameracalibration(path)  # 计算相机模型，并保存到文件
    config = readfromfile(path)
    mtx = np.array(config['mtx'])
    dist = np.array(config['dist'])
    # print("K:", mtx, "\ndist:", dist)
    # 图片地址
    ipath = 'D:/pic/mosaic/room*.jpg'  # 0,1
    # ipath = 'D:/pic/deep/pan*.jpg'  # 3,4
    # ipath = 'D:/pic/mosaic/l*.jpg'   # 1,2

    imglist, sumofimg, height, width = undistortimgs(ipath, mtx, dist)  # 返回校正好的图像序列，高，宽
    # for img in imglist:
    #     simg = scaleimg(img, 4, 4)
    #     cv2.imshow('img', simg)
    #     cv2.waitKey()
    # 位姿检测
    # pose_estimation(path)

    #   从图像序列重建
    #   初始化，对图像0，1，利用对极几何约束，找特征点，匹配，绘制极线,极线矫正，
    #   立体匹配（找到每个点的对应点），计算视差图（对应点的x之差），计算三维坐标(三角测量)z=f*b(基线长)/视差

    img1 = imglist[1]
    img2 = imglist[0]
    F, R, t = epipolar_geometric(img1, img2, mtx)  # 通过特征点计算基础矩阵,并解出R,t
    print("R:", R, "\nt:", t)
    # 计算校正变换
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, mtx, dist, R, t)
    #
    rectifyed_img1, rectifyed_img2 = rectifyImage(img1, img2, map1x, map1y, map2x, map2y)
    # 立体匹配
    disparity = sgbm(rectifyed_img1, rectifyed_img2)

    # 畸变校正后的图
    plt.subplot(231), plt.imshow(img1, cmap='gray')
    plt.subplot(232), plt.imshow(img2, cmap='gray')
    # 极线校正后的图
    plt.subplot(233), plt.imshow(rectifyed_img1, cmap='gray')
    plt.subplot(234), plt.imshow(rectifyed_img2, cmap='gray')
    # 视差图
    plt.subplot(235), plt.imshow(disparity, cmap="hot")
    plt.show()

    points = threeD(disparity, Q)

    # 筛选点
    left = points[[points[i, 2] != np.inf for i in range(len(points))], :]
    # for i in range(len(left)):
    #     left[i]=left[i]/10
    print("可用点：", left.shape)

    pcd = open3d.geometry.PointCloud()  # 首先建立一个pcd类型的数据（这是open3d中的数据形式）
    # 将点云转换成open3d中的数据形式并用pcd来保存，以方便用open3d处理
    pcd.points = open3d.utility.Vector3dVector(left)
    # 将点云从oepn3d形式转换为矩阵形式
    np_points = np.asarray(pcd.points)
    # 用open3d可视化生成的点云
    open3d.visualization.draw_geometries([pcd])

    #   方向一：
    #   根据前面求得三维坐标，对后续每个图片角点匹配，通过solvepnp求解相机运动---->slam
    #   方向二：
    #   同上，找匹配点，根据每两张图像计算出一批三维坐标，然后点云匹配，减小累积误差

    # 对于三维点云，表面三角化
