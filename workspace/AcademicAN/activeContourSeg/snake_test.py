
from warnings import warn
import numpy as np
import cv2 as cv
from scipy.interpolate import RectBivariateSpline
from skimage.draw import circle_perimeter
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
from snake import getCircleContour

# img = np.zeros((100, 100))
# rr, cc = circle_perimeter(35, 45, 25)
# img[rr, cc] = 1
# img = gaussian(img, 2)

# s = np.linspace(0, 2*np.pi, 100)
# init = 50 * np.array([np.sin(s), np.cos(s)]).T + 50

# snake = active_contour(img, init, w_edge=0, w_line=1)  # doctest: +SKIP
# dist = np.sqrt((45-snake[:, 0])**2 + (35-snake[:, 1])**2)  # doctest: +SKIP
# int(np.mean(dist))  # doctest: +SKIP

# plt.imshow(snake)
# plt.show()


img_ = cv.imread("../TwoStage/袁洪稳-右-硬化-横切.jpg")  # 读入图像
img = rgb2gray(img_) # 灰度化
seg_img = cv.Canny(img_, 150, 200)
lalps_img = cv.Laplacian(img_, cv.CV_16S, ksize= 3)

# 圆的参数方程：(220, 100) r=100
t = np.linspace(0, 2*np.pi, 1000) # 参数t, [0,2π]
x = 559 + 120*np.cos(t)
y = 260 + 120*np.sin(t)

# 构造初始Snake
init = np.array([x, y]).T # shape=(400, 2)
#init = getCircleContour((2689, 1547), (510, 380), N=200)

# Snake模型迭代输出
snake = active_contour(gaussian(img, 3), snake=init, alpha=0.8,
                        beta=10, gamma=0.01, w_line=-200, w_edge=500
                        , convergence=0.1)
snake_ = active_contour(gaussian(img, 3), snake=init, alpha=0.8,
                       beta=10, gamma=0.01, w_line=200, w_edge=500, convergence=0.5)

# 圆的参数方程：(220, 100) r=100
t = np.linspace(0, 2*np.pi, 1000) # 参数t, [0,2π]
x_plaque = 559 + 63*np.cos(t)
y_plaque = 350 + 34*np.sin(t)

# 构造初始Snake
init_plaque = np.array([x_plaque, y_plaque]).T # shape=(400, 2)
#init = getCircleContour((2689, 1547), (510, 380), N=200)

# # Snake模型迭代输出
# snake_plaque = active_contour(gaussian(img, 3), snake=init_plaque, alpha=0.15,
#                               beta=10, gamma=0.001, w_line=-1, w_edge=10, convergence=0.5)
# snake_plaque_ = active_contour(gaussian(img, 3), snake=init_plaque, alpha=0.15,
#                               beta=10, gamma=0.001, w_line=1, w_edge=-10, convergence=0.5)

# 绘图显示
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap="gray")
plt.plot(init[:, 0], init[:, 1], '--r', lw=1)
plt.plot(snake[:, 0], snake[:, 1], '-b', lw=1)
plt.plot(snake_[:, 0], snake_[:, 1], 'g', lw=1)
plt.plot(init_plaque[:, 0], init_plaque[:, 1], '--r', lw=1)
# plt.plot(snake_plaque[:, 0], snake_plaque[:, 1], '-b', lw=1)
# plt.plot(snake_plaque_[:, 0], snake_plaque_[:, 1], 'g', lw=1)
plt.xticks([]), plt.yticks([]), plt.axis("off")

# plt.figure()
# plt.imshow(lalps_img)
plt.show()
