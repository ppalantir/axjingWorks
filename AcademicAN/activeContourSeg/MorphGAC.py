from skimage.segmentation.morphsnakes import morphological_geodesic_active_contour, inverse_gaussian_gradient,circle_level_set, ellipse_level_set
import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import time

Img = cv2.imread("../TwoStage/袁洪稳-右-硬化-横切.jpg")  # 读入原图
Image = Img
image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
img = np.array(image, dtype=np.float64)  # 读入到np的array中，并转化浮点类型

# 画初始轮廓
# Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
print(image.shape)
sobel_img = cv2.Canny(image, 10, 20)
gimg = inverse_gaussian_gradient(image, alpha=300, sigma=4.6)
plt.figure(1)
plt.imshow(gimg)

# 圆的参数方程：(220, 100) r=100
x_centre, y_centre = (563, 257)
x_axis_ellipse, y_axis_ellipse = (139, 129)

thre_area = image[125:384, 424:702]

threshold1 = np.sum(thre_area)/np.sum(image)
ils = circle_level_set(image.shape, (y_centre, x_centre), y_axis_ellipse+3)
start = time.time()
eils = ellipse_level_set(image.shape, (y_centre, x_centre),x_axis_ellipse, y_axis_ellipse+3)
print(time.time()-start)
u = morphological_geodesic_active_contour(gimg, 20, eils,threshold=0.002, smoothing=2,balloon=-1)


x_centre_s, y_centre_s = (567, 351)
x_axis_ellipse_s, y_axis_ellipse_s = (60, 20)

eils_s = ellipse_level_set(image.shape, (y_centre_s,x_centre_s), x_axis_ellipse_s+4,y_axis_ellipse_s+4)
u_s = morphological_geodesic_active_contour(gimg, 40, eils_s, smoothing=1,balloon=-1)

plt.figure(2)
plt.contour(u, [0.3], colors="r")
plt.contour(eils, colors='g')
plt.contour(eils_s, colors='g')
plt.contour(u_s, [0.3], colors='y')
plt.imshow(Image)
plt.show()
