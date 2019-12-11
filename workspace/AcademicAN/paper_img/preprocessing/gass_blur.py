import cv2
import numpy as np
img_path = './blur_15_100.png'
kernel_size = (15, 15)
sigma = 100

img = cv2.imread(img_path)
img = cv2.GaussianBlur(img, kernel_size, sigma)
img1 = np.float32(img) #转化数值类型
kernel = np.ones((20,20),np.float32)/25
dst = cv2.filter2D(img1,-1,kernel)
#cv2.filter2D(src,dst,kernel,auchor=(-1,-1))函数：
blur_img = 'blur_' + str(kernel_size[0]) + '_' + str(sigma) + '.png'
cv2.imwrite(blur_img, img)
