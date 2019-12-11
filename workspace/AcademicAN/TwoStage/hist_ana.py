from matplotlib import pyplot as plt
import cv2
import numpy as np

img = cv2.imread('./test_tmp/hist/蔡春灵-左-硬化-横切.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#plt.imshow(img, cmap=plt.cm.gray)
#plt.imshow(img)

hist = cv2.calcHist([img], [1], None, [256], [0, 256])

##plt.figure()
#plt.title("Grayscale Histogram")
#plt.xlabel("Bins")
#plt.ylabel("# of Pixels")
#plt.plot(hist)
#plt.xlim([0, 256])
#plt.ylim([0, 4000])
#plt.show()
#
#plt.figure()
#plt.hist(img.ravel(), 256, [0, 256])
#plt.ylim([0, 4000])
#
#
#color = ('b','g','r')
#for i,col in enumerate(color):
#    histr = cv2.calcHist([img],[i],None,[256],[0,256])
#    plt.plot(histr,color = col)
#    plt.xlim([0,256])
#    plt.ylim([0, 4000])
#plt.show()
#
mask = np.zeros(img.shape[:2], np.uint8)
mask[110:370, 320:645] = 255
#mask[246:530, 162:880] = 225
mask_img = cv2.bitwise_and(img, img, mask = mask)

hist_full = cv2.calcHist([img],[0],None,[256],[0,2])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,2])

hist_full_1 = cv2.calcHist([img],[0],None,[256],[0.6,256])
hist_mask_1 = cv2.calcHist([img],[0],mask,[256],[0.6,256])
#cv2.imshow('fd', mask_img)
#cv2.waitKey(0)
#plt.figure(figsize=(9,7))
#plt.subplot(221), plt.imshow(img, 'gray'), plt.title("Longitudinal Raw Image")
##plt.subplot(222), plt.imshow(mask,'gray')
#plt.subplot(222), plt.plot(hist_full_1), plt.xlim([0.5, 256]), plt.title("Histogram")
#plt.subplot(223), plt.imshow(mask_img, 'gray'), plt.title("Longitudinal Vessel Image")
#plt.subplot(224), plt.plot(hist_mask_1), plt.xlim([0.5,256]), plt.title("Histogram")
#plt.savefig(fname="Longitudinal.svg",format="svg")
#plt.show()
plt.figure(figsize=(9,7))

plt.subplot(221), plt.imshow(img, 'gray'), plt.title("Transverse Raw Image")
#plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(222), plt.plot(hist_full_1), plt.xlim([0.5, 256]), plt.title("Histogram")
plt.subplot(223), plt.imshow(mask_img, 'gray'), plt.title("Transverse Vessel Image")
plt.subplot(224), plt.plot(hist_mask_1), plt.xlim([0.5,256]), plt.title("Histogram")
plt.savefig(fname="Transverse.svg",format="svg")
plt.show()

#plt.figure(figsize=(10,8))
#plt.subplot(231), plt.imshow(img, 'gray'), plt.title("Longitudinal Raw Image")
##plt.subplot(222), plt.imshow(mask,'gray')
#plt.subplot(232), plt.plot(hist_full), plt.xlim([0, 2.1])
#plt.subplot(233), plt.plot(hist_full_1), plt.xlim([0.8, 256])
#plt.subplot(234), plt.imshow(mask_img, 'gray'), plt.title("Longitudinal Vessel Image")
#plt.subplot(235), plt.plot(hist_mask),plt.xlim([0, 2.1]) 
#plt.subplot(236), plt.plot(hist_mask_1), plt.xlim([0.8,256])
#plt.savefig(fname="Longitudinal.svg",format="svg")
#plt.show()
