"""
梯度简单来说就是求导，OpenCV 提供了三种不同的梯度滤波器，
或者说高通滤波器： Sobel，Scharr 和 Laplacian。
Sobel， Scharr 其实就是求一阶或二阶导数。
Scharr 是对 Sobel（使用小的卷积核求解求解梯度角度时）的优化。
Laplacian 是求二阶导数。

"""
import cv2
import numpy as np 
import matplotlib.pyplot as plt 




def Sobel_(img_path):
    img = cv2.imread(img_path, 0)
    print(img, img.shape)
    """
    在Sobel函数的第二个参数这里使用了cv2.CV_16S。
    因为OpenCV文档中对Sobel算子的介绍中有这么一句：
    “in the case of 8-bit input images it will result in truncated derivatives”。
    即Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，
    所以Sobel建立的图像位数不够，会有截断。因此要使用16位有符号的数据类型，即cv2.CV_16S
    """
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0,1)

    """
    在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。否则将无法显示图像，而只是一副灰色的窗口。
    dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
    可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片
    """
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)

    """
    由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
    dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
    其中alpha是第一幅图片中元素的权重，beta是第二个的权重，gamma是加到最后结果上的一个值。
    """
    dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0,5, 0)
    img_h1 = np.hstack([img, abs_x])
    img_h2 = np.hstack([img, dst])
    img_all = np.vstack([img_h1, img_h2])

    plt.figure(figsize=(20,10))
    plt.imshow(img_all, cmap=plt.cm.gray)
    #plt.show()

def Laplacian_(img_path):
    '''
    拉普拉斯对噪声敏感，会产生双边效果。
    不能检测出边的方向。通常不直接用于边的检测，
    只起辅助的角色，检测一个像素是在边的亮的一边
    还是暗的一边利用零跨越，确定边的位置。
    '''
    img = cv2.imread(img_path, 0)
    gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    dst = cv2.convertScaleAbs(gray_lap)
    plt.imshow(dst, cmap=plt.cm.gray)
    #plt.show()

def edge_detect(img):
    
    #高斯模糊,降低噪声
    blurred = cv2.GaussianBlur(img,(3,3),0)
    #灰度图像
    gray = cv2.cvtColor(blurred,cv2.COLOR_RGB2GRAY)
    #图像梯度
    xgrad = cv2.Sobel(gray,cv2.CV_16SC1,1,0)
    ygrad = cv2.Sobel(gray,cv2.CV_16SC1,0,1)
    #计算边缘
    #50和150参数必须符合1：3或者1：2
    edge_output = cv2.Canny(xgrad,ygrad,50,150)
    
    dst = cv2.bitwise_and(img,img,mask=edge_output)
    
    return edge_output, dst

if __name__ == "__main__":
    img_path = r"./丁细宜-右-斑块-纵切.jpg"
    Sobel_(img_path)
    Laplacian_(img_path)

    img = cv2.imread(img_path)
    edge_output, canny_edge = edge_detect(img.copy())

    plt.figure(figsize=(20, 8))

    plt.subplot(131)
    plt.imshow(img[:,:,::-1])

    plt.subplot(132)
    plt.imshow(canny_edge[:,:,::-1])

    plt.subplot(133)
    plt.imshow(edge_output, cmap=plt.cm.gray)

    plt.tight_layout()
    plt.show()

