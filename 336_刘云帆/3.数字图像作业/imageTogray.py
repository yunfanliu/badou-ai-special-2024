import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
# RGB转化为Gray
img=cv2.imread("lenna.png")
#获取图片尺寸
h,w=img.shape[:2]


# 1.浮点算法:Gray=R0.3+G0.59+B0.11;
fudian_imgeGray=np.zeros((h,w),img.dtype)
# 2.整数方法:Gray=(R30+G59+B11)/100;
zhengshu_imgeGray=np.zeros((h,w),img.dtype)
# 3.移位方法:Gray =(R76+G151+B*28)>>8;
yiwei_imgeGray=np.zeros((h,w),img.dtype)
# 4.平均值法:Gray=(R+G+B)/3;
pingjun_imgeGray=np.zeros((h,w),img.dtype)
# 5.仅取绿色:Gray=G;
onlyGreen_imgeGray=np.zeros((h,w),img.dtype)


for i in range(h):
    for j in range(w):
        m=img[i,j]
        fudian_imgeGray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
        zhengshu_imgeGray[i,j]=int(m[0]*11+m[1]*59+m[2]*30)/100
        yiwei_imgeGray[i,j]=int(m[0]*28+m[1]*151+m[2]*76)/256
        pingjun_imgeGray[i,j]=int(m[0]+m[1]+m[2])/3
        onlyGreen_imgeGray[i,j]=int(m[1])


print(fudian_imgeGray)
print(zhengshu_imgeGray)
print(yiwei_imgeGray)
print(pingjun_imgeGray)
print(onlyGreen_imgeGray)

img_gray = rgb2gray(img)
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)


plt.subplot(231)
plt.imshow(fudian_imgeGray, cmap='gray')


plt.subplot(232)
plt.imshow(zhengshu_imgeGray, cmap='gray')


plt.subplot(233)
plt.imshow(yiwei_imgeGray, cmap='gray')


plt.subplot(234)
plt.imshow(pingjun_imgeGray, cmap='gray')


plt.subplot(235)
plt.imshow(onlyGreen_imgeGray, cmap='gray')



plt.subplot(236)
plt.imshow(img_binary, cmap='gray')
plt.show()