#-*- coding: utf-8 -*-
from PIL import Image
from pylab import *

#读取图片,灰度化，并转为数组
im = array(Image.open("cat/cat.9.jpg").convert('L'))
im2 = 255 - im # 对图像进行反相处理
im3 = (100.0/255) * im + 100 # 将图像像素值变换到 100...200 区间
im4 = 255.0 * (im/255.0)**2 # 对图像像素值求平方后得到的图像(二次函数变换，使较暗的像素值变得更小)
#2x2显示结果 使用第一个显示原灰度图
subplot(221)
title('f(x) = x')
gray()
imshow(im)
#2x2显示结果 使用第二个显示反相图
subplot(222)
title('f(x) = 255 - x')
gray()
imshow(im2)
#2x2显示结果 使用第三个显示100-200图
subplot(223)
title('f(x) = (100/255)*x + 100')
gray()
imshow(im3)
#2x2显示结果 使用第四个显示二次函数变换图
subplot(224)
title('f(x) =255 *(x/255)^2')
gray()
imshow(im4)
#输出图中的最大和最小像素值
print(int(im.min()),int(im.max()))
print(int(im2.min()),int(im2.max()))
print(int(im3.min()),int(im3.max()))
print(int(im4.min()),int(im4.max()))
show()
