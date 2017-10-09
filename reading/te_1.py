# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *

#读取图片并转为数组
im = array(Image.open("cat/cat.9.jpg"))
#输出数组的各维度长度以及类型
print(im.shape,im.dtype)
#输出位于坐标100,100，颜色通道为r的像素值
print(im[100,100,0])
#输出坐标100,100的rgb值
print(im[100,100])
print(im.shape,im.dtype)
