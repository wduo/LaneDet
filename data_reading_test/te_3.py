# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *

#读取图片,灰度化，并转为数组
im = array(Image.open("cat/cat.9.jpg").convert('L'),'f')
#输出数组的各维度长度以及类型
print(im.shape,im.dtype)
#输出坐标100,100的值
print(im[100,100])
