# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *

#读取图片并转为数组
im = array(Image.open("cat/cat.9.jpg"))
figure(1),title("yuantu"),imshow(im)
#红色通道
r = im[:,:,0]
#交换红蓝通道并显示
im[:,:,0] = im[:,:,2]
im[:,:,2] = r
figure(2),imshow(im)
show()
