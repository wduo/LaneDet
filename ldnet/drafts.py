import tensorflow as tf
import numpy as np

# print(str(99))
#
# factor_for_h = 24  # 图片height方向上的划分因子
# factor_for_w = 12  # 图片width方向上的划分因子
# factor_unused = 3  # 不使用图片的上面 1/factor_unused 部分
# print(range(factor_for_h // factor_unused - 1, factor_for_h))

# from PIL import Image, ImageDraw
#
# im01 = Image.open("dir6_initial_imgs/f00060.png")
# # draw = ImageDraw.Draw(im01)
# # # draw.line([(0, 0), (100, 300), (200, 500)], fill=(255, 0, 0), width=5)
# # # draw.line([50, 10, 100, 200, 400, 300], fill=(0, 255, 0), width=10)
# # # im01.show()
# #
# # draw.rectangle((0, 0, 40, 20), outline="green")
# # draw.text([0, 0], "0_0", "red")
# # draw.rectangle((40, 0, 80, 20), outline="green")
# # draw.text([40, 0], "0_1", "red")
# # im01.show()
# #
# # re = tf.contrib.layers.l2_regularizer(0.0001)
#
# # im01 = np.array(im01)
# # print(im01.shape)
# #
# # cell = im01[0:100, 40:200]
# # print(cell.shape)
# #
# # for i in range(4, 6):
# #     print(i)
#
# # im01.convert('RGB')
#
# im2 = Image.new("RGB", (640, 480))
# draw = ImageDraw.Draw(im2)
# draw.rectangle(xy=(10, 100, 50, 200), fill=(0, 255, 0))
# draw.rectangle(xy=(60, 100, 100, 200), fill=(0, 255, 0))
#
# # merge two images using blend
# blend = Image.blend(im01, im2, 0.25)
#
# blend.show()

for i in range(3, 6):
    print(i)

ori = np.array([0, 1, 2, 3, 4, 5, 6])
part = ori[0:3]
print(part)
