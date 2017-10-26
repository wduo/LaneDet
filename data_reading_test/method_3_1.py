import matplotlib.pyplot as plt
import tensorflow as tf
# import numpy as np

print(tf.__version__)

image_raw = tf.gfile.FastGFile('imgs/21.jpg','rb').read()   #bytes
img = tf.image.decode_jpeg(image_raw)  #Tensor
img2 = tf.image.convert_image_dtype(img, dtype = tf.uint8)

with tf.Session() as sess:
    print(type(image_raw)) # bytes
    print(type(img)) # Tensor
    print(type(img2))

    print(type(img.eval())) # ndarray !!!
    print(img.eval().shape)
    print(img.eval().dtype)

    print(type(img2.eval()))
    print(img2.eval().shape)
    print(img2.eval().dtype)
    plt.figure(1)
    plt.imshow(img.eval())
    plt.show()
