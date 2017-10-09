import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

print(tf.__version__)

image_value = tf.read_file('imgs/21.jpg')
img = tf.image.decode_jpeg(image_value, channels=3)

with tf.Session() as sess:
    print(type(image_value)) # bytes
    print(type(img)) # Tensor
    #print(type(img2))

    print(type(img.eval())) # ndarray !!!
    print(img.eval().shape)
    print(img.eval().dtype)

#    print(type(img2.eval()))
#    print(img2.eval().shape)
#    print(img2.eval().dtype)
    plt.figure(1)
    plt.imshow(img.eval())
    plt.show()
