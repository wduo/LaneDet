import os
import tensorflow as tf 
# import matplotlib.pyplot as plt
from pylab import *
from PIL import Image

cwd = os.getcwd()

filename_queue = tf.train.string_input_producer(["train.tfrecords"])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                   })
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [224, 224, 3])
label = tf.cast(features['label'], tf.int32)

with tf.Session() as sess:
    init_op = tf.local_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(20):
        example, l = sess.run([image,label])
        img=Image.fromarray(example, 'RGB')
        # img.save(cwd+'/'+str(i)+'_Label_'+str(l)+'.jpg')
        # print(example, l)
        figure(i),imshow(img)
        print(example[100,100,2])
    show()
    coord.request_stop()
    coord.join(threads)
