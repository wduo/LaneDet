import os
import tensorflow as tf
import numpy
from pylab import *
from PIL import Image

cwd = os.getcwd()

def create_record():
    classes=["dog","cat"]
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224))
            img_raw = img.tobytes()              #将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())  #序列化为字符串
    writer.close()
    
    '''
    for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
    
        image = example.features.feature['image'].bytes_list.value
        label = example.features.feature['label'].int64_list.value
        # 可以做一些预处理之类的
        print(image, label)
    '''

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label

if __name__=='__main__':
    create_record()
    img, label = read_and_decode("train.tfrecords")

    #使用shuffle_batch可以随机打乱输入
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=2, capacity=2000,
                                                min_after_dequeue=1000)
    # init = tf.initialize_all_variables()
    init = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(coord=coord)
        # threads = tf.train.start_queue_runners(sess=sess)
        for i in range(2):
            val, l= sess.run([img_batch, label_batch])
            print(type(val), type(l))
            print(val.shape, l.shape)
            print(val.dtype, l.dtype)
            val0=val[1,:,:,:]
            figure(i), imshow(val0), show()

        coord.request_stop()
        coord.join(threads)
