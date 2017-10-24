import os
import tensorflow as tf
from PIL import Image

cwd = os.getcwd()


def create_tfrecords(isTrain):
    if isTrain:
      classes = ["train_dog","train_cat"]
      records_name = "train.tfrecords"
    else:
      classes=["eval_dog","eval_cat"]
      records_name = "eval.tfrecords"

    writer = tf.python_io.TFRecordWriter(records_name)
    for index, name in enumerate(classes):
        class_path = cwd + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((32, 32))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


def main(unused):
    create_tfrecords(isTrain=1)


if __name__ == '__main__':
    tf.app.run()
