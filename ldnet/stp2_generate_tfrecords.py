import os
import tensorflow as tf
from PIL import Image

cwd = os.getcwd()


def generate_tfrecords(labeled_merged_cells_dir, generated_records_name):
    """
    Generate tfrecords file.
    :param labeled_merged_cells_dir: labeled and merged cells dir.
    :param generated_records_name: the neme of tfrecords file.
    :return: no return. but generate tfrecords in current dir.
    """
    classes = ["lane_cells", "road_surface_cells", "cluttered_cells"]
    writer = tf.python_io.TFRecordWriter(generated_records_name)

    for class_index, class_name in enumerate(classes):
        class_path = cwd + "/" + labeled_merged_cells_dir + "/" + class_name
        for cell_name in os.listdir(class_path):
            cell_path = class_path + "/" + cell_name
            cell = Image.open(cell_path)
            # the initial [height, width] of cells is [30, 40].
            cell = cell.resize((34, 34))
            cell_raw = cell.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[class_index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[cell_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


def main(_):
    # 0: train, 1: validation
    flag = 1
    train_or_validation = ["train", "validation"]
    generate_tfrecords(labeled_merged_cells_dir="dir1_merged_cells/" + train_or_validation[flag],
                       generated_records_name="ldnet_" + train_or_validation[flag] + ".tfrecords")
    print("generated " + train_or_validation[flag] + " tfrecord.")


if __name__ == '__main__':
    tf.app.run()
