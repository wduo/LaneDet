import os
import tensorflow as tf
from PIL import Image

cwd = os.getcwd()


def _generate_tfrecords(all_cells_dir, generated_records_name):
    """
    生成 tfrecords 文件, 执行该文件之前, 请先将每个图片的 所有的正样本cell 从all_cells文件夹 移动到 该图片对应的positive_cells文件夹中
    :param all_cells_dir: 由generate_subimgs.py生成的所有图片的cell所在的文件夹
    :param generated_records_name: 即将生成的 tfrecords 文件的名称
    :return: no return. but generate tfrecords in current dir.
    """
    classes = ["all_cells", "positive_cells"]
    writer = tf.python_io.TFRecordWriter(generated_records_name)
    # 遍历所有图片, single_img_folder_name指当前图片的所有cell所在的文件夹
    for single_img_folder_name in os.listdir(all_cells_dir):
        # 遍历当前图片的 正样本和负样本
        for index, name in enumerate(classes):
            # 当前图片的 正样本或负样本 所在的文件夹的路径
            positive_or_negative_samples_path = cwd + "/" + all_cells_dir + "/" + single_img_folder_name + "/" + name
            # 正样本或负样本文件夹中的cell
            for single_cell_name in os.listdir(positive_or_negative_samples_path):
                single_cell_path = positive_or_negative_samples_path + "/" + single_cell_name
                single_cell = Image.open(single_cell_path)
                single_cell = single_cell.resize((32, 32))
                img_raw = single_cell.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())

    writer.close()


def main(_):
    _generate_tfrecords(all_cells_dir="emg_cells", generated_records_name="ld_train.tfrecords")


if __name__ == '__main__':
    tf.app.run()
