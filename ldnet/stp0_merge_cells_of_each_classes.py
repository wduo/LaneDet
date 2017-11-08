import os
import shutil
import tensorflow as tf

# All classes:      class 0,           class 1,      class 2
list_of_classes = ["cluttered_cells", "lane_cells", "road_surface_cells"]

cells_count = 0  # 统计复制了多少个cells


def merge_cells_of_each_classes(emg_cells_root, merged_cells):
    """
    将每张图片的三类cells分别地合并到对应的三个文件夹中
    The three types of cells of each picture were merged into the corresponding three folders.
    :param emg_cells_root: 此目录中 一张图片一个文件夹 包含当前图片和它的三类cells所在的目录
    :param merged_cells: 此目录中 一个类一个文件夹 合并所有图片的对应类的cells到对应的文件夹
    :return: none
    """
    if not os.path.exists(merged_cells):
        os.mkdir(merged_cells)
        for name_list_of_classes in list_of_classes:
            os.mkdir(merged_cells + "/" + name_list_of_classes)

    global cells_count  # 统计复制了多少个cells

    # 返回所包含的文件夹名称的列表: [f00001.png, f00002.png, ...]
    sub_dirs = [x for x in os.listdir(emg_cells_root)]

    # sub_dir包含三个子目录 对应该图片的三个类 每个子目录中存放对应类的cells
    for sub_dir in sub_dirs:
        for class_dir in list_of_classes:
            # 从根目录到当前图片的一个类的路径
            root_to_one_class_of_current_img = emg_cells_root + "/" + sub_dir + "/" + class_dir
            for single_cell_name in os.listdir(root_to_one_class_of_current_img):
                cells_count += 1  # 统计复制了多少个cells
                # 从根目录到当前图片的一个cell的路径
                root_to_one_cell = root_to_one_class_of_current_img + "/" + single_cell_name
                # 给当前图片的cells加上图片名前缀 以防合并所有图片的cells时重名
                shutil.copy(root_to_one_cell, merged_cells + "/" + class_dir + "/" + sub_dir + "_" + single_cell_name)


def main(_):
    merge_cells_of_each_classes(emg_cells_root="emg_cells_root_dir", merged_cells="merged_cells_dir")
    print("cells count: ", cells_count)


if __name__ == '__main__':
    tf.app.run()
