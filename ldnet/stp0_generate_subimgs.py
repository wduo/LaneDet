import os
import tensorflow as tf
from PIL import Image, ImageDraw

cwd = os.getcwd()

# constants describing the cells data set.
factor_for_h = 16  # 图片height方向上的划分因子 即行方向上划分出 factor_for_h 个 cell
factor_for_w = 16  # 图片width方向上的划分因子 即列方向上划分出 factor_for_w 个 cell
# RIO Selection. If don't select ROI, set follow four factors to 0.
factor_top_unused = 6  # 不使用图片top的 factor_top_unused 行
factor_bottom_unused = 4  # 不使用图片bottom的 factor_bottom_unused 行
factor_left_unused = 0  # 不使用图片top的 factor_left_unused 列
factor_right_unused = 0  # 不使用图片bottom的 factor_right_unused 列

# variables using for current file.
images_amount_counter = 0  # 图片数量计数器, 手动更改


def generate_subimgs(data_dir, cells_dir, drawlines):
    """
    生成可直接用于制作 tfRecords 的cell
    :param data_dir: 图片文件夹所在的根目录
    :param cells_dir: generate cells in this dir.
    :param drawlines: 是否在原图上画标记以便于打标签
    :return: no return. but generate dirs(cell_dir) in current dir.
    """
    files_path = cwd + "/" + data_dir
    sub_dirs = [x[0] for x in os.walk(files_path)]  # 返回的是绝对路径
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        sub_dir_basename = os.path.basename(sub_dir)
        for img_name in os.listdir(files_path + "/" + sub_dir_basename):
            global images_amount_counter  # 统计处理了多少张图片
            images_amount_counter += 1

            img_path = files_path + "/" + sub_dir_basename + "/" + img_name
            img = Image.open(img_path)

            cell_height = img.height // factor_for_h  # 扁矩形的高
            cell_width = img.width // factor_for_w  # 扁矩形的宽

            for hh in range(factor_top_unused, factor_for_h - factor_bottom_unused):
                for ww in range(factor_left_unused, factor_for_w - factor_right_unused):
                    # crop() returns a rectangular region from this image.
                    # The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
                    box = (
                        cell_width * ww,
                        cell_height * hh,
                        cell_width * ww + cell_width - 1,
                        cell_height * hh + cell_height - 1
                    )
                    cell = img.crop(box)  # cell is one port of img.

                    if not os.path.exists(cells_dir):
                        os.mkdir(cells_dir)
                    if not os.path.exists(cells_dir + "/" + sub_dir_basename):
                        os.mkdir(cells_dir + "/" + sub_dir_basename)
                    if not os.path.exists(cells_dir + "/" + sub_dir_basename + "/" + img_name):
                        os.mkdir(cells_dir + "/" + sub_dir_basename + "/" + img_name)
                        # Store road surface cells
                        os.mkdir(cells_dir + "/" + sub_dir_basename + "/" + img_name + "/road_surface_cells")
                        # Store cluttered cells
                        os.mkdir(cells_dir + "/" + sub_dir_basename + "/" + img_name + "/cluttered_cells")
                        # Store lane cells
                        os.mkdir(cells_dir + "/" + sub_dir_basename + "/" + img_name + "/lane_cells")
                    # Save cell to disk.
                    # And temporarily store to '/road_surface_cells' for manually labeling subsequently.
                    cell.save(
                        cwd + "/" + cells_dir + "/" + sub_dir_basename + "/" + img_name + "/road_surface_cells/" + str(
                            hh) + "_" + str(ww) + ".png")
                    # cell.show()

            if drawlines:
                img1 = img
                draw = ImageDraw.Draw(img1)

                # 在原图上画两条直线标示ROI区域
                # line1 = cell_height * factor_top_unused
                # line2 = img.height - cell_height * factor_bottom_unused
                # draw.line([(0, line1), (img.width, line1)], fill=(255, 0, 0), width=1)
                # draw.line([0, line2, img.width, line2], fill=(0, 255, 0), width=1)

                # 此for循环是为了在原图上生成小方格以及每个cell的坐标
                for hh in range(factor_top_unused, factor_for_h - factor_bottom_unused):
                    for ww in range(factor_left_unused, factor_for_w - factor_right_unused):
                        draw.rectangle(xy=(cell_width * ww, cell_height * hh, cell_width * (ww + 1),
                                           cell_height * (hh + 1)), outline="green")
                        draw.text(xy=[cell_width * ww, cell_height * hh], text=str(hh) + "_" + str(ww),
                                  fill=(255, 0, 0))

                img1.save(cwd + "/" + cells_dir + "/" + sub_dir_basename + "/" + img_name + "/" + img_name)
                # img1.show()


def main(_):
    generate_subimgs(data_dir="dir0_initial_imgs", cells_dir="dir0_unlabeled_cells", drawlines=1)
    print("Imgs amount:", images_amount_counter)


if __name__ == '__main__':
    tf.app.run()
