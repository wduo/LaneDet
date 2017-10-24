import os
import tensorflow as tf
from PIL import Image, ImageDraw

cwd = os.getcwd()

# Global constants describing the cells data set.
factor_for_h = 16  # 图片height方向上的划分因子 即行方向上划分出 factor_for_h 个 cell
factor_for_w = 16  # 图片width方向上的划分因子 即列方向上划分出 factor_for_w 个 cell
# RIO Selection. If don't select ROI, set follow four factors to 0.
factor_top_unused = 4  # 不使用图片top的 factor_top_unused 行
factor_bottom_unused = 4  # 不使用图片bottom的 factor_bottom_unused 行
factor_left_unused = 0  # 不使用图片top的 factor_left_unused 列
factor_right_unused = 0  # 不使用图片bottom的 factor_right_unused 列

images_amount_counter = 10  # 图片数量计数器, 手动更改


def _generate_subimgs(data_dir, cells_dir, drawlines):
    """
    生成可直接用于制作 tfRecords 的cell
    :param data_dir: 图片文件所在文件夹
    :param cells_dir: generate cells in this dir.
    :param drawlines: 是否在原图上画两条直线标示ROI区域
    :return: no return. but generate dirs(cell_dir) in current dir.
    """
    files_path = cwd + "/" + data_dir + "/"
    for img_name in os.listdir(files_path):
        img_path = files_path + img_name
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
                if not os.path.exists(cells_dir + "/" + img_name):
                    os.mkdir(cells_dir + "/" + img_name)
                    os.mkdir(cells_dir + "/" + img_name + "/all_cells")
                    os.mkdir(cells_dir + "/" + img_name + "/positive_cells")
                # save cell to disk.
                cell.save(cwd + "/" + cells_dir + "/" + img_name + "/all_cells/" + str(hh) + "_" + str(ww) + ".png")
                # cell.show()

        # 在原图上画两条直线标示ROI区域
        if drawlines:
            img1 = img
            draw = ImageDraw.Draw(img1)
            line1 = cell_height * factor_top_unused
            line2 = img.height - cell_height * factor_bottom_unused
            draw.line([(0, line1), (img.width, line1)], fill=(255, 0, 0), width=1)
            draw.line([0, line2, img.width, line2], fill=(0, 255, 0), width=1)
            img1.save(cwd + "/" + cells_dir + "/" + img_name + "/" + img_name)
            # img1.show()


def main(unused):
    _generate_subimgs(data_dir="emgs", cells_dir="emg_cells", drawlines=1)
    print("Imgs amount:", images_amount_counter)


if __name__ == '__main__':
    tf.app.run()
