import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import arg_scope
from PIL import Image, ImageDraw

from stp0_generate_subimgs import factor_for_h
from stp0_generate_subimgs import factor_for_w
from stp0_generate_subimgs import factor_top_unused
from stp0_generate_subimgs import factor_bottom_unused
from stp0_generate_subimgs import factor_left_unused
from stp0_generate_subimgs import factor_right_unused

from stp5_ldnet_train import images_size
from stp5_ldnet_train import MODEL_SAVE_PATH
from stp5_ldnet_train import NUM_CLASS
from stp5_ldnet_train import MOVING_AVERAGE_DECAY

# import stp4_ldnet
import stp4_ldnet_v1

# constants describing the current file.
EVALUATION_PATH = "/tmp/ldnet/evaluation"


def read_images_as_cells_batch(images_path):
    """
    Read images as cells which dtype is float32, and return all cells as a batch.
    :param images_path: image path.
    :return:
        initial_image: initial image.
        cells_batch_of_one_image: size is [batch, cell_height, cell_width, cell_channel]
    """
    initial_image = Image.open(images_path)

    cell_height = initial_image.height // factor_for_h  # the height of cells
    cell_width = initial_image.width // factor_for_w  # the width of cells

    # the number of cells
    batch_num = (factor_for_h - factor_left_unused - factor_right_unused) * (
        factor_for_w - factor_top_unused - factor_bottom_unused)

    # convert img to array
    array_image = np.array(initial_image)
    # all cells of one image
    cells_batch_of_one_image = []

    for hh in range(factor_top_unused, factor_for_h - factor_bottom_unused):
        for ww in range(factor_left_unused, factor_for_w - factor_right_unused):
            row_start = cell_height * hh
            row_end = cell_height * hh + cell_height
            column_start = cell_width * ww
            column_end = cell_width * ww + cell_width
            cell = array_image[row_start:row_end, column_start:column_end]
            cell = tf.image.resize_images(cell, (images_size[0], images_size[1]))
            cells_batch_of_one_image.append(cell)

    # print(cells_batch_of_one_image)
    cells_batch_of_one_image = tf.reshape(cells_batch_of_one_image,
                                          [-1, images_size[0], images_size[1], images_size[2]])
    print(cells_batch_of_one_image)
    # print(cells_batch_of_one_image.dtype)
    # print(cells_batch_of_one_image.shape)

    # ---------- test start ------------------------------
    # img = Image.fromarray(array_image)
    # img.show()

    # have problems in this code snippet.
    # for ii in range(batch_num):
    #     img = Image.fromarray(int(cells_batch_of_one_image[ii]))
    #     img.show()

    # ---------- test end --------------------------------

    if cells_batch_of_one_image.shape[0] != batch_num:
        print('batch_size and batch_num inconsistent.')

    return initial_image, cells_batch_of_one_image


def evaluate(num_class, images_placeholder_tensor):
    """
    return one cell's final probability value.
    :param num_class:
    :param images_placeholder_tensor:
    :return:
        final_tensor_value: one cell's final probability value which size is [batch_size, num_class].
    """
    with tf.name_scope('input'):
        # the size of images_placeholder: [batch_size, image_width, image_height, image_channel]
        images_placeholder = tf.placeholder(tf.float32, [None, images_size[0], images_size[1], images_size[2]],
                                            name='ImagesPlaceholder')

    # the size of logits: [batch_size, 3]
    # ldnet-v0
    # logits = stp4_ldnet.ldnet(inputs=images_placeholder, num_classes=num_class, print_current_tensor=False)

    # ldnet-v1
    with arg_scope(stp4_ldnet_v1.ldnet_v1_arg_scope()):
        logits = stp4_ldnet_v1.ldnet_v1(inputs=images_placeholder, num_classes=num_class, print_current_tensor=False)

    # the size of final_tensor: [batch_size, 3]
    final_tensor = tf.nn.softmax(logits, name="Prediction")

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        images_placeholder_value = sess.run(images_placeholder_tensor)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/train/ldnet_model.ckpt-0,
            # extract global_step from it.
            # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            # print(global_step)

            final_tensor_value = sess.run(final_tensor, feed_dict={images_placeholder: images_placeholder_value})
            # print(final_tensor_value)
            # print(final_tensor_value.shape)

        else:
            print('No checkpoint file found')

    return final_tensor_value


def show_lane(initial_image, final_tensor_value):
    # print('flourish')

    # show initial image.
    # initial_image.show()

    sess = tf.Session()

    position = 0  # lane's position
    target_cells_counter = 0  # lane's number.
    global_position = []
    for cell_prediction in final_tensor_value:
        position += 1
        # if (cell_prediction[1] >= 0.5) and (cell_prediction[2] < 0.5):
        if sess.run(tf.argmax(cell_prediction)) == 1:
            target_cells_counter += 1
            global_position.append(position + factor_for_w * factor_top_unused)
    print('global_position', global_position)
    print('target_cells_counter', target_cells_counter)

    # the cells' global xy positions.
    global_row = [x // factor_for_w for x in global_position]
    global_column = [x % factor_for_w for x in global_position]
    print('global_row', global_row)
    print('global_column', global_column)

    # draw target cells in mask image.
    mask_image = Image.new("RGB", (initial_image.width, initial_image.height))
    draw_mask = ImageDraw.Draw(mask_image)
    cell_width = initial_image.width // factor_for_w  # the width of cells
    cell_height = initial_image.height // factor_for_h  # the height of cells
    for i in range(target_cells_counter):
        x1_left_margin = cell_width * (global_column[i] - 1)
        x1_top_margin = cell_height * (global_row[i] - 1)
        x2_left_margin = cell_width * global_column[i]
        x2_top_margin = cell_height * global_row[i]
        draw_mask.rectangle(xy=(x1_left_margin, x1_top_margin, x2_left_margin, x2_top_margin), fill=(0, 255, 0))

    # merge two images using blend, and show target cells in initial image.
    lane_cells = Image.blend(initial_image, mask_image, 0.25)
    lane_cells.show()


def main(_):
    initial_image, cells_batch_of_one_image = read_images_as_cells_batch("dir6_initial_imgs/washington1/f00320.png")
    final_tensor_value = evaluate(num_class=NUM_CLASS, images_placeholder_tensor=cells_batch_of_one_image)
    show_lane(initial_image=initial_image, final_tensor_value=final_tensor_value)


if __name__ == '__main__':
    tf.app.run()
