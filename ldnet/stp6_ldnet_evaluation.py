import tensorflow as tf

import stp4_ldnet
from stp5_ldnet_train import images_size
from stp5_ldnet_train import MODEL_SAVE_PATH
from stp5_ldnet_train import NUM_CLASS
from stp5_ldnet_train import MOVING_AVERAGE_DECAY

# constants describing the current file.
EVALUATION_PATH = "/tmp/ldnet/evaluation"


def read_images_as_cell_arrays():
    print('Flourish')


def evaluate(num_class):
    with tf.name_scope('input'):
        # the size of images_placeholder: [batch_size, image_width, image_height, image_channel]
        images_placeholder = tf.placeholder(tf.float32, [None, images_size[0], images_size[1], images_size[2]],
                                            name='ImagesPlaceholder')

    # the size of logits: [batch_size, 3]
    logits = stp4_ldnet.ldnet(inputs=images_placeholder, num_classes=num_class, print_current_tensor=False)

    # the size of final_tensor: [batch_size, 3]
    final_tensor = tf.nn.softmax(logits, name="Prediction")

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/train/ldnet_model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print(global_step)
        else:
            print('No checkpoint file found')
            return


def main(_):
    evaluate(num_class=NUM_CLASS)


if __name__ == '__main__':
    tf.app.run()
