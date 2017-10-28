import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64, """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_string('data_dir', '/home/ipprdl//www/dog_cat', """Path to the data directory.""")
# tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")

# Constants describing the current file.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 22500

cwd = os.getcwd()


def _generate_image_and_label_batch(result, batch_size, min_queue_examples, shuffle):
    """
    Construct a batch of images and labels.
    :param result:
        result.uint8image: 3-D Tensor of [height, width, 3] of type.float32.
        result.label: 1-D Tensor of type.int32.
    :param batch_size: Number of images per batch.
    :param min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
    :param shuffle: boolean indicating whether to use a shuffling queue.

    :return:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [result.uint8image, result.label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [result.image, result.label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # print(images, label_batch)

    return images, tf.reshape(label_batch, [batch_size])


def generate_batches_from_tfrecords(records_name, image_size):
    """
    Generate batches from tfrecords file.
    :param records_name: The name of tfrecords file.
    :param image_size: image size.
    :return:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """

    class TFRecords(object):
        pass

    result = TFRecords()
    result.height = image_size[0]
    result.width = image_size[1]
    result.depth = image_size[2]

    filenames = [os.path.join(cwd, records_name)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    reader = tf.TFRecordReader()
    result.key, value = reader.read(filename_queue)
    value = tf.parse_single_example(value,
                                    features={
                                        'label': tf.FixedLenFeature([], tf.int64),
                                        'img_raw': tf.FixedLenFeature([], tf.string),
                                    })

    result.label = tf.cast(value['label'], tf.int32)

    # we reshape the img_raw from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.decode_raw(value['img_raw'], tf.uint8), [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    depth_major = tf.transpose(depth_major, [1, 2, 0])
    # print(result.uint8image, result.label)

    result.uint8image = tf.cast(depth_major, tf.float32)

    # Image processing for training the network. Note the many random
    # distortions applied to the image. 此处可以对图像做处理

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.1
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print('Filling queue with %d images before starting to train. This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_batch(result, FLAGS.batch_size, min_queue_examples, shuffle=True)

# def main(unused):
#     generate_batches_from_tfrecords(records_name="ld_train.tfrecords")
#
#
# if __name__ == '__main__':
#     tf.app.run()
